// src/main.rs

use anyhow::{anyhow, Result};
use chrono::Local;
use crossbeam_channel::{bounded, Receiver};
use jamtrack_rs::byte_tracker::ByteTracker;
use jamtrack_rs::{Object, Rect as JamRect};
use ndarray::Array3;
use opencv::{
    core::{self, AlgorithmHint, Mat, Point, Rect, Scalar, Size},
    highgui, imgproc, prelude::*, videoio,
};
use sysinfo::{get_current_pid, ProcessRefreshKind, ProcessesToUpdate, System};
use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
};
use std::time::{Duration, Instant};
use serde::Deserialize;
use ultralytics_inference::{Device, InferenceConfig, YOLOModel};

mod state;
use state::*;
mod mediamtx;
use mediamtx::*;

// entry point
fn main() -> Result<()> {
    process_video()
}

#[derive(Clone, Copy, Debug)]
enum OutputFormat {
    Mkv,
    Mp4,
}

const DEFAULT_IO_BUFFER: usize = 120;

#[derive(Debug)]
struct CliOptions {
    source: Option<String>,
    use_rtsp: bool,
    rtsp_index: Option<usize>,
    output: OutputFormat,
    device: Option<Device>,
    threads: Option<usize>,
    io_buffer: usize,
}

fn parse_args() -> Result<CliOptions> {
    let mut opts = CliOptions {
        source: None,
        use_rtsp: false,
        rtsp_index: None,
        output: OutputFormat::Mkv,
        device: None,
        threads: None,
        io_buffer: DEFAULT_IO_BUFFER,
    };

    let mut iter = env::args().skip(1).peekable();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--rtsp" => opts.use_rtsp = true,
            "--mp4" => opts.output = OutputFormat::Mp4,
            "--mkv" => opts.output = OutputFormat::Mkv,
            "--index" => {
                let idx = iter.next().ok_or_else(|| anyhow!("--index requires a value"))?;
                let parsed = idx.parse::<usize>()
                    .map_err(|_| anyhow!("--index must be a positive integer"))?;
                opts.rtsp_index = Some(parsed.saturating_sub(1));
            }
            "--threads" => {
                let threads = iter.next().ok_or_else(|| anyhow!("--threads requires a value"))?;
                let parsed = threads
                    .parse::<usize>()
                    .map_err(|_| anyhow!("--threads must be a non-negative integer"))?;
                opts.threads = Some(parsed);
            }
            "--buffer" | "--io-buffer" => {
                let buf = iter.next().ok_or_else(|| anyhow!("--buffer requires a value"))?;
                let parsed = buf
                    .parse::<usize>()
                    .map_err(|_| anyhow!("--buffer must be a non-negative integer"))?;
                opts.io_buffer = parsed.max(1);
            }
            "--device" => {
                let device = iter.next().ok_or_else(|| anyhow!("--device requires a value"))?;
                let parsed = Device::from_str(&device)
                    .map_err(|err| anyhow!("Invalid --device value '{device}': {err}"))?;
                opts.device = Some(parsed);
            }
            "--cpu" => {
                opts.device = Some(Device::Cpu);
            }
            "--gpu" => {
                let device = if cfg!(target_os = "macos") {
                    Device::Mps
                } else {
                    Device::Cuda(0)
                };
                opts.device = Some(device);
            }
            _ if arg.starts_with("--") => {
                let rest = arg.trim_start_matches("--");
                if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                    let parsed = rest.parse::<usize>()
                        .map_err(|_| anyhow!("Invalid RTSP index flag: {arg}"))?;
                    opts.rtsp_index = Some(parsed.saturating_sub(1));
                } else if opts.source.is_none() {
                    opts.source = Some(arg);
                }
            }
            _ => {
                if opts.source.is_none() {
                    opts.source = Some(arg);
                }
            }
        }
    }

    Ok(opts)
}

struct CaptureWorker {
    rx: Receiver<CapturedFrame>,
    stop: Arc<AtomicBool>,
    join: JoinHandle<Result<()>>,
}

impl CaptureWorker {
    fn stop(self) -> Result<()> {
        self.stop.store(true, Ordering::Relaxed);
        match self.join.join() {
            Ok(res) => res,
            Err(_) => Err(anyhow!("Capture thread panicked")),
        }
    }
}

struct CapturedFrame {
    frame: Mat,
    idx: usize,
}

fn start_capture_thread(
    mut cap: videoio::VideoCapture,
    stride: usize,
    buffer: usize,
) -> CaptureWorker {
    let (tx, rx) = bounded::<CapturedFrame>(buffer.max(1));
    let stop = Arc::new(AtomicBool::new(false));
    let stop_thread = Arc::clone(&stop);
    let join = thread::spawn(move || -> Result<()> {
        let mut frame = Mat::default();
        let mut idx = 0usize;
        loop {
            if stop_thread.load(Ordering::Relaxed) {
                break;
            }
            if !cap.read(&mut frame)? {
                break;
            }
            if idx % stride != 0 {
                idx += 1;
                continue;
            }
            let owned = frame.try_clone()?;
            let packet = CapturedFrame { frame: owned, idx };
            if tx.try_send(packet).is_err() {
                // Drop newest frame when the buffer is full to avoid blocking I/O.
            }
            idx += 1;
        }
        Ok(())
    });

    CaptureWorker { rx, stop, join }
}

// full pipeline
fn process_video() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data_dir = root.join("data");

    let default_input = data_dir.join("drone8.mp4");
    let rtsp_config = data_dir.join("rtsp_links.yml");

    let rtsp_links = load_rtsp_links(&rtsp_config)?;
    let mut opts = parse_args()?;
    if opts.device.is_none() && cfg!(target_os = "macos") {
        opts.device = Some(Device::Mps);
    }
    let (mut sources, mut source_idx) = match opts.source {
        Some(arg) => {
            if is_rtsp_source(&arg) {
                if let Some(pos) = rtsp_links.iter().position(|s| s == &arg) {
                    (rtsp_links, pos)
                } else {
                    (vec![arg], 0)
                }
            } else {
                (vec![arg], 0)
            }
        }
        None => {
            if opts.use_rtsp {
                if rtsp_links.is_empty() {
                    return Err(anyhow!("No RTSP links found in {}", rtsp_config.display()));
                }
                let idx = opts.rtsp_index.unwrap_or(0).min(rtsp_links.len().saturating_sub(1));
                (rtsp_links, idx)
            } else if !rtsp_links.is_empty() {
                let idx = opts.rtsp_index.unwrap_or(0).min(rtsp_links.len().saturating_sub(1));
                (rtsp_links, idx)
            } else {
                (vec![default_input.to_string_lossy().to_string()], 0)
            }
        }
    };

    sources.retain(|s| !s.trim().is_empty());
    if sources.is_empty() {
        sources.push(default_input.to_string_lossy().to_string());
        source_idx = 0;
    }

    let out_dir = root.join("runs/drone_analysis");
    fs::create_dir_all(&out_dir)?;

    let mut cfg = InferenceConfig::new()
        .with_confidence(CONF_THRESH)
        .with_iou(0.45)
        .with_max_det(500);
    let default_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    cfg = cfg.with_threads(opts.threads.unwrap_or(default_threads));
    if let Some(device) = opts.device.clone() {
        cfg = cfg.with_device(device);
    }
    let overlay_device = opts.device.clone().unwrap_or(Device::Cpu);

    let model_path = data_dir.join("yolov11n-visdrone.onnx");
    if !model_path.exists() {
        return Err(anyhow!("Model not found: {}", model_path.display()));
    }

    let mut model = YOLOModel::load_with_config(model_path, cfg)?;

    highgui::named_window("Drone Analytics", highgui::WINDOW_NORMAL)?;
    let (mut state, cap) = init_source_state(&sources[source_idx], &out_dir, opts.output, overlay_device.clone())?;
    let mut is_rtsp = is_rtsp_source(&sources[source_idx]);
    let mut capture = Some(start_capture_thread(cap, state.stride, opts.io_buffer));

    if is_rtsp {
        highgui::resize_window("Drone Analytics", state.w, state.h)?;
    } else {
        highgui::resize_window("Drone Analytics", 1280, 720)?;
    }

    'source_loop: loop {
        let Some(rx) = capture.as_ref().map(|worker| worker.rx.clone()) else {
            break;
        };
        let mut action: Option<i32> = None;
        while let Ok(packet) = rx.recv() {
            let mut frame = packet.frame;
            let frame_idx = packet.idx;

            let input_arr = mat_to_array3_rgb(&frame)?;
            let results = model.predict_array(&input_arr, String::new())?;

            let mut objects: Vec<Object> = Vec::new();
            let mut classes: Vec<i64> = Vec::new();
            let mut det_confs: Vec<f32> = Vec::new();

            if let Some(r0) = results.get(0) {
                if let Some(boxes) = r0.boxes.as_ref() {
                    let xyxy = boxes.xyxy().to_owned();
                    let conf = boxes.conf().to_owned();
                    let cls = boxes.cls().to_owned();

                    for i in 0..boxes.len() {
                        let cid = cls[i] as i64;
                        if !TARGET_CLASSES.contains(&cid) {
                            continue;
                        }

                        let x1 = xyxy[[i, 0]];
                        let y1 = xyxy[[i, 1]];
                        let x2 = xyxy[[i, 2]];
                        let y2 = xyxy[[i, 3]];

                        if x2 <= x1 || y2 <= y1 {
                            continue;
                        }

                        objects.push(Object::new(
                            JamRect::from_xyxy(x1, y1, x2, y2),
                            conf[i],
                            None,
                        ));
                        classes.push(cid);
                        det_confs.push(conf[i]);
                    }
                }
            }

            let tracked = state.tracker.update(&objects)?;
            let dets = apply_tracks(
                tracked,
                &classes,
                &det_confs,
                &mut state.tracks,
                &mut state.next_local_id,
                frame_idx,
                state.out_fps,
            );

            decay_heatmap(&mut state.heatmap)?;
            update_heatmap(&mut state.heatmap, &dets)?;

            let mut counts = [0usize; 4];
            for d in &dets {
                counts[d.speed.bucket()] += 1;
            }

            let total = dets.len();
            let congestion = if total > 0 {
                let score =
                    counts[0] as f32 * 0.85 +
                    counts[1] as f32 * 0.55 +
                    counts[2] as f32 * 0.25 +
                    counts[3] as f32 * 0.05;

                let base = (score / total as f32) * 100.0;

                let dens = density_factor(&state.heatmap, total)?;
                let clus = cluster_factor(&state.heatmap)?;

                let dens_adj = (dens - 1.0) * 18.0;
                let clus_adj = (clus - 1.0) * 22.0;

                (base + dens_adj + clus_adj).clamp(0.0, 100.0) as i32
            } else {
                0
            };

            let out = &mut frame;
            overlay_heatmap(&state.heatmap, out)?;
            draw_trails(out, &state.tracks)?;
            draw_detections(out, &dets)?;
            draw_source_label(out, &state.source_label)?;
            update_fps(&mut state);
            update_sysinfo(&mut state);
            draw_sysinfo(out, &state.sys_snapshot, &state.device)?;
            let proc_fps = state.fps_value.max(0.0);
            draw_hud(out, congestion, proc_fps, state.fps)?;
            if let Some(pub_) = state.mediamtx.as_mut() {
                pub_.send(out)?;
            }
            if state.writer_active {
                if let Some(limit) = state.write_limit {
                    if state.frames_written >= limit {
                        state.writer.release()?;
                        state.writer_active = false;
                    }
                }
                if state.writer_active {
                    state.writer.write(out)?;
                    state.frames_written += 1;
                }
            }
            highgui::imshow("Drone Analytics", out)?;
            let key = highgui::wait_key(1)?;
            if key == 113 {
                action = Some(key);
                break;
            }
            if key == 116 && sources.len() > 1 {
                action = Some(key);
                break;
            }
        }
        if let Some(key) = action {
            if let Some(worker) = capture.take() {
                worker.stop()?;
            }
            if key == 113 {
                break 'source_loop;
            }
            if key == 116 && sources.len() > 1 {
                source_idx = (source_idx + 1) % sources.len();
                let next_source = sources[source_idx].clone();
                state.writer.release()?;
                let (next_state, cap) = init_source_state(&next_source, &out_dir, opts.output, overlay_device.clone())?;
                state = next_state;
                capture = Some(start_capture_thread(cap, state.stride, opts.io_buffer));
                is_rtsp = is_rtsp_source(&next_source);
                if is_rtsp {
                    highgui::resize_window("Drone Analytics", state.w, state.h)?;
                } else {
                    highgui::resize_window("Drone Analytics", 1280, 720)?;
                }
                continue;
            }
        }
        break;
    }

    state.writer.release()?;
    if let Some(worker) = capture.take() {
        worker.stop()?;
    }
    highgui::destroy_all_windows()?;
    Ok(())
}

// apply tracking
fn apply_tracks(
    objects: Vec<Object>,
    det_classes: &[i64],
    det_confs: &[f32],
    tracks: &mut HashMap<i64, Track>,
    next_local_id: &mut i64,
    frame_idx: usize,
    fps: f64,
) -> Vec<Detection> {
    let mut out = Vec::with_capacity(objects.len());
    let mut used_ids: HashSet<i64> = HashSet::new();

    for (i, obj) in objects.into_iter().enumerate() {
        let [x1, y1, x2, y2] = obj.get_rect().get_xyxy();
        let bbox = BBox { x1, y1, x2, y2 };
        let center = bbox_bottom_center(bbox);

        let mut tid = obj.get_track_id().map(|id| id as i64);
        let det_cid = det_classes.get(i).copied().unwrap_or(0);
        let det_conf = det_confs.get(i).copied().unwrap_or(0.0);

        if tid.is_none() {
            let mut best_id: Option<i64> = None;
            let mut best_dist = ORPHAN_MATCH_DIST;
            for (id, t) in tracks.iter() {
                if used_ids.contains(id) {
                    continue;
                }
                if t.class_id != det_cid {
                    continue;
                }
                if frame_idx.saturating_sub(t.last_seen) > MAX_TRACK_AGE {
                    continue;
                }
                let dx = center.0 - t.last_center.0;
                let dy = center.1 - t.last_center.1;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < best_dist {
                    best_dist = dist;
                    best_id = Some(*id);
                }
            }
            if let Some(id) = best_id {
                tid = Some(id);
            }
        }

        let id = tid.unwrap_or_else(|| {
            let id = *next_local_id;
            *next_local_id -= 1;
            id
        });

        let t = tracks.entry(id).or_insert(Track {
            id,
            bbox,
            last_center: center,
            last_seen: frame_idx,
            trail: Vec::new(),
            class_id: det_cid,
            class_conf: det_conf,
            speed_px_s: 0.0,
        });

        if det_conf >= t.class_conf + 0.05 {
            t.class_id = det_cid;
            t.class_conf = det_conf;
        }

        let dx = center.0 - t.last_center.0;
        let dy = center.1 - t.last_center.1;
        let raw = (dx * dx + dy * dy).sqrt() * fps as f32;

        t.speed_px_s = smooth_speed(t.speed_px_s, raw);
        t.last_center = center;
        t.last_seen = frame_idx;
        t.bbox = bbox;

        t.trail.push(center);
        if t.trail.len() > MAX_TRAIL_LEN {
            t.trail.remove(0);
        }

        used_ids.insert(id);
        let stable_class_id = t.class_id;
        let speed_class = classify_speed(t.speed_px_s);

        out.push(Detection {
            bbox,
            score: obj.get_prob(),
            class_id: stable_class_id,
            track_id: Some(id),
            speed: speed_class,
        });
    }

    tracks.retain(|_, t| frame_idx.saturating_sub(t.last_seen) <= MAX_TRACK_AGE);
    out
}

// decay heatmap
fn decay_heatmap(h: &mut Mat) -> Result<()> {
    let mut tmp = Mat::default();
    core::multiply(h, &Scalar::all(HEATMAP_DECAY as f64), &mut tmp, 1.0, -1)?;
    tmp.copy_to(h)?;
    Ok(())
}

// update heatmap
fn update_heatmap(h: &mut Mat, dets: &[Detection]) -> Result<()> {
    for d in dets {
        let (x, y) = bbox_bottom_center(d.bbox);
        imgproc::circle(
            h,
            Point::new(x as i32, y as i32),
            HEATMAP_RADIUS,
            Scalar::all(2.0),
            -1,
            imgproc::LINE_AA,
            0,
        )?;
    }
    Ok(())
}

// overlay heatmap
fn overlay_heatmap(h: &Mat, frame: &mut Mat) -> Result<()> {
    let mut blur = Mat::default();
    imgproc::gaussian_blur(
        h,
        &mut blur,
        Size::new(31, 31),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut max_val = 0.0;
    core::min_max_loc(&blur, None, Some(&mut max_val), None, None, &core::no_array())?;
    if max_val < 5.0 {
        return Ok(());
    }

    let mut norm = Mat::default();
    core::convert_scale_abs(&blur, &mut norm, 255.0 / max_val, 0.0)?;

    let mut jet = Mat::default();
    imgproc::apply_color_map(&norm, &mut jet, imgproc::COLORMAP_JET)?;

    let mut blended = Mat::default();
    core::add_weighted(frame, 0.55, &jet, 0.45, 0.0, &mut blended, -1)?;
    blended.copy_to(frame)?;
    Ok(())
}

// draw trails
fn draw_trails(frame: &mut Mat, tracks: &HashMap<i64, Track>) -> Result<()> {
    for t in tracks.values() {
        if t.trail.len() < 2 {
            continue;
        }
        for w in t.trail.windows(2) {
            imgproc::line(
                frame,
                Point::new(w[0].0 as i32, w[0].1 as i32),
                Point::new(w[1].0 as i32, w[1].1 as i32),
                Scalar::new(0.0, 0.0, 255.0, 0.0),
                TRAIL_THICKNESS,
                imgproc::LINE_AA,
                0,
            )?;
        }
    }
    Ok(())
}

// draw detections
fn draw_detections(frame: &mut Mat, dets: &[Detection]) -> Result<()> {
    for d in dets {
        let r = Rect::new(
            d.bbox.x1 as i32,
            d.bbox.y1 as i32,
            (d.bbox.x2 - d.bbox.x1).max(2.0) as i32,
            (d.bbox.y2 - d.bbox.y1).max(2.0) as i32,
        );

        let col = class_color(d.class_id);
        imgproc::rectangle(frame, r, col, 1, imgproc::LINE_AA, 0)?;

        let speed = d.speed.as_str();
        let label = if speed.is_empty() {
            match d.track_id {
                Some(id) => format!("#{id} {}", class_name(d.class_id)),
                None => format!("{}", class_name(d.class_id)),
            }
        } else {
            match d.track_id {
                Some(id) => format!("#{id} {} {}", class_name(d.class_id), speed),
                None => format!("{} {}", class_name(d.class_id), speed),
            }
        };

        imgproc::put_text(
            frame,
            &label,
            Point::new(r.x, (r.y - 4).max(0)),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.4,
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;
    }
    Ok(())
}

// draw HUD
fn draw_hud(frame: &mut Mat, pct: i32, proc_fps: f32, src_fps: f64) -> Result<()> {
    let w = frame.cols();
    let margin = 30;
    let label_col = Scalar::new(0.0, 255.0, 255.0, 0.0);

    let title = "CONGESTION INDEX";
    let title_scale = 0.7;
    let title_thickness = 1;

    let title_size = imgproc::get_text_size(
        title,
        imgproc::FONT_HERSHEY_DUPLEX,
        title_scale,
        title_thickness,
        &mut 0,
    )?;

    let title_x = (w - margin - title_size.width).max(0);
    imgproc::put_text(
        frame,
        title,
        Point::new(title_x, 50),
        imgproc::FONT_HERSHEY_DUPLEX,
        title_scale,
        label_col,
        title_thickness,
        imgproc::LINE_AA,
        false,
    )?;

    let pct_text = format!("{pct}%");
    let pct_scale = 2.0;
    let pct_thickness = 3;

    let pct_size = imgproc::get_text_size(
        &pct_text,
        imgproc::FONT_HERSHEY_DUPLEX,
        pct_scale,
        pct_thickness,
        &mut 0,
    )?;

    let pct_x = (w - margin - pct_size.width).max(0);

    let col = value_color_pct(pct as f64, 30.0, 70.0);

    imgproc::put_text(
        frame,
        &pct_text,
        Point::new(pct_x, 120),
        imgproc::FONT_HERSHEY_DUPLEX,
        pct_scale,
        col,
        pct_thickness,
        imgproc::LINE_AA,
        false,
    )?;

    let fps_text = format!("FPS {:.1}", proc_fps.max(0.0));
    let src_fps_text = format!("SRC {:.1}", src_fps.max(0.0));
    draw_label_value(
        frame,
        "FPS ",
        &format!("{:.1}", proc_fps.max(0.0)),
        pct_x,
        150,
        0.6,
        1,
        label_col,
        value_color_pct(proc_fps as f64, 10.0, 30.0),
    )?;
    draw_label_value(
        frame,
        "SRC ",
        &format!("{:.1}", src_fps.max(0.0)),
        pct_x,
        172,
        0.5,
        1,
        label_col,
        value_color_pct(src_fps, 10.0, 30.0),
    )?;

    Ok(())
}

fn update_fps(state: &mut SourceState) {
    state.fps_frames += 1;
    let elapsed = state.fps_last_update.elapsed();
    if elapsed >= Duration::from_secs(1) {
        let secs = elapsed.as_secs_f32().max(0.001);
        state.fps_value = state.fps_frames as f32 / secs;
        state.fps_frames = 0;
        state.fps_last_update = Instant::now();
    }
}

#[derive(Clone, Copy)]
struct SysSnapshot {
    proc_cpu_pct: f32,
    proc_mem: u64,
    num_cpus: usize,
    gpu_util_pct: Option<f32>,
}

fn update_sysinfo(state: &mut SourceState) {
    if state.sys_last_update.elapsed() < Duration::from_secs(1) {
        return;
    }

    if let Some(pid) = state.sys_pid {
        state.sys.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            false,
            ProcessRefreshKind::nothing().with_cpu().with_memory(),
        );
    }

    let gpu_util_pct = if cfg!(target_os = "macos") && matches!(state.device, Device::CoreMl) {
        read_coreml_gpu_util_pct()
    } else {
        None
    };
    state.sys_snapshot = build_sys_snapshot(&state.sys, state.sys_pid, gpu_util_pct);
    state.sys_last_update = Instant::now();
}

fn build_sys_snapshot(
    sys: &System,
    pid: Option<sysinfo::Pid>,
    gpu_util_pct: Option<f32>,
) -> SysSnapshot {
    let (proc_cpu_pct, proc_mem) = pid
        .and_then(|p| sys.process(p))
        .map(|proc| (proc.cpu_usage(), proc.memory()))
        .unwrap_or((0.0, 0));

    SysSnapshot {
        proc_cpu_pct,
        proc_mem,
        num_cpus: sys.cpus().len(),
        gpu_util_pct,
    }
}

fn draw_sysinfo(frame: &mut Mat, snap: &SysSnapshot, device: &Device) -> Result<()> {
    let x = 30;
    let y0 = 60;
    let line = 22;
    let scale = 0.55;
    let thickness = 1;
    let label_col = Scalar::new(0.0, 255.0, 255.0, 0.0);

    draw_label_value(
        frame,
        "DEVICE ",
        &format_device_label(device),
        x,
        y0,
        scale,
        thickness,
        label_col,
        label_col,
    )?;

    let cpu_val = if snap.num_cpus > 1 {
        format!("{:>4.1}% ({}c)", snap.proc_cpu_pct.max(0.0), snap.num_cpus)
    } else {
        format!("{:>4.1}%", snap.proc_cpu_pct.max(0.0))
    };
    draw_label_value(
        frame,
        "PROC ",
        &cpu_val,
        x,
        y0 + line,
        scale,
        thickness,
        label_col,
        value_color_pct(snap.proc_cpu_pct as f64, 35.0, 75.0),
    )?;

    let mut row = 2;
    if !matches!(device, Device::Cpu) {
        let gpu_text = match snap.gpu_util_pct {
            Some(val) => {
                draw_label_value(
                    frame,
                    "GPU ",
                    &format!("{:>4.1}%", val.max(0.0)),
                    x,
                    y0 + (row as i32 * line),
                    scale,
                    thickness,
                    label_col,
                    value_color_pct(val as f64, 35.0, 75.0),
                )?;
                None
            }
            None => Some("N/A".to_string()),
        };
        if let Some(na) = gpu_text {
            draw_label_value(
                frame,
                "GPU ",
                &na,
                x,
                y0 + (row as i32 * line),
                scale,
                thickness,
                label_col,
                label_col,
            )?;
        }
        row += 1;
    }

    let mem_gb = bytes_to_gb(snap.proc_mem);
    draw_label_value(
        frame,
        "RAM ",
        &format!("{:.2} GB", mem_gb),
        x,
        y0 + (row as i32 * line),
        scale,
        thickness,
        label_col,
        value_color_pct(mem_gb, 2.0, 8.0),
    )?;
    Ok(())
}

fn value_color_pct(value: f64, low: f64, high: f64) -> Scalar {
    if value >= high {
        Scalar::new(0.0, 0.0, 255.0, 0.0)
    } else if value <= low {
        Scalar::new(0.0, 255.0, 0.0, 0.0)
    } else {
        Scalar::new(0.0, 165.0, 255.0, 0.0)
    }
}

fn draw_label_value(
    frame: &mut Mat,
    label: &str,
    value: &str,
    x: i32,
    y: i32,
    scale: f64,
    thickness: i32,
    label_col: Scalar,
    value_col: Scalar,
) -> Result<()> {
    let mut base = 0;
    let label_size = imgproc::get_text_size(
        label,
        imgproc::FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
        &mut base,
    )?;
    imgproc::put_text(
        frame,
        label,
        Point::new(x, y),
        imgproc::FONT_HERSHEY_SIMPLEX,
        scale,
        label_col,
        thickness,
        imgproc::LINE_AA,
        false,
    )?;
    imgproc::put_text(
        frame,
        value,
        Point::new(x + label_size.width, y),
        imgproc::FONT_HERSHEY_SIMPLEX,
        scale,
        value_col,
        thickness,
        imgproc::LINE_AA,
        false,
    )?;
    Ok(())
}

fn format_device_label(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".to_string(),
        Device::Cuda(idx) => format!("CUDA:{idx}"),
        Device::Mps => "MPS".to_string(),
        Device::CoreMl => "COREML".to_string(),
        Device::DirectMl(idx) => format!("DIRECTML:{idx}"),
        Device::OpenVino => "OPENVINO".to_string(),
        Device::Xnnpack => "XNNPACK".to_string(),
        Device::TensorRt(idx) => format!("TENSORRT:{idx}"),
        Device::Rocm(idx) => format!("ROCM:{idx}"),
    }
}

fn bytes_to_gb(bytes: u64) -> f64 {
    bytes as f64 / 1_073_741_824.0
}

fn read_coreml_gpu_util_pct() -> Option<f32> {
    let output = Command::new("sudo")
        .args([
            "-n",
            "powermetrics",
            "--samplers",
            "gpu_power",
            "-n",
            "1",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_powermetrics_gpu_util(&stdout)
}

fn parse_powermetrics_gpu_util(text: &str) -> Option<f32> {
    for line in text.lines() {
        let lower = line.to_lowercase();
        if !(lower.contains("gpu") && lower.contains("active")) {
            continue;
        }
        if let Some((val, has_pct)) = extract_first_number(line) {
            let pct = if has_pct {
                val
            } else if val <= 1.0 {
                val * 100.0
            } else {
                val
            };
            return Some(pct);
        }
    }
    None
}

fn extract_first_number(line: &str) -> Option<(f32, bool)> {
    let mut buf = String::new();
    let mut has_digit = false;
    let mut has_dot = false;
    let mut has_pct = false;

    for ch in line.chars() {
        if ch.is_ascii_digit() {
            has_digit = true;
            buf.push(ch);
            continue;
        }
        if ch == '.' && !has_dot {
            has_dot = true;
            buf.push(ch);
            continue;
        }
        if ch == '%' && has_digit {
            has_pct = true;
            break;
        }
        if has_digit {
            break;
        }
    }

    if !has_digit {
        return None;
    }

    let val = buf.parse::<f32>().ok()?;
    Some((val, has_pct))
}

// convert Mat -> RGB ndarray
fn mat_to_array3_rgb(mat: &Mat) -> Result<Array3<u8>> {
    let mut rgb = Mat::default();
    imgproc::cvt_color(
        mat,
        &mut rgb,
        imgproc::COLOR_BGR2RGB,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let rows = rgb.rows() as usize;
    let cols = rgb.cols() as usize;

    let data = rgb.data_bytes()?.to_vec();
    Ok(Array3::from_shape_vec((rows, cols, 3), data)?)
}

// color by class
fn class_color(class_id: i64) -> Scalar {
    match class_id {
        3 => Scalar::new(20.0, 190.0, 255.0, 0.0),    // car (orange)
        4 => Scalar::new(255.0, 120.0, 40.0, 0.0),    // van (blue)
        5 => Scalar::new(60.0, 255.0, 120.0, 0.0),    // truck (green)
        7 => Scalar::new(255.0, 220.0, 50.0, 0.0),    // awning-tricycle (gold)
        8 => Scalar::new(255.0, 70.0, 200.0, 0.0),    // bus (pink)
        9 => Scalar::new(80.0, 220.0, 60.0, 0.0),     // motor (lime)
        _ => Scalar::new(210.0, 210.0, 210.0, 0.0),
    }
}

struct SourceState {
    fps: f64,
    w: i32,
    h: i32,
    stride: usize,
    out_fps: f64,
    writer: videoio::VideoWriter,
    writer_active: bool,
    write_limit: Option<usize>,
    frames_written: usize,
    tracker: ByteTracker,
    tracks: HashMap<i64, Track>,
    next_local_id: i64,
    heatmap: Mat,
    source_label: String,
    sys: System,
    sys_pid: Option<sysinfo::Pid>,
    sys_last_update: Instant,
    sys_snapshot: SysSnapshot,
    fps_last_update: Instant,
    fps_frames: usize,
    fps_value: f32,
    mediamtx: Option<RtspPublisher>,
    device: Device,
}

fn init_source_state(
    source: &str,
    out_dir: &Path,
    output: OutputFormat,
    device: Device,
) -> Result<(SourceState, videoio::VideoCapture)> {
    // Drop corrupt frames for FFmpeg backend (helps with damaged MP4/RTSP streams).
    // SAFETY: We only set a process-wide env var before opening the capture.
    unsafe {
        if is_rtsp_source(source) {
            std::env::set_var(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                "rtsp_transport;tcp|fflags;discardcorrupt|err_detect;ignore_err",
            );
        } else {
            std::env::set_var(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                "fflags;discardcorrupt|err_detect;ignore_err",
            );
        }
    }
    let cap = videoio::VideoCapture::from_file(source, videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        return Err(anyhow!("Could not open video"));
    }

    let fps = cap.get(videoio::CAP_PROP_FPS)?;
    if fps <= 0.0 {
        return Err(anyhow!("Invalid FPS"));
    }

let w = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
let h = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;

    let mediamtx = if is_rtsp_source(source) {
        let publish_url = std::env::var("MEDIAMTX_PUBLISH_URL")
            .unwrap_or_else(|_| "rtsp://127.0.0.1:8554/analytics".to_string());
        Some(start_rtsp_publisher(w, h, fps, &publish_url)?)
    } else {
        None
    };



    let stride = 1usize;
    let out_fps = fps;

    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let (out_path, codecs, err_label) = match output {
        OutputFormat::Mp4 => (
            out_dir.join(format!("drone_analysis_{timestamp}.mp4")),
            vec![
                videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?,
                videoio::VideoWriter::fourcc('a', 'v', 'c', '1')?,
                videoio::VideoWriter::fourcc('H', '2', '6', '4')?,
                videoio::VideoWriter::fourcc('X', '2', '6', '4')?,
            ],
            "mp4",
        ),
        OutputFormat::Mkv => (
            out_dir.join(format!("drone_analysis_{timestamp}.mkv")),
            vec![
                videoio::VideoWriter::fourcc('H', '2', '6', '4')?,
                videoio::VideoWriter::fourcc('X', '2', '6', '4')?,
                videoio::VideoWriter::fourcc('a', 'v', 'c', '1')?,
                videoio::VideoWriter::fourcc('M', 'J', 'P', 'G')?,
            ],
            "mkv",
        ),
    };

    let writer = {
        let mut wtr = None;
        for codec in codecs {
            let candidate = videoio::VideoWriter::new(
                out_path.to_str().unwrap(),
                codec,
                out_fps,
                Size::new(w, h),
                true,
            )?;
            if candidate.is_opened()? {
                wtr = Some(candidate);
                break;
            }
        }
        wtr.ok_or_else(|| anyhow!("VideoWriter failed to open ({err_label})."))?
    };

    let write_limit = if is_rtsp_source(source) {
        Some((out_fps * RTSP_SAVE_SECONDS).round().max(1.0) as usize)
    } else {
        None
    };

    let tracker = ByteTracker::new(
        out_fps.round().max(1.0) as usize,
        TRACK_BUFFER,
        TRACK_THRESH,
        HIGH_THRESH,
        MATCH_THRESH,
    );

    let mut sys = System::new_all();
    let sys_pid = get_current_pid().ok();
    if let Some(pid) = sys_pid {
        sys.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            false,
            ProcessRefreshKind::nothing().with_cpu().with_memory(),
        );
    }
    let gpu_util_pct = if cfg!(target_os = "macos") && matches!(device, Device::CoreMl) {
        read_coreml_gpu_util_pct()
    } else {
        None
    };
    let sys_snapshot = build_sys_snapshot(&sys, sys_pid, gpu_util_pct);

    Ok((SourceState {
        fps,
        w,
        h,
        stride,
        out_fps,
        writer,
        writer_active: true,
        write_limit,
        frames_written: 0,
        tracker,
        tracks: HashMap::new(),
        next_local_id: -1,
        heatmap: Mat::zeros(h, w, core::CV_32F)?.to_mat()?,
        source_label: source_display_name(source),
        sys,
        sys_pid,
        sys_last_update: Instant::now(),
        sys_snapshot,
        fps_last_update: Instant::now(),
        fps_frames: 0,
        fps_value: 0.0,
        mediamtx,
        device,
    }, cap))
}


fn draw_source_label(frame: &mut Mat, label: &str) -> Result<()> {
    let title = if label.is_empty() { "source" } else { label };
    let scale = 0.7;
    let thickness = 2;
    imgproc::put_text(
        frame,
        title,
        Point::new(30, 30),
        imgproc::FONT_HERSHEY_DUPLEX,
        scale,
        Scalar::all(255.0),
        thickness,
        imgproc::LINE_AA,
        false,
    )?;
    Ok(())
}

fn source_display_name(source: &str) -> String {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if is_rtsp_source(trimmed) {
        let base = trimmed.split('?').next().unwrap_or(trimmed);
        let last = base.rsplit('/').next().unwrap_or(base);
        return last.to_string();
    }

    let base = trimmed.split('?').next().unwrap_or(trimmed);
    Path::new(base)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(base)
        .to_string()
}

// class name mapping
fn class_name(cid: i64) -> &'static str {
    match cid {
        3 => "car",
        4 => "van",
        5 => "truck",
        7 => "awning-tricycle",
        8 => "bus",
        9 => "motor",
        _ => "obj",
    }
}
fn density_factor(heatmap: &Mat, obj_count: usize) -> Result<f32> {
    if obj_count == 0 {
        return Ok(1.0);
    }

    let sum = core::sum_elems(heatmap)?;
    let area = (heatmap.rows() * heatmap.cols()) as f32;

    let heat_density = (sum[0] as f32 / area).min(5.0);
    let obj_density = (obj_count as f32 / 40.0).min(2.0);

    Ok((0.7 + heat_density + obj_density).clamp(0.6, 2.2))
}

fn cluster_factor(heatmap: &Mat) -> Result<f32> {
    let mut blur = Mat::default();
    imgproc::gaussian_blur(
        heatmap,
        &mut blur,
        Size::new(31, 31),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut max_val = 0.0;
    core::min_max_loc(&blur, None, Some(&mut max_val), None, None, &core::no_array())?;

    let sum = core::sum_elems(heatmap)?;
    if sum[0] < 1.0 {
        return Ok(1.0);
    }

    if max_val < 1.0 {
        return Ok(1.0);
    }

    let mut mask = Mat::default();
    core::compare(&blur, &Scalar::all(max_val * 0.6), &mut mask, core::CMP_GT)?;
    let hot = core::count_non_zero(&mask)? as f32;
    let area = (blur.rows() * blur.cols()) as f32;
    let frac = (hot / area).clamp(0.0001, 1.0);

    let compactness = (1.0 - frac).clamp(0.0, 1.0);
    Ok((1.0 + compactness * 0.6).clamp(0.9, 1.6))
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RtspYaml {
    List(Vec<String>),
    Map { rtsp_links: Vec<String> },
}

fn load_rtsp_links(path: &Path) -> Result<Vec<String>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = fs::read_to_string(path)?;
    let parsed: RtspYaml = serde_yaml::from_str(&raw)?;
    Ok(match parsed {
        RtspYaml::List(list) => list,
        RtspYaml::Map { rtsp_links } => rtsp_links,
    })
}

fn is_rtsp_source(source: &str) -> bool {
    let s = source.trim();
    s.starts_with("rtsp://") || s.starts_with("rtsps://")
}
