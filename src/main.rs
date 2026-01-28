// src/main.rs

use anyhow::{anyhow, Result};
use chrono::Local;
use jamtrack_rs::byte_tracker::ByteTracker;
use jamtrack_rs::{Object, Rect as JamRect};
use opencv::{
    core::{self, Mat, Point, Scalar, Size},
    highgui, imgproc, prelude::*, videoio,
};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread::JoinHandle,
};
use std::time::Instant;
use std::sync::atomic::AtomicBool;
use sysinfo::System;
use ultralytics_inference::{Device, InferenceConfig, YOLOModel};

mod analytics;
use analytics::*;
mod helpers;
use helpers::*;
mod state;
use state::*;
mod mediamtx;
use mediamtx::*;

// entry point
fn main() -> Result<()> {
    process_video()
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
            let (
                congestion,
                traffic_density_pct,
                mobility_index_pct,
                stalled_pct,
                slow_pct,
                medium_pct,
                fast_pct,
            ) = if total > 0 {
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

                let congestion = (base + dens_adj + clus_adj).clamp(0.0, 100.0) as i32;

                let slow_ratio = (counts[0] + counts[1]) as f32 / total as f32;
                let dens_norm = normalize_density_factor(dens);
                let traffic_density =
                    (0.6 * dens_norm + 0.4 * slow_ratio).clamp(0.0, 1.0) * 100.0;
                let traffic_density_pct = traffic_density.round() as i32;

                let mobility_raw = (counts[0] as f32 * 0.0
                    + counts[1] as f32 * 0.35
                    + counts[2] as f32 * 0.7
                    + counts[3] as f32 * 1.0)
                    / total as f32;
                let mobility_index_pct = (mobility_raw.clamp(0.0, 1.0) * 100.0).round() as i32;

                let stalled_pct = ((counts[0] as f32 / total as f32) * 100.0).round() as i32;
                let slow_pct = ((counts[1] as f32 / total as f32) * 100.0).round() as i32;
                let medium_pct = ((counts[2] as f32 / total as f32) * 100.0).round() as i32;
                let fast_pct = ((counts[3] as f32 / total as f32) * 100.0).round() as i32;

                (
                    congestion,
                    traffic_density_pct,
                    mobility_index_pct,
                    stalled_pct,
                    slow_pct,
                    medium_pct,
                    fast_pct,
                )
            } else {
                (0, 0, 0, 0, 0, 0, 0)
            };

            let out = &mut frame;
            overlay_heatmap(&state.heatmap, out)?;
            draw_trails(out, &state.tracks)?;
            draw_detections(out, &dets)?;
            draw_source_label(out, &state.source_label)?;
            update_fps(&mut state.fps_frames, &mut state.fps_last_update, &mut state.fps_value);
            update_sysinfo(
                &mut state.sys,
                state.sys_pid,
                &mut state.sys_last_update,
                &mut state.sys_snapshot,
                &state.gpu_util_shared,
            );
            draw_sysinfo(out, &state.sys_snapshot, &state.device)?;
            let proc_fps = state.fps_value.max(0.0);
            draw_hud(
                out,
                congestion,
                traffic_density_pct,
                mobility_index_pct,
                stalled_pct,
                slow_pct,
                medium_pct,
                fast_pct,
                proc_fps,
                state.fps,
            )?;
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
                stop_gpu_poller(
                    &mut state.gpu_poll_stop,
                    &mut state.gpu_poll_join,
                    &mut state.gpu_util_shared,
                );
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

    stop_gpu_poller(
        &mut state.gpu_poll_stop,
        &mut state.gpu_poll_join,
        &mut state.gpu_util_shared,
    );
    state.writer.release()?;
    if let Some(worker) = capture.take() {
        worker.stop()?;
    }
    highgui::destroy_all_windows()?;
    Ok(())
}

// apply tracking
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
    gpu_util_shared: Option<Arc<Mutex<Option<f32>>>>,
    gpu_poll_stop: Option<Arc<AtomicBool>>,
    gpu_poll_join: Option<JoinHandle<()>>,
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

    let (sys, sys_pid, sys_last_update, sys_snapshot, gpu_util_shared, gpu_poll_stop, gpu_poll_join) =
        init_system_snapshot(&device);

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
        sys_last_update,
        sys_snapshot,
        gpu_util_shared,
        gpu_poll_stop,
        gpu_poll_join,
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
