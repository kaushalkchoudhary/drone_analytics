use anyhow::Result;
use jamtrack_rs::Object;
use opencv::{
    core::{self, AlgorithmHint, Mat, Point, Rect, Scalar, Size},
    imgproc,
    prelude::MatTraitConst,
};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};
use ultralytics_inference::Device;

use crate::state::{
    bbox_bottom_center, classify_speed, smooth_speed, BBox, Detection, Track, HEATMAP_DECAY,
    HEATMAP_RADIUS, MAX_TRACK_AGE, MAX_TRAIL_LEN, ORPHAN_MATCH_DIST,
};

// ================== TRACKING ==================

pub fn apply_tracks(
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

// ================== HEATMAP ==================

pub fn decay_heatmap(h: &mut Mat) -> Result<()> {
    let mut tmp = Mat::default();
    core::multiply(h, &Scalar::all(HEATMAP_DECAY as f64), &mut tmp, 1.0, -1)?;
    tmp.copy_to(h)?;
    Ok(())
}

pub fn update_heatmap(h: &mut Mat, dets: &[Detection]) -> Result<()> {
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

pub fn overlay_heatmap(h: &Mat, frame: &mut Mat) -> Result<()> {
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

// ================== HUD ==================

pub fn draw_trails(frame: &mut Mat, tracks: &HashMap<i64, Track>) -> Result<()> {
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
                crate::state::TRAIL_THICKNESS,
                imgproc::LINE_AA,
                0,
            )?;
        }
    }
    Ok(())
}

pub fn draw_detections(frame: &mut Mat, dets: &[Detection]) -> Result<()> {
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

pub fn draw_hud(
    frame: &mut Mat,
    congestion_pct: i32,
    traffic_density_pct: i32,
    mobility_index_pct: i32,
    stalled_pct: i32,
    slow_pct: i32,
    medium_pct: i32,
    fast_pct: i32,
    proc_fps: f32,
    src_fps: f64,
) -> Result<()> {
    let w = frame.cols();
    let margin = 30;
    let label_col = Scalar::new(230.0, 230.0, 230.0, 0.0);
    let shadow_col = Scalar::new(0.0, 0.0, 0.0, 0.0);

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
    draw_text_with_shadow(
        frame,
        title,
        Point::new(title_x, 50),
        imgproc::FONT_HERSHEY_SIMPLEX,
        title_scale,
        label_col,
        shadow_col,
        title_thickness,
    )?;

    let pct_text = format!("{congestion_pct}%");
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

    let col = value_color_pct(congestion_pct as f64, 30.0, 70.0);

    draw_text_with_shadow(
        frame,
        &pct_text,
        Point::new(pct_x, 120),
        imgproc::FONT_HERSHEY_SIMPLEX,
        pct_scale,
        col,
        shadow_col,
        pct_thickness,
    )?;

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

    let density_title = "TRAFFIC DENSITY";
    let density_title_size = imgproc::get_text_size(
        density_title,
        imgproc::FONT_HERSHEY_DUPLEX,
        0.6,
        1,
        &mut 0,
    )?;
    let density_title_x = (w - margin - density_title_size.width).max(0);
    draw_text_with_shadow(
        frame,
        density_title,
        Point::new(density_title_x, 200),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        label_col,
        shadow_col,
        1,
    )?;

    let density_text = format!("{traffic_density_pct}%");
    let density_col = value_color_pct(traffic_density_pct as f64, 30.0, 70.0);
    draw_text_with_shadow(
        frame,
        &density_text,
        Point::new(density_title_x, 228),
        imgproc::FONT_HERSHEY_SIMPLEX,
        1.2,
        density_col,
        shadow_col,
        2,
    )?;

    let mobility_title = "MOBILITY INDEX";
    let mobility_title_size = imgproc::get_text_size(
        mobility_title,
        imgproc::FONT_HERSHEY_DUPLEX,
        0.6,
        1,
        &mut 0,
    )?;
    let mobility_title_x = (w - margin - mobility_title_size.width).max(0);
    draw_text_with_shadow(
        frame,
        mobility_title,
        Point::new(mobility_title_x, 258),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        label_col,
        shadow_col,
        1,
    )?;

    let mobility_text = format!("{mobility_index_pct}%");
    let mobility_col = value_color_pct(mobility_index_pct as f64, 40.0, 75.0);
    draw_text_with_shadow(
        frame,
        &mobility_text,
        Point::new(mobility_title_x, 286),
        imgproc::FONT_HERSHEY_SIMPLEX,
        1.2,
        mobility_col,
        shadow_col,
        2,
    )?;

    let breakdown_y = 318;
    let row = 18;
    draw_label_value(
        frame,
        "STALLED ",
        &format!("{stalled_pct}%"),
        mobility_title_x,
        breakdown_y,
        0.5,
        1,
        label_col,
        Scalar::new(0.0, 70.0, 255.0, 0.0),
    )?;
    draw_label_value(
        frame,
        "SLOW ",
        &format!("{slow_pct}%"),
        mobility_title_x,
        breakdown_y + row,
        0.5,
        1,
        label_col,
        Scalar::new(0.0, 165.0, 255.0, 0.0),
    )?;
    draw_label_value(
        frame,
        "MEDIUM ",
        &format!("{medium_pct}%"),
        mobility_title_x,
        breakdown_y + row * 2,
        0.5,
        1,
        label_col,
        Scalar::new(0.0, 215.0, 255.0, 0.0),
    )?;
    draw_label_value(
        frame,
        "FAST ",
        &format!("{fast_pct}%"),
        mobility_title_x,
        breakdown_y + row * 3,
        0.5,
        1,
        label_col,
        Scalar::new(0.0, 210.0, 120.0, 0.0),
    )?;

    Ok(())
}

fn draw_text_with_shadow(
    frame: &mut Mat,
    text: &str,
    org: Point,
    font_face: i32,
    scale: f64,
    color: Scalar,
    shadow: Scalar,
    thickness: i32,
) -> Result<()> {
    let shadow_org = Point::new(org.x + 1, org.y + 1);
    imgproc::put_text(
        frame,
        text,
        shadow_org,
        font_face,
        scale,
        shadow,
        thickness + 1,
        imgproc::LINE_AA,
        false,
    )?;
    imgproc::put_text(
        frame,
        text,
        org,
        font_face,
        scale,
        color,
        thickness,
        imgproc::LINE_AA,
        false,
    )?;
    Ok(())
}

pub fn value_color_pct(value: f64, low: f64, high: f64) -> Scalar {
    if value >= high {
        Scalar::new(0.0, 0.0, 255.0, 0.0)
    } else if value <= low {
        Scalar::new(0.0, 255.0, 0.0, 0.0)
    } else {
        Scalar::new(0.0, 165.0, 255.0, 0.0)
    }
}

pub fn draw_label_value(
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

pub fn class_color(class_id: i64) -> Scalar {
    match class_id {
        3 => Scalar::new(20.0, 190.0, 255.0, 0.0),
        4 => Scalar::new(255.0, 120.0, 40.0, 0.0),
        5 => Scalar::new(60.0, 255.0, 120.0, 0.0),
        7 => Scalar::new(255.0, 220.0, 50.0, 0.0),
        8 => Scalar::new(255.0, 70.0, 200.0, 0.0),
        9 => Scalar::new(80.0, 220.0, 60.0, 0.0),
        _ => Scalar::new(210.0, 210.0, 210.0, 0.0),
    }
}

pub fn class_name(cid: i64) -> &'static str {
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

// ================== STATS ==================

pub fn update_fps(fps_frames: &mut usize, fps_last_update: &mut Instant, fps_value: &mut f32) {
    *fps_frames += 1;
    let elapsed = fps_last_update.elapsed();
    if elapsed >= Duration::from_secs(1) {
        let secs = elapsed.as_secs_f32().max(0.001);
        *fps_value = *fps_frames as f32 / secs;
        *fps_frames = 0;
        *fps_last_update = Instant::now();
    }
}

pub fn density_factor(heatmap: &Mat, obj_count: usize) -> Result<f32> {
    if obj_count == 0 {
        return Ok(1.0);
    }

    let sum = core::sum_elems(heatmap)?;
    let area = (heatmap.rows() * heatmap.cols()) as f32;

    let heat_density = (sum[0] as f32 / area).min(5.0);
    let obj_density = (obj_count as f32 / 40.0).min(2.0);

    Ok((0.7 + heat_density + obj_density).clamp(0.6, 2.2))
}

pub fn normalize_density_factor(dens: f32) -> f32 {
    ((dens - 1.0) / 1.2).clamp(0.0, 1.0)
}

pub fn cluster_factor(heatmap: &Mat) -> Result<f32> {
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

// ================== SYSINFO ==================

#[derive(Clone, Copy)]
pub struct SysSnapshot {
    pub proc_cpu_pct: f32,
    pub proc_mem: u64,
    pub num_cpus: usize,
    pub gpu_util_pct: Option<f32>,
}

pub fn update_sysinfo(
    sys: &mut System,
    sys_pid: Option<sysinfo::Pid>,
    sys_last_update: &mut Instant,
    sys_snapshot: &mut SysSnapshot,
    gpu_util_shared: &Option<std::sync::Arc<std::sync::Mutex<Option<f32>>>>,
) {
    if sys_last_update.elapsed() < Duration::from_secs(1) {
        return;
    }

    if let Some(pid) = sys_pid {
        sys.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            false,
            ProcessRefreshKind::nothing().with_cpu().with_memory(),
        );
    }

    let gpu_util_pct = gpu_util_shared
        .as_ref()
        .and_then(|shared| shared.lock().ok().and_then(|v| *v));
    *sys_snapshot = build_sys_snapshot(sys, sys_pid, gpu_util_pct);
    *sys_last_update = Instant::now();
}

pub fn build_sys_snapshot(
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

pub fn draw_sysinfo(frame: &mut Mat, snap: &SysSnapshot, device: &Device) -> Result<()> {
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

pub fn format_device_label(device: &Device) -> String {
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

pub fn bytes_to_gb(bytes: u64) -> f64 {
    bytes as f64 / 1_073_741_824.0
}

pub fn read_coreml_gpu_util_pct() -> Option<f32> {
    let output = std::process::Command::new("sudo")
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

pub fn start_gpu_poll_thread(
) -> (
    std::sync::Arc<std::sync::Mutex<Option<f32>>>,
    std::sync::Arc<std::sync::atomic::AtomicBool>,
    std::thread::JoinHandle<()>,
) {
    let shared = std::sync::Arc::new(std::sync::Mutex::new(None));
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let shared_thread = std::sync::Arc::clone(&shared);
    let stop_thread = std::sync::Arc::clone(&stop);
    let join = std::thread::spawn(move || {
        while !stop_thread.load(std::sync::atomic::Ordering::Relaxed) {
            let val = read_coreml_gpu_util_pct();
            if let Ok(mut guard) = shared_thread.lock() {
                *guard = val;
            }
            for _ in 0..10 {
                if stop_thread.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    });
    (shared, stop, join)
}

pub fn stop_gpu_poller(
    gpu_poll_stop: &mut Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    gpu_poll_join: &mut Option<std::thread::JoinHandle<()>>,
    gpu_util_shared: &mut Option<std::sync::Arc<std::sync::Mutex<Option<f32>>>>,
) {
    if let Some(stop) = gpu_poll_stop.take() {
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    if let Some(join) = gpu_poll_join.take() {
        let _ = join.join();
    }
    *gpu_util_shared = None;
}

pub fn parse_powermetrics_gpu_util(text: &str) -> Option<f32> {
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

pub fn extract_first_number(line: &str) -> Option<(f32, bool)> {
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

pub fn init_system_snapshot(
    device: &Device,
) -> (
    System,
    Option<sysinfo::Pid>,
    Instant,
    SysSnapshot,
    Option<std::sync::Arc<std::sync::Mutex<Option<f32>>>>,
    Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    Option<std::thread::JoinHandle<()>>,
) {
    let mut sys = System::new_all();
    let sys_pid = sysinfo::get_current_pid().ok();
    if let Some(pid) = sys_pid {
        sys.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            false,
            ProcessRefreshKind::nothing().with_cpu().with_memory(),
        );
    }

    let (gpu_util_shared, gpu_poll_stop, gpu_poll_join, gpu_util_pct) =
        if cfg!(target_os = "macos") && matches!(device, Device::CoreMl) {
            let (shared, stop, join) = start_gpu_poll_thread();
            let initial = shared.lock().ok().and_then(|v| *v);
            (Some(shared), Some(stop), Some(join), initial)
        } else {
            (None, None, None, None)
        };
    let sys_snapshot = build_sys_snapshot(&sys, sys_pid, gpu_util_pct);

    (
        sys,
        sys_pid,
        Instant::now(),
        sys_snapshot,
        gpu_util_shared,
        gpu_poll_stop,
        gpu_poll_join,
    )
}
