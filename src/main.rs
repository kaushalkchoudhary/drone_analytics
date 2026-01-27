// src/main.rs

use anyhow::{anyhow, Result};
use chrono::Local;
use jamtrack_rs::byte_tracker::ByteTracker;
use jamtrack_rs::{Object, Rect as JamRect};
use ndarray::Array3;
use opencv::{
    core::{self, AlgorithmHint, Mat, Point, Rect, Scalar, Size},
    highgui, imgproc, prelude::*, videoio,
};
use std::{collections::{HashMap, HashSet}, env, fs, path::{Path, PathBuf}};
use serde::Deserialize;
use ultralytics_inference::{InferenceConfig, YOLOModel};

mod state;
use state::*;

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

    let input_arg = env::args().nth(1);
    let input_source = match input_arg {
        Some(arg) => arg,
        None => {
            let rtsp_links = load_rtsp_links(&rtsp_config)?;
            if let Some(first) = rtsp_links.into_iter().find(|s| !s.trim().is_empty()) {
                first
            } else {
                default_input.to_string_lossy().to_string()
            }
        }
    };

    let is_rtsp = is_rtsp_source(&input_source);
    let input_path = PathBuf::from(&input_source);
    if !is_rtsp && !input_path.exists() {
        return Err(anyhow!("Input not found: {}", input_path.display()));
    }

    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let out_dir = root.join("runs/drone_analysis");
    fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join(format!("drone_analysis_{timestamp}.mp4"));

    let mut cap = videoio::VideoCapture::from_file(&input_source, videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        return Err(anyhow!("Could not open video"));
    }

    let fps = cap.get(videoio::CAP_PROP_FPS)?;
    if fps <= 0.0 {
        return Err(anyhow!("Invalid FPS"));
    }

    let w = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let h = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;

    let stride = (fps / TARGET_FPS).round().max(1.0) as usize;
    let out_fps = fps / stride as f64;

    let mut writer = videoio::VideoWriter::new(
        out_path.to_str().unwrap(),
        videoio::VideoWriter::fourcc('a', 'v', 'c', '1')?,
        out_fps,
        Size::new(w, h),
        true,
    )?;
    if !writer.is_opened()? {
        return Err(anyhow!("VideoWriter failed to open (avc1)."));
    }

    let cfg = InferenceConfig::new()
        .with_confidence(CONF_THRESH)
        .with_iou(0.45)
        .with_max_det(500);

    let model_path = data_dir.join("yolov11n-visdrone.onnx");
    if !model_path.exists() {
        return Err(anyhow!("Model not found: {}", model_path.display()));
    }

    let mut model = YOLOModel::load_with_config(model_path, cfg)?;

    let mut tracker = ByteTracker::new(
        out_fps.round().max(1.0) as usize,
        TRACK_BUFFER,
        TRACK_THRESH,
        HIGH_THRESH,
        MATCH_THRESH,
    );

    let mut tracks: HashMap<i64, Track> = HashMap::new();
    let mut next_local_id: i64 = -1;
    let mut heatmap = Mat::zeros(h, w, core::CV_32F)?.to_mat()?;

    highgui::named_window("Drone Analytics", highgui::WINDOW_NORMAL)?;
    highgui::resize_window("Drone Analytics", 1280, 720)?;

    let mut frame = Mat::default();
    let mut idx = 0usize;

    while cap.read(&mut frame)? {
        if idx % stride != 0 {
            idx += 1;
            continue;
        }

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

        let tracked = tracker.update(&objects)?;
        let dets = apply_tracks(
            tracked,
            &classes,
            &det_confs,
            &mut tracks,
            &mut next_local_id,
            idx,
            out_fps,
        );

        decay_heatmap(&mut heatmap)?;
        update_heatmap(&mut heatmap, &dets)?;

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

            let dens = density_factor(&heatmap, total)?;
            let clus = cluster_factor(&heatmap)?;

            let dens_adj = (dens - 1.0) * 18.0;
            let clus_adj = (clus - 1.0) * 22.0;

            (base + dens_adj + clus_adj).clamp(0.0, 100.0) as i32
        } else {
            0
        };



        let mut out = frame.clone();
        overlay_heatmap(&heatmap, &mut out)?;
        draw_trails(&mut out, &tracks)?;
        draw_detections(&mut out, &dets)?;
        draw_hud(&mut out, congestion)?;

        writer.write(&out)?;
        highgui::imshow("Drone Analytics", &out)?;
        if highgui::wait_key(1)? == 113 {
            break;
        }

        idx += 1;
    }

    writer.release()?;
    cap.release()?;
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
                Scalar::new(180.0, 180.0, 180.0, 0.0),
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
            col,
            1,
            imgproc::LINE_AA,
            false,
        )?;
    }
    Ok(())
}

// draw HUD
fn draw_hud(frame: &mut Mat, pct: i32) -> Result<()> {
    let w = frame.cols();
    let margin = 30;

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
        Scalar::all(255.0),
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

    let col = Scalar::new(
        0.0,
        (255.0 - pct as f64 * 2.5).max(0.0),
        (pct as f64 * 2.5).min(255.0),
        0.0,
    );

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

    Ok(())
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
        8 => Scalar::new(255.0, 70.0, 200.0, 0.0),    // bus (pink)
        9 => Scalar::new(80.0, 220.0, 60.0, 0.0),     // motor (lime)
        _ => Scalar::new(210.0, 210.0, 210.0, 0.0),
    }
}


// class name mapping
fn class_name(cid: i64) -> &'static str {
    match cid {
        3 => "car",
        4 => "van",
        5 => "truck",
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
