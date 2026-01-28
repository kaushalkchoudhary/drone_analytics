// src/main.rs

use anyhow::{anyhow, Result};
use chrono::Local;
use jamtrack_rs::{Object, Rect as JamRect};
use opencv::highgui;
use opencv::prelude::VideoWriterTrait;
use std::{
    fs,
    path::PathBuf,
};
use std::time::Instant;
use ultralytics_inference::{Device, InferenceConfig, YOLOModel};

mod analytics;
use analytics::*;
mod helpers;
use helpers::*;
mod state;
use state::*;
mod mediamtx;
mod db;
use db::Db;

const EMA_ALPHA: f32 = 0.2;
const DB_AGG_INTERVAL_SECS: u64 = 60;

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
    let db_path = out_dir.join("analytics.db");
    let mut db = Db::open(&db_path)?;

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

    let mut model = YOLOModel::load_with_config(&model_path, cfg)?;

    highgui::named_window("Drone Analytics", highgui::WINDOW_NORMAL)?;
    let (mut state, cap) =
        init_source_state(&sources[source_idx], &out_dir, opts.output, overlay_device.clone())?;
    let mut is_rtsp = is_rtsp_source(&sources[source_idx]);
    let mut capture = Some(start_capture_thread(cap, state.stride, opts.io_buffer));

    if is_rtsp {
        highgui::resize_window("Drone Analytics", state.w, state.h)?;
    } else {
        highgui::resize_window("Drone Analytics", 1280, 720)?;
    }

    let session_started = Local::now().to_rfc3339();
    let source_label = source_display_name(&sources[source_idx]);
    let model_label = model_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let config_json = format!(
        "{{\"heatmap\":{},\"trails\":{},\"bboxes\":{},\"threads\":{},\"device\":\"{}\"}}",
        opts.show_heatmap,
        opts.show_trails,
        opts.show_bboxes,
        opts.threads.unwrap_or(0),
        format_device_label(&overlay_device)
    );
    let mut session_id = db.create_session(
        &session_started,
        &sources[source_idx],
        &source_label,
        state.fps,
        state.w,
        state.h,
        model_label,
        &format_device_label(&overlay_device),
        &config_json,
    )?;
    let mut session_start = Instant::now();
    let mut last_flush = Instant::now();
    let mut agg = MinuteAgg::default();

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
            update_heatmap(&mut state.heatmap, &state.tracks)?;

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
                let traffic_density_smoothed =
                    ema_update(&mut state.traffic_density_ema, traffic_density, EMA_ALPHA);
                let traffic_density_pct = traffic_density_smoothed.round() as i32;

                let mobility_raw = (counts[0] as f32 * 0.0
                    + counts[1] as f32 * 0.35
                    + counts[2] as f32 * 0.7
                    + counts[3] as f32 * 1.0)
                    / total as f32;
                let mobility_index = mobility_raw.clamp(0.0, 1.0) * 100.0;
                let mobility_smoothed =
                    ema_update(&mut state.mobility_index_ema, mobility_index, EMA_ALPHA);
                let mobility_index_pct = mobility_smoothed.round() as i32;

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
                let traffic_density_smoothed =
                    ema_update(&mut state.traffic_density_ema, 0.0, EMA_ALPHA);
                let mobility_smoothed = ema_update(&mut state.mobility_index_ema, 0.0, EMA_ALPHA);
                (
                    0,
                    traffic_density_smoothed.round() as i32,
                    mobility_smoothed.round() as i32,
                    0,
                    0,
                    0,
                    0,
                )
            };

            let ts_ms = session_start.elapsed().as_millis() as i64;
            agg.push(
                frame_idx,
                ts_ms,
                total,
                congestion,
                traffic_density_pct,
                mobility_index_pct,
                stalled_pct,
                slow_pct,
                medium_pct,
                fast_pct,
            );
            if last_flush.elapsed().as_secs() >= DB_AGG_INTERVAL_SECS {
                if let Some(row) = agg.take_averages() {
                    db.insert_frame_metrics(
                        session_id,
                        row.frame_idx,
                        row.ts_ms,
                        row.detections,
                        row.congestion,
                        row.traffic_density,
                        row.mobility_index,
                        row.stalled_pct,
                        row.slow_pct,
                        row.medium_pct,
                        row.fast_pct,
                        None,
                    )?;
                }
                last_flush = Instant::now();
            }

            let out = &mut frame;
            if opts.show_heatmap {
                overlay_heatmap(&state.heatmap, out)?;
            }
            if opts.show_trails {
                draw_trails(out, &state.tracks)?;
            }
            if opts.show_bboxes {
                draw_detections(out, &dets)?;
            }
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
                if let Some(row) = agg.take_averages() {
                    db.insert_frame_metrics(
                        session_id,
                        row.frame_idx,
                        row.ts_ms,
                        row.detections,
                        row.congestion,
                        row.traffic_density,
                        row.mobility_index,
                        row.stalled_pct,
                        row.slow_pct,
                        row.medium_pct,
                        row.fast_pct,
                        None,
                    )?;
                }
                let session_ended = Local::now().to_rfc3339();
                let _ = db.finish_session(session_id, &session_ended);
                state.writer.release()?;
                let (next_state, cap) =
                    init_source_state(&next_source, &out_dir, opts.output, overlay_device.clone())?;
                state = next_state;
                capture = Some(start_capture_thread(cap, state.stride, opts.io_buffer));
                is_rtsp = is_rtsp_source(&next_source);
                let session_started = Local::now().to_rfc3339();
                let source_label = source_display_name(&next_source);
                session_id = db.create_session(
                    &session_started,
                    &next_source,
                    &source_label,
                    state.fps,
                    state.w,
                    state.h,
                    model_label,
                    &format_device_label(&overlay_device),
                    &config_json,
                )?;
                session_start = Instant::now();
                last_flush = Instant::now();
                agg = MinuteAgg::default();
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
    if let Some(row) = agg.take_averages() {
        db.insert_frame_metrics(
            session_id,
            row.frame_idx,
            row.ts_ms,
            row.detections,
            row.congestion,
            row.traffic_density,
            row.mobility_index,
            row.stalled_pct,
            row.slow_pct,
            row.medium_pct,
            row.fast_pct,
            None,
        )?;
    }
    let session_ended = Local::now().to_rfc3339();
    let _ = db.finish_session(session_id, &session_ended);
    Ok(())
}
