// src/state.rs
use jamtrack_rs::byte_tracker::ByteTracker;
use opencv::core::Mat;
use std::collections::HashMap;
use std::sync::{atomic::AtomicBool, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;
use sysinfo::System;
use ultralytics_inference::Device;

use crate::analytics::SysSnapshot;
use crate::mediamtx::RtspPublisher;
pub const CONF_THRESH: f32 = 0.10;
pub const RTSP_SAVE_SECONDS: f64 = 60.0;

// ByteTrack params
pub const TRACK_BUFFER: usize = 30;
pub const TRACK_THRESH: f32 = 0.5;
pub const HIGH_THRESH: f32 = 0.6;
pub const MATCH_THRESH: f32 = 0.8;
pub const MAX_TRACK_AGE: usize = 30;
pub const ORPHAN_MATCH_DIST: f32 = 60.0;

// Heatmap
pub const HEATMAP_DECAY: f32 = 0.95;
pub const HEATMAP_RADIUS: i32 = 6;

// Speed thresholds (px/sec)
pub const THRESH_STALLED: f32 = 8.0;
pub const THRESH_SLOW: f32 = 50.0;
pub const THRESH_MEDIUM: f32 = 160.0;

// ALL VisDrone classes
pub const TARGET_CLASSES: [i64; 6] = [3, 4, 5, 7, 8, 9];

// Trails
pub const MAX_TRAIL_LEN: usize = 15;
pub const TRAIL_THICKNESS: i32 = 1;

// ================== CORE TYPES ==================

#[derive(Clone, Copy)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Clone)]
pub struct Detection {
    pub bbox: BBox,
    pub score: f32,
    pub class_id: i64,
    pub track_id: Option<i64>,
    pub speed: SpeedClass,
}

#[derive(Clone)]
pub struct Track {
    pub id: i64,
    pub bbox: BBox,
    pub last_center: (f32, f32),
    pub last_seen: usize,
    pub trail: Vec<(f32, f32)>,
    pub class_id: i64,
    pub class_conf: f32,
    pub speed_px_s: f32,
}

// ================== SPEED ==================

#[derive(Clone, Copy)]
pub enum SpeedClass {
    Stalled,
    Slow,
    Medium,
    Fast,
}

impl SpeedClass {
    pub fn bucket(self) -> usize {
        match self {
            SpeedClass::Stalled => 0,
            SpeedClass::Slow => 1,
            SpeedClass::Medium => 2,
            SpeedClass::Fast => 3,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SpeedClass::Stalled => "STALLED",
            SpeedClass::Slow => "SLOW",
            SpeedClass::Medium => "MEDIUM",
            SpeedClass::Fast => "FAST",
        }
    }
}

// ================== HELPERS ==================

pub fn bbox_bottom_center(b: BBox) -> (f32, f32) {
    ((b.x1 + b.x2) * 0.5, b.y2)
}

pub fn smooth_speed(prev: f32, curr: f32) -> f32 {
    if prev == 0.0 { curr } else { prev * 0.4 + curr * 0.6 }
}

pub fn classify_speed(speed: f32) -> SpeedClass {
    if speed < THRESH_STALLED {
        SpeedClass::Stalled
    } else if speed < THRESH_SLOW {
        SpeedClass::Slow
    } else if speed < THRESH_MEDIUM {
        SpeedClass::Medium
    } else {
        SpeedClass::Fast
    }
}

// ================== PIPELINE STATE ==================

pub struct SourceState {
    pub fps: f64,
    pub w: i32,
    pub h: i32,
    pub stride: usize,
    pub out_fps: f64,
    pub writer: opencv::videoio::VideoWriter,
    pub writer_active: bool,
    pub write_limit: Option<usize>,
    pub frames_written: usize,
    pub tracker: ByteTracker,
    pub tracks: HashMap<i64, Track>,
    pub next_local_id: i64,
    pub heatmap: Mat,
    pub source_label: String,
    pub sys: System,
    pub sys_pid: Option<sysinfo::Pid>,
    pub sys_last_update: Instant,
    pub sys_snapshot: SysSnapshot,
    pub gpu_util_shared: Option<Arc<Mutex<Option<f32>>>>,
    pub gpu_poll_stop: Option<Arc<AtomicBool>>,
    pub gpu_poll_join: Option<JoinHandle<()>>,
    pub fps_last_update: Instant,
    pub fps_frames: usize,
    pub fps_value: f32,
    pub traffic_density_ema: f32,
    pub mobility_index_ema: f32,
    pub mediamtx: Option<RtspPublisher>,
    pub device: Device,
}

#[derive(Default)]
pub struct MinuteAgg {
    count: u64,
    sum_detections: u64,
    sum_congestion: i64,
    sum_density: i64,
    sum_mobility: i64,
    sum_stalled: i64,
    sum_slow: i64,
    sum_medium: i64,
    sum_fast: i64,
    last_frame_idx: usize,
    last_ts_ms: i64,
}

pub struct MinuteRow {
    pub frame_idx: usize,
    pub ts_ms: i64,
    pub detections: usize,
    pub congestion: i32,
    pub traffic_density: i32,
    pub mobility_index: i32,
    pub stalled_pct: i32,
    pub slow_pct: i32,
    pub medium_pct: i32,
    pub fast_pct: i32,
}

impl MinuteAgg {
    pub fn push(
        &mut self,
        frame_idx: usize,
        ts_ms: i64,
        detections: usize,
        congestion: i32,
        traffic_density: i32,
        mobility_index: i32,
        stalled_pct: i32,
        slow_pct: i32,
        medium_pct: i32,
        fast_pct: i32,
    ) {
        self.count += 1;
        self.sum_detections += detections as u64;
        self.sum_congestion += congestion as i64;
        self.sum_density += traffic_density as i64;
        self.sum_mobility += mobility_index as i64;
        self.sum_stalled += stalled_pct as i64;
        self.sum_slow += slow_pct as i64;
        self.sum_medium += medium_pct as i64;
        self.sum_fast += fast_pct as i64;
        self.last_frame_idx = frame_idx;
        self.last_ts_ms = ts_ms;
    }

    pub fn take_averages(&mut self) -> Option<MinuteRow> {
        if self.count == 0 {
            return None;
        }
        let denom = self.count as f64;
        let row = MinuteRow {
            frame_idx: self.last_frame_idx,
            ts_ms: self.last_ts_ms,
            detections: (self.sum_detections as f64 / denom).round() as usize,
            congestion: (self.sum_congestion as f64 / denom).round() as i32,
            traffic_density: (self.sum_density as f64 / denom).round() as i32,
            mobility_index: (self.sum_mobility as f64 / denom).round() as i32,
            stalled_pct: (self.sum_stalled as f64 / denom).round() as i32,
            slow_pct: (self.sum_slow as f64 / denom).round() as i32,
            medium_pct: (self.sum_medium as f64 / denom).round() as i32,
            fast_pct: (self.sum_fast as f64 / denom).round() as i32,
        };
        *self = MinuteAgg::default();
        Some(row)
    }
}
