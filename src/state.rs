// src/state.rs


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
pub const HEATMAP_RADIUS: i32 = 10;

// Speed thresholds (px/sec)
pub const THRESH_STALLED: f32 = 15.0;
pub const THRESH_SLOW: f32 = 80.0;
pub const THRESH_MEDIUM: f32 = 250.0;

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
            SpeedClass::Stalled => "",
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
    if prev == 0.0 { curr } else { prev * 0.7 + curr * 0.3 }
}

pub fn classify_speed(speed: f32) -> SpeedClass {
    if speed < THRESH_SLOW {
        SpeedClass::Slow
    } else if speed < THRESH_MEDIUM {
        SpeedClass::Medium
    } else {
        SpeedClass::Fast
    }
}
