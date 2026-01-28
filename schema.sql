PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  source TEXT NOT NULL,
  source_label TEXT,
  fps REAL,
  width INTEGER,
  height INTEGER,
  model TEXT,
  device TEXT,
  config_json TEXT
);

CREATE TABLE IF NOT EXISTS minute_frame_agg_metrics (
  id INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL,
  frame_idx INTEGER NOT NULL,
  ts_ms INTEGER NOT NULL,
  detections INTEGER NOT NULL,
  congestion INTEGER NOT NULL,
  traffic_density INTEGER NOT NULL,
  mobility_index INTEGER NOT NULL,
  stalled_pct INTEGER NOT NULL,
  slow_pct INTEGER NOT NULL,
  medium_pct INTEGER NOT NULL,
  fast_pct INTEGER NOT NULL,
  avg_speed_px_s REAL,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_minute_metrics_session_frame
  ON minute_frame_agg_metrics(session_id, frame_idx);

CREATE INDEX IF NOT EXISTS idx_minute_metrics_session_ts
  ON minute_frame_agg_metrics(session_id, ts_ms);
