use anyhow::Result;
use rusqlite::{params, Connection};
use std::path::Path;

pub struct Db {
    conn: Connection,
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute("PRAGMA foreign_keys = ON", [])?;
        Self::ensure_schema(&conn)?;
        Ok(Self { conn })
    }

    pub fn create_session(
        &mut self,
        started_at: &str,
        source: &str,
        source_label: &str,
        fps: f64,
        width: i32,
        height: i32,
        model: &str,
        device: &str,
        config_json: &str,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO sessions (started_at, source, source_label, fps, width, height, model, device, config_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                started_at,
                source,
                source_label,
                fps,
                width,
                height,
                model,
                device,
                config_json
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn finish_session(&mut self, session_id: i64, ended_at: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE sessions SET ended_at = ?1 WHERE id = ?2",
            params![ended_at, session_id],
        )?;
        Ok(())
    }

    pub fn insert_frame_metrics(
        &mut self,
        session_id: i64,
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
        avg_speed_px_s: Option<f32>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO minute_frame_agg_metrics (
                session_id, frame_idx, ts_ms, detections,
                congestion, traffic_density, mobility_index,
                stalled_pct, slow_pct, medium_pct, fast_pct, avg_speed_px_s
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                session_id,
                frame_idx as i64,
                ts_ms,
                detections as i64,
                congestion,
                traffic_density,
                mobility_index,
                stalled_pct,
                slow_pct,
                medium_pct,
                fast_pct,
                avg_speed_px_s.map(|v| v as f64)
            ],
        )?;
        Ok(())
    }

    fn ensure_schema(conn: &Connection) -> Result<()> {
        let user_version: i32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if user_version < 2 {
            let _ = conn.execute("DROP TABLE IF EXISTS frame_metrics", []);
            let _ = conn.execute("DROP TABLE IF EXISTS minute_frame_agg_metrics", []);
            let _ = conn.execute("DROP TABLE IF EXISTS sessions", []);
            let schema = include_str!("../schema.sql");
            conn.execute_batch(schema)?;
            conn.execute("PRAGMA user_version = 2", [])?;
        } else {
            let schema = include_str!("../schema.sql");
            conn.execute_batch(schema)?;
        }
        Ok(())
    }
}
