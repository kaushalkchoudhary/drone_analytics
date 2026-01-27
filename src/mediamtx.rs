use anyhow::Result;
use opencv::core::Mat;
use std::io::Write;
use std::process::{Child, ChildStdin, Command, Stdio};
use opencv::prelude::MatTraitConstManual;

pub struct RtspPublisher {
    _child: Child,
    stdin: ChildStdin,
}

pub fn start_rtsp_publisher(
    w: i32,
    h: i32,
    fps: f64,
    path: &str,
) -> Result<RtspPublisher> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", &format!("{}x{}", w, h),
            "-r", &format!("{:.2}", fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-f", "rtsp",
            path,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()?;

    let stdin = child.stdin.take().expect("ffmpeg stdin");

    Ok(RtspPublisher {
        _child: child,
        stdin,
    })
}

impl RtspPublisher {
    #[inline]
    pub fn send(&mut self, frame: &Mat) -> Result<()> {
        self.stdin.write_all(frame.data_bytes()?)?;
        Ok(())
    }
}
