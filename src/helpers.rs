use anyhow::{anyhow, Result};
use crossbeam_channel::{bounded, Receiver};
use ndarray::Array3;
use opencv::{
    core::{AlgorithmHint, Mat},
    imgproc,
    prelude::{MatTraitConst, MatTraitConstManual, VideoCaptureTrait},
    videoio,
};
use serde::Deserialize;
use std::{
    env, fs,
    path::Path,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
};
use ultralytics_inference::Device;

#[derive(Clone, Copy, Debug)]
pub enum OutputFormat {
    Mkv,
    Mp4,
}

pub const DEFAULT_IO_BUFFER: usize = 120;

#[derive(Debug)]
pub struct CliOptions {
    pub source: Option<String>,
    pub use_rtsp: bool,
    pub rtsp_index: Option<usize>,
    pub output: OutputFormat,
    pub device: Option<Device>,
    pub threads: Option<usize>,
    pub io_buffer: usize,
}

pub fn parse_args() -> Result<CliOptions> {
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
                let parsed = idx
                    .parse::<usize>()
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
                    let parsed = rest
                        .parse::<usize>()
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

pub struct CaptureWorker {
    pub rx: Receiver<CapturedFrame>,
    stop: Arc<AtomicBool>,
    join: JoinHandle<Result<()>>,
}

impl CaptureWorker {
    pub fn stop(self) -> Result<()> {
        self.stop.store(true, Ordering::Relaxed);
        match self.join.join() {
            Ok(res) => res,
            Err(_) => Err(anyhow!("Capture thread panicked")),
        }
    }
}

pub struct CapturedFrame {
    pub frame: Mat,
    pub idx: usize,
}

pub fn start_capture_thread(
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

pub fn mat_to_array3_rgb(mat: &Mat) -> Result<Array3<u8>> {
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

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RtspYaml {
    List(Vec<String>),
    Map { rtsp_links: Vec<String> },
}

pub fn load_rtsp_links(path: &Path) -> Result<Vec<String>> {
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

pub fn is_rtsp_source(source: &str) -> bool {
    let s = source.trim();
    s.starts_with("rtsp://") || s.starts_with("rtsps://")
}

pub fn source_display_name(source: &str) -> String {
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
