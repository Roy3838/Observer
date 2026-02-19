use crate::error::Result;
use base64::{engine::general_purpose::STANDARD, Engine};
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::RgbaImage;
use parking_lot::RwLock;
use screenshots::Screen;
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::{plugin::PluginApi, AppHandle, Runtime};
use tokio::sync::watch;

// Capture settings - optimized for performance
const TARGET_FPS: u64 = 10; // 10fps is plenty for screen capture
const JPEG_QUALITY: u8 = 50; // Lower quality for smaller files
const MAX_WIDTH: u32 = 1280; // Max width for captured frames

/// Shared capture state accessible across async contexts
struct CaptureState {
    /// Latest frame as JPEG bytes with timestamp
    latest_frame: RwLock<Option<(Vec<u8>, f64)>>,
    /// Whether capture is currently active
    is_active: AtomicBool,
    /// Total frames captured
    frame_count: AtomicU64,
    /// Signal to stop the capture thread
    stop_signal: watch::Sender<bool>,
}

/// Global capture state - initialized on first use
static CAPTURE_STATE: std::sync::OnceLock<Arc<CaptureState>> = std::sync::OnceLock::new();

fn get_capture_state() -> Arc<CaptureState> {
    CAPTURE_STATE
        .get_or_init(|| {
            let (tx, _rx) = watch::channel(false);
            Arc::new(CaptureState {
                latest_frame: RwLock::new(None),
                is_active: AtomicBool::new(false),
                frame_count: AtomicU64::new(0),
                stop_signal: tx,
            })
        })
        .clone()
}

pub fn init<R: Runtime, C: serde::de::DeserializeOwned>(
    _app: &AppHandle<R>,
    _api: PluginApi<R, C>,
) -> Result<()> {
    log::info!("[ScreenCapture] Desktop plugin initialized");
    Ok(())
}

pub async fn start_capture() -> Result<bool> {
    let state = get_capture_state();

    // Check if already capturing
    if state.is_active.load(Ordering::SeqCst) {
        log::info!("[ScreenCapture] Capture already active");
        return Ok(true);
    }

    log::info!("[ScreenCapture] Starting desktop screen capture...");

    // Reset stop signal
    let _ = state.stop_signal.send(false);

    // Clear any stale frame from previous capture
    {
        let mut frame = state.latest_frame.write();
        *frame = None;
    }

    // Mark as active before spawning thread
    state.is_active.store(true, Ordering::SeqCst);
    state.frame_count.store(0, Ordering::SeqCst);

    // Create a receiver for the stop signal
    let stop_rx = state.stop_signal.subscribe();
    let capture_state = state.clone();

    // Spawn the capture thread
    std::thread::spawn(move || {
        log::info!("[ScreenCapture] Capture thread started");

        // Get the primary screen
        let screens = match Screen::all() {
            Ok(s) => s,
            Err(e) => {
                log::error!("[ScreenCapture] Failed to get screens: {:?}", e);
                capture_state.is_active.store(false, Ordering::SeqCst);
                return;
            }
        };

        let screen = match screens.into_iter().next() {
            Some(s) => s,
            None => {
                log::error!("[ScreenCapture] No screens found");
                capture_state.is_active.store(false, Ordering::SeqCst);
                return;
            }
        };

        log::info!(
            "[ScreenCapture] Capturing screen: {}x{} (downscaling to max {}px wide, {}fps, {}% quality)",
            screen.display_info.width,
            screen.display_info.height,
            MAX_WIDTH,
            TARGET_FPS,
            JPEG_QUALITY
        );

        let target_frame_time = Duration::from_millis(1000 / TARGET_FPS);

        loop {
            let frame_start = Instant::now();

            // Check stop signal (non-blocking)
            if *stop_rx.borrow() {
                log::info!("[ScreenCapture] Stop signal received");
                break;
            }

            // Capture frame
            match screen.capture() {
                Ok(image) => {
                    let width = image.width();
                    let height = image.height();

                    // Downscale if image is too large
                    let (final_width, final_height, raw_bytes) = if width > MAX_WIDTH {
                        let scale = MAX_WIDTH as f32 / width as f32;
                        let new_height = (height as f32 * scale) as u32;

                        // Create RgbaImage from raw bytes and resize
                        let rgba_image = RgbaImage::from_raw(width, height, image.into_raw())
                            .expect("Failed to create image");
                        let resized = image::imageops::resize(&rgba_image, MAX_WIDTH, new_height, FilterType::Triangle);

                        (MAX_WIDTH, new_height, resized.into_raw())
                    } else {
                        (width, height, image.into_raw())
                    };

                    // Convert to JPEG
                    let mut jpeg_buffer = Cursor::new(Vec::new());
                    let mut encoder = JpegEncoder::new_with_quality(&mut jpeg_buffer, JPEG_QUALITY);

                    if let Err(e) = encoder.encode(&raw_bytes, final_width, final_height, image::ColorType::Rgba8) {
                        log::error!("[ScreenCapture] Failed to encode JPEG: {:?}", e);
                        continue;
                    }

                    let jpeg_bytes = jpeg_buffer.into_inner();
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64();

                    // Store the frame
                    {
                        let mut frame = capture_state.latest_frame.write();
                        *frame = Some((jpeg_bytes, timestamp));
                    }

                    let count = capture_state.frame_count.fetch_add(1, Ordering::SeqCst) + 1;
                    if count == 1 {
                        log::info!("[ScreenCapture] First frame captured ({}x{}, {} bytes)",
                            final_width, final_height,
                            capture_state.latest_frame.read().as_ref().map(|(b, _)| b.len()).unwrap_or(0));
                    } else if count % (TARGET_FPS * 10) as u64 == 0 {
                        log::debug!("[ScreenCapture] Captured {} frames", count);
                    }
                }
                Err(e) => {
                    log::error!("[ScreenCapture] Capture failed: {:?}", e);
                }
            }

            // Maintain target fps
            let elapsed = frame_start.elapsed();
            if elapsed < target_frame_time {
                std::thread::sleep(target_frame_time - elapsed);
            }
        }

        log::info!("[ScreenCapture] Capture thread exiting");
        capture_state.is_active.store(false, Ordering::SeqCst);
    });

    log::info!("[ScreenCapture] Capture started");
    Ok(true)
}

pub async fn stop_capture() -> Result<()> {
    let state = get_capture_state();

    if !state.is_active.load(Ordering::SeqCst) {
        log::info!("[ScreenCapture] Capture not active");
        return Ok(());
    }

    log::info!("[ScreenCapture] Stopping capture...");

    // Mark as inactive FIRST to prevent race condition on restart
    // (otherwise start_capture may see is_active=true and return early)
    state.is_active.store(false, Ordering::SeqCst);

    // Send stop signal to the capture thread
    let _ = state.stop_signal.send(true);

    // Clear the latest frame
    {
        let mut frame = state.latest_frame.write();
        *frame = None;
    }

    log::info!("[ScreenCapture] Capture stopped");
    Ok(())
}

pub async fn get_frame() -> Result<String> {
    let state = get_capture_state();

    let frame = state.latest_frame.read();
    match frame.as_ref() {
        Some((jpeg_bytes, _timestamp)) => {
            let base64_frame = STANDARD.encode(jpeg_bytes);
            Ok(base64_frame)
        }
        None => Err(crate::Error::NoFrame),
    }
}

/// Get broadcast status with optional frame data (unified API for mobile/desktop)
pub fn get_broadcast_status() -> Result<serde_json::Value> {
    let state = get_capture_state();

    let is_active = state.is_active.load(Ordering::SeqCst);
    let frame_count = state.frame_count.load(Ordering::SeqCst);

    let frame_data = state.latest_frame.read();
    let (frame, timestamp, is_stale) = match frame_data.as_ref() {
        Some((jpeg_bytes, ts)) => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            // Consider stale if frame is more than 1 second old
            let stale = (now - ts) > 1.0;
            let base64_frame = STANDARD.encode(jpeg_bytes);
            (Some(base64_frame), Some(*ts), stale)
        }
        None => (None, None, false),
    };

    Ok(serde_json::json!({
        "isActive": is_active,
        "isStale": is_stale,
        "frame": frame,
        "timestamp": timestamp,
        "frameCount": frame_count
    }))
}
