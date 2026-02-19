use crate::error::Result;
use crate::targets::{self, CaptureTarget, TargetKind};
use base64::{engine::general_purpose::STANDARD, Engine};
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::RgbaImage;
use parking_lot::RwLock;
use serde::Serialize;
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::ipc::Channel;
use tauri::{plugin::PluginApi, AppHandle, Runtime};
use tokio::sync::watch;
use xcap::{Monitor, Window};

/// Frame data sent through the channel to the frontend
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameData {
    /// Base64-encoded JPEG frame
    pub frame: String,
    /// Unix timestamp in seconds
    pub timestamp: f64,
    /// Frame dimensions
    pub width: u32,
    pub height: u32,
    /// Frame sequence number
    pub frame_count: u64,
}

// Capture settings - optimized for performance
const TARGET_FPS: u64 = 10; // 10fps is plenty for screen capture
const JPEG_QUALITY: u8 = 50; // Lower quality for smaller files
const MAX_WIDTH: u32 = 1280; // Max width for captured frames

/// Shared capture state accessible across async contexts
struct CaptureState {
    /// Whether capture is currently active
    is_active: AtomicBool,
    /// Total frames captured
    frame_count: AtomicU64,
    /// Signal to stop the capture thread
    stop_signal: watch::Sender<bool>,
    /// Currently selected capture target (None = primary monitor)
    selected_target: RwLock<Option<String>>,
}

/// Global capture state - initialized on first use
static CAPTURE_STATE: std::sync::OnceLock<Arc<CaptureState>> = std::sync::OnceLock::new();

fn get_capture_state() -> Arc<CaptureState> {
    CAPTURE_STATE
        .get_or_init(|| {
            let (tx, _rx) = watch::channel(false);
            Arc::new(CaptureState {
                is_active: AtomicBool::new(false),
                frame_count: AtomicU64::new(0),
                stop_signal: tx,
                selected_target: RwLock::new(None),
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

pub async fn stop_capture() -> Result<()> {
    let state = get_capture_state();

    if !state.is_active.load(Ordering::SeqCst) {
        log::info!("[ScreenCapture] Capture not active");
        return Ok(());
    }

    log::info!("[ScreenCapture] Stopping capture...");

    // Mark as inactive FIRST to prevent race condition on restart
    state.is_active.store(false, Ordering::SeqCst);

    // Send stop signal to the capture thread
    let _ = state.stop_signal.send(true);

    // Clear the selected target
    {
        let mut target = state.selected_target.write();
        *target = None;
    }

    log::info!("[ScreenCapture] Capture stopped");
    Ok(())
}

/// Get broadcast status
pub fn get_broadcast_status() -> Result<serde_json::Value> {
    let state = get_capture_state();

    let is_active = state.is_active.load(Ordering::SeqCst);
    let frame_count = state.frame_count.load(Ordering::SeqCst);
    let selected_target = state.selected_target.read().clone();

    Ok(serde_json::json!({
        "isActive": is_active,
        "frameCount": frame_count,
        "targetId": selected_target
    }))
}

/// Get all available capture targets
pub fn get_capture_targets(include_thumbnails: bool) -> Result<Vec<CaptureTarget>> {
    targets::get_all_targets(include_thumbnails)
}

/// Start capture with channel-based streaming (push instead of poll)
/// Frames are pushed to the frontend as they're captured
pub fn start_capture_stream(
    target_id: Option<String>,
    on_frame: Channel<FrameData>,
) -> Result<()> {
    let state = get_capture_state();

    log::info!("[ScreenCapture] Starting channel-based capture stream with target: {:?}", target_id);

    // Send stop signal first to ensure any existing thread stops
    let _ = state.stop_signal.send(true);

    // Wait for any existing capture thread to exit
    if state.is_active.load(Ordering::SeqCst) {
        log::info!("[ScreenCapture] Waiting for existing capture to stop...");
        std::thread::sleep(Duration::from_millis(100));
    }

    // Now reset the stop signal to false BEFORE creating the receiver
    let _ = state.stop_signal.send(false);

    // Small delay to ensure the channel value is settled
    std::thread::sleep(Duration::from_millis(10));

    // Update selected target
    {
        let mut target = state.selected_target.write();
        *target = target_id.clone();
    }

    // Mark as active
    state.is_active.store(true, Ordering::SeqCst);
    state.frame_count.store(0, Ordering::SeqCst);

    // Create a receiver for the stop signal AFTER resetting it
    let stop_rx = state.stop_signal.subscribe();

    // Verify the stop signal is actually false
    if *stop_rx.borrow() {
        log::error!("[ScreenCapture] Stop signal is still true after reset! This is a bug.");
        let _ = state.stop_signal.send(false);
        std::thread::sleep(Duration::from_millis(10));
    }

    let capture_state = state.clone();

    // Spawn the capture thread with channel
    std::thread::spawn(move || {
        log::info!("[ScreenCapture] Channel capture thread started");

        let capture_result = match &target_id {
            Some(id) => {
                match targets::parse_target_id(id) {
                    Ok((kind, numeric_id)) => {
                        log::info!("[ScreenCapture] Stream capturing {:?} with id {}", kind, numeric_id);
                        run_capture_loop_with_channel(capture_state, stop_rx, Some((kind, numeric_id)), on_frame)
                    }
                    Err(e) => {
                        log::error!("[ScreenCapture] Failed to parse target ID: {:?}", e);
                        Err(e)
                    }
                }
            }
            None => {
                log::info!("[ScreenCapture] Stream capturing primary monitor");
                run_capture_loop_with_channel(capture_state, stop_rx, None, on_frame)
            }
        };

        if let Err(e) = capture_result {
            log::error!("[ScreenCapture] Channel capture loop failed: {:?}", e);
        }
    });

    log::info!("[ScreenCapture] Channel capture stream started");
    Ok(())
}

/// Run the capture loop, pushing frames through a channel
fn run_capture_loop_with_channel(
    capture_state: Arc<CaptureState>,
    stop_rx: watch::Receiver<bool>,
    target: Option<(TargetKind, u32)>,
    on_frame: Channel<FrameData>,
) -> Result<()> {
    let target_frame_time = Duration::from_millis(1000 / TARGET_FPS);

    enum CaptureSource {
        Monitor(Monitor),
        Window(Window),
    }

    let source = match &target {
        Some((TargetKind::Monitor, id)) => {
            let monitors = Monitor::all()
                .map_err(|e| crate::error::Error::Platform(format!("Failed to get monitors: {}", e)))?;
            let monitor = monitors.into_iter()
                .find(|m| m.id() == *id)
                .ok_or_else(|| crate::error::Error::Platform(format!("Monitor {} not found", id)))?;
            log::info!(
                "[ScreenCapture] Channel capturing monitor: {} ({}x{})",
                monitor.name(),
                monitor.width(),
                monitor.height()
            );
            CaptureSource::Monitor(monitor)
        }
        Some((TargetKind::Window, id)) => {
            let windows = Window::all()
                .map_err(|e| crate::error::Error::Platform(format!("Failed to get windows: {}", e)))?;
            let window = windows.into_iter()
                .find(|w| w.id() == *id)
                .ok_or_else(|| crate::error::Error::Platform(format!("Window {} not found", id)))?;
            log::info!(
                "[ScreenCapture] Channel capturing window: {} ({}x{})",
                window.title(),
                window.width(),
                window.height()
            );
            CaptureSource::Window(window)
        }
        None => {
            let monitors = Monitor::all()
                .map_err(|e| crate::error::Error::Platform(format!("Failed to get monitors: {}", e)))?;
            let monitor = monitors.into_iter()
                .find(|m| m.is_primary())
                .or_else(|| Monitor::all().ok().and_then(|m| m.into_iter().next()))
                .ok_or_else(|| crate::error::Error::Platform("No monitors found".to_string()))?;
            log::info!(
                "[ScreenCapture] Channel capturing primary monitor: {} ({}x{}, {}fps)",
                monitor.name(),
                monitor.width(),
                monitor.height(),
                TARGET_FPS
            );
            CaptureSource::Monitor(monitor)
        }
    };

    let mut frame_count: u64 = 0;

    loop {
        let frame_start = Instant::now();

        // Check stop signal
        if *stop_rx.borrow() {
            log::info!("[ScreenCapture] Channel capture stop signal received");
            break;
        }

        // Capture frame
        let capture_result = match &source {
            CaptureSource::Monitor(monitor) => monitor.capture_image(),
            CaptureSource::Window(window) => window.capture_image(),
        };

        match capture_result {
            Ok(image) => {
                // Process and send frame through channel
                if let Some(frame_data) = process_frame_for_channel(&image, frame_count) {
                    frame_count += 1;

                    if frame_count == 1 {
                        log::info!(
                            "[ScreenCapture] First channel frame sent ({}x{}, {} bytes)",
                            frame_data.width,
                            frame_data.height,
                            frame_data.frame.len()
                        );
                    }

                    // Push frame to frontend via channel
                    if let Err(e) = on_frame.send(frame_data) {
                        log::error!("[ScreenCapture] Failed to send frame through channel: {:?}", e);
                        // Channel closed, stop capture
                        break;
                    }

                    // Update shared state frame count
                    capture_state.frame_count.store(frame_count, Ordering::SeqCst);
                }
            }
            Err(e) => {
                log::error!("[ScreenCapture] Channel capture failed: {:?}", e);
            }
        }

        // Maintain target fps
        let elapsed = frame_start.elapsed();
        if elapsed < target_frame_time {
            std::thread::sleep(target_frame_time - elapsed);
        }
    }

    log::info!("[ScreenCapture] Channel capture thread exiting after {} frames", frame_count);
    capture_state.is_active.store(false, Ordering::SeqCst);
    Ok(())
}

/// Process a frame and return FrameData ready for channel transmission
fn process_frame_for_channel(image: &RgbaImage, frame_count: u64) -> Option<FrameData> {
    let width = image.width();
    let height = image.height();

    // Downscale if too large
    let resized = if width > MAX_WIDTH {
        let scale = MAX_WIDTH as f32 / width as f32;
        let new_height = (height as f32 * scale) as u32;
        image::imageops::resize(image, MAX_WIDTH, new_height, FilterType::Nearest)
    } else {
        image.clone()
    };

    let final_width = resized.width();
    let final_height = resized.height();

    // Convert RGBA to RGB
    let rgba_bytes = resized.as_raw();
    let mut rgb_bytes = Vec::with_capacity((final_width * final_height * 3) as usize);
    for chunk in rgba_bytes.chunks_exact(4) {
        rgb_bytes.push(chunk[0]);
        rgb_bytes.push(chunk[1]);
        rgb_bytes.push(chunk[2]);
    }

    // Encode to JPEG
    let mut jpeg_buffer = Cursor::new(Vec::new());
    let mut encoder = JpegEncoder::new_with_quality(&mut jpeg_buffer, JPEG_QUALITY);

    if let Err(e) = encoder.encode(&rgb_bytes, final_width, final_height, image::ExtendedColorType::Rgb8) {
        log::error!("[ScreenCapture] Failed to encode JPEG for channel: {:?}", e);
        return None;
    }

    let jpeg_bytes = jpeg_buffer.into_inner();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    Some(FrameData {
        frame: STANDARD.encode(&jpeg_bytes),
        timestamp,
        width: final_width,
        height: final_height,
        frame_count,
    })
}
