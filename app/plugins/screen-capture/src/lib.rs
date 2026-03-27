use tauri::{
    plugin::{Builder as PluginBuilder, TauriPlugin},
    Manager, Runtime,
};

mod error;

// Audio pipeline module - shared resampling utilities for all desktop platforms
#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub mod audio_pipeline;

// Audio module - only needed for Windows/Linux (macOS uses unified desktop module)
#[cfg(all(
    not(any(target_os = "android", target_os = "ios")),
    not(target_os = "macos")
))]
pub mod audio;

#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub mod targets;

// Platform-specific desktop implementations
// macOS uses unified ScreenCaptureKit for BOTH video and audio
// Windows/Linux use xcap for video + WASAPI/ALSA for audio
#[cfg(all(target_os = "macos", not(any(target_os = "android", target_os = "ios"))))]
#[path = "macos.rs"]
pub mod desktop;

#[cfg(all(not(target_os = "macos"), not(any(target_os = "android", target_os = "ios"))))]
pub mod desktop;

#[cfg(any(target_os = "android", target_os = "ios"))]
mod mobile;

// Android JNI module for channel-based capture
#[cfg(target_os = "android")]
pub mod android;

pub use error::{Error, Result};

#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub use targets::CaptureTarget;

/// Initializes the screen capture plugin
pub fn init<R: Runtime>() -> TauriPlugin<R> {
    PluginBuilder::new("screen-capture")
        .invoke_handler(tauri::generate_handler![
            #[cfg(any(target_os = "android", target_os = "ios"))]
            start_capture_cmd,
            stop_capture_cmd,
            stop_video_cmd,
            stop_audio_cmd,
            #[cfg(any(target_os = "android", target_os = "ios"))]
            get_frame_cmd,
            #[cfg(target_os = "ios")]
            get_app_group_path_cmd,
            #[cfg(target_os = "ios")]
            read_broadcast_debug_log_cmd,
            #[cfg(target_os = "ios")]
            list_app_group_files_cmd,
            #[cfg(not(any(target_os = "android", target_os = "ios")))]
            get_capture_targets_cmd,
            #[cfg(not(any(target_os = "android", target_os = "ios")))]
            start_capture_stream_cmd,
            #[cfg(not(any(target_os = "android", target_os = "ios")))]
            start_video_stream_cmd,
            #[cfg(not(any(target_os = "android", target_os = "ios")))]
            start_audio_stream_cmd,
            // Android channel-based streaming commands
            #[cfg(target_os = "android")]
            start_video_stream_cmd,
            #[cfg(target_os = "android")]
            start_audio_stream_cmd,
            #[cfg(target_os = "android")]
            stop_video_stream_cmd,
            #[cfg(target_os = "android")]
            stop_audio_stream_cmd
        ])
        .setup(|app, api| {
            #[cfg(any(target_os = "android", target_os = "ios"))]
            {
                let screen_capture = mobile::init(app, api)?;
                app.manage(screen_capture);
            }

            #[cfg(not(any(target_os = "android", target_os = "ios")))]
            desktop::init(app, api)?;

            Ok(())
        })
        .build()
}

// ==================== Mobile-only commands ====================

#[cfg(any(target_os = "android", target_os = "ios"))]
#[tauri::command]
async fn start_capture_cmd<R: Runtime>(
    app: tauri::AppHandle<R>,
) -> Result<bool> {
    let screen_capture = app.state::<mobile::ScreenCapture<R>>();

    // On Android, check if a JNI channel was set up (via start_capture_stream_cmd)
    // If so, use video stream mode which enables JNI callbacks
    #[cfg(target_os = "android")]
    {
        if android::is_video_streaming() {
            log::info!("[ScreenCapture] Android: JNI channel detected, using video stream mode");
            return screen_capture.start_video_stream().map(|_| true);
        }
    }

    // Default behavior: show permission dialog and start capture
    screen_capture.start_capture()
}

#[cfg(any(target_os = "android", target_os = "ios"))]
#[tauri::command]
async fn get_frame_cmd<R: Runtime>(
    app: tauri::AppHandle<R>,
) -> Result<String> {
    let screen_capture = app.state::<mobile::ScreenCapture<R>>();
    screen_capture.get_frame()
}

// ==================== iOS-only debug commands ====================

/// Get the App Group container path (for debugging shared memory setup)
#[cfg(target_os = "ios")]
#[tauri::command]
async fn get_app_group_path_cmd<R: Runtime>(
    app: tauri::AppHandle<R>,
) -> Result<String> {
    let screen_capture = app.state::<mobile::ScreenCapture<R>>();
    screen_capture.get_app_group_path()
}

/// Read the broadcast extension debug log
#[cfg(target_os = "ios")]
#[tauri::command]
async fn read_broadcast_debug_log_cmd<R: Runtime>(
    app: tauri::AppHandle<R>,
) -> Result<String> {
    let screen_capture = app.state::<mobile::ScreenCapture<R>>();
    screen_capture.read_broadcast_debug_log()
}

/// List files in the App Group container (for debugging)
#[cfg(target_os = "ios")]
#[tauri::command]
async fn list_app_group_files_cmd<R: Runtime>(
    app: tauri::AppHandle<R>,
) -> Result<serde_json::Value> {
    let screen_capture = app.state::<mobile::ScreenCapture<R>>();
    screen_capture.list_app_group_files()
}

// ==================== Cross-platform commands ====================

/// Stop all capture (video + audio)
#[tauri::command]
async fn stop_capture_cmd<R: Runtime>(
    #[allow(unused_variables)] app: tauri::AppHandle<R>,
) -> Result<()> {
    #[cfg(any(target_os = "android", target_os = "ios"))]
    {
        let screen_capture = app.state::<mobile::ScreenCapture<R>>();
        return screen_capture.stop_capture();
    }

    // macOS: unified module handles both video and audio
    #[cfg(target_os = "macos")]
    {
        let _ = desktop::stop_audio();
        return desktop::stop_capture().await;
    }

    // Windows/Linux: separate audio module
    #[cfg(all(
        not(any(target_os = "android", target_os = "ios")),
        not(target_os = "macos")
    ))]
    {
        let _ = audio::stop_audio();
        return desktop::stop_capture().await;
    }
}

/// Stop only video capture
#[tauri::command]
async fn stop_video_cmd<R: Runtime>(
    #[allow(unused_variables)] app: tauri::AppHandle<R>,
) -> Result<()> {
    #[cfg(any(target_os = "android", target_os = "ios"))]
    {
        let screen_capture = app.state::<mobile::ScreenCapture<R>>();
        return screen_capture.stop_capture();
    }

    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    {
        return desktop::stop_capture().await;
    }
}

/// Stop only audio capture
#[tauri::command]
async fn stop_audio_cmd<R: Runtime>(
    #[allow(unused_variables)] _app: tauri::AppHandle<R>,
) -> Result<()> {
    // macOS: unified module handles audio
    #[cfg(target_os = "macos")]
    {
        return desktop::stop_audio();
    }

    // Windows/Linux: separate audio module
    #[cfg(all(
        not(any(target_os = "android", target_os = "ios")),
        not(target_os = "macos")
    ))]
    {
        return audio::stop_audio();
    }

    #[cfg(any(target_os = "android", target_os = "ios"))]
    {
        // Audio stop is handled by mobile crate's stop_audio_stream_cmd
        // Frontend should call that command directly on iOS/Android
        Ok(())
    }
}

// ==================== Desktop-only commands ====================

/// Get all available capture targets (monitors and windows)
#[cfg(not(any(target_os = "android", target_os = "ios")))]
#[tauri::command]
async fn get_capture_targets_cmd<R: Runtime>(
    _app: tauri::AppHandle<R>,
    include_thumbnails: Option<bool>,
) -> Result<Vec<targets::CaptureTarget>> {
    let include_thumbnails = include_thumbnails.unwrap_or(true);
    desktop::get_capture_targets(include_thumbnails)
}

/// Start capture with channel-based streaming (desktop only)
/// Frames are pushed to frontend via channel instead of polling
/// Audio capture is also started and streamed via separate channel
/// NOTE: For independent control, use start_video_stream_cmd and start_audio_stream_cmd instead
#[cfg(target_os = "macos")]
#[tauri::command]
fn start_capture_stream_cmd<R: Runtime>(
    _app: tauri::AppHandle<R>,
    target_id: Option<String>,
    on_frame: tauri::ipc::Channel<desktop::FrameData>,
    on_audio: tauri::ipc::Channel<desktop::AudioData>,
) -> Result<()> {
    // macOS: unified module handles both - start video first, then audio
    desktop::start_capture_stream(target_id, on_frame)?;

    if let Err(e) = desktop::start_audio_stream(on_audio) {
        log::warn!("[ScreenCapture] Audio capture failed to start: {:?}", e);
    }

    Ok(())
}

/// Start capture with channel-based streaming (Windows/Linux)
#[cfg(all(
    not(any(target_os = "android", target_os = "ios")),
    not(target_os = "macos")
))]
#[tauri::command]
fn start_capture_stream_cmd<R: Runtime>(
    _app: tauri::AppHandle<R>,
    target_id: Option<String>,
    on_frame: tauri::ipc::Channel<desktop::FrameData>,
    on_audio: tauri::ipc::Channel<audio::AudioData>,
) -> Result<()> {
    // Start video capture
    desktop::start_capture_stream(target_id, on_frame)?;

    // Start audio capture
    if let Err(e) = audio::start_audio_stream(on_audio) {
        log::warn!("[ScreenCapture] Audio capture failed to start: {:?}", e);
    }

    Ok(())
}

/// Start video-only capture with channel-based streaming (desktop only)
/// Frames are pushed to frontend via channel instead of polling
#[cfg(not(any(target_os = "android", target_os = "ios")))]
#[tauri::command]
fn start_video_stream_cmd<R: Runtime>(
    _app: tauri::AppHandle<R>,
    target_id: Option<String>,
    on_frame: tauri::ipc::Channel<desktop::FrameData>,
) -> Result<()> {
    desktop::start_capture_stream(target_id, on_frame)
}

/// Start audio-only capture with channel-based streaming (macOS)
/// System audio is captured via unified ScreenCaptureKit module
#[cfg(target_os = "macos")]
#[tauri::command]
fn start_audio_stream_cmd<R: Runtime>(
    _app: tauri::AppHandle<R>,
    on_audio: tauri::ipc::Channel<desktop::AudioData>,
) -> Result<()> {
    desktop::start_audio_stream(on_audio)
}

/// Start audio-only capture with channel-based streaming (Windows/Linux)
/// System audio is captured via WASAPI/ALSA
#[cfg(all(
    not(any(target_os = "android", target_os = "ios")),
    not(target_os = "macos")
))]
#[tauri::command]
fn start_audio_stream_cmd<R: Runtime>(
    _app: tauri::AppHandle<R>,
    on_audio: tauri::ipc::Channel<audio::AudioData>,
) -> Result<()> {
    audio::start_audio_stream(on_audio)
}

// ==================== Android channel-based streaming commands ====================

/// Start video streaming on Android with channel-based delivery
/// Stores the channel for JNI callbacks and triggers Kotlin to start capture
#[cfg(target_os = "android")]
#[tauri::command]
fn start_video_stream_cmd<R: Runtime>(
    app: tauri::AppHandle<R>,
    on_frame: tauri::ipc::Channel<android::FrameData>,
) -> Result<()> {
    log::info!("[ScreenCapture] Android: Starting video stream");

    // Store the channel for JNI callbacks
    android::set_frame_channel(on_frame);

    // Trigger Kotlin to start capture (it will call nativeOnFrame via JNI)
    let screen_capture = app.state::<mobile::ScreenCapture<R>>();
    match screen_capture.start_video_stream() {
        Ok(_) => {
            log::info!("[ScreenCapture] Android: Video stream started");
            Ok(())
        }
        Err(e) => {
            // Clear channel on failure
            android::clear_frame_channel();
            log::error!("[ScreenCapture] Android: Failed to start video stream: {:?}", e);
            Err(e)
        }
    }
}

/// Start audio streaming on Android with channel-based delivery
/// Stores the channel for JNI callbacks and triggers Kotlin to start audio capture
#[cfg(target_os = "android")]
#[tauri::command]
fn start_audio_stream_cmd<R: Runtime>(
    app: tauri::AppHandle<R>,
    on_audio: tauri::ipc::Channel<android::AudioData>,
) -> Result<()> {
    log::info!("[ScreenCapture] Android: Starting audio stream");

    // Store the channel for JNI callbacks
    android::set_audio_channel(on_audio);

    // Trigger Kotlin to start audio capture (it will call nativeOnAudio via JNI)
    let screen_capture = app.state::<mobile::ScreenCapture<R>>();
    match screen_capture.start_audio_stream() {
        Ok(_) => {
            log::info!("[ScreenCapture] Android: Audio stream started");
            Ok(())
        }
        Err(e) => {
            // Clear channel on failure
            android::clear_audio_channel();
            log::error!("[ScreenCapture] Android: Failed to start audio stream: {:?}", e);
            Err(e)
        }
    }
}

/// Stop video streaming on Android
#[cfg(target_os = "android")]
#[tauri::command]
fn stop_video_stream_cmd<R: Runtime>(
    app: tauri::AppHandle<R>,
) -> Result<()> {
    log::info!("[ScreenCapture] Android: Stopping video stream");

    // Clear the channel first
    android::clear_frame_channel();

    // Tell Kotlin to stop capture
    let screen_capture = app.state::<mobile::ScreenCapture<R>>();
    screen_capture.stop_video_stream()
}

/// Stop audio streaming on Android
#[cfg(target_os = "android")]
#[tauri::command]
fn stop_audio_stream_cmd<R: Runtime>(
    app: tauri::AppHandle<R>,
) -> Result<()> {
    log::info!("[ScreenCapture] Android: Stopping audio stream");

    // Clear the channel first
    android::clear_audio_channel();

    // Tell Kotlin to stop audio capture
    let screen_capture = app.state::<mobile::ScreenCapture<R>>();
    screen_capture.stop_audio_stream()
}
