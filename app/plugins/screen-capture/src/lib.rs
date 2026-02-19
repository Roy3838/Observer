use tauri::{
    plugin::{Builder as PluginBuilder, TauriPlugin},
    Manager, Runtime,
};

mod error;

#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub mod desktop;
#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub mod targets;

#[cfg(any(target_os = "android", target_os = "ios"))]
mod mobile;

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
            #[cfg(any(target_os = "android", target_os = "ios"))]
            get_frame_cmd,
            #[cfg(not(any(target_os = "android", target_os = "ios")))]
            get_capture_targets_cmd,
            #[cfg(not(any(target_os = "android", target_os = "ios")))]
            start_capture_stream_cmd
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

// ==================== Cross-platform commands ====================

#[tauri::command]
async fn stop_capture_cmd<R: Runtime>(
    #[allow(unused_variables)] app: tauri::AppHandle<R>,
) -> Result<()> {
    #[cfg(any(target_os = "android", target_os = "ios"))]
    {
        let screen_capture = app.state::<mobile::ScreenCapture<R>>();
        return screen_capture.stop_capture();
    }

    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    return desktop::stop_capture().await;
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
#[cfg(not(any(target_os = "android", target_os = "ios")))]
#[tauri::command]
fn start_capture_stream_cmd<R: Runtime>(
    _app: tauri::AppHandle<R>,
    target_id: Option<String>,
    on_frame: tauri::ipc::Channel<desktop::FrameData>,
) -> Result<()> {
    desktop::start_capture_stream(target_id, on_frame)
}
