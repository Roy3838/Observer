use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{ipc::Channel, AppHandle, Manager, State};
use base64::Engine;

mod server;
use server::{AudioData, FrameData, ServerState, start_server};

#[cfg(target_os = "ios")]
mod audio_ring;

#[cfg(target_os = "ios")]
mod video_frame;

pub struct AppSettings {
    pub ollama_url: Mutex<Option<String>>,
}

#[tauri::command]
async fn set_ollama_url(
    new_url: Option<String>,
    settings: State<'_, AppSettings>,
    app_handle: AppHandle,
) -> Result<(), String> {
    eprintln!("set_ollama_url called with: {:?}", new_url);

    // Update in-memory
    *settings.ollama_url.lock().unwrap() = new_url.clone();
    eprintln!("Updated in-memory ollama_url");

    // Persist to file
    let config_path = app_handle.path().app_data_dir()
        .map_err(|e| {
            eprintln!("ERROR getting app_data_dir: {}", e);
            e.to_string()
        })?
        .join("settings.json");

    eprintln!("Config path: {:?}", config_path);

    // Ensure directory exists
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            eprintln!("ERROR creating directory: {}", e);
            e.to_string()
        })?;
    }

    let config = serde_json::json!({ "ollama_url": new_url });
    std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap())
        .map_err(|e| {
            eprintln!("ERROR writing settings: {}", e);
            e.to_string()
        })?;

    eprintln!("Saved ollama_url to {:?}", config_path);
    Ok(())
}

#[tauri::command]
async fn get_ollama_url(
    settings: State<'_, AppSettings>,
) -> Result<Option<String>, String> {
    Ok(settings.ollama_url.lock().unwrap().clone())
}

/// Unified command: returns broadcast state + latest frame in one call
#[tauri::command]
async fn get_broadcast_status(
    state: State<'_, ServerState>
) -> Result<serde_json::Value, String> {
    let broadcast = state.broadcast.read().await;
    let frame = state.latest_frame.read().await;

    let current_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    // Consider stale if active but no frames for >3 seconds
    let is_stale = broadcast.is_active && broadcast.last_frame_at
        .map(|t| current_time - t > 3.0)
        .unwrap_or(true);

    // Build frame data if available
    let (frame_base64, frame_timestamp) = match frame.as_ref() {
        Some((data, timestamp)) => {
            let base64 = base64::prelude::BASE64_STANDARD.encode(data);
            (Some(base64), Some(*timestamp))
        }
        None => (None, None)
    };

    Ok(serde_json::json!({
        "isActive": broadcast.is_active,
        "isStale": is_stale,
        "frame": frame_base64,
        "timestamp": frame_timestamp,
        "frameCount": broadcast.frame_count
    }))
}

/// Start capture stream with channel-based frame delivery
/// On iOS: Uses shared memory buffer (written by broadcast extension)
/// On Android: Sets up JNI channel for frame callbacks
#[tauri::command]
async fn start_capture_stream_cmd(
    state: State<'_, ServerState>,
    on_frame: Channel<FrameData>,
    #[allow(unused_variables)] app_group_path: Option<String>,
) -> Result<(), String> {
    eprintln!("Starting capture stream with channel");

    // Store the channel in ServerState (used by iOS)
    state.set_frame_channel(Some(on_frame.clone())).await;

    // On iOS, start the video frame reader
    #[cfg(target_os = "ios")]
    {
        // Set App Group path if provided
        if let Some(path) = app_group_path {
            eprintln!("Setting App Group path for video: {}", path);
            video_frame::set_app_group_path(std::path::PathBuf::from(path));
        }

        eprintln!("Starting iOS video frame reader");
        let reader_state = std::sync::Arc::new(video_frame::VideoFrameReaderState::new(
            state.frame_channel.clone()
        ));
        video_frame::start_video_frame_reader(reader_state);
    }

    // On Android, set up the JNI channel for frame callbacks
    #[cfg(target_os = "android")]
    {
        eprintln!("Setting up Android JNI frame channel");
        // SAFETY: Both FrameData types (server::FrameData and android::FrameData) have
        // identical structure and serde serialization. Channel<T> serializes T to JSON,
        // so as long as the JSON format matches (which it does), this transmute is safe.
        // Both types have: frame (Vec<u8> with serde_bytes), timestamp (f64), width (u32),
        // height (u32), frame_count (u64) - all with camelCase rename.
        let android_channel: tauri::ipc::Channel<tauri_plugin_screen_capture::android::FrameData> =
            unsafe { std::mem::transmute(on_frame) };
        tauri_plugin_screen_capture::android::set_frame_channel(android_channel);
    }

    Ok(())
}

/// Stop the capture stream channel
#[tauri::command]
async fn stop_capture_stream_cmd(
    state: State<'_, ServerState>,
) -> Result<(), String> {
    eprintln!("Stopping capture stream channel");

    // On iOS, stop the video frame reader
    #[cfg(target_os = "ios")]
    {
        video_frame::stop_video_frame_reader();
    }

    // On Android, clear the JNI frame channel
    #[cfg(target_os = "android")]
    {
        tauri_plugin_screen_capture::android::clear_frame_channel();
    }

    // Clear the channel in ServerState
    state.set_frame_channel(None).await;

    Ok(())
}

/// Read broadcast extension debug log from App Group container
#[tauri::command]
async fn read_broadcast_debug_log() -> Result<String, String> {
    #[cfg(target_os = "ios")]
    {
        // Scan App Group containers for the debug log
        let shared_containers = std::path::PathBuf::from("/private/var/mobile/Containers/Shared/AppGroup");
        if shared_containers.exists() {
            if let Ok(entries) = std::fs::read_dir(&shared_containers) {
                for entry in entries.flatten() {
                    let log_path = entry.path().join("broadcast_debug.log");
                    if log_path.exists() {
                        match std::fs::read_to_string(&log_path) {
                            Ok(content) => return Ok(content),
                            Err(e) => return Err(format!("Failed to read log: {}", e)),
                        }
                    }
                }
            }
        }
        Ok("No broadcast_debug.log found in App Group containers".to_string())
    }

    #[cfg(not(target_os = "ios"))]
    {
        Ok("Debug log only available on iOS".to_string())
    }
}

/// Set the App Group container path (called from iOS to enable shared memory audio)
#[tauri::command]
async fn set_app_group_path_cmd(path: String) -> Result<(), String> {
    eprintln!("📁 Setting App Group path: {}", path);

    #[cfg(target_os = "ios")]
    {
        audio_ring::set_app_group_path(std::path::PathBuf::from(path));
    }

    #[cfg(not(target_os = "ios"))]
    {
        let _ = path; // Silence unused variable warning
        eprintln!("📁 App Group path ignored on non-iOS");
    }

    Ok(())
}

/// Start audio stream with channel-based audio delivery
/// On iOS: Uses shared memory ring buffer (written by broadcast extension)
/// On Android: Sets up JNI channel for audio callbacks
#[tauri::command]
async fn start_audio_stream_cmd(
    state: State<'_, ServerState>,
    on_audio: Channel<AudioData>,
    #[allow(unused_variables)] app_group_path: Option<String>,
) -> Result<(), String> {
    eprintln!("🎵 Starting audio stream with channel");

    // Store the channel in ServerState (used by iOS)
    state.set_audio_channel(Some(on_audio.clone())).await;

    // On iOS, start the ring buffer reader
    #[cfg(target_os = "ios")]
    {
        // Set App Group path if provided
        if let Some(path) = app_group_path {
            eprintln!("🎵 Setting App Group path: {}", path);
            audio_ring::set_app_group_path(std::path::PathBuf::from(path));
        }

        eprintln!("🎵 Starting iOS audio ring buffer reader");
        let reader_state = std::sync::Arc::new(audio_ring::AudioRingReaderState::new(
            state.audio_channel.clone()
        ));
        audio_ring::start_audio_ring_reader(reader_state);
    }

    // On Android, set up the JNI channel for audio callbacks
    #[cfg(target_os = "android")]
    {
        eprintln!("🎵 Setting up Android JNI audio channel");
        // SAFETY: Both AudioData types (server::AudioData and android::AudioData) serialize
        // to compatible JSON. The main difference is field names but both use camelCase.
        // server::AudioData has: samples, timestamp, sample_rate, channels, chunk_count
        // android::AudioData has: samples, timestamp, sample_rate, sample_count
        // The frontend handles both formats, so this transmute is safe for the channel.
        let android_channel: tauri::ipc::Channel<tauri_plugin_screen_capture::android::AudioData> =
            unsafe { std::mem::transmute(on_audio) };
        tauri_plugin_screen_capture::android::set_audio_channel(android_channel);
    }

    Ok(())
}

/// Stop the audio stream channel
#[tauri::command]
async fn stop_audio_stream_cmd(
    state: State<'_, ServerState>,
) -> Result<(), String> {
    eprintln!("🔇 Stopping audio stream channel");

    // On iOS, stop the ring buffer reader
    #[cfg(target_os = "ios")]
    {
        audio_ring::stop_audio_ring_reader();
    }

    // On Android, clear the JNI audio channel
    #[cfg(target_os = "android")]
    {
        tauri_plugin_screen_capture::android::clear_audio_channel();
    }

    // Clear the channel in ServerState
    state.set_audio_channel(None).await;

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // EARLY LOG - Check if app is starting
    eprintln!("🚀🚀🚀 OBSERVER APP STARTING 🚀🚀🚀");

    // Initialize server state for broadcast frames
    let server_state = ServerState::new();
    let server_state_for_setup = server_state.clone();

    eprintln!("📦 ServerState created, spawning frame server...");

    let mut builder = tauri::Builder::default()
        .plugin(tauri_plugin_screen_capture::init())
        .plugin(tauri_plugin_pip::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_edge_to_edge::init())
        .plugin(tauri_plugin_deep_link::init())
        .plugin(tauri_plugin_web_auth::init())
        .plugin(tauri_plugin_iap::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Debug)
            .build());

    #[cfg(target_os = "ios")]
    {
        builder = builder.plugin(tauri_plugin_ios_keyboard::init());
    }

    builder.setup(move |app| {
            // Load persisted ollama_url from settings.json
            let config_path = app.path().app_data_dir()
                .ok()
                .map(|p| p.join("settings.json"));

            eprintln!("Looking for settings at: {:?}", config_path);

            let ollama_url = config_path
                .and_then(|path| {
                    let result = std::fs::read_to_string(&path);
                    eprintln!("Read settings file result: {:?}", result.as_ref().map(|_| "OK").map_err(|e| e.to_string()));
                    result.ok()
                })
                .and_then(|s| {
                    eprintln!("Settings content: {}", s);
                    serde_json::from_str::<serde_json::Value>(&s).ok()
                })
                .and_then(|v| v["ollama_url"].as_str().map(String::from))
                .or_else(|| Some("http://localhost:11434".to_string()));

            eprintln!("Loaded ollama_url: {:?}", ollama_url);

            app.manage(AppSettings {
                ollama_url: Mutex::new(ollama_url),
            });

            // Start HTTP server in background using Tauri's async runtime
            eprintln!("🌐 About to spawn server task...");
            let server_state_clone = server_state_for_setup.clone();
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                eprintln!("🔥 Server task starting...");
                start_server(server_state_clone, app_handle).await;
                eprintln!("⚠️ Server task ended (this shouldn't happen)");
            });
            eprintln!("✅ Server task spawned");

            Ok(())
        })
        .manage(server_state) // Make state available to commands
        .invoke_handler(tauri::generate_handler![
            set_ollama_url,
            get_ollama_url,
            get_broadcast_status,
            start_capture_stream_cmd,
            stop_capture_stream_cmd,
            start_audio_stream_cmd,
            stop_audio_stream_cmd,
            set_app_group_path_cmd,
            read_broadcast_debug_log
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
