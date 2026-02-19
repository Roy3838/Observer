use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{ipc::Channel, AppHandle, Manager, State};
use base64::Engine;

mod server;
use server::{FrameData, ServerState, start_server};

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
/// Frames from the broadcast extension will be pushed through the channel
#[tauri::command]
async fn start_capture_stream_cmd(
    state: State<'_, ServerState>,
    on_frame: Channel<FrameData>,
) -> Result<(), String> {
    eprintln!("🎬 Starting capture stream with channel");

    // Store the channel so HTTP handler can push frames to it
    state.set_frame_channel(Some(on_frame)).await;

    Ok(())
}

/// Stop the capture stream channel
#[tauri::command]
async fn stop_capture_stream_cmd(
    state: State<'_, ServerState>,
) -> Result<(), String> {
    eprintln!("🛑 Stopping capture stream channel");

    // Clear the channel
    state.set_frame_channel(None).await;

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

    tauri::Builder::default()
        .plugin(tauri_plugin_screen_capture::init())
        .plugin(tauri_plugin_pip::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_edge_to_edge::init())
        .plugin(tauri_plugin_deep_link::init())
        .plugin(tauri_plugin_auth::init())
        .plugin(tauri_plugin_ios_keyboard::init())
        .plugin(tauri_plugin_iap::init())
        .plugin(tauri_plugin_opener::init())
        .setup(move |app| {
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
            stop_capture_stream_cmd
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
