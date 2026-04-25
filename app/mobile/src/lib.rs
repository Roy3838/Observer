use std::sync::{Mutex, atomic::{AtomicBool, Ordering}};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{ipc::Channel, AppHandle, Manager, State};
use base64::Engine;

// Global flag to signal download cancellation
static DOWNLOAD_CANCELLED: AtomicBool = AtomicBool::new(false);

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

// ============================================================================
// LLM Engine Commands (iOS only)
// ============================================================================

/// List downloaded GGUF models from the models directory
/// Now includes multimodal detection (checks for associated mmproj files)
#[tauri::command]
async fn llm_list_models(
    app_handle: AppHandle,
) -> Result<Vec<serde_json::Value>, String> {
    #[cfg(target_os = "ios")]
    {
        use tauri_plugin_llm_engine::{NativeModelInfo, model_id_from_filename, find_mmproj_for_model};

        let models_dir = app_handle.path().app_data_dir()
            .map_err(|e| e.to_string())?
            .join("models");

        let mut models: Vec<serde_json::Value> = Vec::new();

        if models_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&models_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(ext) = path.extension() {
                        if ext == "gguf" || ext == "GGUF" {
                            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                                // Skip mmproj files - they're not standalone models
                                if filename.to_lowercase().contains("mmproj") {
                                    continue;
                                }

                                let size_bytes = std::fs::metadata(&path)
                                    .map(|m| m.len())
                                    .unwrap_or(0);

                                // Check if this model has an associated mmproj file
                                let mmproj_path = find_mmproj_for_model(&path);
                                let mmproj_filename = mmproj_path.as_ref()
                                    .and_then(|p| p.file_name())
                                    .and_then(|n| n.to_str())
                                    .map(String::from);
                                let is_multimodal = mmproj_path.is_some();

                                let model_id = model_id_from_filename(filename);
                                let info = NativeModelInfo {
                                    id: model_id.clone(),
                                    name: model_id,
                                    filename: filename.to_string(),
                                    size_bytes,
                                    hf_url: None,
                                    mmproj_filename,
                                    is_multimodal,
                                };

                                if let Ok(json) = serde_json::to_value(info) {
                                    models.push(json);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(models)
    }

    #[cfg(not(target_os = "ios"))]
    {
        let _ = app_handle;
        Ok(vec![])
    }
}

/// Download a GGUF model from a HuggingFace URL with progress reporting
#[tauri::command]
async fn llm_download_model(
    app_handle: AppHandle,
    url: String,
    on_progress: Channel<serde_json::Value>,
) -> Result<String, String> {
    #[cfg(target_os = "ios")]
    {
        use tauri_plugin_llm_engine::filename_from_hf_url;

        // Reset cancellation flag at start of new download
        DOWNLOAD_CANCELLED.store(false, Ordering::SeqCst);

        // Extract filename from URL
        let filename = filename_from_hf_url(&url)
            .ok_or_else(|| "Could not extract filename from URL".to_string())?;

        if !filename.ends_with(".gguf") && !filename.ends_with(".GGUF") {
            return Err("URL must point to a .gguf file".to_string());
        }

        let models_dir = app_handle.path().app_data_dir()
            .map_err(|e| e.to_string())?
            .join("models");

        std::fs::create_dir_all(&models_dir)
            .map_err(|e| format!("Failed to create models dir: {}", e))?;

        let output_path = models_dir.join(&filename);
        eprintln!("[LLM] Downloading {} to {:?}", url, output_path);

        // Send initial progress
        let _ = on_progress.send(serde_json::json!({
            "status": "downloading",
            "progress": 0,
            "downloadedBytes": 0,
            "totalBytes": 0,
            "filename": filename
        }));

        // Download with progress
        let client = reqwest::Client::new();
        let response = client.get(&url)
            .send()
            .await
            .map_err(|e| format!("Download failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Download failed with status: {}", response.status()));
        }

        let total_size = response.content_length().unwrap_or(0);
        let mut downloaded: u64 = 0;

        let mut file = std::fs::File::create(&output_path)
            .map_err(|e| format!("Failed to create file: {}", e))?;

        use std::io::Write;
        let mut stream = response.bytes_stream();
        use futures_util::StreamExt;

        while let Some(chunk) = stream.next().await {
            // Check if download was cancelled
            if DOWNLOAD_CANCELLED.load(Ordering::SeqCst) {
                eprintln!("[LLM] Download cancelled by user: {}", filename);
                drop(file); // Close file handle before deleting
                let _ = std::fs::remove_file(&output_path); // Delete partial file
                let _ = on_progress.send(serde_json::json!({
                    "status": "cancelled",
                    "filename": filename
                }));
                return Err("Download cancelled".to_string());
            }

            let chunk = chunk.map_err(|e| format!("Download error: {}", e))?;
            file.write_all(&chunk)
                .map_err(|e| format!("Write error: {}", e))?;

            downloaded += chunk.len() as u64;
            let progress = if total_size > 0 {
                (downloaded as f64 / total_size as f64 * 100.0) as u32
            } else {
                0
            };

            let _ = on_progress.send(serde_json::json!({
                "status": "downloading",
                "progress": progress,
                "downloadedBytes": downloaded,
                "totalBytes": total_size,
                "filename": filename
            }));
        }

        eprintln!("[LLM] Download complete: {:?}", output_path);

        let _ = on_progress.send(serde_json::json!({
            "status": "complete",
            "progress": 100,
            "downloadedBytes": downloaded,
            "totalBytes": downloaded,
            "filename": filename
        }));

        Ok(filename)
    }

    #[cfg(not(target_os = "ios"))]
    {
        let _ = (app_handle, url, on_progress);
        Err("LLM download only available on iOS".to_string())
    }
}

/// Cancel an ongoing download
#[tauri::command]
async fn llm_cancel_download() -> Result<(), String> {
    eprintln!("[LLM] Cancel download requested");
    DOWNLOAD_CANCELLED.store(true, Ordering::SeqCst);
    Ok(())
}

/// Delete a downloaded model by filename
#[tauri::command]
async fn llm_delete_model(
    app_handle: AppHandle,
    filename: String,
) -> Result<(), String> {
    #[cfg(target_os = "ios")]
    {
        let models_dir = app_handle.path().app_data_dir()
            .map_err(|e| e.to_string())?
            .join("models");

        let model_path = models_dir.join(&filename);

        if model_path.exists() {
            std::fs::remove_file(&model_path)
                .map_err(|e| format!("Failed to delete model: {}", e))?;
            eprintln!("[LLM] Deleted model: {:?}", model_path);
        }

        Ok(())
    }

    #[cfg(not(target_os = "ios"))]
    {
        let _ = (app_handle, filename);
        Err("LLM delete only available on iOS".to_string())
    }
}

/// Load a model into memory for inference by filename.
/// Optionally specify mmproj_filename explicitly — if omitted, falls back to auto-detection.
#[tauri::command]
async fn llm_load_model(
    app_handle: AppHandle,
    filename: String,
    mmproj_filename: Option<String>,
) -> Result<(), String> {
    #[cfg(target_os = "ios")]
    {
        use tauri_plugin_llm_engine::{with_engine, model_id_from_filename};

        let models_dir = app_handle.path().app_data_dir()
            .map_err(|e| e.to_string())?
            .join("models");

        let model_path = models_dir.join(&filename);

        eprintln!("[LLM] ========== LOAD MODEL START ==========");
        eprintln!("[LLM] Filename: {}", filename);
        eprintln!("[LLM] Full path: {:?}", model_path);
        eprintln!("[LLM] Path exists: {}", model_path.exists());
        eprintln!("[LLM] mmproj: {:?}", mmproj_filename);

        if !model_path.exists() {
            eprintln!("[LLM] ERROR: Model file not found!");
            return Err(format!("Model not found: {}", filename));
        }

        let file_size = std::fs::metadata(&model_path)
            .map(|m| m.len())
            .unwrap_or(0);
        eprintln!("[LLM] File size: {} bytes ({:.2} GB)", file_size, file_size as f64 / 1_073_741_824.0);

        if file_size == 0 {
            eprintln!("[LLM] ERROR: Model file is empty!");
            return Err("Model file is empty (0 bytes)".to_string());
        }

        let model_id = model_id_from_filename(&filename);
        eprintln!("[LLM] Model ID: {}", model_id);

        // Resolve explicit mmproj path if provided
        let explicit_mmproj = mmproj_filename.as_ref().map(|f| models_dir.join(f));
        if let Some(ref p) = explicit_mmproj {
            if !p.exists() {
                eprintln!("[LLM] ERROR: mmproj file not found: {:?}", p);
                return Err(format!("mmproj not found: {}", mmproj_filename.unwrap()));
            }
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_engine(|engine| {
                engine.load_model(model_path.clone(), model_id.clone(), explicit_mmproj.clone())
            })
        }));

        match result {
            Ok(Ok(())) => {
                eprintln!("[LLM] ✅ Model loaded successfully!");
                eprintln!("[LLM] ========== LOAD MODEL END ==========");
                Ok(())
            }
            Ok(Err(e)) => {
                eprintln!("[LLM] ❌ Load error: {}", e);
                eprintln!("[LLM] ========== LOAD MODEL END ==========");
                Err(e)
            }
            Err(panic_info) => {
                let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };
                eprintln!("[LLM] 💥 PANIC during load: {}", panic_msg);
                eprintln!("[LLM] ========== LOAD MODEL END ==========");
                Err(format!("Panic during model load: {}", panic_msg))
            }
        }
    }

    #[cfg(not(target_os = "ios"))]
    {
        let _ = (app_handle, filename, mmproj_filename);
        Err("LLM load only available on iOS".to_string())
    }
}

/// Generate text from messages with streaming tokens
#[tauri::command]
async fn llm_generate(
    messages: Vec<serde_json::Value>,
    on_token: Channel<String>,
) -> Result<String, String> {
    #[cfg(target_os = "ios")]
    {
        use tauri_plugin_llm_engine::{with_engine, ChatMessage, ChatContent, ChatContentPart, LLM_ENGINE};

        eprintln!("[LLM] ========== GENERATE START ==========");
        eprintln!("[LLM] Received {} raw messages", messages.len());

        // Check engine state first
        {
            let guard = LLM_ENGINE.lock().map_err(|e| format!("Lock error: {}", e))?;
            match guard.as_ref() {
                Some(engine) => {
                    eprintln!("[LLM] Engine exists, is_loaded: {}", engine.is_loaded());
                    if !engine.is_loaded() {
                        eprintln!("[LLM] ❌ ERROR: Engine exists but no model loaded!");
                        return Err("No model loaded - please load a model first".to_string());
                    }
                }
                None => {
                    eprintln!("[LLM] ❌ ERROR: Engine not initialized!");
                    return Err("LLM engine not initialized".to_string());
                }
            }
        }

        // Parse messages from JSON - supports both text-only and multimodal content
        let chat_messages: Vec<ChatMessage> = messages.into_iter()
            .filter_map(|m| {
                let role = m.get("role")?.as_str()?.to_string();
                let content_value = m.get("content")?;

                // Content can be either a string or an array of content parts
                let content = if let Some(text) = content_value.as_str() {
                    // Simple text content
                    eprintln!("[LLM] Message: role={}, content_len={}", role, text.len());
                    ChatContent::Text(text.to_string())
                } else if let Some(parts_array) = content_value.as_array() {
                    // Multimodal content parts
                    let parts: Vec<ChatContentPart> = parts_array.iter()
                        .filter_map(|part| {
                            let part_type = part.get("type")?.as_str()?;
                            match part_type {
                                "text" => {
                                    let text = part.get("text")?.as_str()?.to_string();
                                    Some(ChatContentPart::Text { text })
                                }
                                "image" => {
                                    let image = part.get("image")?.as_str()?.to_string();
                                    Some(ChatContentPart::Image { image })
                                }
                                _ => None
                            }
                        })
                        .collect();
                    eprintln!("[LLM] Message: role={}, multimodal parts={}", role, parts.len());
                    ChatContent::Parts(parts)
                } else {
                    return None;
                };

                Some(ChatMessage { role, content })
            })
            .collect();

        if chat_messages.is_empty() {
            eprintln!("[LLM] ❌ ERROR: No valid messages after parsing!");
            return Err("No valid messages provided".to_string());
        }

        eprintln!("[LLM] Parsed {} chat messages, starting generation...", chat_messages.len());

        // Use catch_unwind to catch any panics from llama.cpp
        let on_token_clone = on_token.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_engine(|engine| {
                engine.generate(chat_messages, 2048, |token| {
                    let _ = on_token_clone.send(token.to_string());
                })
            })
        }));

        match result {
            Ok(Ok(response)) => {
                eprintln!("[LLM] ✅ Generation complete, response_len={}", response.len());
                eprintln!("[LLM] ========== GENERATE END ==========");
                Ok(response)
            }
            Ok(Err(e)) => {
                eprintln!("[LLM] ❌ Generation error: {}", e);
                eprintln!("[LLM] ========== GENERATE END ==========");
                Err(e)
            }
            Err(panic_info) => {
                let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };
                eprintln!("[LLM] 💥 PANIC during generation: {}", panic_msg);
                eprintln!("[LLM] ========== GENERATE END ==========");
                Err(format!("Panic during generation: {}", panic_msg))
            }
        }
    }

    #[cfg(not(target_os = "ios"))]
    {
        let _ = (messages, on_token);
        Err("LLM generate only available on iOS".to_string())
    }
}

/// Unload the current model to free memory
#[tauri::command]
async fn llm_unload_model() -> Result<(), String> {
    #[cfg(target_os = "ios")]
    {
        use tauri_plugin_llm_engine::with_engine;

        with_engine(|engine| {
            engine.unload();
            Ok(())
        })?;

        eprintln!("[LLM] Model unloaded");
        Ok(())
    }

    #[cfg(not(target_os = "ios"))]
    {
        Ok(())
    }
}

/// Check if a model is currently loaded
#[tauri::command]
async fn llm_is_loaded() -> Result<bool, String> {
    #[cfg(target_os = "ios")]
    {
        use tauri_plugin_llm_engine::with_engine;

        with_engine(|engine| Ok(engine.is_loaded()))
    }

    #[cfg(not(target_os = "ios"))]
    {
        Ok(false)
    }
}

/// Check if the loaded model supports multimodal (vision)
#[tauri::command]
async fn llm_is_multimodal() -> Result<bool, String> {
    #[cfg(target_os = "ios")]
    {
        use tauri_plugin_llm_engine::with_engine;

        with_engine(|engine| Ok(engine.is_multimodal()))
    }

    #[cfg(not(target_os = "ios"))]
    {
        Ok(false)
    }
}

/// Test LLM backend initialization - call this to check if llama.cpp works
#[tauri::command]
async fn llm_test_init() -> Result<String, String> {
    #[cfg(target_os = "ios")]
    {
        eprintln!("[LLM TEST] ========== TESTING BACKEND INIT ==========");

        // Try to catch any panic during initialization
        let result = std::panic::catch_unwind(|| {
            eprintln!("[LLM TEST] Calling init_engine()...");
            tauri_plugin_llm_engine::init_engine()
        });

        match result {
            Ok(Ok(())) => {
                eprintln!("[LLM TEST] ✅ Backend initialized successfully!");
                Ok("Backend initialized successfully".to_string())
            }
            Ok(Err(e)) => {
                let err_msg = format!("Backend init error: {}", e);
                eprintln!("[LLM TEST] ❌ {}", err_msg);
                Err(err_msg)
            }
            Err(panic) => {
                let panic_msg = if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };
                eprintln!("[LLM TEST] 💥 PANIC: {}", panic_msg);
                Err(format!("Panic during init: {}", panic_msg))
            }
        }
    }

    #[cfg(not(target_os = "ios"))]
    {
        Ok("LLM not available on this platform".to_string())
    }
}

/// Debug command to get detailed LLM engine state for frontend debugging
#[tauri::command]
async fn llm_debug_state(app_handle: AppHandle) -> Result<serde_json::Value, String> {
    #[cfg(target_os = "ios")]
    {
        use tauri_plugin_llm_engine::LLM_ENGINE;

        // Check models directory
        let models_dir = app_handle.path().app_data_dir()
            .map_err(|e| e.to_string())?
            .join("models");

        let models_exist = models_dir.exists();
        let model_files: Vec<String> = if models_exist {
            std::fs::read_dir(&models_dir)
                .map(|entries| {
                    entries.flatten()
                        .filter_map(|e| e.file_name().to_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default()
        } else {
            vec![]
        };

        // Check engine state
        let engine_state = match LLM_ENGINE.lock() {
            Ok(guard) => {
                match guard.as_ref() {
                    Some(engine) => serde_json::json!({
                        "initialized": true,
                        "isLoaded": engine.is_loaded(),
                        "loadedModelId": engine.loaded_model_id(),
                        "isMultimodal": engine.is_multimodal(),
                    }),
                    None => serde_json::json!({
                        "initialized": false,
                        "isLoaded": false,
                        "loadedModelId": null,
                        "isMultimodal": false,
                    }),
                }
            },
            Err(e) => serde_json::json!({
                "error": format!("Lock poisoned: {}", e),
            }),
        };

        Ok(serde_json::json!({
            "modelsDir": models_dir.to_string_lossy(),
            "modelsDirExists": models_exist,
            "modelFiles": model_files,
            "engine": engine_state,
        }))
    }

    #[cfg(not(target_os = "ios"))]
    {
        let _ = app_handle;
        Ok(serde_json::json!({
            "platform": "not-ios",
            "llmAvailable": false,
        }))
    }
}

/// Get comprehensive debug info including sampler params and metrics
#[tauri::command]
async fn llm_get_debug_info(app_handle: AppHandle) -> Result<serde_json::Value, String> {
    #[cfg(target_os = "ios")]
    {
        use tauri_plugin_llm_engine::LLM_ENGINE;

        let models_dir = app_handle.path().app_data_dir()
            .map_err(|e| e.to_string())?
            .join("models");

        let engine_state = match LLM_ENGINE.lock() {
            Ok(guard) => {
                match guard.as_ref() {
                    Some(engine) => {
                        let sampler_params = engine.get_sampler_params();
                        let last_metrics = engine.get_last_metrics();
                        let model_path = engine.get_model_path()
                            .map(|p| p.to_string_lossy().to_string());
                        let mmproj_path = engine.get_mmproj_path()
                            .map(|p| p.to_string_lossy().to_string());

                        serde_json::json!({
                            "initialized": true,
                            "isLoaded": engine.is_loaded(),
                            "loadedModelId": engine.loaded_model_id(),
                            "isMultimodal": engine.is_multimodal(),
                            "modelPath": model_path,
                            "mmprojPath": mmproj_path,
                            "samplerParams": {
                                "temperature": sampler_params.temperature,
                                "topP": sampler_params.top_p,
                                "topK": sampler_params.top_k,
                                "seed": sampler_params.seed,
                                "repeatPenalty": sampler_params.repeat_penalty,
                            },
                            "lastMetrics": last_metrics.map(|m| serde_json::json!({
                                "tokensGenerated": m.tokens_generated,
                                "promptTokens": m.prompt_tokens,
                                "timeToFirstTokenMs": m.time_to_first_token_ms,
                                "totalGenerationTimeMs": m.total_generation_time_ms,
                                "tokensPerSecond": m.tokens_per_second,
                            })),
                        })
                    },
                    None => serde_json::json!({
                        "initialized": false,
                        "isLoaded": false,
                        "loadedModelId": null,
                        "isMultimodal": false,
                        "modelPath": null,
                        "mmprojPath": null,
                        "samplerParams": null,
                        "lastMetrics": null,
                    }),
                }
            },
            Err(e) => serde_json::json!({
                "error": format!("Lock poisoned: {}", e),
            }),
        };

        Ok(serde_json::json!({
            "modelsDir": models_dir.to_string_lossy(),
            "engine": engine_state,
        }))
    }

    #[cfg(not(target_os = "ios"))]
    {
        let _ = app_handle;
        Ok(serde_json::json!({
            "platform": "not-ios",
            "llmAvailable": false,
        }))
    }
}

/// Set sampler parameters for text generation
#[tauri::command]
async fn llm_set_sampler_params(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    seed: Option<u32>,
    repeat_penalty: Option<f32>,
) -> Result<(), String> {
    #[cfg(target_os = "ios")]
    {
        use tauri_plugin_llm_engine::{with_engine, SamplerParams};

        with_engine(|engine| {
            let current = engine.get_sampler_params().clone();
            let new_params = SamplerParams {
                temperature: temperature.unwrap_or(current.temperature),
                top_p: top_p.unwrap_or(current.top_p),
                top_k: top_k.unwrap_or(current.top_k),
                seed: seed.unwrap_or(current.seed),
                repeat_penalty: repeat_penalty.unwrap_or(current.repeat_penalty),
            };
            engine.set_sampler_params(new_params);
            Ok(())
        })
    }

    #[cfg(not(target_os = "ios"))]
    {
        let _ = (temperature, top_p, top_k, seed, repeat_penalty);
        Err("LLM only available on iOS".to_string())
    }
}

/// Set whether to use GPU acceleration (Metal)
/// Must be called before loading a model to take effect
#[tauri::command]
async fn llm_set_use_gpu(use_gpu: bool) -> Result<(), String> {
    #[cfg(target_os = "ios")]
    {
        use tauri_plugin_llm_engine::with_engine;

        with_engine(|engine| {
            engine.set_use_gpu(use_gpu);
            Ok(())
        })
    }

    #[cfg(not(target_os = "ios"))]
    {
        let _ = use_gpu;
        Err("LLM only available on iOS".to_string())
    }
}

/// Get whether GPU acceleration is enabled
#[tauri::command]
async fn llm_get_use_gpu() -> Result<bool, String> {
    #[cfg(target_os = "ios")]
    {
        use tauri_plugin_llm_engine::with_engine;

        with_engine(|engine| {
            Ok(engine.get_use_gpu())
        })
    }

    #[cfg(not(target_os = "ios"))]
    {
        Err("LLM only available on iOS".to_string())
    }
}

/// Test generation with a simple prompt, returns response and metrics
#[tauri::command]
async fn llm_test_generate(
    prompt: String,
    max_tokens: Option<u32>,
    on_token: Channel<String>,
) -> Result<serde_json::Value, String> {
    #[cfg(target_os = "ios")]
    {
        use tauri_plugin_llm_engine::{with_engine, ChatMessage, ChatContent};

        let max_tokens = max_tokens.unwrap_or(256);

        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: ChatContent::Text(prompt),
        }];

        let on_token_clone = on_token.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_engine(|engine| {
                let response = engine.generate(messages, max_tokens, |token| {
                    let _ = on_token_clone.send(token.to_string());
                })?;

                let metrics = engine.get_last_metrics().map(|m| serde_json::json!({
                    "tokensGenerated": m.tokens_generated,
                    "promptTokens": m.prompt_tokens,
                    "timeToFirstTokenMs": m.time_to_first_token_ms,
                    "totalGenerationTimeMs": m.total_generation_time_ms,
                    "tokensPerSecond": m.tokens_per_second,
                }));

                Ok(serde_json::json!({
                    "response": response,
                    "metrics": metrics,
                }))
            })
        }));

        match result {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(e)) => Err(e),
            Err(panic_info) => {
                let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };
                Err(format!("Panic during test generation: {}", panic_msg))
            }
        }
    }

    #[cfg(not(target_os = "ios"))]
    {
        let _ = (prompt, max_tokens, on_token);
        Err("LLM only available on iOS".to_string())
    }
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
            read_broadcast_debug_log,
            // LLM commands 
            llm_list_models,
            llm_download_model,
            llm_cancel_download,
            llm_delete_model,
            llm_load_model,
            llm_generate,
            llm_unload_model,
            llm_is_loaded,
            llm_is_multimodal,
            llm_debug_state,
            llm_test_init,
            // LLM debug commands
            llm_get_debug_info,
            llm_set_sampler_params,
            llm_set_use_gpu,
            llm_get_use_gpu,
            llm_test_generate
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
