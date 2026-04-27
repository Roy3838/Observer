// In src-tauri/src/lib.rs

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod controls;
mod install_cli;
mod notifications;
mod overlay;
mod shortcuts;

// Import unified shortcut types (desktop only)
use shortcuts::UnifiedShortcutState;

// ---- Final, Corrected Imports (Desktop only) ----
use axum::{
    body::Body,
    extract::State as AxumState,
    http::{HeaderMap, Method, StatusCode, Uri},
    response::Response,
    routing::any,
    Router,
};
use futures::future::join_all;

use http_body_util::BodyExt;

use reqwest::Client;
use std::sync::{Mutex, atomic::{AtomicBool, Ordering}};
use tauri::{AppHandle, Emitter, Manager, State};

// Global flag to signal download cancellation
static DOWNLOAD_CANCELLED: AtomicBool = AtomicBool::new(false);
// Global flag to signal generation cancellation
static GENERATION_CANCELLED: AtomicBool = AtomicBool::new(false);

use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    WebviewUrl, WebviewWindowBuilder,
};

use tauri_plugin_dialog::DialogExt;

use tauri_plugin_updater::UpdaterExt;

use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
};

struct AppSettings {
    ollama_url: Mutex<Option<String>>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct OverlayMessage {
    id: String,
    content: String,
    timestamp: u64,
}

struct OverlayState {
    messages: Mutex<Vec<OverlayMessage>>,
}

use tokio::sync::broadcast;

#[derive(Clone, serde::Serialize, Debug)]
pub struct CommandMessage {
    #[serde(rename = "type")]
    pub message_type: String,
    #[serde(rename = "agentId")]
    pub agent_id: String,
    pub action: String,
}

struct CommandState {
    pending_commands: Mutex<std::collections::HashMap<String, String>>,
    // SSE broadcast channel for real-time commands
    command_broadcaster: broadcast::Sender<CommandMessage>,
}

#[tauri::command]
async fn set_ollama_url(
    new_url: Option<String>,
    settings: State<'_, AppSettings>,
    shortcut_state: State<'_, UnifiedShortcutState>,
    app_handle: AppHandle,
) -> Result<(), String> {
    log::info!("Setting Ollama URL to: {:?}", new_url);

    // Update in-memory AppSettings
    *settings.ollama_url.lock().unwrap() = new_url.clone();

    // Persist to disk (also updates UnifiedShortcutState)
    shortcuts::save_ollama_url(&app_handle, &shortcut_state, new_url)?;

    Ok(())
}

#[tauri::command]
async fn get_ollama_url(settings: State<'_, AppSettings>) -> Result<Option<String>, String> {
    log::info!("Getting Ollama URL");
    // Lock the mutex, clone the value inside, and return it.
    // We clone so we don't hold the lock longer than necessary.
    let url = settings.ollama_url.lock().unwrap().clone();
    Ok(url)
}

#[tauri::command]
async fn check_ollama_servers(urls: Vec<String>) -> Result<Vec<String>, String> {
    // <-- No State parameter
    log::info!(
        "Rust backend received request to check servers (using dedicated client): {:?}",
        urls
    );

    // Create a new, temporary client just for this operation.
    let client = Client::new();

    // The rest of the logic is identical.
    let checks = urls.into_iter().map(|url| {
        let client = client.clone();
        let check_url = format!("{}/v1/models", url);

        tokio::spawn(async move {
            match client
                .get(&check_url)
                .timeout(std::time::Duration::from_millis(2500))
                .send()
                .await
            {
                Ok(response) if response.status().is_success() => {
                    log::info!("Success checking server at {}", url);
                    Some(url)
                }
                Ok(response) => {
                    log::warn!("Failed check for {}: Status {}", url, response.status());
                    None
                }
                Err(e) => {
                    log::warn!("Failed check for {}: Error: {}", url, e);
                    None
                }
            }
        })
    });

    let results = join_all(checks).await;

    let successful_urls: Vec<String> = results
        .into_iter()
        .filter_map(|res| res.ok().flatten())
        .collect();

    log::info!("Found running servers at: {:?}", successful_urls);

    Ok(successful_urls)
}

/// Get broadcast status (matches mobile API)
#[tauri::command]
async fn get_broadcast_status() -> Result<serde_json::Value, String> {
    tauri_plugin_screen_capture::desktop::get_broadcast_status()
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_overlay_messages(
    overlay_state: State<'_, OverlayState>,
) -> Result<Vec<OverlayMessage>, String> {
    log::info!("Getting overlay messages");
    let messages = overlay_state.messages.lock().unwrap().clone();
    Ok(messages)
}

#[tauri::command]
async fn clear_overlay_messages(
    overlay_state: State<'_, OverlayState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    log::info!("Clearing overlay messages");
    overlay_state.messages.lock().unwrap().clear();

    // Emit event to notify frontend of cleared messages
    let empty_messages: Vec<OverlayMessage> = vec![];
    if let Err(e) = app_handle.emit("overlay-messages-updated", &empty_messages) {
        log::warn!(
            "Failed to emit overlay-messages-updated event after clear: {}",
            e
        );
    } else {
        log::debug!("Emitted overlay-messages-updated event with 0 messages after clear");
    }

    Ok(())
}

#[tauri::command]
async fn show_overlay(app_handle: tauri::AppHandle) -> Result<(), String> {
    log::info!("Showing overlay window");
    if let Some(window) = app_handle.get_webview_window("overlay") {
        window.show().map_err(|e| e.to_string())?;
        // Re-enable click-through after showing
        if let Err(e) = window.set_ignore_cursor_events(true) {
            log::warn!("Failed to enable click-through on overlay: {}", e);
        }
        Ok(())
    } else {
        Err("Overlay window not found".to_string())
    }
}

#[tauri::command]
async fn hide_overlay(app_handle: tauri::AppHandle) -> Result<(), String> {
    log::info!("Hiding overlay window");
    if let Some(window) = app_handle.get_webview_window("overlay") {
        window.hide().map_err(|e| e.to_string())
    } else {
        Err("Overlay window not found".to_string())
    }
}

// Shortcut commands moved to shortcuts module

// Shortcut helper functions moved to shortcuts module

// ============================================================================
// LLM Engine Commands (using shared plugin)
// ============================================================================

use tauri::ipc::Channel;

/// List all GGUF files in the models directory (models and projectors alike).
/// No filtering or auto-detection — the frontend decides how to use each file.
#[tauri::command]
async fn llm_list_gguf(app_handle: AppHandle) -> Result<Vec<serde_json::Value>, String> {
    let models_dir = app_handle.path().app_data_dir()
        .map_err(|e| e.to_string())?
        .join("models");

    let mut files: Vec<serde_json::Value> = Vec::new();

    if models_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "gguf" || ext == "GGUF" {
                        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                            let size_bytes = std::fs::metadata(&path)
                                .map(|m| m.len())
                                .unwrap_or(0);

                            files.push(serde_json::json!({
                                "filename": filename,
                                "sizeBytes": size_bytes,
                            }));
                        }
                    }
                }
            }
        }
    }

    Ok(files)
}

/// Download a GGUF model from a HuggingFace URL with progress reporting
#[tauri::command]
async fn llm_download_model(
    app_handle: AppHandle,
    url: String,
    on_progress: Channel<serde_json::Value>,
) -> Result<String, String> {
    use tauri_plugin_llm_engine::filename_from_hf_url;
    use futures_util::StreamExt;

    // Reset cancellation flag at start of new download
    DOWNLOAD_CANCELLED.store(false, Ordering::SeqCst);

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

    let _ = on_progress.send(serde_json::json!({
        "status": "downloading",
        "progress": 0,
        "downloadedBytes": 0,
        "totalBytes": 0,
        "filename": filename
    }));

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

    while let Some(chunk) = stream.next().await {
        // Check if download was cancelled
        if DOWNLOAD_CANCELLED.load(Ordering::SeqCst) {
            log::info!("Download cancelled by user: {}", filename);
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

    let _ = on_progress.send(serde_json::json!({
        "status": "complete",
        "progress": 100,
        "downloadedBytes": downloaded,
        "totalBytes": downloaded,
        "filename": filename
    }));

    Ok(filename)
}

/// Cancel an ongoing download
#[tauri::command]
async fn llm_cancel_download() -> Result<(), String> {
    log::info!("Cancel download requested");
    DOWNLOAD_CANCELLED.store(true, Ordering::SeqCst);
    Ok(())
}

/// Cancel an ongoing generation
#[tauri::command]
async fn llm_cancel_generation() -> Result<(), String> {
    log::info!("Cancel generation requested");
    GENERATION_CANCELLED.store(true, Ordering::SeqCst);
    Ok(())
}

/// Delete a downloaded model by filename
#[tauri::command]
async fn llm_delete_model(app_handle: AppHandle, filename: String) -> Result<(), String> {
    let models_dir = app_handle.path().app_data_dir()
        .map_err(|e| e.to_string())?
        .join("models");

    let model_path = models_dir.join(&filename);

    if model_path.exists() {
        std::fs::remove_file(&model_path)
            .map_err(|e| format!("Failed to delete model: {}", e))?;
    }

    Ok(())
}

/// Load a model into memory for inference
#[tauri::command]
async fn llm_load_model(
    app_handle: AppHandle,
    filename: String,
    mmproj_filename: Option<String>,
) -> Result<(), String> {
    use tauri_plugin_llm_engine::{with_engine, model_id_from_filename};

    let models_dir = app_handle.path().app_data_dir()
        .map_err(|e| e.to_string())?
        .join("models");

    let model_path = models_dir.join(&filename);

    if !model_path.exists() {
        return Err(format!("Model not found: {}", filename));
    }

    let model_id = model_id_from_filename(&filename);
    let explicit_mmproj = mmproj_filename.as_ref().map(|f| models_dir.join(f));

    if let Some(ref p) = explicit_mmproj {
        if !p.exists() {
            return Err(format!("mmproj not found: {}", mmproj_filename.unwrap()));
        }
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_engine(|engine| {
            engine.load_model(model_path.clone(), model_id.clone(), explicit_mmproj.clone())
        })
    }));

    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(panic_info) => {
            let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            Err(format!("Panic during model load: {}", panic_msg))
        }
    }
}

/// Generate text from messages with streaming tokens
#[tauri::command]
async fn llm_generate(
    messages: Vec<serde_json::Value>,
    enable_thinking: bool,
    on_token: Channel<String>,
) -> Result<serde_json::Value, String> {
    use tauri_plugin_llm_engine::{with_engine, ChatMessage, ChatContent, ChatContentPart, LLM_ENGINE};

    {
        let guard = LLM_ENGINE.lock().map_err(|e| format!("Lock error: {}", e))?;
        match guard.as_ref() {
            Some(engine) => {
                if !engine.is_loaded() {
                    return Err("No model loaded - please load a model first".to_string());
                }
            }
            None => {
                return Err("LLM engine not initialized".to_string());
            }
        }
    }

    let chat_messages: Vec<ChatMessage> = messages.into_iter()
        .filter_map(|m| {
            let role = m.get("role")?.as_str()?.to_string();
            let content_value = m.get("content")?;

            let content = if let Some(text) = content_value.as_str() {
                ChatContent::Text(text.to_string())
            } else if let Some(parts_array) = content_value.as_array() {
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
                ChatContent::Parts(parts)
            } else {
                return None;
            };

            Some(ChatMessage { role, content })
        })
        .collect();

    if chat_messages.is_empty() {
        return Err("No valid messages provided".to_string());
    }

    GENERATION_CANCELLED.store(false, Ordering::SeqCst);
    let on_token_clone = on_token.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_engine(|engine| {
            let response = engine.generate(chat_messages, enable_thinking, |token| {
                if GENERATION_CANCELLED.load(Ordering::SeqCst) {
                    return false;
                }
                let _ = on_token_clone.send(token.to_string());
                true
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
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) => Err(e),
        Err(panic_info) => {
            let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            Err(format!("Panic during generation: {}", panic_msg))
        }
    }
}

/// Unload the current model
#[tauri::command]
async fn llm_unload_model() -> Result<(), String> {
    use tauri_plugin_llm_engine::with_engine;
    with_engine(|engine| {
        engine.unload();
        Ok(())
    })
}

/// Check if a model is loaded
#[tauri::command]
async fn llm_is_loaded() -> Result<bool, String> {
    use tauri_plugin_llm_engine::with_engine;
    with_engine(|engine| Ok(engine.is_loaded()))
}

/// Check if the loaded model supports multimodal
#[tauri::command]
async fn llm_is_multimodal() -> Result<bool, String> {
    use tauri_plugin_llm_engine::with_engine;
    with_engine(|engine| Ok(engine.is_multimodal()))
}

/// Initialize the LLM backend engine (idempotent, safe to call multiple times)
#[tauri::command]
async fn llm_init_engine() -> Result<(), String> {
    tauri_plugin_llm_engine::init_engine()
}

/// Test LLM backend initialization
#[tauri::command]
async fn llm_test_init() -> Result<String, String> {
    use tauri_plugin_llm_engine::init_engine;

    let result = std::panic::catch_unwind(|| {
        init_engine()
    });

    match result {
        Ok(Ok(_)) => Ok("Backend initialized successfully".to_string()),
        Ok(Err(e)) => Err(format!("Backend init error: {}", e)),
        Err(panic) => {
            let panic_msg = if let Some(s) = panic.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            Err(format!("Panic during init: {}", panic_msg))
        }
    }
}

/// Get LLM engine debug state
#[tauri::command]
async fn llm_debug_state(app_handle: AppHandle) -> Result<serde_json::Value, String> {
    use tauri_plugin_llm_engine::LLM_ENGINE;

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

/// Get detailed debug info with metrics
#[tauri::command]
async fn llm_get_debug_info(app_handle: AppHandle) -> Result<serde_json::Value, String> {
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

/// Set sampler parameters
#[tauri::command]
async fn llm_set_sampler_params(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    seed: Option<u32>,
    repeat_penalty: Option<f32>,
) -> Result<(), String> {
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

/// Set whether to use GPU acceleration (Metal)
/// Must be called before loading a model to take effect
#[tauri::command]
async fn llm_set_use_gpu(use_gpu: bool) -> Result<(), String> {
    use tauri_plugin_llm_engine::with_engine;

    with_engine(|engine| {
        engine.set_use_gpu(use_gpu);
        Ok(())
    })
}

/// Get whether GPU acceleration is enabled
#[tauri::command]
async fn llm_get_use_gpu() -> Result<bool, String> {
    use tauri_plugin_llm_engine::with_engine;

    with_engine(|engine| {
        Ok(engine.get_use_gpu())
    })
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct MemoryInfo {
    total_bytes: u64,
    used_bytes: u64,
    available_bytes: u64,
}

#[tauri::command]
fn get_memory_info() -> MemoryInfo {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    MemoryInfo {
        total_bytes: sys.total_memory(),
        used_bytes: sys.used_memory(),
        available_bytes: sys.available_memory(),
    }
}

// Shared state for our application (desktop only)
#[derive(Clone)]
struct AppState {
    app_handle: AppHandle,
    http_client: Client,
}

async fn proxy_handler(
    AxumState(state): AxumState<AppState>,
    method: Method,
    headers: HeaderMap,
    uri: Uri,
    body: Body,
) -> Result<Response, StatusCode> {
    let path = uri.path();
    let query = uri.query().unwrap_or("");

    let target_url = {
        // This whole block will evaluate to a single String value.

        let settings = state.app_handle.state::<AppSettings>();
        let ollama_url_guard = settings.ollama_url.lock().unwrap();

        let base_url = ollama_url_guard
            .as_deref()
            .unwrap_or("http://127.0.0.1:11434");

        // 2. This is the last line. With no semicolon, its value is "returned"
        //    from the block and assigned to `target_url`.
        format!("{}{}?{}", base_url, path, query)
    };

    log::info!("Proxying {} request to: {}", method, target_url);

    let body_bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            log::error!("Failed to collect request body: {}", e);
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };

    // Strip Origin header before forwarding - Ollama rejects non-web origins like tauri://localhost
    let mut forwarded_headers = headers.clone();
    forwarded_headers.remove(axum::http::header::ORIGIN);

    let reqwest_request = state
        .http_client
        .request(method, &target_url)
        .headers(forwarded_headers)
        .body(body_bytes);

    match reqwest_request.send().await {
        Ok(upstream_response) => {
            let mut response_builder = Response::builder()
                .status(upstream_response.status())
                .version(upstream_response.version());

            if let Some(headers) = response_builder.headers_mut() {
                headers.extend(upstream_response.headers().clone());
            }

            let response_stream = upstream_response.bytes_stream();
            let response_body = Body::from_stream(response_stream);

            Ok(response_builder.body(response_body).unwrap())
        }
        Err(e) => {
            log::error!("Proxy request to Ollama failed: {}", e);
            Err(StatusCode::BAD_GATEWAY)
        }
    }
}

#[derive(Clone)]
struct ServerUrl(String);

#[tauri::command]
fn get_server_url(server_url: State<Mutex<ServerUrl>>) -> String {
    server_url.lock().unwrap().0.clone()
}

#[cfg(all(not(debug_assertions)))]
fn start_static_server(app_handle: tauri::AppHandle) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        const SERVER_PORT: u16 = 3838;
        let url = format!("http://127.0.0.1:{}", SERVER_PORT);
        let addr_str = url.replace("http://", "");

        let server_url_state = app_handle.state::<Mutex<ServerUrl>>();
        *server_url_state.lock().unwrap() = ServerUrl(url.clone());

        let resource_path = app_handle
            .path()
            .resource_dir()
            .expect("failed to get resource directory")
            .join("_up_/dist");

        log::info!("Serving static files from: {:?}", resource_path);

        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);

        let state = AppState {
            app_handle: app_handle.clone(),
            http_client: Client::new(),
        };

        let app = Router::new()
            .route("/v1/*path", any(proxy_handler))
            .route("/api/*path", any(proxy_handler))
            .route("/ask", axum::routing::post(notifications::ask_handler))
            .route(
                "/ping",
                axum::routing::get(|| async {
                    log::info!("==== PING-PONG ====");
                    "pong"
                }),
            )
            .route(
                "/message",
                axum::routing::post(notifications::message_handler),
            )
            .route(
                "/notification",
                axum::routing::post(notifications::notification_handler),
            )
            .route("/overlay", axum::routing::post(overlay::overlay_handler))
            .route("/click", axum::routing::post(controls::click_handler))
            .route(
                "/commands-stream",
                axum::routing::get(commands::commands_stream_handler),
            )
            // Legacy HTTP endpoints (for backward compatibility during migration)
            .route(
                "/commands",
                axum::routing::get(commands::get_commands_handler),
            )
            .route(
                "/commands",
                axum::routing::post(commands::post_commands_handler),
            )
            .fallback_service(ServeDir::new(resource_path))
            .with_state(state)
            .layer(cors);

        let listener = tokio::net::TcpListener::bind(&addr_str).await;

        match listener {
            Ok(l) => {
                log::info!("Web server listening on {}", url);
                if let Err(e) = axum::serve(l, app.into_make_service()).await {
                    log::error!("Server error: {}", e);
                }
            }
            Err(e) => {
                log::error!(
                    "FATAL: Failed to bind to address {}. Is another instance running? Error: {}",
                    addr_str,
                    e
                );
            }
        }
    });
}

// register_global_shortcuts function moved to shortcuts module

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let mut builder = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_screen_capture::init());

    // Updater
    let builder = {
        builder = builder.plugin(tauri_plugin_updater::Builder::new().build());
        builder.manage(Mutex::new(ServerUrl("".to_string())))
    };

    builder
        .setup(|app| {
            // Load app config early so we can initialize everything with persisted values
            let loaded_config = shortcuts::load_config_from_disk(app.handle());

            // Initialize AppSettings with loaded ollama_url
            app.manage(AppSettings {
                ollama_url: Mutex::new(loaded_config.ollama_url.clone()),
            });

            {
                app.manage(OverlayState {
                    messages: Mutex::new(Vec::new()),
                });

                app.manage({
                    let (tx, _rx) = broadcast::channel(100); // Buffer up to 100 commands
                    CommandState {
                        pending_commands: Mutex::new(std::collections::HashMap::new()),
                        command_broadcaster: tx,
                    }
                });
            }

            app.manage(UnifiedShortcutState {
                config: Mutex::new(loaded_config),
                registered_shortcuts: Mutex::new(Vec::new()),
            });

            // We use the handle to call updater and restart
            {
                let handle = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    // Notice we use the handle to get the updater
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        handle.updater()
                    })) {
                    Ok(updater_result) => {
                        match updater_result {
                            Ok(updater) => {
                                match updater.check().await {
                                    Ok(Some(update)) => {
                        log::info!("Update {} is available!", update.version);

                        // ---- V2 UPDATER DIALOG LOGIC ----
                        let question = format!(
                            "A new version ({}) of Observer is available. Would you like to install it now and restart?",
                            update.version
                        );
                        
                        // Use the new non-blocking dialog with a callback
                        handle.dialog().message(question)
                            .title("Update Available")
                            .buttons(tauri_plugin_dialog::MessageDialogButtons::YesNo)
                            .kind(tauri_plugin_dialog::MessageDialogKind::Info)
                            .show(move |answer_is_yes| {
                                if answer_is_yes {
                                    log::info!("User agreed to update. Downloading and installing...");
                                    
                                    // We need a new async runtime to run the update download within the callback
                                    let update_handle = handle.clone();
                                    tauri::async_runtime::spawn(async move {
                                        if let Err(e) = update.download_and_install(|_, _| {}, || {}).await {
                                            log::error!("Failed to install update: {}", e);
                                        } else {
                                            // Relaunch after successful install
                                            update_handle.restart();
                                        }
                                    });
                                } else {
                                    log::info!("User deferred the update.");
                                }
                            });

                    }
                                    Ok(None) => {
                                        log::info!("You are running the latest version!");
                                    }
                                    Err(e) => {
                                        log::error!("Updater check failed: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                log::error!("Failed to get updater: {}", e);
                            }
                        }
                    }
                    Err(_) => {
                        log::error!("Updater panicked - continuing without update check");
                    }
                }
                });
            }

            app.handle().plugin(
                tauri_plugin_log::Builder::default()
                    .level(log::LevelFilter::Info)
                    .build(),
            )?;

            // Forward LlmEngine log lines to the frontend via a Tauri event.
            // JS subscribes to "llm-log" and pipes entries into the Logger singleton.
            {
                let emit_handle = app.handle().clone();
                tauri_plugin_llm_engine::set_log_emitter(move |level, message| {
                    let _ = emit_handle.emit("llm-log", serde_json::json!({
                        "level": level,
                        "message": message,
                    }));
                });
            }

            // HTTP server
            #[cfg(not(debug_assertions))]
            {
                let app_handle = app.handle().clone();
                std::thread::spawn(move || {
                    start_static_server(app_handle);
                });
            }

            #[cfg(debug_assertions)]
            {
                let server_url_state = app.state::<Mutex<ServerUrl>>();
                let dev_url = app.config().build.dev_url.clone().unwrap();
                *server_url_state.lock().unwrap() = ServerUrl(dev_url.to_string());
            }

            // System tray
            {
                let menu_handle = app.handle();

                let show = MenuItem::with_id(menu_handle, "show", "Show Launcher", true, None::<&str>)?;
                let quit = MenuItem::with_id(menu_handle, "quit", "Quit", true, None::<&str>)?;
                let menu = Menu::with_items(menu_handle, &[&show, &quit])?;

                let _tray = TrayIconBuilder::new()
                    .tooltip("Observer AI is running")
                    .icon(app.default_window_icon().cloned().unwrap())
                    .menu(&menu)
                    .on_menu_event(move |app, event| match event.id.as_ref() {
                        "quit" => {
                            log::info!("Exit called");
                            app.exit(0);
                        }
                        "show" => {
                            if let Some(window) = app.get_webview_window("main") {
                                window.show().unwrap();
                                window.set_focus().unwrap();
                            }
                        }
                        _ => {}
                    })
                    .build(app)?;
            }

            // Create the overlay window synchronously to avoid race conditions
            match WebviewWindowBuilder::new(
                app,
                "overlay",
                WebviewUrl::App("/overlay".into()),
            )
            .title("Observer Overlay")
            .inner_size(700.0, 700.0)
            .position(50.0, 50.0)
            .decorations(false)
            .transparent(true)
            .shadow(false)
            .always_on_top(true)
            .skip_taskbar(true)
            .visible(false)
            .resizable(false)
            .content_protected(true)
            .build() {
                Ok(window) => {
                    log::info!("Overlay window created successfully with content protection");

                    // Explicitly set content protection after window creation
                    if let Err(e) = window.set_content_protected(true) {
                        log::warn!("Could not set content protection on overlay window: {}", e);
                    } else {
                        log::info!("Content protection explicitly enabled on overlay window");
                    }

                    // Note: set_ignore_cursor_events is deferred to show_overlay/toggle — calling
                    // it here on Linux panics because the GDK window isn't realized until shown.
                }
                Err(e) => {
                    log::error!("Failed to create overlay window: {}", e);
                    // Don't panic, just log the error
                }
            }

            // Create the screen selector window (hidden by default)
            match WebviewWindowBuilder::new(
                app,
                "screen-selector",
                WebviewUrl::App("/screen-selector".into()),
            )
            .title("Select Screen or Window")
            .inner_size(900.0, 650.0)
            .center()
            .decorations(true)
            .transparent(false)
            .resizable(true)
            .visible(false)
            .build() {
                Ok(_window) => {
                    log::info!("Screen selector window created successfully");
                }
                Err(e) => {
                    log::error!("Failed to create screen selector window: {}", e);
                }
            }

            // Register shortcuts (config already loaded at app initialization)
            #[cfg(desktop)]
            {
                shortcuts::register_shortcuts_on_startup(app)?;
            }

            #[cfg(not(desktop))]
            {
                log::info!("Global shortcuts not available on this platform");
            }

            // Silently install/update the bundled observe CLI in the background
            std::thread::spawn(|| {
                install_cli::try_install_cli();
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            match event {
                tauri::WindowEvent::CloseRequested { api, .. } => {
                    window.hide().unwrap();
                    api.prevent_close();
                }
                _ => {}
            }
        })
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_os::init())
        .invoke_handler(tauri::generate_handler![
            get_server_url,
            set_ollama_url,
            get_ollama_url,
            check_ollama_servers,
            get_overlay_messages,
            clear_overlay_messages,
            show_overlay,
            hide_overlay,
            get_broadcast_status,
            shortcuts::get_shortcut_config,
            shortcuts::get_registered_shortcuts,
            shortcuts::set_shortcut_config,
            // LLM commands
            llm_list_gguf,
            llm_download_model,
            llm_cancel_download,
            llm_cancel_generation,
            llm_delete_model,
            llm_load_model,
            llm_generate,
            llm_unload_model,
            llm_is_loaded,
            llm_is_multimodal,
            llm_init_engine,
            llm_test_init,
            llm_debug_state,
            llm_get_debug_info,
            llm_set_sampler_params,
            llm_set_use_gpu,
            llm_get_use_gpu,
            get_memory_info,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
