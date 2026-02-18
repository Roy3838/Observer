// In src-tauri/src/lib.rs

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod controls;
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
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};

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

// Shortcut commands moved to shortcuts module

// Shortcut helper functions moved to shortcuts module

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

    let reqwest_request = state
        .http_client
        .request(method, &target_url)
        .headers(headers)
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

                    // Make the window draggable by setting it as focusable
                    if let Err(e) = window.set_focus() {
                        log::warn!("Could not focus overlay window: {}", e);
                    }
                }
                Err(e) => {
                    log::error!("Failed to create overlay window: {}", e);
                    // Don't panic, just log the error
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
            get_broadcast_status,
            shortcuts::get_shortcut_config,
            shortcuts::get_registered_shortcuts,
            shortcuts::set_shortcut_config
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
