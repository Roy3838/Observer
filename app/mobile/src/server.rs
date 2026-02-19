// server.rs - Localhost HTTP server for receiving broadcast frames and proxying inference requests

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, Method, StatusCode, Uri},
    response::Response,
    routing::{any, get, post},
    Router,
    body::Bytes,
    Json,
};
use base64::Engine;
use http_body_util::BodyExt;
use reqwest::Client;
use serde::Serialize;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{ipc::Channel, AppHandle, Manager};
use tokio::sync::RwLock;

use crate::AppSettings;

/// Frame data sent through the channel to the frontend
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameData {
    /// Base64-encoded JPEG frame
    pub frame: String,
    /// Unix timestamp in seconds
    pub timestamp: f64,
    /// Frame dimensions (0 for iOS since we don't know without decoding)
    pub width: u32,
    pub height: u32,
    /// Frame sequence number
    pub frame_count: u64,
}

/// Broadcast lifecycle state
#[derive(Clone, Default)]
pub struct BroadcastState {
    pub is_active: bool,
    pub started_at: Option<f64>,
    pub last_frame_at: Option<f64>,
    pub frame_count: u64,
}

/// Shared state holding the latest frame and broadcast status
#[derive(Clone)]
pub struct ServerState {
    pub latest_frame: Arc<RwLock<Option<(Vec<u8>, f64)>>>,
    pub broadcast: Arc<RwLock<BroadcastState>>,
    /// Channel for pushing frames to frontend (when streaming is active)
    pub frame_channel: Arc<RwLock<Option<Channel<FrameData>>>>,
}

impl ServerState {
    pub fn new() -> Self {
        Self {
            latest_frame: Arc::new(RwLock::new(None)),
            broadcast: Arc::new(RwLock::new(BroadcastState::default())),
            frame_channel: Arc::new(RwLock::new(None)),
        }
    }

    /// Set the frame channel for streaming
    pub async fn set_frame_channel(&self, channel: Option<Channel<FrameData>>) {
        let mut ch = self.frame_channel.write().await;
        *ch = channel;
    }
}

/// Combined state for the axum router
#[derive(Clone)]
struct AppState {
    server_state: ServerState,
    app_handle: AppHandle,
    http_client: Client,
}

/// Response for broadcast status endpoint
#[derive(Serialize)]
struct BroadcastStatusResponse {
    is_active: bool,
    is_stale: bool,
    started_at: Option<f64>,
    last_frame_at: Option<f64>,
    frame_count: u64,
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

/// Handle incoming frame data from broadcast extension
async fn handle_frame(
    State(state): State<AppState>,
    body: Bytes,
) -> &'static str {
    let frame_bytes = body.to_vec();
    let timestamp = now();

    // Update broadcast state and get frame count
    let frame_count = {
        let mut broadcast = state.server_state.broadcast.write().await;
        broadcast.last_frame_at = Some(timestamp);
        broadcast.frame_count += 1;
        broadcast.frame_count
    };

    // Try to push frame through channel if one is registered
    {
        let channel_guard = state.server_state.frame_channel.read().await;
        if let Some(channel) = channel_guard.as_ref() {
            let frame_data = FrameData {
                frame: base64::prelude::BASE64_STANDARD.encode(&frame_bytes),
                timestamp,
                width: 0,  // Unknown without decoding JPEG
                height: 0,
                frame_count,
            };

            if let Err(e) = channel.send(frame_data) {
                eprintln!("Failed to send frame through channel: {:?}", e);
            }
        }
    }

    // Also store latest frame for polling fallback
    let mut frame = state.server_state.latest_frame.write().await;
    *frame = Some((frame_bytes, timestamp));

    "OK"
}

/// Handle broadcast start event
async fn handle_broadcast_start(State(state): State<AppState>) -> &'static str {
    let timestamp = now();

    let mut broadcast = state.server_state.broadcast.write().await;
    broadcast.is_active = true;
    broadcast.started_at = Some(timestamp);
    broadcast.last_frame_at = None;
    broadcast.frame_count = 0;

    // Clear any stale frame data
    let mut frame = state.server_state.latest_frame.write().await;
    *frame = None;

    eprintln!("🎥 Broadcast started");
    "OK"
}

/// Handle broadcast stop event
async fn handle_broadcast_stop(State(state): State<AppState>) -> &'static str {
    let mut broadcast = state.server_state.broadcast.write().await;
    broadcast.is_active = false;

    eprintln!("🎥 Broadcast stopped (received {} frames)", broadcast.frame_count);
    "OK"
}

/// Get broadcast status
async fn handle_broadcast_status(State(state): State<AppState>) -> Json<BroadcastStatusResponse> {
    let broadcast = state.server_state.broadcast.read().await;
    let current_time = now();

    // Consider stale if active but no frames for >3 seconds
    let is_stale = broadcast.is_active && broadcast.last_frame_at
        .map(|t| current_time - t > 3.0)
        .unwrap_or(true);

    Json(BroadcastStatusResponse {
        is_active: broadcast.is_active,
        is_stale,
        started_at: broadcast.started_at,
        last_frame_at: broadcast.last_frame_at,
        frame_count: broadcast.frame_count,
    })
}

/// Health check endpoint
async fn health_check() -> &'static str {
    "Observer server running"
}

/// Proxy handler for inference requests (same as desktop)
async fn proxy_handler(
    State(state): State<AppState>,
    method: Method,
    headers: HeaderMap,
    uri: Uri,
    body: Body,
) -> Result<Response<Body>, StatusCode> {
    let path = uri.path();
    let query = uri.query().unwrap_or("");

    let target_url = {
        let settings = state.app_handle.state::<AppSettings>();
        let ollama_url_guard = settings.ollama_url.lock().unwrap();

        let base_url = ollama_url_guard
            .as_deref()
            .unwrap_or("http://127.0.0.1:11434");

        format!("{}{}?{}", base_url, path, query)
    };

    eprintln!("Proxying {} request to: {}", method, target_url);

    let body_bytes = match body.collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            eprintln!("Failed to collect request body: {}", e);
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
            eprintln!("Proxy response status: {}", upstream_response.status());
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
            eprintln!("Proxy request to inference server failed: {}", e);
            Err(StatusCode::BAD_GATEWAY)
        }
    }
}

/// Start the HTTP server on localhost:3838
pub async fn start_server(state: ServerState, app_handle: AppHandle) {
    eprintln!("Server starting...");

    let app_state = AppState {
        server_state: state,
        app_handle,
        http_client: Client::new(),
    };

    let app = Router::new()
        // Frame routes (for broadcast extension)
        .route("/frames", post(handle_frame))
        .route("/broadcast/start", post(handle_broadcast_start))
        .route("/broadcast/stop", post(handle_broadcast_stop))
        .route("/broadcast/status", get(handle_broadcast_status))
        .route("/health", get(health_check))
        // Proxy routes (same as desktop)
        .route("/v1/*path", any(proxy_handler))
        .route("/api/*path", any(proxy_handler))
        .with_state(app_state);

    eprintln!("Server app instantiated");

    let listener = match tokio::net::TcpListener::bind("127.0.0.1:3838").await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Failed to bind to localhost:3838: {}", e);
            return;
        }
    };

    eprintln!("Server listening on http://127.0.0.1:3838");
    eprintln!("Ready to receive frames and proxy inference requests...");

    if let Err(e) = axum::serve(listener, app).await {
        log::error!("Server error: {}", e);
    }
}
