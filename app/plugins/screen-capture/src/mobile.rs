use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tauri::{plugin::{PluginApi, PluginHandle}, AppHandle, Runtime};
use crate::error::Result;

#[cfg(target_os = "ios")]
tauri::ios_plugin_binding!(init_plugin_screen_capture);

/// Response from Android plugin (returns {value: bool})
#[derive(Deserialize)]
#[allow(dead_code)]
struct AndroidBoolResponse {
    value: Option<bool>,
}

/// Broadcast status response from native plugins
#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct BroadcastStatusResponse {
    pub is_active: bool,
    pub is_stale: bool,
    pub frame: Option<String>,
    pub timestamp: Option<f64>,
    pub frame_count: u64,
}

// Initialize the mobile plugin and return a handle
pub fn init<R: Runtime, C: DeserializeOwned>(
    _app: &AppHandle<R>,
    api: PluginApi<R, C>,
) -> Result<ScreenCapture<R>> {
    log::info!("[ScreenCapture] Mobile plugin initialized");
    #[cfg(target_os = "ios")]
    let handle = api.register_ios_plugin(init_plugin_screen_capture)?;
    #[cfg(target_os = "android")]
    let handle = api.register_android_plugin("com.plugin.screencapture", "ScreenCapturePlugin")?;
    Ok(ScreenCapture(handle))
}

/// Access to the screen capture APIs
pub struct ScreenCapture<R: Runtime>(PluginHandle<R>);

impl<R: Runtime> ScreenCapture<R> {
    pub fn start_capture(&self) -> Result<bool> {
        log::info!("[ScreenCapture] Calling native startCapture");

        #[cfg(target_os = "ios")]
        {
            // iOS returns raw bool
            self.0
                .run_mobile_plugin("startCapture", ())
                .map_err(Into::into)
        }

        #[cfg(target_os = "android")]
        {
            // Android returns {value: bool}
            let response: AndroidBoolResponse = self.0
                .run_mobile_plugin("startCapture", ())
                .map_err(|e| {
                    log::error!("[ScreenCapture] Android plugin error: {:?}", e);
                    e
                })?;
            Ok(response.value.unwrap_or(false))
        }
    }

    pub fn stop_capture(&self) -> Result<()> {
        log::info!("[ScreenCapture] Calling native stopCapture");
        self.0
            .run_mobile_plugin("stopCapture", ())
            .map_err(Into::into)
    }

    pub fn get_frame(&self) -> Result<String> {
        log::debug!("[ScreenCapture] Calling native getFrame");
        self.0
            .run_mobile_plugin("getFrame", ())
            .map_err(Into::into)
    }

    /// Get broadcast status including active state and latest frame
    pub fn get_broadcast_status(&self) -> Result<serde_json::Value> {
        log::debug!("[ScreenCapture] Calling native getBroadcastStatus");

        // Try to get status from native plugin
        match self.0.run_mobile_plugin::<_, BroadcastStatusResponse>("getBroadcastStatus", ()) {
            Ok(status) => {
                Ok(serde_json::json!({
                    "isActive": status.is_active,
                    "isStale": status.is_stale,
                    "frame": status.frame,
                    "timestamp": status.timestamp,
                    "frameCount": status.frame_count
                }))
            }
            Err(e) => {
                log::warn!("[ScreenCapture] Native getBroadcastStatus not available: {:?}", e);
                // Return default status - capture might be managed by app's ServerState
                Ok(serde_json::json!({
                    "isActive": false,
                    "isStale": false,
                    "frame": null,
                    "timestamp": null,
                    "frameCount": 0
                }))
            }
        }
    }
}
