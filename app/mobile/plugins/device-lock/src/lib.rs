//! Device lock plugin: blocks every other app on the phone until Observer
//! unlocks it or a native timeout expires.
//!
//! - iOS: Screen Time API (FamilyControls + ManagedSettings shields), with a
//!   DeviceActivityMonitor extension that clears the shield when the timeout
//!   ends even if Observer has been killed.
//! - Android: full-screen `SYSTEM_ALERT_WINDOW` overlay kept alive by a
//!   foreground service. The overlay dies with the process, so there is no
//!   way to get stuck locked.
//! - Desktop: not supported (`status.supported == false`, `lock` errors).

use serde::{Deserialize, Serialize};
use tauri::{
    plugin::{Builder as PluginBuilder, TauriPlugin},
    Runtime,
};

#[cfg(any(target_os = "ios", target_os = "android"))]
use tauri::Manager;

#[cfg(any(target_os = "ios", target_os = "android"))]
mod mobile;

/// Default lock length when the caller does not pass one.
pub const DEFAULT_MAX_MINUTES: u32 = 30;
/// Upper bound for a single lock. Keeps the iOS DeviceActivity schedule
/// inside one time-of-day window and avoids "locked for a week" mistakes.
pub const MAX_MAX_MINUTES: u32 = 720;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    /// Platform can lock at all (false on desktop, and on iOS < 16).
    pub supported: bool,
    /// User granted the permission (Screen Time on iOS, overlay on Android).
    pub authorized: bool,
    /// A lock is currently active.
    pub locked: bool,
    /// When the active lock expires, in ms since the Unix epoch.
    #[serde(default)]
    pub locked_until: Option<f64>,
    /// How the last lock ended: "manual" | "timeout" | "emergency".
    #[serde(default)]
    pub last_unlock: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LockArgs {
    pub reason: String,
    pub max_minutes: u32,
    pub allow_emergency_unlock: bool,
}

/// Initializes the device-lock plugin.
pub fn init<R: Runtime>() -> TauriPlugin<R> {
    PluginBuilder::new("device-lock")
        .invoke_handler(tauri::generate_handler![
            get_status,
            request_permission,
            lock,
            unlock
        ])
        .setup(|_app, _api| {
            #[cfg(any(target_os = "ios", target_os = "android"))]
            {
                let device_lock = mobile::init(_app, _api)?;
                _app.manage(device_lock);
            }
            Ok(())
        })
        .build()
}

#[cfg(not(any(target_os = "ios", target_os = "android")))]
const UNSUPPORTED: &str = "Device lock is only available on iOS and Android";

#[tauri::command]
async fn get_status<R: Runtime>(_app: tauri::AppHandle<R>) -> Result<Status, String> {
    #[cfg(any(target_os = "ios", target_os = "android"))]
    {
        _app.state::<mobile::DeviceLock<R>>()
            .get_status()
            .map_err(|e| e.to_string())
    }

    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    Ok(Status::default())
}

/// iOS: shows the Screen Time authorization sheet (Face ID).
/// Android: opens the "Display over other apps" settings page.
/// Resolves with the status after the user returns.
#[tauri::command]
async fn request_permission<R: Runtime>(_app: tauri::AppHandle<R>) -> Result<Status, String> {
    #[cfg(any(target_os = "ios", target_os = "android"))]
    {
        _app.state::<mobile::DeviceLock<R>>()
            .request_permission()
            .map_err(|e| e.to_string())
    }

    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    Err(UNSUPPORTED.into())
}

/// Locks the phone. Calling it while already locked replaces the reason and
/// restarts the timer.
#[tauri::command]
async fn lock<R: Runtime>(
    _app: tauri::AppHandle<R>,
    reason: Option<String>,
    max_minutes: Option<u32>,
    allow_emergency_unlock: Option<bool>,
) -> Result<Status, String> {
    let args = LockArgs {
        reason: reason
            .map(|r| r.trim().to_string())
            .filter(|r| !r.is_empty())
            .unwrap_or_else(|| "Observer locked your phone".to_string()),
        max_minutes: max_minutes
            .unwrap_or(DEFAULT_MAX_MINUTES)
            .clamp(1, MAX_MAX_MINUTES),
        allow_emergency_unlock: allow_emergency_unlock.unwrap_or(true),
    };

    #[cfg(any(target_os = "ios", target_os = "android"))]
    {
        log::info!("[DeviceLock] lock for {} min", args.max_minutes);
        _app.state::<mobile::DeviceLock<R>>()
            .lock(args)
            .map_err(|e| e.to_string())
    }

    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    {
        let _ = args;
        Err(UNSUPPORTED.into())
    }
}

#[tauri::command]
async fn unlock<R: Runtime>(_app: tauri::AppHandle<R>) -> Result<Status, String> {
    #[cfg(any(target_os = "ios", target_os = "android"))]
    {
        _app.state::<mobile::DeviceLock<R>>()
            .unlock()
            .map_err(|e| e.to_string())
    }

    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    Err(UNSUPPORTED.into())
}
