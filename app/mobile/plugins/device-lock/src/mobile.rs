use serde::de::DeserializeOwned;
use tauri::{
    plugin::{PluginApi, PluginHandle},
    AppHandle, Runtime,
};

use crate::{LockArgs, Status};

#[cfg(target_os = "ios")]
tauri::ios_plugin_binding!(init_plugin_device_lock);

pub fn init<R: Runtime, C: DeserializeOwned>(
    _app: &AppHandle<R>,
    api: PluginApi<R, C>,
) -> Result<DeviceLock<R>, Box<dyn std::error::Error>> {
    log::info!("[DeviceLock] Mobile plugin initialized");
    #[cfg(target_os = "ios")]
    let handle = api.register_ios_plugin(init_plugin_device_lock)?;
    #[cfg(target_os = "android")]
    let handle = api.register_android_plugin("com.plugin.devicelock", "DeviceLockPlugin")?;
    Ok(DeviceLock(handle))
}

pub struct DeviceLock<R: Runtime>(PluginHandle<R>);

impl<R: Runtime> DeviceLock<R> {
    pub fn get_status(&self) -> Result<Status, Box<dyn std::error::Error>> {
        self.0
            .run_mobile_plugin("getStatus", ())
            .map_err(Into::into)
    }

    pub fn request_permission(&self) -> Result<Status, Box<dyn std::error::Error>> {
        self.0
            .run_mobile_plugin("requestPermission", ())
            .map_err(Into::into)
    }

    pub fn lock(&self, args: LockArgs) -> Result<Status, Box<dyn std::error::Error>> {
        self.0
            .run_mobile_plugin("lock", args)
            .map_err(Into::into)
    }

    pub fn unlock(&self) -> Result<Status, Box<dyn std::error::Error>> {
        self.0
            .run_mobile_plugin("unlock", ())
            .map_err(Into::into)
    }
}
