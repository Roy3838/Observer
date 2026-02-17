use serde::{Serialize, Serializer};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Screen capture not available")]
    NotAvailable,

    #[error("Permission denied")]
    PermissionDenied,

    #[error("Capture not started")]
    NotStarted,

    #[error("No frame available")]
    NoFrame,

    #[error("Platform error: {0}")]
    Platform(String),

    #[error(transparent)]
    Tauri(#[from] tauri::Error),

    #[cfg(any(target_os = "android", target_os = "ios"))]
    #[error(transparent)]
    PluginInvoke(#[from] tauri::plugin::mobile::PluginInvokeError),
}

impl Serialize for Error {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
