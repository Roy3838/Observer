//! Target enumeration for screen capture
//! Provides cross-platform window and monitor discovery using xcap

use crate::error::{Error, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use xcap::{Monitor, Window};

/// Thumbnail settings
const THUMBNAIL_MAX_WIDTH: u32 = 320;
const THUMBNAIL_JPEG_QUALITY: u8 = 60;

/// Kind of capture target
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TargetKind {
    Monitor,
    Window,
}

/// A capture target (monitor or window) with metadata and optional thumbnail
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureTarget {
    /// Unique identifier: "monitor:{id}" or "window:{id}"
    pub id: String,
    /// Type of target
    pub kind: TargetKind,
    /// Display name (e.g., "Built-in Display" or "Chrome - Google")
    pub name: String,
    /// For windows: the application name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app_name: Option<String>,
    /// Base64-encoded JPEG thumbnail
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumbnail: Option<String>,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Whether this is the primary monitor (always false for windows)
    pub is_primary: bool,
    /// X position (for monitors)
    pub x: i32,
    /// Y position (for monitors)
    pub y: i32,
}

/// Get all available capture targets (monitors and windows)
pub fn get_all_targets(include_thumbnails: bool) -> Result<Vec<CaptureTarget>> {
    let mut targets = Vec::new();

    // Get monitors
    let monitors = Monitor::all().map_err(|e| Error::Platform(format!("Failed to enumerate monitors: {}", e)))?;

    for monitor in monitors {
        let id = format!("monitor:{}", monitor.id());
        let name = monitor.name().to_string();
        let width = monitor.width();
        let height = monitor.height();
        let x = monitor.x();
        let y = monitor.y();
        let is_primary = monitor.is_primary();

        let thumbnail = if include_thumbnails {
            capture_monitor_thumbnail(&monitor).ok()
        } else {
            None
        };

        targets.push(CaptureTarget {
            id,
            kind: TargetKind::Monitor,
            name,
            app_name: None,
            thumbnail,
            width,
            height,
            is_primary,
            x,
            y,
        });
    }

    // Get windows
    let windows = Window::all().map_err(|e| Error::Platform(format!("Failed to enumerate windows: {}", e)))?;

    for window in windows {
        // Skip windows with no title or very small windows
        let title = window.title();
        if title.is_empty() {
            continue;
        }

        let width = window.width();
        let height = window.height();

        // Skip tiny windows (likely hidden or utility windows)
        if width < 100 || height < 100 {
            continue;
        }

        // Skip minimized windows (they can't be captured without native picker)
        if window.is_minimized() {
            continue;
        }

        let id = format!("window:{}", window.id());
        let app_name = window.app_name().to_string();

        let thumbnail = if include_thumbnails {
            capture_window_thumbnail(&window).ok()
        } else {
            None
        };

        let x = window.x();
        let y = window.y();

        targets.push(CaptureTarget {
            id,
            kind: TargetKind::Window,
            name: title.to_string(),
            app_name: Some(app_name),
            thumbnail,
            width,
            height,
            is_primary: false,
            x,
            y,
        });
    }

    // Sort: monitors first (primary first), then windows sorted by app name
    targets.sort_by(|a, b| {
        match (&a.kind, &b.kind) {
            (TargetKind::Monitor, TargetKind::Window) => std::cmp::Ordering::Less,
            (TargetKind::Window, TargetKind::Monitor) => std::cmp::Ordering::Greater,
            (TargetKind::Monitor, TargetKind::Monitor) => {
                // Primary monitor first
                b.is_primary.cmp(&a.is_primary)
            }
            (TargetKind::Window, TargetKind::Window) => {
                // Sort by app name, then by window title
                match (&a.app_name, &b.app_name) {
                    (Some(a_app), Some(b_app)) => {
                        a_app.cmp(b_app).then(a.name.cmp(&b.name))
                    }
                    _ => a.name.cmp(&b.name)
                }
            }
        }
    });

    Ok(targets)
}

/// Parse a target ID into its components
pub fn parse_target_id(target_id: &str) -> Result<(TargetKind, u32)> {
    let parts: Vec<&str> = target_id.splitn(2, ':').collect();
    if parts.len() != 2 {
        return Err(Error::Platform(format!("Invalid target ID format: {}", target_id)));
    }

    let kind = match parts[0] {
        "monitor" => TargetKind::Monitor,
        "window" => TargetKind::Window,
        _ => return Err(Error::Platform(format!("Unknown target kind: {}", parts[0]))),
    };

    let id = parts[1].parse::<u32>()
        .map_err(|_| Error::Platform(format!("Invalid target ID number: {}", parts[1])))?;

    Ok((kind, id))
}

/// Capture a thumbnail of a monitor
fn capture_monitor_thumbnail(monitor: &Monitor) -> Result<String> {
    let image = monitor.capture_image()
        .map_err(|e| Error::Platform(format!("Failed to capture monitor: {}", e)))?;

    encode_thumbnail(&image)
}

/// Capture a thumbnail of a window
fn capture_window_thumbnail(window: &Window) -> Result<String> {
    let image = window.capture_image()
        .map_err(|e| Error::Platform(format!("Failed to capture window: {}", e)))?;

    encode_thumbnail(&image)
}

/// Encode an image as a base64 JPEG thumbnail
fn encode_thumbnail(image: &image::RgbaImage) -> Result<String> {
    let width = image.width();
    let height = image.height();

    // Downscale if needed
    let resized = if width > THUMBNAIL_MAX_WIDTH {
        let scale = THUMBNAIL_MAX_WIDTH as f32 / width as f32;
        let new_height = (height as f32 * scale) as u32;
        image::imageops::resize(image, THUMBNAIL_MAX_WIDTH, new_height, FilterType::Nearest)
    } else {
        image.clone()
    };

    let final_width = resized.width();
    let final_height = resized.height();

    // Convert RGBA to RGB directly (avoid DynamicImage overhead)
    let rgba_bytes = resized.as_raw();
    let mut rgb_bytes = Vec::with_capacity((final_width * final_height * 3) as usize);
    for chunk in rgba_bytes.chunks_exact(4) {
        rgb_bytes.push(chunk[0]); // R
        rgb_bytes.push(chunk[1]); // G
        rgb_bytes.push(chunk[2]); // B
    }

    // Encode as JPEG
    let mut jpeg_buffer = Cursor::new(Vec::new());
    let mut encoder = JpegEncoder::new_with_quality(&mut jpeg_buffer, THUMBNAIL_JPEG_QUALITY);

    encoder.encode(&rgb_bytes, final_width, final_height, image::ExtendedColorType::Rgb8)
        .map_err(|e| Error::Platform(format!("Failed to encode thumbnail: {}", e)))?;

    let jpeg_bytes = jpeg_buffer.into_inner();
    Ok(STANDARD.encode(&jpeg_bytes))
}
