use crate::CommandState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};

// Comprehensive app configuration
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct AppConfig {
    pub shortcuts: UnifiedShortcutConfig,
    pub ollama_url: Option<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            shortcuts: UnifiedShortcutConfig::default(),
            ollama_url: Some("http://localhost:11434".to_string()),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct UnifiedShortcutConfig {
    // Overlay shortcuts
    pub overlay_toggle: Option<String>,
    pub overlay_move_up: Option<String>,
    pub overlay_move_down: Option<String>,
    pub overlay_move_left: Option<String>,
    pub overlay_move_right: Option<String>,
    pub overlay_resize_up: Option<String>,
    pub overlay_resize_down: Option<String>,
    pub overlay_resize_left: Option<String>,
    pub overlay_resize_right: Option<String>,

    // Agent shortcuts: agent_id -> shortcut_key
    pub agent_shortcuts: HashMap<String, String>,
}

impl Default for UnifiedShortcutConfig {
    fn default() -> Self {
        // Platform-specific defaults
        #[cfg(target_os = "windows")]
        {
            Self {
                overlay_toggle: Some("Alt+B".to_string()),
                overlay_move_up: Some("Alt+ArrowUp".to_string()),
                overlay_move_down: Some("Alt+ArrowDown".to_string()),
                overlay_move_left: Some("Alt+ArrowLeft".to_string()),
                overlay_move_right: Some("Alt+ArrowRight".to_string()),
                overlay_resize_up: Some("Alt+Shift+ArrowUp".to_string()),
                overlay_resize_down: Some("Alt+Shift+ArrowDown".to_string()),
                overlay_resize_left: Some("Alt+Shift+ArrowLeft".to_string()),
                overlay_resize_right: Some("Alt+Shift+ArrowRight".to_string()),
                agent_shortcuts: HashMap::new(),
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            Self {
                overlay_toggle: Some("Cmd+B".to_string()),
                overlay_move_up: Some("Cmd+ArrowUp".to_string()),
                overlay_move_down: Some("Cmd+ArrowDown".to_string()),
                overlay_move_left: Some("Cmd+ArrowLeft".to_string()),
                overlay_move_right: Some("Cmd+ArrowRight".to_string()),
                overlay_resize_up: Some("Cmd+Shift+ArrowUp".to_string()),
                overlay_resize_down: Some("Cmd+Shift+ArrowDown".to_string()),
                overlay_resize_left: Some("Cmd+Shift+ArrowLeft".to_string()),
                overlay_resize_right: Some("Cmd+Shift+ArrowRight".to_string()),
                agent_shortcuts: HashMap::new(),
            }
        }
    }
}

pub struct UnifiedShortcutState {
    pub config: Mutex<AppConfig>,
    pub registered_shortcuts: Mutex<Vec<String>>,
}

#[derive(Debug, Clone)]
enum ShortcutAction {
    OverlayToggle,
    OverlayMoveUp,
    OverlayMoveDown,
    OverlayMoveLeft,
    OverlayMoveRight,
    OverlayResizeUp,
    OverlayResizeDown,
    OverlayResizeLeft,
    OverlayResizeRight,
    AgentToggle(String), // agent_id
}

// Tauri commands
#[tauri::command]
pub async fn get_shortcut_config(
    shortcut_state: State<'_, UnifiedShortcutState>,
) -> Result<UnifiedShortcutConfig, String> {
    let app_config = shortcut_state.config.lock().unwrap().clone();
    Ok(app_config.shortcuts)
}

#[tauri::command]
pub async fn get_registered_shortcuts(
    shortcut_state: State<'_, UnifiedShortcutState>,
) -> Result<Vec<String>, String> {
    let shortcuts = shortcut_state.registered_shortcuts.lock().unwrap().clone();
    Ok(shortcuts)
}

#[tauri::command]
pub async fn set_shortcut_config(
    config: UnifiedShortcutConfig,
    shortcut_state: State<'_, UnifiedShortcutState>,
    app_handle: AppHandle,
) -> Result<(), String> {
    log::info!("Setting unified shortcut config");

    // Preserve ollama_url from current config
    let ollama_url = shortcut_state.config.lock().unwrap().ollama_url.clone();

    let new_app_config = AppConfig {
        shortcuts: config,
        ollama_url,
    };

    // Save to disk
    save_config_to_disk(&app_handle, &new_app_config)?;

    // Update in-memory config
    *shortcut_state.config.lock().unwrap() = new_app_config;

    log::info!("Shortcut config saved. Application restart required for changes to take effect.");
    Ok(())
}

// Settings.json management
fn get_settings_path(
    app_handle: &AppHandle,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let app_data_dir = app_handle.path().app_data_dir()?;
    std::fs::create_dir_all(&app_data_dir)?;
    Ok(app_data_dir.join("settings.json"))
}

pub fn load_config_from_disk(app_handle: &AppHandle) -> AppConfig {
    match get_settings_path(app_handle) {
        Ok(settings_path) => {
            if settings_path.exists() {
                match std::fs::read_to_string(&settings_path) {
                    Ok(content) => {
                        // Try to load as new AppConfig format first
                        match serde_json::from_str::<AppConfig>(&content) {
                            Ok(config) => {
                                log::info!("Loaded app config from {:?}", settings_path);
                                return config;
                            }
                            Err(_) => {
                                // Try to load as old UnifiedShortcutConfig format (migration)
                                log::info!("Attempting to migrate old settings format...");
                                match serde_json::from_str::<UnifiedShortcutConfig>(&content) {
                                    Ok(old_config) => {
                                        log::info!("Migrating settings to new AppConfig format");
                                        let new_config = AppConfig {
                                            shortcuts: old_config,
                                            ollama_url: None,
                                        };
                                        // Save the migrated config in new format
                                        if let Err(e) = save_config_to_disk(app_handle, &new_config)
                                        {
                                            log::warn!("Failed to save migrated config: {}", e);
                                        } else {
                                            log::info!("Migration successful");
                                        }
                                        return new_config;
                                    }
                                    Err(e) => {
                                        log::warn!(
                                            "Failed to parse settings.json (old or new format): {}",
                                            e
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to read settings.json: {}", e);
                    }
                }
            } else {
                log::info!("No settings.json found, using defaults");
            }
        }
        Err(e) => {
            log::error!("Failed to get settings path: {}", e);
        }
    }

    AppConfig::default()
}

fn save_config_to_disk(app_handle: &AppHandle, config: &AppConfig) -> Result<(), String> {
    match get_settings_path(app_handle) {
        Ok(settings_path) => match serde_json::to_string_pretty(config) {
            Ok(json_content) => match std::fs::write(&settings_path, json_content) {
                Ok(_) => {
                    log::info!("Saved app config to {:?}", settings_path);
                    Ok(())
                }
                Err(e) => {
                    let error_msg = format!("Failed to write settings.json: {}", e);
                    log::error!("{}", error_msg);
                    Err(error_msg)
                }
            },
            Err(e) => {
                let error_msg = format!("Failed to serialize config: {}", e);
                log::error!("{}", error_msg);
                Err(error_msg)
            }
        },
        Err(e) => {
            let error_msg = format!("Failed to get settings path: {}", e);
            log::error!("{}", error_msg);
            Err(error_msg)
        }
    }
}

// Helper function to save ollama URL while preserving shortcuts
pub fn save_ollama_url(
    app_handle: &AppHandle,
    shortcut_state: &State<UnifiedShortcutState>,
    ollama_url: Option<String>,
) -> Result<(), String> {
    // Get current config and update ollama_url
    let mut app_config = shortcut_state.config.lock().unwrap().clone();
    app_config.ollama_url = ollama_url;

    // Save to disk
    save_config_to_disk(app_handle, &app_config)?;

    // Update in-memory state
    *shortcut_state.config.lock().unwrap() = app_config;

    Ok(())
}

// Shortcut parsing
fn parse_shortcut_string(shortcut_str: &str) -> Option<tauri_plugin_global_shortcut::Shortcut> {
    use tauri_plugin_global_shortcut::{Code, Modifiers, Shortcut};

    let parts: Vec<&str> = shortcut_str.split('+').map(|s| s.trim()).collect();
    if parts.is_empty() {
        return None;
    }

    let mut modifiers = Modifiers::empty();
    let key_part = if parts.len() == 1 {
        parts[0]
    } else {
        // Parse modifiers
        for i in 0..parts.len() - 1 {
            match parts[i] {
                "Cmd" | "Super" => modifiers |= Modifiers::SUPER,
                "Alt" => modifiers |= Modifiers::ALT,
                "Ctrl" => modifiers |= Modifiers::CONTROL,
                "Shift" => modifiers |= Modifiers::SHIFT,
                _ => return None,
            }
        }
        parts[parts.len() - 1]
    };

    let key = match key_part {
        // Letters
        "A" => Code::KeyA,
        "B" => Code::KeyB,
        "C" => Code::KeyC,
        "D" => Code::KeyD,
        "E" => Code::KeyE,
        "F" => Code::KeyF,
        "G" => Code::KeyG,
        "H" => Code::KeyH,
        "I" => Code::KeyI,
        "J" => Code::KeyJ,
        "K" => Code::KeyK,
        "L" => Code::KeyL,
        "M" => Code::KeyM,
        "N" => Code::KeyN,
        "O" => Code::KeyO,
        "P" => Code::KeyP,
        "Q" => Code::KeyQ,
        "R" => Code::KeyR,
        "S" => Code::KeyS,
        "T" => Code::KeyT,
        "U" => Code::KeyU,
        "V" => Code::KeyV,
        "W" => Code::KeyW,
        "X" => Code::KeyX,
        "Y" => Code::KeyY,
        "Z" => Code::KeyZ,

        // Numbers
        "0" => Code::Digit0,
        "1" => Code::Digit1,
        "2" => Code::Digit2,
        "3" => Code::Digit3,
        "4" => Code::Digit4,
        "5" => Code::Digit5,
        "6" => Code::Digit6,
        "7" => Code::Digit7,
        "8" => Code::Digit8,
        "9" => Code::Digit9,

        // Function keys
        "F1" => Code::F1,
        "F2" => Code::F2,
        "F3" => Code::F3,
        "F4" => Code::F4,
        "F5" => Code::F5,
        "F6" => Code::F6,
        "F7" => Code::F7,
        "F8" => Code::F8,
        "F9" => Code::F9,
        "F10" => Code::F10,
        "F11" => Code::F11,
        "F12" => Code::F12,

        // Arrow keys
        "ArrowUp" => Code::ArrowUp,
        "ArrowDown" => Code::ArrowDown,
        "ArrowLeft" => Code::ArrowLeft,
        "ArrowRight" => Code::ArrowRight,

        // Special keys
        "Space" => Code::Space,
        "Enter" => Code::Enter,
        "Tab" => Code::Tab,
        "Escape" => Code::Escape,
        "Backspace" => Code::Backspace,
        "Delete" => Code::Delete,
        "Home" => Code::Home,
        "End" => Code::End,
        "PageUp" => Code::PageUp,
        "PageDown" => Code::PageDown,

        _ => return None,
    };

    Some(Shortcut::new(Some(modifiers), key))
}

// Helper function to ensure overlay always ignores cursor events
fn ensure_overlay_click_through(window: &tauri::WebviewWindow) {
    if let Err(e) = window.set_ignore_cursor_events(true) {
        log::warn!("Failed to re-enable click-through on overlay: {}", e);
    }
}

// Main registration function - called ONLY at startup
#[cfg(desktop)]
pub fn register_shortcuts_on_startup(
    app: &mut tauri::App,
) -> Result<(), Box<dyn std::error::Error>> {
    use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

    let shortcut_state = app.state::<UnifiedShortcutState>();
    let app_config = shortcut_state.config.lock().unwrap().clone();
    let config = app_config.shortcuts;

    // Collect all shortcuts with their actions
    let mut shortcuts_to_register: Vec<(
        tauri_plugin_global_shortcut::Shortcut,
        String,
        ShortcutAction,
    )> = Vec::new();

    // Overlay shortcuts
    if let Some(key) = &config.overlay_toggle {
        if let Some(shortcut) = parse_shortcut_string(key) {
            shortcuts_to_register.push((shortcut, key.clone(), ShortcutAction::OverlayToggle));
        }
    }

    if let Some(key) = &config.overlay_move_up {
        if let Some(shortcut) = parse_shortcut_string(key) {
            shortcuts_to_register.push((shortcut, key.clone(), ShortcutAction::OverlayMoveUp));
        }
    }

    if let Some(key) = &config.overlay_move_down {
        if let Some(shortcut) = parse_shortcut_string(key) {
            shortcuts_to_register.push((shortcut, key.clone(), ShortcutAction::OverlayMoveDown));
        }
    }

    if let Some(key) = &config.overlay_move_left {
        if let Some(shortcut) = parse_shortcut_string(key) {
            shortcuts_to_register.push((shortcut, key.clone(), ShortcutAction::OverlayMoveLeft));
        }
    }

    if let Some(key) = &config.overlay_move_right {
        if let Some(shortcut) = parse_shortcut_string(key) {
            shortcuts_to_register.push((shortcut, key.clone(), ShortcutAction::OverlayMoveRight));
        }
    }

    if let Some(key) = &config.overlay_resize_up {
        if let Some(shortcut) = parse_shortcut_string(key) {
            shortcuts_to_register.push((shortcut, key.clone(), ShortcutAction::OverlayResizeUp));
        }
    }

    if let Some(key) = &config.overlay_resize_down {
        if let Some(shortcut) = parse_shortcut_string(key) {
            shortcuts_to_register.push((shortcut, key.clone(), ShortcutAction::OverlayResizeDown));
        }
    }

    if let Some(key) = &config.overlay_resize_left {
        if let Some(shortcut) = parse_shortcut_string(key) {
            shortcuts_to_register.push((shortcut, key.clone(), ShortcutAction::OverlayResizeLeft));
        }
    }

    if let Some(key) = &config.overlay_resize_right {
        if let Some(shortcut) = parse_shortcut_string(key) {
            shortcuts_to_register.push((shortcut, key.clone(), ShortcutAction::OverlayResizeRight));
        }
    }

    // Agent shortcuts
    for (agent_id, shortcut_key) in &config.agent_shortcuts {
        if !shortcut_key.is_empty() {
            if let Some(shortcut) = parse_shortcut_string(shortcut_key) {
                shortcuts_to_register.push((
                    shortcut,
                    shortcut_key.clone(),
                    ShortcutAction::AgentToggle(agent_id.clone()),
                ));
            }
        }
    }

    // Create action mapping for the handler
    let actions: Vec<ShortcutAction> = shortcuts_to_register
        .iter()
        .map(|(_, _, action)| action.clone())
        .collect();
    let registered_shortcuts: Vec<tauri_plugin_global_shortcut::Shortcut> = shortcuts_to_register
        .iter()
        .map(|(s, _, _)| s.clone())
        .collect();
    let shortcut_keys: Vec<String> = shortcuts_to_register
        .iter()
        .map(|(_, key, _)| key.clone())
        .collect();

    // Register the single global shortcut handler
    app.handle().plugin(
        tauri_plugin_global_shortcut::Builder::new()
            .with_handler(move |app_handle, shortcut, event| {
                if event.state() != ShortcutState::Pressed {
                    return;
                }

                // Find which shortcut was pressed and emit the event immediately for visual feedback
                if let Some(index) = registered_shortcuts.iter().position(|s| s == shortcut) {
                    let action = &actions[index];

                    // Emit shortcut-pressed event for visual feedback (before executing action)
                    if let Some(shortcut_key) = shortcut_keys.get(index) {
                        if let Err(e) = app_handle.emit("shortcut-pressed", shortcut_key) {
                            log::warn!("Failed to emit shortcut-pressed event: {}", e);
                        }
                    }

                    match action {
                        ShortcutAction::OverlayToggle => {
                            if let Some(window) = app_handle.get_webview_window("overlay") {
                                match window.is_visible() {
                                    Ok(visible) => {
                                        let result = if visible {
                                            window.hide()
                                        } else {
                                            window.show()
                                        };
                                        match result {
                                            Ok(_) => {
                                                log::info!(
                                                    "Overlay {} via toggle shortcut",
                                                    if visible { "hidden" } else { "shown" }
                                                );
                                                // Re-enable click-through after showing
                                                if !visible {
                                                    ensure_overlay_click_through(&window);
                                                }
                                            }
                                            Err(e) => log::error!(
                                                "Failed to {} overlay: {}",
                                                if visible { "hide" } else { "show" },
                                                e
                                            ),
                                        }
                                    }
                                    Err(e) => {
                                        log::error!("Failed to check overlay visibility: {}", e)
                                    }
                                }
                            }
                        }

                        ShortcutAction::OverlayMoveUp
                        | ShortcutAction::OverlayMoveDown
                        | ShortcutAction::OverlayMoveLeft
                        | ShortcutAction::OverlayMoveRight => {
                            if let Some(window) = app_handle.get_webview_window("overlay") {
                                if let Ok(current_pos) = window.outer_position() {
                                    let (dx, dy) = match action {
                                        ShortcutAction::OverlayMoveUp => (0, -50),
                                        ShortcutAction::OverlayMoveDown => (0, 50),
                                        ShortcutAction::OverlayMoveLeft => (-50, 0),
                                        ShortcutAction::OverlayMoveRight => (50, 0),
                                        _ => (0, 0),
                                    };

                                    let new_x = current_pos.x + dx;
                                    let new_y = current_pos.y + dy;

                                    if window
                                        .set_position(tauri::Position::Physical(
                                            tauri::PhysicalPosition { x: new_x, y: new_y },
                                        ))
                                        .is_ok()
                                    {
                                        let direction = match action {
                                            ShortcutAction::OverlayMoveUp => "up",
                                            ShortcutAction::OverlayMoveDown => "down",
                                            ShortcutAction::OverlayMoveLeft => "left",
                                            ShortcutAction::OverlayMoveRight => "right",
                                            _ => "unknown",
                                        };
                                        log::info!(
                                            "Overlay moved {} to ({}, {})",
                                            direction,
                                            new_x,
                                            new_y
                                        );
                                        ensure_overlay_click_through(&window);
                                    }
                                }
                            }
                        }

                        ShortcutAction::OverlayResizeUp
                        | ShortcutAction::OverlayResizeDown
                        | ShortcutAction::OverlayResizeLeft
                        | ShortcutAction::OverlayResizeRight => {
                            if let Some(window) = app_handle.get_webview_window("overlay") {
                                if let Ok(current_size) = window.inner_size() {
                                    let size_delta = 50.0;
                                    let (new_width, new_height) = match action {
                                        ShortcutAction::OverlayResizeUp => {
                                            let new_h = (current_size.height as f64 - size_delta)
                                                .max(200.0);
                                            (current_size.width as f64, new_h)
                                        }
                                        ShortcutAction::OverlayResizeDown => {
                                            let new_h = (current_size.height as f64 + size_delta)
                                                .max(200.0);
                                            (current_size.width as f64, new_h)
                                        }
                                        ShortcutAction::OverlayResizeLeft => {
                                            let new_w =
                                                (current_size.width as f64 - size_delta).max(200.0);
                                            (new_w, current_size.height as f64)
                                        }
                                        ShortcutAction::OverlayResizeRight => {
                                            let new_w =
                                                (current_size.width as f64 + size_delta).max(200.0);
                                            (new_w, current_size.height as f64)
                                        }
                                        _ => {
                                            (current_size.width as f64, current_size.height as f64)
                                        }
                                    };

                                    if window
                                        .set_size(tauri::Size::Physical(tauri::PhysicalSize {
                                            width: new_width as u32,
                                            height: new_height as u32,
                                        }))
                                        .is_ok()
                                    {
                                        let direction = match action {
                                            ShortcutAction::OverlayResizeUp => "up",
                                            ShortcutAction::OverlayResizeDown => "down",
                                            ShortcutAction::OverlayResizeLeft => "left",
                                            ShortcutAction::OverlayResizeRight => "right",
                                            _ => "unknown",
                                        };
                                        log::info!(
                                            "Overlay resized {} to {}x{}",
                                            direction,
                                            new_width,
                                            new_height
                                        );
                                        ensure_overlay_click_through(&window);
                                    }
                                }
                            }
                        }

                        ShortcutAction::AgentToggle(agent_id) => {
                            log::info!("Agent hotkey pressed for agent: {}", agent_id);
                            let command_state = app_handle.state::<CommandState>();
                            crate::commands::broadcast_command(
                                &command_state,
                                agent_id.clone(),
                                "toggle".to_string(),
                            );
                        }
                    }
                }
            })
            .build(),
    )?;

    // Register all shortcuts
    let mut registered_keys = Vec::new();

    for (shortcut, key, action) in shortcuts_to_register {
        match app.global_shortcut().register(shortcut) {
            Ok(_) => {
                let description = match action {
                    ShortcutAction::OverlayToggle => "overlay toggle",
                    ShortcutAction::OverlayMoveUp => "overlay move up",
                    ShortcutAction::OverlayMoveDown => "overlay move down",
                    ShortcutAction::OverlayMoveLeft => "overlay move left",
                    ShortcutAction::OverlayMoveRight => "overlay move right",
                    ShortcutAction::OverlayResizeUp => "overlay resize up",
                    ShortcutAction::OverlayResizeDown => "overlay resize down",
                    ShortcutAction::OverlayResizeLeft => "overlay resize left",
                    ShortcutAction::OverlayResizeRight => "overlay resize right",
                    ShortcutAction::AgentToggle(agent_id) => {
                        registered_keys.push(format!("{} -> toggle agent {}", key, agent_id));
                        continue;
                    }
                };

                log::info!("✓ Registered shortcut '{}' for {}", key, description);
                registered_keys.push(format!("{} -> {}", key, description));
            }
            Err(e) => {
                log::warn!("✗ Failed to register shortcut '{}': {}", key, e);
            }
        }
    }

    // Update registered shortcuts state
    *shortcut_state.registered_shortcuts.lock().unwrap() = registered_keys;

    log::info!(
        "Shortcut registration complete - {} shortcuts active",
        shortcut_state.registered_shortcuts.lock().unwrap().len()
    );
    Ok(())
}
