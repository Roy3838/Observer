const COMMANDS: &[&str] = &["get_status", "request_permission", "lock", "unlock"];

fn main() {
    tauri_plugin::Builder::new(COMMANDS)
        .ios_path("ios")
        .android_path("android")
        .build();
}
