# Add project specific ProGuard rules here.

# Keep screen capture plugin classes
-keep class com.plugin.screencapture.** { *; }

# Keep Tauri plugin annotations
-keep @app.tauri.annotation.** class * { *; }
-keepclassmembers class * {
    @app.tauri.annotation.* <methods>;
}
