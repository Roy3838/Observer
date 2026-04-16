import AVKit
import AVFoundation
import UIKit
import WebKit
import Tauri

/// Minimal PiP plugin that enables web Picture-in-Picture API in the WebView.
/// The actual PiP is handled by the web standard `video.requestPictureInPicture()`.
@objc public class PipPlugin: Plugin {

    public override func load(webview: WKWebView) {
        super.load(webview: webview)

        // Enable Picture-in-Picture for HTML5 video elements
        // This allows the web PiP API (video.requestPictureInPicture()) to work
        if #available(iOS 14.2, *) {
            webview.configuration.allowsPictureInPictureMediaPlayback = true
            webview.configuration.allowsInlineMediaPlayback = true
        }

        // Configure audio session for background playback
        // Use .mixWithOthers to not interrupt existing WebView audio capture
        do {
            try AVAudioSession.sharedInstance().setCategory(
                .playback,
                mode: .default,
                options: [.mixWithOthers]
            )
            try AVAudioSession.sharedInstance().setActive(true)
        } catch {
            print("[PiP] Warning: Failed to configure audio session: \(error.localizedDescription)")
        }

        print("[PiP] Plugin loaded - web PiP API enabled")
    }
}

@_cdecl("init_plugin_pip")
public func initPlugin() -> Plugin {
    return PipPlugin()
}
