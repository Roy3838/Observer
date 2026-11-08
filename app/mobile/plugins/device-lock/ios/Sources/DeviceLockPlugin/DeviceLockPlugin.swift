import DeviceLockCore
import FamilyControls
import Foundation
import Tauri
import UIKit
import WebKit

struct LockArgs: Decodable {
    let reason: String
    let maxMinutes: Int
    let allowEmergencyUnlock: Bool
}

struct StatusResponse: Encodable {
    let supported: Bool
    let authorized: Bool
    let locked: Bool
    let lockedUntil: Double?  // ms since epoch
    let lastUnlock: String?
}

@objc public class DeviceLockPlugin: Plugin {

    public override func load(webview: WKWebView) {
        super.load(webview: webview)
        if #available(iOS 16.0, *) {
            // Backstop in case the monitor extension didn't get to run.
            DeviceLockCore.clearIfExpired()
        }
    }

    @objc public func getStatus(_ invoke: Invoke) {
        invoke.resolve(currentStatus())
    }

    @objc public func requestPermission(_ invoke: Invoke) {
        guard #available(iOS 16.0, *) else {
            invoke.reject("Locking requires iOS 16 or later")
            return
        }
        Task { @MainActor in
            do {
                try await AuthorizationCenter.shared.requestAuthorization(for: .individual)
                invoke.resolve(self.currentStatus())
            } catch {
                invoke.reject("Screen Time permission was not granted: \(error.localizedDescription)")
            }
        }
    }

    @objc public func lock(_ invoke: Invoke) {
        guard #available(iOS 16.0, *) else {
            invoke.reject("Locking requires iOS 16 or later")
            return
        }
        guard AuthorizationCenter.shared.authorizationStatus == .approved else {
            invoke.reject("Screen Time permission not granted. Call requestPermission() first.")
            return
        }
        do {
            let args = try invoke.parseArgs(LockArgs.self)
            try DeviceLockCore.lock(
                reason: args.reason,
                minutes: args.maxMinutes,
                allowEmergencyUnlock: args.allowEmergencyUnlock
            )
            invoke.resolve(currentStatus())
        } catch {
            // DeviceLockCore.lock only shields after the timeout is scheduled,
            // but make sure nothing is left behind.
            DeviceLockCore.unlock(cause: .manual)
            invoke.reject("Failed to lock: \(error.localizedDescription)")
        }
    }

    @objc public func unlock(_ invoke: Invoke) {
        if #available(iOS 16.0, *) {
            DeviceLockCore.unlock(cause: .manual)
        }
        invoke.resolve(currentStatus())
    }

    private func currentStatus() -> StatusResponse {
        guard #available(iOS 16.0, *) else {
            return StatusResponse(
                supported: false, authorized: false, locked: false,
                lockedUntil: nil, lastUnlock: nil)
        }
        DeviceLockCore.clearIfExpired()
        let snap = DeviceLockCore.snapshot()
        return StatusResponse(
            supported: true,
            authorized: AuthorizationCenter.shared.authorizationStatus == .approved,
            locked: snap.isLocked,
            lockedUntil: snap.isLocked ? snap.lockedUntil.map { $0.timeIntervalSince1970 * 1000 } : nil,
            lastUnlock: snap.lastUnlock
        )
    }
}

@_cdecl("init_plugin_device_lock")
public func initPlugin() -> Plugin {
    return DeviceLockPlugin()
}
