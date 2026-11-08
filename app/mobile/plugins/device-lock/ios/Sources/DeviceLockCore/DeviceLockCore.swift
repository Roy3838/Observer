import DeviceActivity
import Foundation
import ManagedSettings

/// Lock state shared by the app and its Screen Time extensions.
///
/// Also compiled directly into the DeviceLockMonitor / ShieldConfiguration /
/// ShieldAction extension targets (see ../../ios-extensions), so keep it free
/// of Tauri and UIKit.
@available(iOS 16.0, *)
public enum DeviceLockCore {
    public static let appGroup = "group.com.observer.ai"
    public static let activityName = DeviceActivityName("observer.devicelock.timeout")

    /// Seconds a tap on "Emergency unlock" stays armed waiting for the second tap.
    public static let emergencyConfirmWindow: TimeInterval = 10

    private static let storeName = ManagedSettingsStore.Name("observer.devicelock")

    private enum Key {
        static let lockedUntil = "deviceLock.lockedUntil"       // Double, epoch seconds
        static let reason = "deviceLock.reason"                 // String
        static let allowEmergency = "deviceLock.allowEmergency" // Bool
        static let emergencyArmedAt = "deviceLock.emergencyArmedAt" // Double, epoch seconds
        static let lastUnlock = "deviceLock.lastUnlock"         // String
    }

    public enum UnlockCause: String {
        case manual, timeout, emergency
    }

    public struct Snapshot {
        public let lockedUntil: Date?
        public let reason: String
        public let allowEmergencyUnlock: Bool
        public let emergencyArmed: Bool
        public let lastUnlock: String?

        public var isLocked: Bool {
            guard let until = lockedUntil else { return false }
            return until > Date()
        }
    }

    private static var defaults: UserDefaults {
        UserDefaults(suiteName: appGroup) ?? .standard
    }

    private static var store: ManagedSettingsStore {
        ManagedSettingsStore(named: storeName)
    }

    // MARK: - Read

    public static func snapshot() -> Snapshot {
        let d = defaults
        let until = d.double(forKey: Key.lockedUntil)
        let armedAt = d.double(forKey: Key.emergencyArmedAt)
        return Snapshot(
            lockedUntil: until > 0 ? Date(timeIntervalSince1970: until) : nil,
            reason: d.string(forKey: Key.reason) ?? "Observer locked your phone",
            allowEmergencyUnlock: d.object(forKey: Key.allowEmergency) as? Bool ?? true,
            emergencyArmed: armedAt > 0
                && Date().timeIntervalSince1970 - armedAt < emergencyConfirmWindow,
            lastUnlock: d.string(forKey: Key.lastUnlock)
        )
    }

    // MARK: - Write

    /// Shields every app and schedules the native timeout. Throws (and leaves
    /// nothing shielded) if the timeout can't be scheduled: we never lock
    /// without a guaranteed way out.
    public static func lock(reason: String, minutes: Int, allowEmergencyUnlock: Bool) throws {
        let end = Date().addingTimeInterval(TimeInterval(minutes * 60))

        // Restart the timeout for this lock.
        let center = DeviceActivityCenter()
        center.stopMonitoring([activityName])
        try center.startMonitoring(activityName, during: schedule(endingAt: end))

        let d = defaults
        d.set(end.timeIntervalSince1970, forKey: Key.lockedUntil)
        d.set(reason, forKey: Key.reason)
        d.set(allowEmergencyUnlock, forKey: Key.allowEmergency)
        d.removeObject(forKey: Key.emergencyArmedAt)

        let s = store
        // Every app category. System apps Apple doesn't allow shielding (Phone,
        // Settings, emergency SOS) stay usable. Web domains are deliberately not
        // shielded so Observer's own WKWebView keeps working; Safari is covered
        // as an app.
        s.shield.applicationCategories = .all()
    }

    /// Removes the shield and the timeout. Safe to call from any extension.
    public static func unlock(cause: UnlockCause, stopMonitoring: Bool = true) {
        store.clearAllSettings()
        if stopMonitoring {
            DeviceActivityCenter().stopMonitoring([activityName])
        }
        let d = defaults
        d.removeObject(forKey: Key.lockedUntil)
        d.removeObject(forKey: Key.emergencyArmedAt)
        d.set(cause.rawValue, forKey: Key.lastUnlock)
    }

    /// Clears a lock whose deadline has passed. Backstop for the monitor extension.
    @discardableResult
    public static func clearIfExpired() -> Bool {
        let snap = snapshot()
        guard snap.lockedUntil != nil, !snap.isLocked else { return false }
        unlock(cause: .timeout)
        return true
    }

    public static func armEmergencyUnlock() {
        defaults.set(Date().timeIntervalSince1970, forKey: Key.emergencyArmedAt)
    }

    // MARK: - Schedule

    /// DeviceActivity rejects intervals shorter than 15 minutes, so for short
    /// locks the interval starts in the past and only its end matters.
    static func schedule(endingAt end: Date) -> DeviceActivitySchedule {
        let minimum: TimeInterval = 15 * 60 + 30
        let start = min(Date(), end.addingTimeInterval(-minimum))
        let comps: Set<Calendar.Component> = [.hour, .minute, .second]
        return DeviceActivitySchedule(
            intervalStart: Calendar.current.dateComponents(comps, from: start),
            intervalEnd: Calendar.current.dateComponents(comps, from: end),
            repeats: false
        )
    }
}
