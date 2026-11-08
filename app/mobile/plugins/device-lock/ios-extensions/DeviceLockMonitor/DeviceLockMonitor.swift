import DeviceActivity
import Foundation

/// Runs in its own process when the lock's DeviceActivity interval ends, even
/// if Observer is suspended or killed. This is the guarantee that a lock
/// always expires.
class DeviceLockMonitor: DeviceActivityMonitor {
    override func intervalDidEnd(for activity: DeviceActivityName) {
        super.intervalDidEnd(for: activity)
        guard activity == DeviceLockCore.activityName else { return }

        // Ignore a stale end event if the lock was renewed with a later deadline.
        let snap = DeviceLockCore.snapshot()
        if let until = snap.lockedUntil, until.timeIntervalSinceNow > 60 { return }

        DeviceLockCore.unlock(cause: .timeout, stopMonitoring: false)
    }
}
