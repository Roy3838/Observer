import ManagedSettings

/// Handles the shield's buttons. "Emergency unlock" needs two taps within
/// DeviceLockCore.emergencyConfirmWindow seconds; `.defer` makes iOS redraw
/// the shield so the label switches to "Tap again to unlock".
class DeviceLockShieldAction: ShieldActionDelegate {
    override func handle(
        action: ShieldAction, for application: ApplicationToken,
        completionHandler: @escaping (ShieldActionResponse) -> Void
    ) {
        completionHandler(respond(to: action))
    }

    override func handle(
        action: ShieldAction, for webDomain: WebDomainToken,
        completionHandler: @escaping (ShieldActionResponse) -> Void
    ) {
        completionHandler(respond(to: action))
    }

    override func handle(
        action: ShieldAction, for category: ActivityCategoryToken,
        completionHandler: @escaping (ShieldActionResponse) -> Void
    ) {
        completionHandler(respond(to: action))
    }

    private func respond(to action: ShieldAction) -> ShieldActionResponse {
        switch action {
        case .primaryButtonPressed:
            return .close
        case .secondaryButtonPressed:
            let snap = DeviceLockCore.snapshot()
            guard snap.allowEmergencyUnlock else { return .none }
            if snap.emergencyArmed {
                DeviceLockCore.unlock(cause: .emergency)
                return .none
            }
            DeviceLockCore.armEmergencyUnlock()
            return .defer
        @unknown default:
            return .close
        }
    }
}
