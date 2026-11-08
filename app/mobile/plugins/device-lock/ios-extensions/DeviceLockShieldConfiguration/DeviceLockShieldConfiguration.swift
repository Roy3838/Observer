import ManagedSettings
import ManagedSettingsUI
import UIKit

/// What the user sees when opening any app while locked.
class DeviceLockShieldConfiguration: ShieldConfigurationDataSource {
    override func configuration(shielding application: Application) -> ShieldConfiguration {
        makeConfiguration()
    }

    override func configuration(
        shielding application: Application, in category: ActivityCategory
    ) -> ShieldConfiguration {
        makeConfiguration()
    }

    override func configuration(shielding webDomain: WebDomain) -> ShieldConfiguration {
        makeConfiguration()
    }

    override func configuration(
        shielding webDomain: WebDomain, in category: ActivityCategory
    ) -> ShieldConfiguration {
        makeConfiguration()
    }

    private func makeConfiguration() -> ShieldConfiguration {
        let snap = DeviceLockCore.snapshot()

        var subtitle = snap.reason
        if let until = snap.lockedUntil {
            let formatter = DateFormatter()
            formatter.timeStyle = .short
            formatter.dateStyle = .none
            subtitle += "\n\nUnlocks at \(formatter.string(from: until))"
        }

        var secondary: ShieldConfiguration.Label?
        if snap.allowEmergencyUnlock {
            secondary = .init(
                text: snap.emergencyArmed ? "Tap again to unlock" : "Emergency unlock",
                color: snap.emergencyArmed ? .systemRed : .lightGray
            )
        }

        return ShieldConfiguration(
            backgroundBlurStyle: .systemUltraThinMaterialDark,
            backgroundColor: UIColor.black.withAlphaComponent(0.6),
            icon: UIImage(systemName: "lock.fill"),
            title: .init(text: "Locked by Observer", color: .white),
            subtitle: .init(text: subtitle, color: .lightGray),
            primaryButtonLabel: .init(text: "OK", color: .white),
            primaryButtonBackgroundColor: .systemIndigo,
            secondaryButtonLabel: secondary
        )
    }
}
