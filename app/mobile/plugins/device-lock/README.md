# tauri-plugin-device-lock

Blocks every other app on the phone until Observer unlocks it or a native
timeout expires. Backs the `lockPhone()` / `unlockPhone()` agent tools.

| | iOS (16+) | Android (7+) |
|---|---|---|
| Mechanism | Screen Time shield on all app categories (FamilyControls + ManagedSettings) | Full-screen `SYSTEM_ALERT_WINDOW` overlay + `specialUse` foreground service |
| Permission | Screen Time (Face ID sheet) | "Display over other apps" |
| Timeout enforced by | `DeviceLockMonitor` extension (runs even if Observer is killed) | Main-thread timer; overlay dies with the process anyway |
| Emergency unlock | Tap shield's "Emergency unlock" twice within 10 s | Hold the button 5 s |
| Still usable while locked | Phone, emergency SOS, Settings (Apple doesn't allow shielding them) | Notification shade, Home/Recents gestures (overlay stays on top) |

## Commands

All return a `Status`:
`{ supported, authorized, locked, lockedUntil?: msEpoch, lastUnlock?: "manual" | "timeout" | "emergency" }`

```ts
import { invoke } from '@tauri-apps/api/core';

await invoke('plugin:device-lock|get_status');
await invoke('plugin:device-lock|request_permission');
await invoke('plugin:device-lock|lock', { reason: 'Back to work', maxMinutes: 15, allowEmergencyUnlock: true });
await invoke('plugin:device-lock|unlock');
```

`maxMinutes` defaults to 30 and is clamped to 1–720. `lock` while locked
replaces the reason and restarts the timer. On desktop, `get_status` returns
`supported: false` and the other commands reject.

## Wiring (done)

- `app/mobile/Cargo.toml`, `app/mobile/src/lib.rs`, `app/mobile/capabilities/mobile.json` register the plugin.
- Android needs nothing else: the manifest merges in automatically.
- iOS: `com.apple.developer.family-controls` is in `observer-mobile_iOS.entitlements`, and the
  three extension targets (`DeviceLockMonitor`, `DeviceLockShieldConfiguration`,
  `DeviceLockShieldAction`) are in the Xcode project, embedded in the app's
  "Embed Foundation Extensions" phase. They compile `ios-extensions/<Name>/` plus the
  shared `ios/Sources/DeviceLockCore/DeviceLockCore.swift`, with a minimum of iOS 16.0.
- JS: `lockPhone()` / `unlockPhone()` agent tools (`src/utils/handlers/utils.ts`,
  `javascript.ts`), with a permission pre-flight on agent start (`src/utils/pre-flight.ts`).

> ⚠️ Don't run `xcodegen` on `gen/apple/project.yml`. The real `project.pbxproj` is
> hand-maintained (gitignored), and xcodegen rewrites every Info.plist and entitlements
> file it lists, wiping the broadcast extension's config. The DeviceLock targets are in
> `project.yml` only so the file stays in sync with the project.

Each extension bundle ID (`com.observer.ai.DeviceLock*`) needs the Family Controls
capability and the `group.com.observer.ai` App Group on its App ID. The Screen Time APIs
don't work in the Simulator.

Bump `CFBundleShortVersionString` / `CFBundleVersion` in the three extension
`Info.plist`s along with the broadcast extension on each release.

## Things to verify on device

- Whether `.all()` also shields Observer itself. Either way the agent keeps
  running in the background via PiP.
- Locks under 15 minutes: DeviceActivity requires 15-minute intervals, so the
  schedule starts in the past and only its end matters. Make sure
  `intervalDidEnd` fires on time.
