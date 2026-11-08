// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "tauri-plugin-device-lock",
    platforms: [
        .iOS(.v15)
    ],
    products: [
        .library(
            name: "tauri-plugin-device-lock",
            type: .static,
            targets: ["DeviceLockPlugin"]
        )
    ],
    dependencies: [
        .package(name: "Tauri", path: "../.tauri/tauri-api")
    ],
    targets: [
        // No Tauri dependency: the Screen Time extensions in ../ios-extensions
        // compile this file directly so app and extensions share one source of truth.
        .target(
            name: "DeviceLockCore",
            path: "Sources/DeviceLockCore",
            linkerSettings: [
                .linkedFramework("FamilyControls"),
                .linkedFramework("ManagedSettings"),
                .linkedFramework("DeviceActivity"),
            ]
        ),
        .target(
            name: "DeviceLockPlugin",
            dependencies: [
                "DeviceLockCore",
                .product(name: "Tauri", package: "Tauri"),
            ],
            path: "Sources/DeviceLockPlugin"
        ),
    ]
)
