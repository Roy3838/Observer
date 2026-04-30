#!/bin/bash

# Update version strings across all Observer app files
# Usage: ./update_version.sh 2.1.2

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 2.1.2"
    exit 1
fi

VERSION="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Updating version to $VERSION..."

# JSON files - update "version": "X.X.X"
for file in \
    "$SCRIPT_DIR/package.json" \
    "$SCRIPT_DIR/desktop/tauri.conf.json" \
    "$SCRIPT_DIR/mobile/tauri.conf.json"
do
    if [ -f "$file" ]; then
        sed -i '' "s/\"version\": \"[0-9]*\.[0-9]*\.[0-9]*\"/\"version\": \"$VERSION\"/" "$file"
        echo "  Updated: $file"
    else
        echo "  Warning: $file not found"
    fi
done

# iOS Info.plist files - update CFBundleShortVersionString and CFBundleVersion
for file in \
    "$SCRIPT_DIR/mobile/gen/apple/observer-mobile_iOS/Info.plist" \
    "$SCRIPT_DIR/mobile/extensions/ObserverBroadcast/Info.plist" \
    "$SCRIPT_DIR/mobile/gen/apple/broadcast/Info.plist"
do
    if [ -f "$file" ]; then
        # Update CFBundleShortVersionString
        sed -i '' "/<key>CFBundleShortVersionString<\/key>/{ n; s/<string>[0-9]*\.[0-9]*\.[0-9]*<\/string>/<string>$VERSION<\/string>/; }" "$file"
        # Update CFBundleVersion
        sed -i '' "/<key>CFBundleVersion<\/key>/{ n; s/<string>[0-9]*\.[0-9]*\.[0-9]*<\/string>/<string>$VERSION<\/string>/; }" "$file"
        echo "  Updated: $file"
    else
        echo "  Warning: $file not found"
    fi
done

# broadcastSetupUI Info.plist - only CFBundleShortVersionString
file="$SCRIPT_DIR/mobile/gen/apple/broadcastSetupUI/Info.plist"
if [ -f "$file" ]; then
    sed -i '' "/<key>CFBundleShortVersionString<\/key>/{ n; s/<string>[0-9]*\.[0-9]*\.[0-9]*<\/string>/<string>$VERSION<\/string>/; }" "$file"
    echo "  Updated: $file"
else
    echo "  Warning: $file not found"
fi

# Xcode project.pbxproj - update MARKETING_VERSION
file="$SCRIPT_DIR/mobile/gen/apple/observer-mobile.xcodeproj/project.pbxproj"
if [ -f "$file" ]; then
    sed -i '' "s/MARKETING_VERSION = [0-9]*\.[0-9]*\.[0-9]*;/MARKETING_VERSION = $VERSION;/g" "$file"
    echo "  Updated: $file"
else
    echo "  Warning: $file not found"
fi

# Android tauri.properties - update versionName and increment versionCode
file="$SCRIPT_DIR/mobile/gen/android/app/tauri.properties"

if [ -f "$file" ]; then
    CURRENT_CODE=$(grep "tauri.android.versionCode" "$file" | cut -d= -f2)
    VERSION_CODE=$((CURRENT_CODE + 1))
    sed -i '' "s/tauri.android.versionCode=.*/tauri.android.versionCode=$VERSION_CODE/" "$file"
    sed -i '' "s/tauri.android.versionName=.*/tauri.android.versionName=$VERSION/" "$file"
    echo "  Updated: $file (versionCode=$VERSION_CODE)"
else
    # Create the file if it doesn't exist
    mkdir -p "$(dirname "$file")"
    echo "tauri.android.versionCode=1" > "$file"
    echo "tauri.android.versionName=$VERSION" >> "$file"
    echo "  Created: $file (versionCode=1)"
fi

echo ""
echo "Version updated to $VERSION"
