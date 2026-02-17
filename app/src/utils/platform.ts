// Platform detection utilities for Observer
import { platform as getPlatform } from '@tauri-apps/plugin-os';

/**
 * Check if the app is running in Tauri
 */
export const isTauri = (): boolean => {
  return Boolean(
    typeof window !== 'undefined' &&
    (window as any).__TAURI__
  );
};

/**
 * Check if the app is running on a mobile platform (iOS or Android)
 */
export const isMobile = (): boolean => {
  if (!isTauri()) {
    console.log('[PLATFORM] Not Tauri, isMobile = false');
    return false;
  }

  try {
    // Use the official Tauri OS plugin
    const platform = getPlatform();
    console.log('[PLATFORM] Platform detected:', platform);
    return platform === 'android' || platform === 'ios';
  } catch (err) {
    console.warn('[PLATFORM] getPlatform() failed:', err);
    return false;
  }
};

/**
 * Check if the app is running on desktop (Windows, macOS, or Linux)
 */
export const isDesktop = (): boolean => {
  return isTauri() && !isMobile();
};

/**
 * Check if the app is running on iOS
 */
export const isIOS = (): boolean => {
  if (!isTauri()) return false;
  return getPlatform() === 'ios';
};

/**
 * Check if the app is running on Android
 */
export const isAndroid = (): boolean => {
  if (!isTauri()) return false;
  return getPlatform() === 'android';
};

/**
 * Check if the app is running in a web browser (not Tauri)
 */
export const isWeb = (): boolean => {
  return !isTauri();
};

// Feature detection

/**
 * Check if overlay window support is available (desktop only)
 */
export const hasOverlaySupport = (): boolean => {
  return isDesktop();
};

/**
 * Check if global keyboard shortcuts are available (desktop only)
 */
export const hasGlobalShortcuts = (): boolean => {
  return isDesktop();
};

/**
 * Check if system tray is available (desktop only)
 */
export const hasSystemTray = (): boolean => {
  return isDesktop();
};

/**
 * Check if native Tauri screen capture is available
 * Now available on all Tauri platforms (mobile AND desktop)
 */
export const hasNativeScreenCapture = (): boolean => {
  return isTauri();
};

/**
 * Check if web-based screen capture (getDisplayMedia) is available
 */
export const hasWebScreenCapture = (): boolean => {
  return isDesktop() || isWeb();
};

/**
 * Check if camera access is available
 * (Available on all platforms via getUserMedia)
 */
export const hasCameraAccess = (): boolean => {
  return typeof navigator !== 'undefined' &&
         typeof navigator.mediaDevices !== 'undefined' &&
         typeof navigator.mediaDevices.getUserMedia === 'function';
};

/**
 * Get the current platform name
 */
export const getPlatformName = (): string => {
  if (!isTauri()) return 'Web';

  const platform = getPlatform();
  switch (platform) {
    case 'ios':
      return 'iOS';
    case 'android':
      return 'Android';
    case 'macos':
      return 'macOS';
    case 'windows':
      return 'Windows';
    case 'linux':
      return 'Linux';
    default:
      return platform || 'Unknown';
  }
};

/**
 * Cross-platform confirm dialog.
 * Uses @tauri-apps/plugin-dialog in any Tauri context (where window.confirm may be
 * suppressed by the native webview), and falls back to window.confirm in the browser.
 */
export const confirm = async (message: string): Promise<boolean> => {
  if (isTauri()) {
    const { confirm: tauriConfirm } = await import('@tauri-apps/plugin-dialog');
    return tauriConfirm(message);
  }
  return window.confirm(message);
};

/**
 * Log platform information for debugging
 */
export const logPlatformInfo = (): void => {
  console.log('[Platform]', {
    name: getPlatformName(),
    isTauri: isTauri(),
    isMobile: isMobile(),
    isDesktop: isDesktop(),
    hasOverlay: hasOverlaySupport(),
    hasShortcuts: hasGlobalShortcuts(),
    hasNativeScreenCapture: hasNativeScreenCapture(),
    hasWebScreenCapture: hasWebScreenCapture(),
  });
};
