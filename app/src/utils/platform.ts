// Platform detection utilities for Observer
import { platform as getPlatform } from '@tauri-apps/plugin-os';
import { Logger, LogLevel } from './logging';

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
    return false;
  }

  try {
    // Use the official Tauri OS plugin
    const platform = getPlatform();
    return platform === 'android' || platform === 'ios';
  } catch (err) {
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

/**
 * Initialize Tauri log forwarding to the frontend Logger.
 * This captures Rust logs (log::info!, log::debug!, etc.) and forwards them
 * to the TypeScript Logger service.
 *
 * Call this once at app startup.
 */
export const initTauriLogForwarding = async (): Promise<(() => void) | null> => {
  if (!isTauri()) {
    return null;
  }

  try {
    const { attachLogger } = await import('@tauri-apps/plugin-log');

    const detach = await attachLogger(({ level, message }) => {
      // Map Tauri log levels to our LogLevel enum
      const logLevel = level <= 1 ? LogLevel.ERROR
        : level === 2 ? LogLevel.WARNING
        : level === 3 ? LogLevel.INFO
        : LogLevel.DEBUG;

      Logger.log(logLevel, 'rust', message);
    });

    Logger.info('platform', 'Tauri log forwarding initialized');
    return detach;
  } catch (err) {
    console.warn('[Platform] Failed to initialize Tauri log forwarding:', err);
    return null;
  }
};

// Cached Tauri fetch function (loaded lazily on first use)
let tauriFetchFn: typeof fetch | null = null;
let tauriFetchLoading: Promise<void> | null = null;

/**
 * Initialize Tauri HTTP fetch in Tauri builds (desktop and mobile).
 * Call this at app startup to pre-load the Tauri HTTP plugin.
 */
export const initPlatformFetch = async (): Promise<void> => {
  if (!isTauri()) return;

  if (tauriFetchLoading) {
    await tauriFetchLoading;
    return;
  }

  tauriFetchLoading = (async () => {
    try {
      const { fetch } = await import('@tauri-apps/plugin-http');
      tauriFetchFn = fetch;
      Logger.info('platform', 'Tauri HTTP plugin loaded for http:// requests');
    } catch (err) {
      console.warn('[Platform] Failed to load Tauri HTTP plugin:', err);
    }
  })();

  await tauriFetchLoading;
};

/**
 * Platform-aware fetch for local/private network requests.
 *
 * On Windows desktop, WebView2 blocks fetch requests from the tauri.localhost
 * origin to private network addresses due to Private Network Access security
 * restrictions. This function uses Tauri's HTTP plugin (native Rust HTTP via
 * reqwest) on desktop for HTTP requests, bypassing WebView2 entirely.
 * On Android release builds the WebView blocks all cleartext http://, and on
 * iOS native requests skip CORS, so mobile uses the same native path.
 *
 * Logic:
 * - http:// URLs → likely local/private network → use Tauri HTTP plugin in Tauri builds
 * - https:// URLs → external APIs → use regular browser fetch
 *
 * This elegantly covers all cases: localhost, LAN IPs, custom network setups.
 */
export const platformFetch = async (
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<Response> => {
  const url = typeof input === 'string'
    ? input
    : input instanceof URL
      ? input.href
      : input.url;

  // HTTP (not HTTPS) = likely local/private network
  const isHttp = url.startsWith('http://');

  if (isTauri() && isHttp && tauriFetchFn) {
    Logger.debug('platform', `Using Tauri HTTP for: ${url}`);
    return tauriFetchFn(input as any, init as any);
  }

  // Use regular fetch for HTTPS and web
  return fetch(input, init);
};

/**
 * Open a URL outside the app. Tauri webviews (especially mobile) ignore
 * target="_blank" and window.open, so route through the opener plugin there.
 */
export const openExternal = async (url: string): Promise<void> => {
  if (isTauri()) {
    try {
      const { openUrl } = await import('@tauri-apps/plugin-opener');
      await openUrl(url);
      return;
    } catch (err) {
      Logger.error('PLATFORM', `openUrl failed for ${url}:`, err);
      return;
    }
  }
  window.open(url, '_blank', 'noopener,noreferrer');
};

/**
 * On Tauri, intercept clicks on external links (http/https/mailto/tel/sms)
 * and open them via the opener plugin instead of the webview.
 */
export const initExternalLinkHandler = (): void => {
  if (!isTauri()) return;
  document.addEventListener('click', (e) => {
    if (e.defaultPrevented) return;
    const a = (e.target as Element | null)?.closest?.('a[href]') as HTMLAnchorElement | null;
    if (!a) return;
    const href = a.getAttribute('href') || '';
    if (!/^(https?:|mailto:|tel:|sms:)/i.test(href)) return;
    // Same-origin http(s) links are in-app navigation
    if (/^https?:/i.test(href) && new URL(href).origin === window.location.origin) return;
    e.preventDefault();
    void openExternal(href);
  });
};
