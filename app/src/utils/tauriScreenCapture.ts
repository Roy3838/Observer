import { invoke, Channel } from '@tauri-apps/api/core';
import { listen, UnlistenFn } from '@tauri-apps/api/event';
import { isTauri, isDesktop } from './platform';
import { Logger } from '@utils/logging';

export interface BroadcastStatus {
  isActive: boolean;
  isStale: boolean;
  frame: string | null;
  timestamp: number | null;
  frameCount: number;
  targetId?: string | null;
}

export interface CaptureTarget {
  id: string;
  kind: 'monitor' | 'window';
  name: string;
  appName?: string;
  thumbnail?: string;
  width: number;
  height: number;
  isPrimary: boolean;
  x: number;
  y: number;
}

/** Frame data received from Rust via Channel */
export interface FrameData {
  frame: string;      // Base64-encoded JPEG
  timestamp: number;  // Unix timestamp
  width: number;
  height: number;
  frameCount: number;
}

/** Result of starting a capture stream */
export interface CaptureStreamResult {
  /** Clean stream for AI/main_loop - no overlay */
  cleanStream: MediaStream;
  /** Stream with PiP overlay for user display */
  pipStream: MediaStream;
  /** Stop the capture and cleanup */
  stop: () => Promise<void>;
  /** Get the latest base64 frame directly (for pre-processor) */
  getLatestFrame: () => string | null;
}

/** Callback for PiP overlay drawing */
export type PipOverlayDrawer = (
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number
) => void;

/**
 * Unified Tauri screen capture - works on both mobile and desktop
 * Uses the screen-capture plugin which handles platform-specific capture internally
 */
class TauriScreenCapture {
  private capturing = false;
  private latestBase64Frame: string | null = null;
  private pipOverlayDrawer: PipOverlayDrawer | null = null;

  /**
   * Set a custom PiP overlay drawer function.
   * This will be called after each frame is drawn to the PiP canvas.
   */
  setPipOverlayDrawer(drawer: PipOverlayDrawer | null): void {
    this.pipOverlayDrawer = drawer;
  }

  /**
   * Get the latest base64 frame directly (for pre-processor).
   * Avoids re-encoding from canvas.
   */
  getLatestBase64Frame(): string | null {
    return this.latestBase64Frame;
  }

  /**
   * Start capture with channel-based streaming.
   * Works on both desktop and iOS - frames are pushed from Rust as they arrive.
   *
   * On desktop: Shows selector window, waits for target selection, then starts xcap with channel
   * On iOS: Triggers ReplayKit picker, then HTTP-received frames are pushed via channel
   *
   * Returns MediaStreams for display and a cleanup function.
   */
  async startCaptureStream(targetId?: string): Promise<CaptureStreamResult> {
    if (!isTauri()) {
      throw new Error('Screen capture only available in Tauri');
    }

    Logger.info("TAURI_SCREEN", `Starting channel-based capture stream`);

    // On desktop, show selector and wait for target selection (unless targetId provided)
    let selectedTargetId: string | undefined = targetId;
    if (isDesktop() && !selectedTargetId) {
      const selected = await this.waitForTargetSelection();
      if (!selected) {
        throw new Error('Screen capture cancelled by user');
      }
      selectedTargetId = selected;
    }

    Logger.info("TAURI_SCREEN", `Starting capture with target: ${selectedTargetId || 'default'}`);

    // Create two canvases: clean (for AI) and pip (for display with overlay)
    const canvasClean = document.createElement('canvas');
    const canvasPip = document.createElement('canvas');

    // Initial size - will be adapted to actual frame dimensions
    canvasClean.width = 1920;
    canvasClean.height = 1080;
    canvasPip.width = 1920;
    canvasPip.height = 1080;

    const ctxClean = canvasClean.getContext('2d');
    const ctxPip = canvasPip.getContext('2d');

    if (!ctxClean || !ctxPip) {
      throw new Error('Failed to create canvas contexts');
    }

    // Create MediaStreams from canvases
    const cleanStream = canvasClean.captureStream(30);
    const pipStream = canvasPip.captureStream(30);

    let canvasSizeInitialized = false;
    let frameCount = 0;
    let isActive = true;
    const cachedImage = new Image();

    // Create a channel to receive frames from Rust
    const frameChannel = new Channel<FrameData>();

    frameChannel.onmessage = (frameData: FrameData) => {
      if (!isActive) return;

      frameCount++;
      this.latestBase64Frame = frameData.frame;

      if (frameCount === 1) {
        Logger.info("TAURI_SCREEN", `First frame received via channel`);
      }
      if (frameCount % 100 === 0) {
        Logger.debug("TAURI_SCREEN", `Received ${frameCount} frames via channel`);
      }

      // Decode and draw frame
      cachedImage.onload = () => {
        if (!isActive) return;

        // Adapt canvas size on first frame
        if (!canvasSizeInitialized && cachedImage.naturalWidth > 0) {
          canvasClean.width = cachedImage.naturalWidth;
          canvasClean.height = cachedImage.naturalHeight;
          canvasPip.width = cachedImage.naturalWidth;
          canvasPip.height = cachedImage.naturalHeight;
          canvasSizeInitialized = true;
          Logger.info("TAURI_SCREEN", `Canvas adapted to ${cachedImage.naturalWidth}x${cachedImage.naturalHeight}`);
        }

        // Draw to clean canvas (no overlay)
        ctxClean.drawImage(cachedImage, 0, 0);

        // Draw to PiP canvas (with overlay for iOS background mode)
        ctxPip.drawImage(cachedImage, 0, 0);

        // Draw PiP overlay if set (used on iOS for background indicator)
        if (this.pipOverlayDrawer) {
          this.pipOverlayDrawer(ctxPip, canvasPip.width, canvasPip.height);
        }
      };

      cachedImage.src = 'data:image/jpeg;base64,' + frameData.frame;
    };

    // Start the capture stream
    try {
      if (isDesktop()) {
        // Desktop: Use the screen-capture plugin with selected target
        await invoke('plugin:screen-capture|start_capture_stream_cmd', {
          targetId: selectedTargetId || null,
          onFrame: frameChannel,
        });
      } else {
        // Mobile (iOS): First set up the channel, then trigger ReplayKit picker
        // The channel receives frames when broadcast extension sends them via HTTP
        await invoke('start_capture_stream_cmd', {
          onFrame: frameChannel,
        });

        // Trigger the ReplayKit picker to start broadcast
        await invoke<boolean>('plugin:screen-capture|start_capture_cmd');
      }
    } catch (error) {
      Logger.error("TAURI_SCREEN", `Failed to start capture stream: ${error}`);
      throw error;
    }

    this.capturing = true;
    Logger.info("TAURI_SCREEN", "Channel-based capture stream started");

    // Return streams and control functions
    return {
      cleanStream,
      pipStream,
      stop: async () => {
        isActive = false;
        this.capturing = false;
        this.latestBase64Frame = null;

        if (isDesktop()) {
          // Desktop: Stop the xcap capture
          await this.stopCapture();
        } else {
          // Mobile: Stop the channel and broadcast
          try {
            await invoke('stop_capture_stream_cmd');
          } catch (e) {
            Logger.warn("TAURI_SCREEN", `Error stopping capture stream: ${e}`);
          }
          await this.stopCapture();
        }

        // Stop canvas streams
        cleanStream.getTracks().forEach(track => track.stop());
        pipStream.getTracks().forEach(track => track.stop());

        Logger.info("TAURI_SCREEN", `Capture stream stopped after ${frameCount} frames`);
      },
      getLatestFrame: () => this.latestBase64Frame,
    };
  }

  /**
   * Show the selector window and wait for user to pick a target.
   * Returns the selected targetId, or null if cancelled.
   */
  private async waitForTargetSelection(): Promise<string | null> {
    return new Promise(async (resolve) => {
      let unlistenSelected: UnlistenFn | null = null;
      let unlistenCancelled: UnlistenFn | null = null;

      const cleanup = () => {
        if (unlistenSelected) unlistenSelected();
        if (unlistenCancelled) unlistenCancelled();
      };

      try {
        // Listen for target selection
        unlistenSelected = await listen<{ targetId: string }>('screen-capture-target-selected', (event) => {
          Logger.info("TAURI_SCREEN", `Target selected: ${event.payload.targetId}`);
          cleanup();
          resolve(event.payload.targetId);
        });

        // Listen for cancellation
        unlistenCancelled = await listen('screen-capture-target-cancelled', () => {
          Logger.info("TAURI_SCREEN", `Target selection cancelled`);
          cleanup();
          resolve(null);
        });

        // Show the selector window
        await this.showSelector();
      } catch (error) {
        Logger.error("TAURI_SCREEN", `Error showing selector: ${error}`);
        cleanup();
        resolve(null);
      }
    });
  }

  async startCapture(): Promise<boolean> {
    if (!isTauri()) {
      throw new Error('Screen capture only available in Tauri');
    }

    try {
      Logger.info("TAURI_SCREEN", `Started Capturing tauri screen`);

      // On desktop, show the selector first so user can pick what to capture
      if (isDesktop()) {
        Logger.info("TAURI_SCREEN", `Popping up the screen capture window`);
        await this.showSelector();
        // The selector window will call startCaptureWithTarget when user picks
        // Return true to indicate the flow started (actual capture starts after selection)
        return true;
      }

      // On mobile, start capture directly (uses system picker)
      const result = await invoke<boolean>('plugin:screen-capture|start_capture_cmd');
      this.capturing = result;
      Logger.info("TAURI_SCREEN", `start_capture_cmd called`);
      return result;
    } catch (error) {
      throw error;
    }
  }

  async stopCapture(): Promise<void> {
    if (!isTauri()) {
      throw new Error('Screen capture only available in Tauri');
    }

    try {
      await invoke('plugin:screen-capture|stop_capture_cmd');
      this.capturing = false;
    } catch (error) {
      throw error;
    }
  }

  /**
   * Unified status + frame query - the single source of truth for broadcast state.
   * Returns broadcast state and the latest frame in one call.
   * Works on both mobile (iOS/Android) and desktop (macOS/Windows/Linux).
   */
  async getStatus(): Promise<BroadcastStatus> {
    if (!isTauri()) {
      // Return safe defaults when not in Tauri
      return {
        isActive: false,
        isStale: false,
        frame: null,
        timestamp: null,
        frameCount: 0,
      };
    }

    try {
      const result = await invoke<BroadcastStatus>('get_broadcast_status');
      return result;
    } catch (error) {
      console.error("[ScreenCapture]", error);
      // Return safe defaults on error
      return {
        isActive: false,
        isStale: false,
        frame: null,
        timestamp: null,
        frameCount: 0,
      };
    }
  }

  isCapturing(): boolean {
    return this.capturing;
  }

  // ================== Screen/Window Selector Methods ==================

  /**
   * Get all available capture targets (monitors and windows).
   * Desktop only.
   */
  async getTargets(includeThumbnails: boolean = true): Promise<CaptureTarget[]> {
    if (!isTauri() || !isDesktop()) {
      throw new Error('Target selection only available on desktop');
    }

    return invoke<CaptureTarget[]>('plugin:screen-capture|get_capture_targets_cmd', {
      includeThumbnails
    });
  }

  /**
   * Open the screen selector window.
   * This shows a custom UI for selecting screens/windows.
   * Desktop only.
   */
  async showSelector(): Promise<void> {
    if (!isTauri() || !isDesktop()) {
      throw new Error('Screen selector only available on desktop');
    }

    // Import dynamically to avoid bundling desktop-only code
    const { WebviewWindow } = await import('@tauri-apps/api/webviewWindow');

    // Get or create the selector window
    let selectorWindow = await WebviewWindow.getByLabel('screen-selector');

    if (!selectorWindow) {
      // Window doesn't exist, might have been closed - create it
      Logger.warn("TAURI_SCREEN", "Screen selector window not found, it may need to be recreated");
      throw new Error('Screen selector window not available');
    }

    // Show and focus the window
    await selectorWindow.show();
    await selectorWindow.setFocus();
  }
}

export const tauriScreenCapture = new TauriScreenCapture();

// Re-export with old name for backward compatibility
export const mobileScreenCapture = tauriScreenCapture;
