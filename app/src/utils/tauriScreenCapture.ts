import { invoke } from '@tauri-apps/api/core';
import { isTauri } from './platform';

export interface BroadcastStatus {
  isActive: boolean;
  isStale: boolean;
  frame: string | null;
  timestamp: number | null;
  frameCount: number;
}

/**
 * Unified Tauri screen capture - works on both mobile and desktop
 * Uses the screen-capture plugin which handles platform-specific capture internally
 */
class TauriScreenCapture {
  private capturing = false;

  async startCapture(): Promise<boolean> {
    if (!isTauri()) {
      throw new Error('Screen capture only available in Tauri');
    }

    try {
      const result = await invoke<boolean>('plugin:screen-capture|start_capture_cmd');
      this.capturing = result;
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
      const result = await invoke<BroadcastStatus>('plugin:screen-capture|get_broadcast_status');
      return result;
    } catch (error) {
      console.error('[ScreenCapture] Error getting status:', error);
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
}

export const tauriScreenCapture = new TauriScreenCapture();

// Re-export with old name for backward compatibility
export const mobileScreenCapture = tauriScreenCapture;
