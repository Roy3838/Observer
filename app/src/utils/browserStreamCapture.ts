/**
 * Browser-specific stream capture using Web APIs (getDisplayMedia, getUserMedia)
 * Handles browser environment where screen audio requires the display stream
 */

import { Logger } from './logging';

export type MasterStreamType = 'display' | 'camera' | 'microphone' | 'screenAudio';

export interface BrowserCaptureStreams {
  masterDisplayStream: MediaStream | null;
  masterCameraStream: MediaStream | null;
  masterMicrophoneStream: MediaStream | null;
  screenVideoStream: MediaStream | null;
  screenAudioStream: MediaStream | null;
  cameraStream: MediaStream | null;
  microphoneStream: MediaStream | null;
}

class BrowserStreamCapture {
  private streams: BrowserCaptureStreams = {
    masterDisplayStream: null,
    masterCameraStream: null,
    masterMicrophoneStream: null,
    screenVideoStream: null,
    screenAudioStream: null,
    cameraStream: null,
    microphoneStream: null,
  };

  private pendingAcquisitions = new Map<MasterStreamType, Promise<void>>();
  private mockCameraAnimationId: number | null = null;

  /**
   * Called when a master stream dies on its own (user hit "Stop sharing",
   * device unplugged). StreamManager sets this so it can drop its mirrored
   * stream references — otherwise it keeps handing out a dead MediaStream and
   * agents silently capture frozen frames.
   */
  public onStreamEnd: ((type: MasterStreamType) => void) | null = null;

  private createMockCameraStream(): MediaStream {
    const W = 640, H = 480;
    const canvas = document.createElement('canvas');
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d')!;
    let frame = 0;

    const draw = () => {
      frame++;
      ctx.clearRect(0, 0, W, H);

      const ACCENT = '#6366f1'; // single color for everything
      const BG = '#eef2ff';     // indigo-50 — tinted to match

      // Background
      ctx.fillStyle = BG;
      ctx.fillRect(0, 0, W, H);

      const cx = W / 2, cy = H / 2 - 10;

      // Pulse rings — crisp, same accent at low opacity
      for (let i = 0; i < 2; i++) {
        const phase = (frame / 100 + i * 0.5) % 1;
        const radius = 70 + phase * 90;
        const alpha = (1 - phase) * 0.12;
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(99, 102, 241, ${alpha})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Person figure — same accent color
      ctx.fillStyle = ACCENT;

      // Head
      ctx.beginPath();
      ctx.arc(cx, cy - 38, 28, 0, Math.PI * 2);
      ctx.fill();

      // Body
      const bw = 52, bh = 62, bx = cx - bw / 2, by = cy - 6;
      const r = 14;
      ctx.beginPath();
      ctx.moveTo(bx + r, by);
      ctx.lineTo(bx + bw - r, by);
      ctx.quadraticCurveTo(bx + bw, by, bx + bw, by + r);
      ctx.lineTo(bx + bw, by + bh - r);
      ctx.quadraticCurveTo(bx + bw, by + bh, bx + bw - r, by + bh);
      ctx.lineTo(bx + r, by + bh);
      ctx.quadraticCurveTo(bx, by + bh, bx, by + bh - r);
      ctx.lineTo(bx, by + r);
      ctx.quadraticCurveTo(bx, by, bx + r, by);
      ctx.fill();

      // "a person" label above with arrow pointing down
      const labelY = cy - 105;
      ctx.fillStyle = ACCENT;
      ctx.font = 'italic 600 13px -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('a person', cx, labelY);

      const arrowTipY = cy - 70;
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx, labelY + 10);
      ctx.lineTo(cx, arrowTipY - 8);
      ctx.stroke();

      ctx.fillStyle = ACCENT;
      ctx.beginPath();
      ctx.moveTo(cx, arrowTipY);
      ctx.lineTo(cx - 5, arrowTipY - 9);
      ctx.lineTo(cx + 5, arrowTipY - 9);
      ctx.closePath();
      ctx.fill();

      // "No camera access" — same accent, lighter weight
      ctx.fillStyle = `rgba(99, 102, 241, 0.45)`;
      ctx.font = '500 13px -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('No camera access', cx, H / 2 + 115);

      this.mockCameraAnimationId = requestAnimationFrame(draw);
    };

    this.mockCameraAnimationId = requestAnimationFrame(draw);
    return canvas.captureStream(15);
  }

  /**
   * Acquire a master stream (deduplicates concurrent requests)
   */
  async acquireMasterStream(type: MasterStreamType): Promise<void> {
    // If already pending, return existing promise
    if (this.pendingAcquisitions.has(type)) {
      Logger.debug("BrowserCapture", `Joining pending request for '${type}' stream`);
      return this.pendingAcquisitions.get(type)!;
    }

    // Create new acquisition promise
    const promise = this._acquireMasterStreamImpl(type).finally(() => {
      this.pendingAcquisitions.delete(type);
    });

    this.pendingAcquisitions.set(type, promise);
    return promise;
  }

  /**
   * Internal implementation of master stream acquisition
   */
  private async _acquireMasterStreamImpl(type: MasterStreamType): Promise<void> {
    switch (type) {
      case 'display':
        if (this.streams.masterDisplayStream) return;

        Logger.info("BrowserCapture", "Requesting display media with audio");
        const displayStream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
          audio: true,
        });

        this.streams.masterDisplayStream = displayStream;
        this.streams.screenVideoStream = new MediaStream(displayStream.getVideoTracks());

        // Screen audio is part of the display stream in browsers
        if (displayStream.getAudioTracks().length > 0) {
          const audioStream = new MediaStream(displayStream.getAudioTracks());
          this.streams.screenAudioStream = audioStream;
          Logger.info("BrowserCapture", "Screen audio captured from display stream");
        } else {
          Logger.warn("BrowserCapture", "No audio track in display stream");
        }

        // Handle stream end
        displayStream.getVideoTracks()[0]?.addEventListener('ended', () => {
          Logger.warn("BrowserCapture", "Display stream ended unexpectedly");
          this.handleStreamEnd('display');
        });
        break;

      case 'screenAudio':
        // In browser, screen audio is captured via display stream
        // This case handles when screenAudio is requested separately
        Logger.info("BrowserCapture", "Screen audio requested - acquiring via display stream");
        await this._acquireMasterStreamImpl('display');
        break;

      case 'camera':
        if (this.streams.masterCameraStream) return;

        Logger.info("BrowserCapture", "Requesting camera access");

        // Try to use preferred camera device, fallback to default
        const preferredDeviceId = localStorage.getItem('observer_preferred_camera_device');
        let cameraConstraints: MediaStreamConstraints = { video: true };

        if (preferredDeviceId) {
          cameraConstraints = { video: { deviceId: { exact: preferredDeviceId } } };
          Logger.debug("BrowserCapture", `Requesting camera with deviceId: ${preferredDeviceId}`);
        }

        let cameraStream: MediaStream;
        try {
          cameraStream = await navigator.mediaDevices.getUserMedia(cameraConstraints);
        } catch (error) {
          // If preferred device fails, try default camera
          if (preferredDeviceId) {
            Logger.warn("BrowserCapture", `Preferred camera device failed, falling back to default.`, error);
            try {
              cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
            } catch (fallbackError) {
              Logger.warn("BrowserCapture", "Camera access denied, using mock stream.", fallbackError);
              cameraStream = this.createMockCameraStream();
            }
          } else {
            Logger.warn("BrowserCapture", "Camera access denied, using mock stream.", error);
            cameraStream = this.createMockCameraStream();
          }
        }

        this.streams.masterCameraStream = cameraStream;
        this.streams.cameraStream = cameraStream;

        cameraStream.getVideoTracks()[0]?.addEventListener('ended', () => {
          Logger.warn("BrowserCapture", "Camera stream ended unexpectedly");
          this.handleStreamEnd('camera');
        });
        break;

      case 'microphone':
        if (this.streams.masterMicrophoneStream) return;

        Logger.info("BrowserCapture", "Requesting microphone access");
        const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this.streams.masterMicrophoneStream = micStream;
        this.streams.microphoneStream = micStream;

        micStream.getAudioTracks()[0]?.addEventListener('ended', () => {
          Logger.warn("BrowserCapture", "Microphone stream ended unexpectedly");
          this.handleStreamEnd('microphone');
        });
        break;
    }

    Logger.info("BrowserCapture", `Master '${type}' stream acquired`);
  }

  /**
   * Teardown a specific master stream
   */
  teardownMasterStream(type: MasterStreamType): void {
    switch (type) {
      case 'display':
        if (this.streams.masterDisplayStream) {
          Logger.info("BrowserCapture", "Tearing down display stream");
          this.streams.masterDisplayStream.getTracks().forEach(track => track.stop());
          this.streams.masterDisplayStream = null;
          this.streams.screenVideoStream = null;
          this.streams.screenAudioStream = null;
        }
        break;

      case 'screenAudio':
        // In browser, screen audio is part of display stream
        // Tearing down screenAudio alone doesn't make sense, but we handle it gracefully
        Logger.info("BrowserCapture", "Screen audio teardown requested - screen audio is part of display stream");
        // Only clear the audio reference, don't stop the display stream
        this.streams.screenAudioStream = null;
        break;

      case 'camera':
        if (this.streams.masterCameraStream) {
          Logger.info("BrowserCapture", "Tearing down camera stream");
          if (this.mockCameraAnimationId !== null) {
            cancelAnimationFrame(this.mockCameraAnimationId);
            this.mockCameraAnimationId = null;
          }
          this.streams.masterCameraStream.getTracks().forEach(track => track.stop());
          this.streams.masterCameraStream = null;
          this.streams.cameraStream = null;
        }
        break;

      case 'microphone':
        if (this.streams.masterMicrophoneStream) {
          Logger.info("BrowserCapture", "Tearing down microphone stream");
          this.streams.masterMicrophoneStream.getTracks().forEach(track => track.stop());
          this.streams.masterMicrophoneStream = null;
          this.streams.microphoneStream = null;
        }
        break;
    }
  }

  /**
   * Handle unexpected stream end
   */
  private handleStreamEnd(type: MasterStreamType): void {
    Logger.warn("BrowserCapture", `Master ${type} stream ended unexpectedly`);
    this.onStreamEnd?.(type);
  }

  /**
   * Get current streams
   */
  getStreams(): BrowserCaptureStreams {
    return { ...this.streams };
  }

  /**
   * Check if a stream is available
   */
  isStreamAvailable(type: 'camera' | 'screenVideo' | 'screenAudio' | 'microphone'): boolean {
    switch (type) {
      case 'camera':
        return !!this.streams.cameraStream;
      case 'screenVideo':
        return !!this.streams.screenVideoStream;
      case 'screenAudio':
        return !!this.streams.screenAudioStream;
      case 'microphone':
        return !!this.streams.microphoneStream;
    }
  }

  /**
   * Map pseudo-stream type to required master streams (browser-specific logic)
   * In browser, screenAudio requires the display stream (can't be captured separately)
   */
  getMasterStreamsForPseudoStream(pseudoStream: string): MasterStreamType[] {
    switch (pseudoStream) {
      case 'camera':
        return ['camera'];
      case 'screenVideo':
        return ['display'];
      case 'screenAudio':
        return ['display']; // Browser requires display stream for audio
      case 'microphone':
        return ['microphone'];
      case 'allAudio':
        return ['display', 'microphone']; // Need both for combined audio
      default:
        return [];
    }
  }
}

export const browserStreamCapture = new BrowserStreamCapture();
