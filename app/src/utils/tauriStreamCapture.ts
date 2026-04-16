import { invoke, Channel } from '@tauri-apps/api/core';
import { listen, UnlistenFn } from '@tauri-apps/api/event';
import { isTauri, isDesktop } from './platform';
import { Logger } from '@utils/logging';
import { decodeBase64PCM, PCM_SAMPLE_RATE } from './audio/pcmUtils';

/** Callback type for receiving PCM samples from unified pipeline */
export type PCMCallback = (samples: Float32Array, streamType: 'screenAudio' | 'microphone') => void;

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
  frame: Uint8Array;  // Raw JPEG bytes
  timestamp: number;  // Unix timestamp
  width: number;
  height: number;
  frameCount: number;
}

/** Audio data received from Rust via Channel */
export interface AudioData {
  samples: string;      // Base64-encoded PCM (f32 samples, little-endian, mono)
  timestamp: number;    // Unix timestamp
  sampleRate: number;   // e.g., 48000
  chunkCount: number;   // Sequence number
}

/** Result of starting a capture stream */
export interface CaptureStreamResult {
  /** Clean stream for AI/main_loop - no overlay */
  cleanStream: MediaStream;
  /** Stream with PiP overlay for user display */
  pipStream: MediaStream;
  /** Screen audio as MediaStream (like browser's getDisplayMedia audio) */
  screenAudioStream: MediaStream | null;
  /** Stop the capture and cleanup */
  stop: () => Promise<void>;
  /** Get the latest base64 frame directly (for pre-processor) */
  getLatestFrame: () => string | null;
  /** Get the latest audio data (raw PCM) */
  getLatestAudio: () => AudioData | null;
}

/** Callback for PiP overlay drawing */
export type PipOverlayDrawer = (
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number
) => void;

/** Result of starting a video-only stream */
export interface VideoStreamResult {
  /** Clean stream for AI/main_loop - no overlay */
  cleanStream: MediaStream;
  /** Stream with PiP overlay for user display */
  pipStream: MediaStream;
  /** Stop the video capture and cleanup */
  stop: () => Promise<void>;
  /** Get the latest base64 frame directly (for pre-processor) */
  getLatestFrame: () => string | null;
}

/** Result of starting an audio-only stream */
export interface AudioStreamResult {
  /** Screen audio as MediaStream */
  audioStream: MediaStream;
  /** Stop the audio capture and cleanup */
  stop: () => Promise<void>;
  /** Get the latest audio data (raw PCM) */
  getLatestAudio: () => AudioData | null;
}

/**
 * Unified Tauri screen capture - works on both mobile and desktop
 * Uses the screen-capture plugin which handles platform-specific capture internally
 */
export type MasterStreamType = 'display' | 'camera' | 'microphone' | 'screenAudio';

export interface TauriCaptureStreams {
  screenVideoStream: MediaStream | null;
  screenVideoStreamWithPip: MediaStream | null; // Screen stream with PiP overlay
  screenAudioStream: MediaStream | null;
  cameraStream: MediaStream | null;
  microphoneStream: MediaStream | null;
}

class TauriStreamCapture {
  private capturing = false;
  private latestFrameBytes: Uint8Array | null = null;
  private latestBase64Frame: string | null = null; // Cached base64 conversion
  private latestAudioData: AudioData | null = null;
  private pipOverlayDrawer: PipOverlayDrawer | null = null;

  // Stream state for new interface
  private streams: TauriCaptureStreams = {
    screenVideoStream: null,
    screenVideoStreamWithPip: null,
    screenAudioStream: null,
    cameraStream: null,
    microphoneStream: null,
  };

  private pendingAcquisitions = new Map<MasterStreamType, Promise<void>>();

  // Active stream results (for cleanup)
  private videoStreamResult: VideoStreamResult | null = null;
  private audioStreamResult: AudioStreamResult | null = null;

  // PCM callback for unified transcription pipeline
  private pcmCallback: PCMCallback | null = null;

  /**
   * Acquire a master stream (Tauri implementation)
   * Tauri-specific: screenAudio is independent from display
   */
  async acquireMasterStream(type: MasterStreamType): Promise<void> {
    if (this.pendingAcquisitions.has(type)) {
      Logger.debug("TauriCapture", `Joining pending request for '${type}' stream`);
      return this.pendingAcquisitions.get(type)!;
    }

    const promise = this._acquireMasterStreamImpl(type).finally(() => {
      this.pendingAcquisitions.delete(type);
    });

    this.pendingAcquisitions.set(type, promise);
    return promise;
  }

  private async _acquireMasterStreamImpl(type: MasterStreamType): Promise<void> {
    switch (type) {
      case 'display':
        if (this.streams.screenVideoStream) return;

        Logger.info("TauriCapture", "Starting video capture");
        const videoResult = await this.startVideoStream();
        this.streams.screenVideoStream = videoResult.cleanStream;
        this.streams.screenVideoStreamWithPip = videoResult.pipStream;
        this.videoStreamResult = videoResult;
        break;

      case 'screenAudio':
        if (this.streams.screenAudioStream) return;

        Logger.info("TauriCapture", "Starting independent system audio capture");
        const audioResult = await this.startAudioStream();
        this.streams.screenAudioStream = audioResult.audioStream;
        this.audioStreamResult = audioResult;
        break;

      case 'camera':
        if (this.streams.cameraStream) return;

        Logger.info("TauriCapture", "Starting camera capture");

        // Try to use preferred camera device, fallback to default
        const preferredCameraId = localStorage.getItem('observer_preferred_camera_device');
        let cameraConstraints: MediaStreamConstraints = { video: true };

        if (preferredCameraId) {
          cameraConstraints = { video: { deviceId: { exact: preferredCameraId } } };
          Logger.debug("TauriCapture", `Requesting camera with deviceId: ${preferredCameraId}`);
        }

        let cameraStream: MediaStream;
        try {
          cameraStream = await navigator.mediaDevices.getUserMedia(cameraConstraints);
        } catch (error) {
          // If preferred device fails, try default camera
          if (preferredCameraId) {
            Logger.warn("TauriCapture", `Preferred camera device failed, falling back to default.`, error);
            cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
          } else {
            throw error;
          }
        }

        this.streams.cameraStream = cameraStream;
        break;

      case 'microphone':
        if (this.streams.microphoneStream) return;

        Logger.info("TauriCapture", "Starting microphone capture");
        const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this.streams.microphoneStream = micStream;
        break;
    }

    Logger.info("TauriCapture", `Master '${type}' stream acquired`);
  }

  /**
   * Teardown a specific master stream
   */
  teardownMasterStream(type: MasterStreamType): void {
    switch (type) {
      case 'display':
        if (this.videoStreamResult) {
          Logger.info("TauriCapture", "Tearing down video stream");
          this.videoStreamResult.stop();
          this.videoStreamResult = null;
          this.streams.screenVideoStream = null;
          this.streams.screenVideoStreamWithPip = null;
        }
        break;

      case 'screenAudio':
        if (this.audioStreamResult) {
          Logger.info("TauriCapture", "Tearing down audio stream");
          this.audioStreamResult.stop();
          this.audioStreamResult = null;
          this.streams.screenAudioStream = null;
        }
        break;

      case 'camera':
        if (this.streams.cameraStream) {
          Logger.info("TauriCapture", "Tearing down camera stream");
          this.streams.cameraStream.getTracks().forEach(track => track.stop());
          this.streams.cameraStream = null;
        }
        break;

      case 'microphone':
        if (this.streams.microphoneStream) {
          Logger.info("TauriCapture", "Tearing down microphone stream");
          this.streams.microphoneStream.getTracks().forEach(track => track.stop());
          this.streams.microphoneStream = null;
        }
        break;
    }
  }

  /**
   * Get current streams
   */
  getStreams(): TauriCaptureStreams {
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
   * Map pseudo-stream type to required master streams (Tauri-specific logic)
   * In Tauri, screenAudio is INDEPENDENT from display (unlike browser)
   */
  getMasterStreamsForPseudoStream(pseudoStream: string): MasterStreamType[] {
    switch (pseudoStream) {
      case 'camera':
        return ['camera'];
      case 'screenVideo':
        return ['display'];
      case 'screenAudio':
        return ['screenAudio']; // Tauri: independent audio capture
      case 'microphone':
        return ['microphone'];
      case 'allAudio':
        return ['screenAudio', 'microphone']; // Both independent
      default:
        return [];
    }
  }

  /**
   * Set a custom PiP overlay drawer function.
   * This will be called after each frame is drawn to the PiP canvas.
   */
  setPipOverlayDrawer(drawer: PipOverlayDrawer | null): void {
    this.pipOverlayDrawer = drawer;
  }

  /**
   * Set PCM callback for unified transcription pipeline.
   * When set, decoded PCM samples (Float32Array @ 16kHz) are passed to this callback
   * instead of just being used for playback.
   *
   * @param callback - Function to receive PCM samples, or null to disable
   */
  setPCMCallback(callback: PCMCallback | null): void {
    this.pcmCallback = callback;
    if (callback) {
      Logger.info("TauriCapture", "PCM callback set for unified transcription pipeline");
    } else {
      Logger.info("TauriCapture", "PCM callback cleared");
    }
  }

  /**
   * Get the latest base64 frame directly (for pre-processor).
   * Computes base64 lazily from raw bytes when needed.
   */
  getLatestBase64Frame(): string | null {
    if (!this.latestFrameBytes) {
      return null;
    }
    // Compute base64 lazily (only when AI needs it)
    if (!this.latestBase64Frame) {
      // Convert Uint8Array to base64
      let binary = '';
      const bytes = this.latestFrameBytes;
      for (let i = 0; i < bytes.length; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      this.latestBase64Frame = btoa(binary);
    }
    return this.latestBase64Frame;
  }

  /**
   * Get the latest audio data directly.
   * Returns raw PCM audio data (base64-encoded f32 samples).
   */
  getLatestAudioData(): AudioData | null {
    return this.latestAudioData;
  }

  /**
   * Start video-only capture stream.
   * Independent from audio - use startAudioStream separately if you also need audio.
   * Works on both desktop and iOS.
   */
  async startVideoStream(targetId?: string): Promise<VideoStreamResult> {
    if (!isTauri()) {
      throw new Error('Screen capture only available in Tauri');
    }

    Logger.info("TAURI_STREAM", `Starting video-only capture stream`);

    // Desktop: Show selector and wait for target selection (unless targetId provided)
    // iOS: Will trigger ReplayKit picker later
    let selectedTargetId: string | undefined = targetId;
    if (isDesktop() && !selectedTargetId) {
      const selected = await this.waitForTargetSelection();
      if (!selected) {
        throw new Error('Screen capture cancelled by user');
      }
      selectedTargetId = selected;
    }

    Logger.info("TAURI_STREAM", `Starting video capture with target: ${selectedTargetId || 'iOS broadcast'}`);

    // Create two canvases: clean (for AI) and pip (for display with overlay)
    const canvasClean = document.createElement('canvas');
    const canvasPip = document.createElement('canvas');

    canvasClean.width = 1920;
    canvasClean.height = 1080;
    canvasPip.width = 1920;
    canvasPip.height = 1080;

    const ctxClean = canvasClean.getContext('2d');
    const ctxPip = canvasPip.getContext('2d');

    if (!ctxClean || !ctxPip) {
      throw new Error('Failed to create canvas contexts');
    }

    const cleanStream = canvasClean.captureStream(30);
    const pipStream = canvasPip.captureStream(30);

    let canvasSizeInitialized = false;
    let frameCount = 0;
    let isActive = true;
    const cachedImage = new Image();

    const frameChannel = new Channel<FrameData>();
    let previousObjectUrl: string | null = null;

    frameChannel.onmessage = (frameData: FrameData) => {
      if (!isActive) return;

      frameCount++;

      if (frameCount === 1) {
        Logger.info("TAURI_STREAM", `First video frame received (${frameData.frame.length} bytes, type: ${typeof frameData.frame}, isArray: ${Array.isArray(frameData.frame)}, constructor: ${frameData.frame?.constructor?.name})`);
      }
      if (frameCount % 100 === 0) {
        Logger.debug("TAURI_STREAM", `Received ${frameCount} video frames`);
      }

      // Revoke previous object URL to prevent memory leak
      if (previousObjectUrl) {
        URL.revokeObjectURL(previousObjectUrl);
      }

      // Ensure we have a proper Uint8Array for Blob creation
      // Tauri channels may send as ArrayBuffer or plain array depending on serde config
      const rawFrame = frameData.frame as unknown;
      let frameBytes: Uint8Array;
      if (rawFrame instanceof Uint8Array) {
        frameBytes = rawFrame;
      } else if (rawFrame instanceof ArrayBuffer) {
        frameBytes = new Uint8Array(rawFrame);
      } else if (Array.isArray(rawFrame)) {
        frameBytes = new Uint8Array(rawFrame);
      } else {
        // Fallback: try to use as-is (might be ArrayBufferView)
        frameBytes = new Uint8Array(rawFrame as ArrayBufferLike);
      }

      // Create blob and object URL from raw JPEG bytes
      const blob = new Blob([frameBytes], { type: 'image/jpeg' });
      const objectUrl = URL.createObjectURL(blob);
      previousObjectUrl = objectUrl;

      // Store converted bytes for getLatestFrame() - base64 computed lazily when needed
      this.latestFrameBytes = frameBytes;
      this.latestBase64Frame = null; // Clear cached base64, will be recomputed on demand

      if (frameCount === 1) {
        Logger.info("TAURI_STREAM", `Created blob URL: ${objectUrl}, blob size: ${blob.size}`);
      }

      cachedImage.onerror = (e) => {
        Logger.error("TAURI_STREAM", `Image load error: ${e}`);
      };

      cachedImage.onload = () => {
        if (!isActive) return;

        if (!canvasSizeInitialized && cachedImage.naturalWidth > 0) {
          canvasClean.width = cachedImage.naturalWidth;
          canvasClean.height = cachedImage.naturalHeight;
          canvasPip.width = cachedImage.naturalWidth;
          canvasPip.height = cachedImage.naturalHeight;
          canvasSizeInitialized = true;
          Logger.info("TAURI_STREAM", `Video canvas adapted to ${cachedImage.naturalWidth}x${cachedImage.naturalHeight}`);
        }

        ctxClean.drawImage(cachedImage, 0, 0);
        ctxPip.drawImage(cachedImage, 0, 0);

        if (this.pipOverlayDrawer) {
          this.pipOverlayDrawer(ctxPip, canvasPip.width, canvasPip.height);
        }
      };

      cachedImage.src = objectUrl;
    };

    // Start video-only capture
    if (isDesktop()) {
      // Desktop: Use screen-capture plugin
      await invoke('plugin:screen-capture|start_video_stream_cmd', {
        targetId: selectedTargetId || null,
        onFrame: frameChannel,
      });
    } else {
      // iOS: Get App Group path, register channel then trigger ReplayKit picker
      let appGroupPath: string | null = null;
      try {
        appGroupPath = await invoke<string>('plugin:screen-capture|get_app_group_path_cmd');
        Logger.info("TAURI_STREAM", `iOS App Group path for video: ${appGroupPath}`);
      } catch (e) {
        Logger.warn("TAURI_STREAM", `Could not get App Group path: ${e}`);
      }

      await invoke('start_capture_stream_cmd', {
        onFrame: frameChannel,
        appGroupPath: appGroupPath,
      });
      await invoke<boolean>('plugin:screen-capture|start_capture_cmd');
    }

    this.capturing = true;
    Logger.info("TAURI_STREAM", "Video-only capture stream started");

    return {
      cleanStream,
      pipStream,
      stop: async () => {
        isActive = false;
        this.capturing = false;
        this.latestFrameBytes = null;
        this.latestBase64Frame = null;
        if (previousObjectUrl) {
          URL.revokeObjectURL(previousObjectUrl);
          previousObjectUrl = null;
        }

        if (isDesktop()) {
          await invoke('plugin:screen-capture|stop_video_cmd');
        } else {
          // iOS: Stop capture stream and broadcast
          try {
            await invoke('stop_capture_stream_cmd');
          } catch (e) {
            Logger.warn("TAURI_STREAM", `Error stopping capture stream: ${e}`);
          }
          await invoke('plugin:screen-capture|stop_capture_cmd');
        }

        cleanStream.getTracks().forEach(track => track.stop());
        pipStream.getTracks().forEach(track => track.stop());

        Logger.info("TAURI_STREAM", `Video stream stopped after ${frameCount} frames`);
      },
      getLatestFrame: () => this.getLatestBase64Frame(),
    };
  }

  /**
   * Start audio-only capture stream.
   * Independent from video - use startVideoStream separately if you also need video.
   * Works on both desktop (macOS) and iOS.
   */
  async startAudioStream(): Promise<AudioStreamResult> {
    if (!isTauri()) {
      throw new Error('Audio capture only available in Tauri');
    }

    Logger.info("TAURI_STREAM", `Starting audio-only capture stream`);

    let audioContext: AudioContext | null = null;
    let workletNode: AudioWorkletNode | null = null;
    let audioStream: MediaStream | null = null;
    let audioDestination: MediaStreamAudioDestinationNode | null = null;
    let audioChunkCount = 0;
    let isActive = true;
    let configuredSampleRate = 0;

    // AudioWorklet code for mono PCM playback using efficient ring buffer with prebuffering
    // NOTE: This worklet runs at 16kHz to match the unified PCM pipeline
    const workletCode = `
      class PCMPlayerProcessor extends AudioWorkletProcessor {
        constructor() {
          super();
          // Ring buffer: ~4 seconds at 16kHz (power of 2 for fast modulo)
          this.bufferSize = 65536;
          this.bufferMask = this.bufferSize - 1;
          this.buffer = new Float32Array(this.bufferSize);
          this.writeIndex = 0;
          this.readIndex = 0;

          // Prebuffer: wait for ~100ms of audio before starting playback
          // This absorbs jitter from network/IPC delays
          this.prebufferSamples = 1600; // ~100ms at 16kHz (unified PCM pipeline rate)
          this.prebuffering = true;
          this.underrunCount = 0;

          this.port.onmessage = (e) => {
            const samples = e.data;
            const len = samples.length;
            for (let i = 0; i < len; i++) {
              this.buffer[(this.writeIndex + i) & this.bufferMask] = samples[i];
            }
            this.writeIndex = (this.writeIndex + len) & this.bufferMask;
          };
        }

        process(inputs, outputs, parameters) {
          const output = outputs[0];
          const frameSize = output[0].length;
          const available = (this.writeIndex - this.readIndex) & this.bufferMask;

          // Prebuffering phase: wait until we have enough audio buffered
          if (this.prebuffering) {
            if (available >= this.prebufferSamples) {
              this.prebuffering = false;
            } else {
              // Output silence while prebuffering
              for (let ch = 0; ch < output.length; ch++) {
                output[ch].fill(0);
              }
              return true;
            }
          }

          if (available >= frameSize) {
            // Output mono to all channels (typically L and R)
            for (let i = 0; i < frameSize; i++) {
              const sample = this.buffer[(this.readIndex + i) & this.bufferMask];
              for (let ch = 0; ch < output.length; ch++) {
                output[ch][i] = sample;
              }
            }
            this.readIndex = (this.readIndex + frameSize) & this.bufferMask;
          } else {
            // Underrun: output silence and re-enter prebuffering mode
            this.underrunCount++;
            this.prebuffering = true;
            for (let ch = 0; ch < output.length; ch++) {
              output[ch].fill(0);
            }
          }
          return true;
        }
      }
      registerProcessor('pcm-player-processor', PCMPlayerProcessor);
    `;

    // Use 16kHz sample rate to match resampled audio from Rust backend
    // The unified PCM pipeline resamples all audio to 16kHz for transcription
    const targetSampleRate = PCM_SAMPLE_RATE; // 16kHz
    const blob = new Blob([workletCode], { type: 'application/javascript' });
    const workletUrl = URL.createObjectURL(blob);

    audioContext = new AudioContext({ sampleRate: targetSampleRate });
    await audioContext.audioWorklet.addModule(workletUrl);
    URL.revokeObjectURL(workletUrl);

    workletNode = new AudioWorkletNode(audioContext, 'pcm-player-processor', {
      outputChannelCount: [2],
    });

    audioDestination = audioContext.createMediaStreamDestination();
    workletNode.connect(audioDestination);
    audioStream = audioDestination.stream;
    configuredSampleRate = targetSampleRate;

    Logger.info("TAURI_STREAM", `Audio MediaStream initialized (${targetSampleRate}Hz, mono) - matches unified PCM pipeline`);

    const audioChannel = new Channel<AudioData>();

    audioChannel.onmessage = (audioData: AudioData) => {
      if (!isActive) return;

      audioChunkCount++;
      this.latestAudioData = audioData;

      if (audioChunkCount === 1) {
        Logger.info("TAURI_STREAM", `First audio chunk received (${audioData.sampleRate}Hz, mono)`);
        // Audio is now pre-resampled to 16kHz in Rust for transcription
        if (audioData.sampleRate === PCM_SAMPLE_RATE) {
          Logger.info("TAURI_STREAM", "Audio at 16kHz - unified PCM pipeline active");
        } else if (audioData.sampleRate !== configuredSampleRate) {
          Logger.warn("TAURI_STREAM", `Sample rate mismatch: source=${audioData.sampleRate}Hz, context=${configuredSampleRate}Hz`);
        }
      }
      if (audioChunkCount % 500 === 0) {
        Logger.debug("TAURI_STREAM", `Received ${audioChunkCount} audio chunks`);
      }

      try {
        // Decode base64 PCM to Float32Array
        const samples = decodeBase64PCM(audioData.samples);

        // Call PCM callback for transcription pipeline (if set)
        if (this.pcmCallback) {
          this.pcmCallback(samples, 'screenAudio');
        }

        // Send to worklet for playback
        if (workletNode) {
          workletNode.port.postMessage(samples);
        }
      } catch (error) {
        Logger.error("TAURI_STREAM", `Failed to decode audio chunk: ${error}`);
      }
    };

    // Start audio-only capture
    if (isDesktop()) {
      // Desktop: Use screen-capture plugin
      await invoke('plugin:screen-capture|start_audio_stream_cmd', {
        onAudio: audioChannel,
      });
    } else {
      // iOS: Get App Group path, register audio channel, then trigger ReplayKit picker
      let appGroupPath: string | null = null;
      try {
        appGroupPath = await invoke<string>('plugin:screen-capture|get_app_group_path_cmd');
        Logger.info("TAURI_STREAM", `iOS App Group path: ${appGroupPath}`);
      } catch (e) {
        Logger.warn("TAURI_STREAM", `Could not get App Group path: ${e}`);
      }

      await invoke('start_audio_stream_cmd', {
        onAudio: audioChannel,
        appGroupPath: appGroupPath,
      });
      await invoke<boolean>('plugin:screen-capture|start_capture_cmd');
    }

    Logger.info("TAURI_STREAM", "Audio-only capture stream started");

    return {
      audioStream,
      stop: async () => {
        isActive = false;
        this.latestAudioData = null;

        if (isDesktop()) {
          await invoke('plugin:screen-capture|stop_audio_cmd');
        } else {
          // iOS: Stop audio stream and broadcast
          try {
            await invoke('stop_audio_stream_cmd');
          } catch (e) {
            Logger.warn("TAURI_STREAM", `Error stopping audio stream: ${e}`);
          }
          await invoke('plugin:screen-capture|stop_capture_cmd');
        }

        if (audioStream) {
          audioStream.getTracks().forEach(track => track.stop());
        }
        if (workletNode) {
          workletNode.disconnect();
        }
        if (audioContext) {
          await audioContext.close();
        }

        Logger.info("TAURI_STREAM", `Audio stream stopped after ${audioChunkCount} chunks`);
      },
      getLatestAudio: () => this.latestAudioData,
    };
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

    Logger.info("TAURI_STREAM", `Starting channel-based capture stream`);

    // On desktop, show selector and wait for target selection (unless targetId provided)
    let selectedTargetId: string | undefined = targetId;
    if (isDesktop() && !selectedTargetId) {
      const selected = await this.waitForTargetSelection();
      if (!selected) {
        throw new Error('Screen capture cancelled by user');
      }
      selectedTargetId = selected;
    }

    Logger.info("TAURI_STREAM", `Starting capture with target: ${selectedTargetId || 'default'}`);

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

    // Create channels to receive frames and audio from Rust
    const frameChannel = new Channel<FrameData>();
    const audioChannel = new Channel<AudioData>();
    let audioChunkCount = 0;
    let previousObjectUrl: string | null = null;

    // Set up audio processing to convert PCM to MediaStream
    let audioContext: AudioContext | null = null;
    let workletNode: AudioWorkletNode | null = null;
    let screenAudioStream: MediaStream | null = null;
    let audioInitialized = false;
    let audioDestination: MediaStreamAudioDestinationNode | null = null;
    let configuredSampleRate = 0;

    // AudioWorklet code for mono PCM playback with prebuffering
    // NOTE: This worklet runs at 16kHz to match the unified PCM pipeline
    const workletCode = `
      class PCMPlayerProcessor extends AudioWorkletProcessor {
        constructor() {
          super();
          // Ring buffer: ~4 seconds at 16kHz (power of 2 for fast modulo)
          this.bufferSize = 65536;
          this.bufferMask = this.bufferSize - 1;
          this.buffer = new Float32Array(this.bufferSize);
          this.writeIndex = 0;
          this.readIndex = 0;

          // Prebuffer: wait for ~100ms of audio before starting playback
          // This absorbs jitter from network/IPC delays
          this.prebufferSamples = 1600; // ~100ms at 16kHz (unified PCM pipeline rate)
          this.prebuffering = true;
          this.underrunCount = 0;

          this.port.onmessage = (e) => {
            const samples = e.data;
            const len = samples.length;
            for (let i = 0; i < len; i++) {
              this.buffer[(this.writeIndex + i) & this.bufferMask] = samples[i];
            }
            this.writeIndex = (this.writeIndex + len) & this.bufferMask;
          };
        }

        process(inputs, outputs, parameters) {
          const output = outputs[0];
          const frameSize = output[0].length;
          const available = (this.writeIndex - this.readIndex) & this.bufferMask;

          // Prebuffering phase: wait until we have enough audio buffered
          if (this.prebuffering) {
            if (available >= this.prebufferSamples) {
              this.prebuffering = false;
            } else {
              // Output silence while prebuffering
              for (let ch = 0; ch < output.length; ch++) {
                output[ch].fill(0);
              }
              return true;
            }
          }

          if (available >= frameSize) {
            // Output mono to all channels (typically L and R)
            for (let i = 0; i < frameSize; i++) {
              const sample = this.buffer[(this.readIndex + i) & this.bufferMask];
              for (let ch = 0; ch < output.length; ch++) {
                output[ch][i] = sample;
              }
            }
            this.readIndex = (this.readIndex + frameSize) & this.bufferMask;
          } else {
            // Underrun: output silence and re-enter prebuffering mode
            this.underrunCount++;
            this.prebuffering = true;
            for (let ch = 0; ch < output.length; ch++) {
              output[ch].fill(0);
            }
          }
          return true;
        }
      }
      registerProcessor('pcm-player-processor', PCMPlayerProcessor);
    `;

    // Initialize AudioContext with 16kHz to match unified PCM pipeline
    // (Rust resamples all audio to 16kHz for transcription)
    const initializeAudio = async () => {
      try {
        const targetSampleRate = PCM_SAMPLE_RATE; // 16kHz - matches unified pipeline
        const blob = new Blob([workletCode], { type: 'application/javascript' });
        const workletUrl = URL.createObjectURL(blob);

        audioContext = new AudioContext({ sampleRate: targetSampleRate });
        await audioContext.audioWorklet.addModule(workletUrl);
        URL.revokeObjectURL(workletUrl);

        workletNode = new AudioWorkletNode(audioContext, 'pcm-player-processor', {
          outputChannelCount: [2],
        });

        audioDestination = audioContext.createMediaStreamDestination();
        workletNode.connect(audioDestination);
        screenAudioStream = audioDestination.stream;

        audioInitialized = true;
        configuredSampleRate = targetSampleRate;
        Logger.info("TAURI_STREAM", `Audio MediaStream initialized (${targetSampleRate}Hz, mono) - matches unified PCM pipeline`);
      } catch (error) {
        Logger.error("TAURI_STREAM", `Failed to initialize audio stream: ${error}`);
      }
    };

    // Initialize audio upfront
    await initializeAudio();

    frameChannel.onmessage = (frameData: FrameData) => {
      if (!isActive) return;

      frameCount++;

      if (frameCount === 1) {
        Logger.info("TAURI_STREAM", `First frame received via channel (${frameData.frame.length} bytes, type: ${typeof frameData.frame}, constructor: ${frameData.frame?.constructor?.name})`);
      }
      if (frameCount % 100 === 0) {
        Logger.debug("TAURI_STREAM", `Received ${frameCount} frames via channel`);
      }

      // Revoke previous object URL to prevent memory leak
      if (previousObjectUrl) {
        URL.revokeObjectURL(previousObjectUrl);
      }

      // Ensure we have a proper Uint8Array for Blob creation
      // Tauri channels may send as ArrayBuffer or plain array depending on serde config
      const rawFrame = frameData.frame as unknown;
      let frameBytes: Uint8Array;
      if (rawFrame instanceof Uint8Array) {
        frameBytes = rawFrame;
      } else if (rawFrame instanceof ArrayBuffer) {
        frameBytes = new Uint8Array(rawFrame);
      } else if (Array.isArray(rawFrame)) {
        frameBytes = new Uint8Array(rawFrame);
      } else {
        // Fallback: try to use as-is (might be ArrayBufferView)
        frameBytes = new Uint8Array(rawFrame as ArrayBufferLike);
      }

      // Store converted bytes for getLatestFrame() - base64 computed lazily when needed
      this.latestFrameBytes = frameBytes;
      this.latestBase64Frame = null; // Clear cached base64, will be recomputed on demand

      // Create blob and object URL from raw JPEG bytes
      const blob = new Blob([frameBytes], { type: 'image/jpeg' });
      const objectUrl = URL.createObjectURL(blob);
      previousObjectUrl = objectUrl;

      if (frameCount === 1) {
        Logger.info("TAURI_STREAM", `Created blob URL: ${objectUrl}, blob size: ${blob.size}`);
      }

      cachedImage.onerror = (e) => {
        Logger.error("TAURI_STREAM", `Image load error: ${e}`);
      };

      cachedImage.onload = () => {
        if (!isActive) return;

        // Adapt canvas size on first frame
        if (!canvasSizeInitialized && cachedImage.naturalWidth > 0) {
          canvasClean.width = cachedImage.naturalWidth;
          canvasClean.height = cachedImage.naturalHeight;
          canvasPip.width = cachedImage.naturalWidth;
          canvasPip.height = cachedImage.naturalHeight;
          canvasSizeInitialized = true;
          Logger.info("TAURI_STREAM", `Canvas adapted to ${cachedImage.naturalWidth}x${cachedImage.naturalHeight}`);
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

      cachedImage.src = objectUrl;
    };

    audioChannel.onmessage = (audioData: AudioData) => {
      if (!isActive) return;

      audioChunkCount++;
      this.latestAudioData = audioData;

      if (audioChunkCount === 1) {
        Logger.info("TAURI_STREAM", `First audio chunk received via channel (${audioData.sampleRate}Hz, mono)`);
        // Audio is now pre-resampled to 16kHz in Rust for transcription
        if (audioData.sampleRate === PCM_SAMPLE_RATE) {
          Logger.info("TAURI_STREAM", "Audio at 16kHz - unified PCM pipeline active");
        } else if (audioData.sampleRate !== configuredSampleRate) {
          Logger.warn("TAURI_STREAM", `Sample rate mismatch: source=${audioData.sampleRate}Hz, context=${configuredSampleRate}Hz`);
        }
      }
      if (audioChunkCount % 500 === 0) {
        Logger.debug("TAURI_STREAM", `Received ${audioChunkCount} audio chunks via channel`);
      }

      try {
        // Decode base64 PCM to Float32Array
        const samples = decodeBase64PCM(audioData.samples);

        // Call PCM callback for transcription pipeline (if set)
        if (this.pcmCallback) {
          this.pcmCallback(samples, 'screenAudio');
        }

        // Send to worklet for playback
        if (audioInitialized && workletNode) {
          workletNode.port.postMessage(samples);
        }
      } catch (error) {
        Logger.error("TAURI_STREAM", `Failed to decode audio chunk: ${error}`);
      }
    };

    // Start the capture stream
    try {
      if (isDesktop()) {
        // Desktop: Use the screen-capture plugin with selected target and audio
        await invoke('plugin:screen-capture|start_capture_stream_cmd', {
          targetId: selectedTargetId || null,
          onFrame: frameChannel,
          onAudio: audioChannel,
        });
      } else {
        // Mobile (iOS): First set up the channels, then trigger ReplayKit picker
        // Get App Group path for shared memory video and audio
        let appGroupPath: string | null = null;
        try {
          appGroupPath = await invoke<string>('plugin:screen-capture|get_app_group_path_cmd');
          Logger.info("TAURI_STREAM", `iOS App Group path: ${appGroupPath}`);
        } catch (e) {
          Logger.warn("TAURI_STREAM", `Could not get App Group path: ${e}`);
        }

        // Video: Register frame channel with App Group path for shared memory buffer
        await invoke('start_capture_stream_cmd', {
          onFrame: frameChannel,
          appGroupPath: appGroupPath,
        });

        // Audio: Register audio channel with App Group path for ring buffer
        await invoke('start_audio_stream_cmd', {
          onAudio: audioChannel,
          appGroupPath: appGroupPath,
        });

        // Trigger the ReplayKit picker to start broadcast
        await invoke<boolean>('plugin:screen-capture|start_capture_cmd');
      }
    } catch (error) {
      Logger.error("TAURI_STREAM", `Failed to start capture stream: ${error}`);
      throw error;
    }

    this.capturing = true;
    Logger.info("TAURI_STREAM", "Channel-based capture stream started");

    // Return streams and control functions
    return {
      cleanStream,
      pipStream,
      screenAudioStream,
      stop: async () => {
        isActive = false;
        this.capturing = false;
        this.latestFrameBytes = null;
        this.latestBase64Frame = null;
        this.latestAudioData = null;
        if (previousObjectUrl) {
          URL.revokeObjectURL(previousObjectUrl);
          previousObjectUrl = null;
        }

        if (isDesktop()) {
          // Desktop: Stop the xcap capture (also stops audio)
          await this.stopCapture();
        } else {
          // Mobile: Stop the channels and broadcast
          try {
            await invoke('stop_capture_stream_cmd');
          } catch (e) {
            Logger.warn("TAURI_STREAM", `Error stopping capture stream: ${e}`);
          }
          try {
            await invoke('stop_audio_stream_cmd');
          } catch (e) {
            Logger.warn("TAURI_STREAM", `Error stopping audio stream: ${e}`);
          }
          await this.stopCapture();
        }

        // Stop canvas streams
        cleanStream.getTracks().forEach(track => track.stop());
        pipStream.getTracks().forEach(track => track.stop());

        // Stop audio stream and close context
        if (screenAudioStream) {
          screenAudioStream.getTracks().forEach(track => track.stop());
        }
        if (workletNode) {
          workletNode.disconnect();
        }
        if (audioContext) {
          await audioContext.close();
        }

        Logger.info("TAURI_STREAM", `Capture stream stopped after ${frameCount} frames, ${audioChunkCount} audio chunks`);
      },
      getLatestFrame: () => this.getLatestBase64Frame(),
      getLatestAudio: () => this.latestAudioData,
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
          Logger.info("TAURI_STREAM", `Target selected: ${event.payload.targetId}`);
          cleanup();
          resolve(event.payload.targetId);
        });

        // Listen for cancellation
        unlistenCancelled = await listen('screen-capture-target-cancelled', () => {
          Logger.info("TAURI_STREAM", `Target selection cancelled`);
          cleanup();
          resolve(null);
        });

        // Show the selector window
        await this.showSelector();
      } catch (error) {
        Logger.error("TAURI_STREAM", `Error showing selector: ${error}`);
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
      Logger.info("TAURI_STREAM", `Started Capturing tauri screen`);

      // On desktop, show the selector first so user can pick what to capture
      if (isDesktop()) {
        Logger.info("TAURI_STREAM", `Popping up the screen capture window`);
        await this.showSelector();
        // The selector window will call startCaptureWithTarget when user picks
        // Return true to indicate the flow started (actual capture starts after selection)
        return true;
      }

      // On mobile, start capture directly (uses system picker)
      const result = await invoke<boolean>('plugin:screen-capture|start_capture_cmd');
      this.capturing = result;
      Logger.info("TAURI_STREAM", `start_capture_cmd called`);
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
   * Stop only video capture.
   * Works on both desktop and iOS.
   */
  async stopVideo(): Promise<void> {
    if (!isTauri()) {
      throw new Error('Screen capture only available in Tauri');
    }

    try {
      if (isDesktop()) {
        await invoke('plugin:screen-capture|stop_video_cmd');
      } else {
        // iOS: Stop capture stream and broadcast
        await invoke('stop_capture_stream_cmd');
        await invoke('plugin:screen-capture|stop_capture_cmd');
      }
      this.capturing = false;
      this.latestFrameBytes = null;
      this.latestBase64Frame = null;
    } catch (error) {
      throw error;
    }
  }

  /**
   * Stop only audio capture.
   * Works on desktop (macOS) and iOS.
   */
  async stopAudio(): Promise<void> {
    if (!isTauri()) {
      throw new Error('Audio capture only available in Tauri');
    }

    try {
      if (isDesktop()) {
        await invoke('plugin:screen-capture|stop_audio_cmd');
      } else {
        // iOS: Stop audio stream and broadcast
        await invoke('stop_audio_stream_cmd');
        await invoke('plugin:screen-capture|stop_capture_cmd');
      }
      this.latestAudioData = null;
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
      Logger.warn("TAURI_STREAM", "Screen selector window not found, it may need to be recreated");
      throw new Error('Screen selector window not available');
    }

    // Show and focus the window
    await selectorWindow.show();
    await selectorWindow.setFocus();
  }
}

export const tauriStreamCapture = new TauriStreamCapture();

// Re-export with old names for backward compatibility
export const tauriScreenCapture = tauriStreamCapture;
export const mobileScreenCapture = tauriStreamCapture;
