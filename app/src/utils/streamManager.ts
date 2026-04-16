import { Logger } from './logging';
import { TranscriptionRouter } from './whisper/TranscriptionRouter';
import { TranscriptionSubscriber, TranscriptionSubscriberImpl } from './whisper/TranscriptionSubscriber';
import { UnifiedTranscriptionService } from './whisper/UnifiedTranscriptionService';
import { isWeb } from './platform';
import { browserStreamCapture } from './browserStreamCapture';
import { tauriStreamCapture, PCMCallback } from './tauriStreamCapture';
import { PCMAudioCapture, createPCMAudioCapture } from './audio/PCMAudioCapture';
import { getAgentCrop } from './screenCapture';

// --- Core Type Definitions ---
export type AudioStreamType = 'screenAudio' | 'microphone' | 'allAudio';
export type PseudoStreamType = 'camera' | 'screenVideo' | AudioStreamType;
type MasterStreamType = 'display' | 'camera' | 'microphone' | 'screenAudio';

/**
 * The single source of truth for the stream state provided to the UI.
 * It is clean, specific, and has no legacy/adapter properties.
 */
export interface StreamState {
  cameraStream: MediaStream | null;
  screenVideoStream: MediaStream | null;
  screenVideoStreamWithPip: MediaStream | null; // Screen stream with PiP overlay (mobile only)
  screenAudioStream: MediaStream | null;
  microphoneStream: MediaStream | null;
  allAudioStream: MediaStream | null;
}
type StreamListener = (state: StreamState) => void;

class Manager {
  // --- Clean Pseudo-Stream State ---
  private cameraStream: MediaStream | null = null;
  private screenVideoStream: MediaStream | null = null;
  private screenVideoStreamWithPip: MediaStream | null = null; // Stream with PiP overlay
  private screenAudioStream: MediaStream | null = null;
  private microphoneStream: MediaStream | null = null;
  private allAudioStream: MediaStream | null = null;

  // Persistent video elements for capture (always playing)
  private cameraVideoElement: HTMLVideoElement | null = null;
  private screenVideoElement: HTMLVideoElement | null = null;

  // --- Internal Management State ---
  private masterDisplayStream: MediaStream | null = null;
  private masterCameraStream: MediaStream | null = null;
  private masterMicrophoneStream: MediaStream | null = null;
  
  private userSets = new Map<PseudoStreamType, Set<string>>();
  private listeners = new Set<StreamListener>();
  private pendingMasterStreams = new Map<MasterStreamType, Promise<void>>();

  // Subscriber management for agent-owned transcripts
  // Key format: `${agentId}:${streamType}`
  private subscribers = new Map<string, TranscriptionSubscriber>();

  private audioContext: AudioContext | null = null;
  private sourceNodes = new Map<string, MediaStreamAudioSourceNode>();

  // PCM capture state (services are managed by TranscriptionRouter)
  private browserPCMCaptures = new Map<AudioStreamType, PCMAudioCapture>();
  private tauriPCMCallbackActive = false;

  // PiP overlay status for mobile
  private pipOverlayStatus: {
    state: 'STARTING' | 'CAPTURING' | 'THINKING' | 'RESPONDING' | 'WAITING' | 'SLEEPING' | 'SKIPPED' | 'IDLE';
    progress?: number; // 0-100 for pie chart (WAITING/SLEEPING states)
    timerSeconds?: number;
  } | null = null;

  constructor() {
    const allStreamTypes: PseudoStreamType[] = ['camera', 'screenVideo', 'screenAudio', 'microphone', 'allAudio'];
    allStreamTypes.forEach(type => this.userSets.set(type, new Set()));
  }

  // --- Public API ---
  
  /** The primary method for acquiring streams, using a "blueprint" approach. */
  public async requestStreamsForAgent(agentId: string, requiredStreams: PseudoStreamType[]): Promise<void> {
    Logger.debug("StreamManager", `Processing stream blueprint for agent '${agentId}': [${requiredStreams.join(', ')}]`);

    // Get the appropriate capture implementation based on platform
    const captureImpl = isWeb() ? browserStreamCapture : tauriStreamCapture;

    // Let the platform-specific implementation map pseudo streams to master streams
    const requiredMasterStreams = new Set<MasterStreamType>();
    for (const type of requiredStreams) {
      const masterStreams = captureImpl.getMasterStreamsForPseudoStream(type);
      masterStreams.forEach(ms => requiredMasterStreams.add(ms));
    }

    const acquisitionPromises: Promise<void>[] = [];
    requiredMasterStreams.forEach(masterType => acquisitionPromises.push(this.ensureMasterStream(masterType, captureImpl)));

    try {
      await Promise.all(acquisitionPromises);

      for (const type of requiredStreams) {
        if (!this.isPseudoStreamAvailable(type)) {
          throw new Error(`Failed to acquire required stream component '${type}'. Permission may have been denied.`);
        }
      }

      // --- Centralized Transcription & Mixer Logic ---
      // Start transcription services and auto-subscribe the agent to each stream type.
      // This ensures the agent receives all transcribed text from the moment streams are acquired.
      if (requiredStreams.includes('allAudio')) {
        await this.initializeAudioMixer();
      }
      if (requiredStreams.includes('microphone') && this.microphoneStream) {
        await this.startTranscriptionForStream('microphone', this.microphoneStream);
        this.getOrCreateSubscriber(agentId, 'microphone');
      }
      if (requiredStreams.includes('screenAudio') && this.screenAudioStream) {
        await this.startTranscriptionForStream('screenAudio', this.screenAudioStream);
        this.getOrCreateSubscriber(agentId, 'screenAudio');
      }
      // Notify listeners after subscribers are created so hooks can detect them
      this.notifyListeners();
      // --- END ---

      requiredStreams.forEach(type => this.userSets.get(type)?.add(agentId));

    } catch (error) {
      Logger.error("StreamManager", `Failed to fulfill stream blueprint for agent '${agentId}'.`, error);
      throw error;
    }
  }

  /** Releases all streams used by a specific agent. */
  public releaseStreamsForAgent(agentId: string): void {
    Logger.debug("StreamManager", `Releasing all streams for agent '${agentId}'`);
    this.userSets.forEach((users, _type) => {
        if (users.has(agentId)) {
            users.delete(agentId);
        }
    });
    this.checkForTeardown();
  }

  // --- Subscriber Management API ---

  /**
   * Create subscriber key from agentId and stream type
   */
  private subscriberKey(agentId: string, type: AudioStreamType): string {
    return `${agentId}:${type}`;
  }

  /**
   * Get or create a subscriber for an agent and stream type.
   * The subscriber accumulates transcripts independently for each agent.
   */
  public getOrCreateSubscriber(agentId: string, type: AudioStreamType): TranscriptionSubscriber {
    const key = this.subscriberKey(agentId, type);

    if (!this.subscribers.has(key)) {
      const subscriber = new TranscriptionSubscriberImpl(agentId, type);
      this.subscribers.set(key, subscriber);

      // Register with transcription service to receive pushes
      const router = TranscriptionRouter.getInstance();
      const service = router.getActiveService(type);
      if (service) {
        service.addSubscriber(subscriber);
        Logger.debug("StreamManager", `Registered subscriber ${key} with transcription service for ${type}`);
      } else {
        Logger.warn("StreamManager", `No transcription service found for ${type} when creating subscriber ${key}`);
      }
    }

    return this.subscribers.get(key)!;
  }

  /**
   * Get an existing subscriber for an agent and stream type.
   * Returns undefined if no subscriber exists.
   */
  public getSubscriber(agentId: string, type: AudioStreamType): TranscriptionSubscriber | undefined {
    const key = this.subscriberKey(agentId, type);
    return this.subscribers.get(key);
  }

  /**
   * Clear all subscriber transcripts for an agent.
   * Called at the end of each agent loop iteration.
   */
  public clearSubscriberTranscripts(agentId: string): void {
    let clearedCount = 0;
    for (const [key, subscriber] of this.subscribers) {
      if (key.startsWith(`${agentId}:`)) {
        subscriber.clear();
        clearedCount++;
      }
    }
    if (clearedCount > 0) {
      Logger.debug("StreamManager", `Cleared ${clearedCount} subscriber transcript(s) for agent ${agentId}`);
    }
  }

  /**
   * Destroy all subscribers for an agent.
   * Called when an agent stops.
   */
  public destroySubscribersForAgent(agentId: string): void {
    const keysToDelete: string[] = [];

    const router = TranscriptionRouter.getInstance();
    for (const [key, subscriber] of this.subscribers) {
      if (key.startsWith(`${agentId}:`)) {
        // Remove from transcription service
        const service = router.getActiveService(subscriber.streamType);
        if (service) {
          service.removeSubscriber(subscriber);
        }

        subscriber.destroy();
        keysToDelete.push(key);
      }
    }

    keysToDelete.forEach(key => this.subscribers.delete(key));

    if (keysToDelete.length > 0) {
      Logger.debug("StreamManager", `Destroyed ${keysToDelete.length} subscriber(s) for agent ${agentId}`);
    }
  }

  public addListener(listener: StreamListener): void { this.listeners.add(listener); listener(this.getCurrentState()); }
  public removeListener(listener: StreamListener): void { this.listeners.delete(listener); }
  
  public getCurrentState(): StreamState {
    return {
      cameraStream: this.cameraStream,
      screenVideoStream: this.screenVideoStream,
      screenVideoStreamWithPip: this.screenVideoStreamWithPip,
      screenAudioStream: this.screenAudioStream,
      microphoneStream: this.microphoneStream,
      allAudioStream: this.allAudioStream,
    };
  }

  /** Set the PiP overlay status (for mobile) */
  public setPipOverlayStatus(status: {
    state: 'STARTING' | 'CAPTURING' | 'THINKING' | 'RESPONDING' | 'WAITING' | 'SLEEPING' | 'SKIPPED' | 'IDLE';
    progress?: number;
    timerSeconds?: number;
  } | null): void {
    this.pipOverlayStatus = status;
  }

  // --- Camera Device Management ---

  /** Get list of available camera devices */
  public async getAvailableCameraDevices(): Promise<MediaDeviceInfo[]> {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      Logger.debug("StreamManager", `Found ${videoDevices.length} camera devices.`);
      return videoDevices;
    } catch (error) {
      Logger.error("StreamManager", "Failed to enumerate camera devices.", error);
      return [];
    }
  }

  /** Get the stored preferred camera device ID from localStorage */
  public getPreferredCameraDevice(): string | null {
    return localStorage.getItem('observer_preferred_camera_device');
  }

  /** Set the preferred camera device ID in localStorage */
  public setPreferredCameraDevice(deviceId: string): void {
    localStorage.setItem('observer_preferred_camera_device', deviceId);
    Logger.info("StreamManager", `Preferred camera device set to: ${deviceId}`);
  }

  /** Switch to a different camera device (stops current stream and re-acquires with new device) */
  public async switchCameraDevice(deviceId: string): Promise<void> {
    Logger.info("StreamManager", `Switching camera to device: ${deviceId}`);

    // Save the preference
    this.setPreferredCameraDevice(deviceId);

    // Get all agents currently using the camera
    const cameraUsers = Array.from(this.userSets.get('camera') || []);

    // Teardown current camera stream
    this.teardownCameraStream();

    // Re-acquire camera with new device for all users
    if (cameraUsers.length > 0) {
      try {
        const captureImpl = isWeb() ? browserStreamCapture : tauriStreamCapture;
        await this.ensureMasterStream('camera', captureImpl);
        Logger.info("StreamManager", "Camera switched successfully.");
      } catch (error) {
        Logger.error("StreamManager", "Failed to switch camera device.", error);
        throw error;
      }
    }
  }

  // --- Frame Capture API ---

  /**
   * Create a persistent video element that stays attached to a stream.
   * Waits for the video to be fully ready (first frame rendered) before resolving.
   * This eliminates race conditions on frame capture, especially on iOS.
   */
  private async createPersistentVideoElement(stream: MediaStream, timeoutMs: number = 5000): Promise<HTMLVideoElement> {
    const video = document.createElement('video');
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;

    // Wait for video to be ready with first frame
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        Logger.warn("StreamManager", "Video element ready timeout - proceeding anyway");
        resolve();
      }, timeoutMs);

      const checkReady = () => {
        // readyState >= 2 (HAVE_CURRENT_DATA) means we have at least one frame
        if (video.readyState >= 2 && video.videoWidth > 0) {
          clearTimeout(timeout);
          resolve();
          return true;
        }
        return false;
      };

      // Check if already ready
      if (checkReady()) return;

      // Listen for loadeddata event (fires when first frame is available)
      video.addEventListener('loadeddata', () => {
        if (checkReady()) return;
      }, { once: true });

      // Also listen for canplay as fallback
      video.addEventListener('canplay', () => {
        if (checkReady()) return;
      }, { once: true });

      // Start playing
      video.play().catch(e => {
        Logger.warn("StreamManager", `Video play failed: ${e}`);
        clearTimeout(timeout);
        reject(e);
      });
    });

    Logger.debug("StreamManager", `Video element ready: readyState=${video.readyState}, ${video.videoWidth}x${video.videoHeight}`);
    return video;
  }

  /**
   * Capture a frame from a video stream. Returns base64 PNG or null.
   *
   * For Tauri screen capture: uses raw frame bytes from the channel directly.
   * This bypasses the canvas-backed video element which can freeze when iOS
   * backgrounds the app. The broadcast extension continues sending frames
   * via the channel, ensuring main_loop receives fresh frames in PiP mode.
   *
   * For camera/browser: uses persistent video elements for instant capture.
   */
  public async captureFrame(
    streamType: 'camera' | 'screen',
    agentId?: string
  ): Promise<string | null> {
    // For Tauri screen capture, use raw frame bytes from channel
    // This is more reliable than canvas-backed video element (especially on iOS)
    if (!isWeb() && streamType === 'screen') {
      const rawFrame = tauriStreamCapture.getLatestBase64Frame();
      if (!rawFrame) {
        Logger.warn("StreamManager", "Cannot capture screen: no raw frame available from channel");
        return null;
      }

      const crop = agentId ? getAgentCrop(agentId, streamType) : null;

      // If no crop, return raw frame directly (more efficient - no re-encoding)
      if (!crop) {
        return rawFrame;
      }

      // Apply crop by decoding, cropping, and re-encoding
      try {
        const blob = await fetch(`data:image/jpeg;base64,${rawFrame}`).then(r => r.blob());
        const bitmap = await createImageBitmap(blob);

        const canvas = document.createElement('canvas');
        canvas.width = crop.width;
        canvas.height = crop.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          bitmap.close();
          return rawFrame; // Fallback to uncropped
        }

        const safeX = Math.min(crop.x, bitmap.width - 1);
        const safeY = Math.min(crop.y, bitmap.height - 1);
        const safeWidth = Math.min(crop.width, bitmap.width - safeX);
        const safeHeight = Math.min(crop.height, bitmap.height - safeY);

        ctx.drawImage(bitmap, safeX, safeY, safeWidth, safeHeight, 0, 0, canvas.width, canvas.height);
        bitmap.close();

        return canvas.toDataURL('image/png').split(',')[1];
      } catch (e) {
        Logger.warn("StreamManager", `Failed to apply crop to raw frame: ${e}`);
        return rawFrame; // Fallback to uncropped
      }
    }

    // For camera or browser, use persistent video element
    const video = streamType === 'camera'
      ? this.cameraVideoElement
      : this.screenVideoElement;

    if (!video || video.readyState < 2 || video.videoWidth === 0) {
      Logger.warn("StreamManager", `Cannot capture ${streamType}: video not ready`);
      return null;
    }

    const canvas = document.createElement('canvas');
    const crop = agentId ? getAgentCrop(agentId, streamType) : null;

    if (crop) {
      canvas.width = crop.width;
      canvas.height = crop.height;
    } else {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    // Draw with crop support
    if (crop) {
      const safeX = Math.min(crop.x, video.videoWidth - 1);
      const safeY = Math.min(crop.y, video.videoHeight - 1);
      const safeWidth = Math.min(crop.width, video.videoWidth - safeX);
      const safeHeight = Math.min(crop.height, video.videoHeight - safeY);
      ctx.drawImage(video, safeX, safeY, safeWidth, safeHeight, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.drawImage(video, 0, 0);
    }

    return canvas.toDataURL('image/png').split(',')[1];
  }

  // --- Private Implementation ---

  private ensureMasterStream(type: MasterStreamType, captureImpl: typeof browserStreamCapture | typeof tauriStreamCapture): Promise<void> {
    if (this.pendingMasterStreams.has(type)) {
      Logger.debug("StreamManager", `Joining pending request for '${type}' stream.`);
      return this.pendingMasterStreams.get(type)!;
    }

    const promise = this.acquireMasterStream(type, captureImpl).finally(() => {
      this.pendingMasterStreams.delete(type);
    });

    this.pendingMasterStreams.set(type, promise);
    return promise;
  }

  private async acquireMasterStream(type: MasterStreamType, captureImpl: typeof browserStreamCapture | typeof tauriStreamCapture): Promise<void> {
    try {
      // Set up PiP overlay if using Tauri and acquiring display stream
      if (!isWeb() && type === 'display') {
        tauriStreamCapture.setPipOverlayDrawer((ctx, width, height) => {
          if (this.pipOverlayStatus) {
            this.drawPipOverlay(ctx, width, height);
          }
        });
      }

      // Delegate to platform-specific implementation
      await captureImpl.acquireMasterStream(type);

      // Sync local state from the implementation
      const streams = captureImpl.getStreams();

      if ('screenVideoStream' in streams) {
        // Tauri streams
        this.screenVideoStream = streams.screenVideoStream;
        if (!isWeb()) {
          // Tauri has separate PiP stream
          const tauriStreams = streams as any;
          this.screenVideoStreamWithPip = tauriStreams.screenVideoStreamWithPip || null;
        }
        this.screenAudioStream = streams.screenAudioStream;
        this.cameraStream = streams.cameraStream;
        this.microphoneStream = streams.microphoneStream;
      } else {
        // Browser streams
        const browserStreams = streams as any;
        if (browserStreams.masterDisplayStream) {
          this.masterDisplayStream = browserStreams.masterDisplayStream;
        }
        if (browserStreams.masterCameraStream) {
          this.masterCameraStream = browserStreams.masterCameraStream;
        }
        if (browserStreams.masterMicrophoneStream) {
          this.masterMicrophoneStream = browserStreams.masterMicrophoneStream;
        }
        this.screenVideoStream = browserStreams.screenVideoStream;
        this.screenAudioStream = browserStreams.screenAudioStream;
        this.cameraStream = browserStreams.cameraStream;
        this.microphoneStream = browserStreams.microphoneStream;
      }

      // Create persistent video elements for capture (fixes iOS race condition)
      // We await these to ensure the video has its first frame before the stream is considered "ready"
      if (type === 'camera' && this.cameraStream && !this.cameraVideoElement) {
        Logger.debug("StreamManager", "Creating persistent camera video element...");
        this.cameraVideoElement = await this.createPersistentVideoElement(this.cameraStream);
      }
      if (type === 'display' && this.screenVideoStream && !this.screenVideoElement) {
        Logger.debug("StreamManager", "Creating persistent screen video element...");
        this.screenVideoElement = await this.createPersistentVideoElement(this.screenVideoStream);
      }

      Logger.info("StreamManager", `Master '${type}' stream acquired and synced.`);
      this.notifyListeners();
    } catch (error) {
      Logger.error("StreamManager", `Acquisition of master '${type}' stream failed.`, error);
      throw error;
    }
  }
  
  private isPseudoStreamAvailable(type: PseudoStreamType): boolean {
    const captureImpl = isWeb() ? browserStreamCapture : tauriStreamCapture;

    switch (type) {
        case 'camera': return captureImpl.isStreamAvailable('camera');
        case 'screenVideo': return captureImpl.isStreamAvailable('screenVideo');
        case 'screenAudio': return captureImpl.isStreamAvailable('screenAudio');
        case 'microphone': return captureImpl.isStreamAvailable('microphone');
        case 'allAudio':
          return captureImpl.isStreamAvailable('screenAudio') && captureImpl.isStreamAvailable('microphone');
    }
    return false;
  }

  private async initializeAudioMixer(): Promise<void> {
    if (this.audioContext || !this.screenAudioStream || !this.microphoneStream) return;
    Logger.info("StreamManager", "Initializing audio mixer for 'allAudio'.");
    this.audioContext = new AudioContext();
    const destination = this.audioContext.createMediaStreamDestination();
    const screenSource = this.audioContext.createMediaStreamSource(this.screenAudioStream);
    screenSource.connect(destination);
    this.sourceNodes.set('screen', screenSource);
    const micSource = this.audioContext.createMediaStreamSource(this.microphoneStream);
    micSource.connect(destination);
    this.sourceNodes.set('microphone', micSource);
    this.allAudioStream = destination.stream;
    // This now correctly starts transcription only when the mixer is initialized
    await this.startTranscriptionForStream('allAudio', this.allAudioStream);
    this.notifyListeners();
  }
  
  private checkForTeardown(): void {
    const isUsed = (type: PseudoStreamType) => (this.userSets.get(type)?.size || 0) > 0;
    // This logic is now robust because it correctly reflects the lifecycle based on user sets.
    if (!isUsed('screenVideo')) this.teardownDisplayStream();
    if (!isUsed('screenAudio') && !isUsed('allAudio')) this.teardownScreenAudioStream();
    if (!isUsed('microphone') && !isUsed('allAudio')) this.teardownMicrophoneStream();
    if (!isUsed('camera')) this.teardownCameraStream();
    if (!isUsed('allAudio')) this.teardownAudioMixer();
  }
  

  private teardownDisplayStream(): void {
    const captureImpl = isWeb() ? browserStreamCapture : tauriStreamCapture;

    if (!this.screenVideoStream && !this.masterDisplayStream) return;

    Logger.info("StreamManager", "Tearing down display stream.");
    captureImpl.teardownMasterStream('display');

    // Clean up persistent video element
    if (this.screenVideoElement) {
      this.screenVideoElement.srcObject = null;
      this.screenVideoElement = null;
    }

    // Sync local state
    this.masterDisplayStream = null;
    this.screenVideoStream = null;
    this.screenVideoStreamWithPip = null;

    this.notifyListeners();
  }

  private teardownScreenAudioStream(): void {
    const captureImpl = isWeb() ? browserStreamCapture : tauriStreamCapture;

    if (!this.screenAudioStream) return;

    Logger.info("StreamManager", "Tearing down screen audio stream.");
    this.stopTranscriptionForStream('screenAudio');
    captureImpl.teardownMasterStream('screenAudio');

    // Sync local state
    this.screenAudioStream = null;

    this.notifyListeners();
  }

  private teardownCameraStream(): void {
    const captureImpl = isWeb() ? browserStreamCapture : tauriStreamCapture;

    if (!this.cameraStream && !this.masterCameraStream) return;

    Logger.info("StreamManager", "Tearing down camera stream.");
    captureImpl.teardownMasterStream('camera');

    // Clean up persistent video element
    if (this.cameraVideoElement) {
      this.cameraVideoElement.srcObject = null;
      this.cameraVideoElement = null;
    }

    // Sync local state
    this.masterCameraStream = null;
    this.cameraStream = null;

    this.notifyListeners();
  }

  private teardownMicrophoneStream(): void {
    const captureImpl = isWeb() ? browserStreamCapture : tauriStreamCapture;

    if (!this.microphoneStream && !this.masterMicrophoneStream) return;

    Logger.info("StreamManager", "Tearing down microphone stream.");
    this.stopTranscriptionForStream('microphone');
    captureImpl.teardownMasterStream('microphone');

    // Sync local state
    this.masterMicrophoneStream = null;
    this.microphoneStream = null;

    this.notifyListeners();
  }
  
  private teardownAudioMixer(): void {
    if (!this.audioContext) return;
    Logger.info("StreamManager", "Tearing down audio mixer.");
    this.sourceNodes.forEach(node => node.disconnect());
    this.sourceNodes.clear();
    this.stopTranscriptionForStream('allAudio'); // This correctly stops the mixed audio transcription
    this.audioContext.close();
    this.audioContext = null;
    this.allAudioStream = null;
    this.notifyListeners();
  }

  private async startTranscriptionForStream(type: AudioStreamType, stream: MediaStream): Promise<void> {
    const router = TranscriptionRouter.getInstance();

    // Router handles singleton logic - returns existing or creates new
    const service = await router.acquireService(type);

    // Set up PCM capture to feed samples to the service
    await this.startPCMCapture(type, stream, service);

    // Register any existing subscribers for this stream type
    for (const [key, subscriber] of this.subscribers) {
      if (subscriber.streamType === type) {
        service.addSubscriber(subscriber);
        Logger.debug("StreamManager", `Registered existing subscriber ${key} with transcription service for ${type}`);
      }
    }
  }

  /**
   * Start PCM capture to feed samples to the transcription service
   */
  private async startPCMCapture(type: AudioStreamType, stream: MediaStream, service: UnifiedTranscriptionService): Promise<void> {
    // Skip if already capturing for this type
    if (this.browserPCMCaptures.has(type)) return;

    // screenAudio in Tauri comes from Rust via channel - use Tauri PCM callback
    // microphone uses standard MediaStream (getUserMedia) - use PCMAudioCapture
    const useTauriCallback = !isWeb() && type === 'screenAudio';

    if (useTauriCallback) {
      // Tauri screenAudio: Set up PCM callback on tauriStreamCapture
      // The callback will receive samples at 16kHz from the Rust resampler
      if (!this.tauriPCMCallbackActive) {
        const pcmCallback: PCMCallback = (samples, streamType) => {
          // Route samples to the appropriate service via router
          const activeService = TranscriptionRouter.getInstance().getActiveService(streamType);
          if (activeService) {
            activeService.feedPCM(samples);
          }
        };
        tauriStreamCapture.setPCMCallback(pcmCallback);
        this.tauriPCMCallbackActive = true;
        Logger.info("StreamManager", "Tauri PCM callback set for unified pipeline");
      }
    } else {
      // Standard MediaStream (browser, or Tauri microphone): Use PCMAudioCapture
      const pcmCapture = createPCMAudioCapture();
      await pcmCapture.start(stream, (samples) => {
        service.feedPCM(samples);
      });
      this.browserPCMCaptures.set(type, pcmCapture);
      Logger.info("StreamManager", `PCM capture started for '${type}'`);
    }
  }

  private stopTranscriptionForStream(type: AudioStreamType): void {
    // Stop PCMAudioCapture if used for this type (microphone, allAudio, or browser mode)
    const pcmCapture = this.browserPCMCaptures.get(type);
    if (pcmCapture) {
      pcmCapture.stop();
      this.browserPCMCaptures.delete(type);
    }

    // Clear Tauri PCM callback if stopping screenAudio (the only type that uses it)
    if (!isWeb() && type === 'screenAudio' && this.tauriPCMCallbackActive) {
      tauriStreamCapture.setPCMCallback(null);
      this.tauriPCMCallbackActive = false;
    }

    // Release service (router handles refcount and stopping)
    const router = TranscriptionRouter.getInstance();
    router.releaseService(type);
  }

  private notifyListeners(): void {
    const state = this.getCurrentState();
    this.listeners.forEach(listener => listener(state));
  }

  private drawPipOverlay(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    if (!this.pipOverlayStatus) return;

    const { state, progress, timerSeconds } = this.pipOverlayStatus;

    // Size: ~50% of video width, positioned bottom-right
    const boxWidth = Math.round(width * 0.5);
    const boxHeight = Math.round(boxWidth * 0.6);
    const padding = Math.round(width * 0.03);
    const x = width - boxWidth - padding;
    const y = height - boxHeight - padding;
    const cornerRadius = Math.round(boxWidth * 0.08);

    // Semi-transparent background with rounded corners
    ctx.fillStyle = 'rgba(17, 24, 39, 0.85)'; // gray-900 with transparency
    ctx.beginPath();
    ctx.moveTo(x + cornerRadius, y);
    ctx.lineTo(x + boxWidth - cornerRadius, y);
    ctx.quadraticCurveTo(x + boxWidth, y, x + boxWidth, y + cornerRadius);
    ctx.lineTo(x + boxWidth, y + boxHeight - cornerRadius);
    ctx.quadraticCurveTo(x + boxWidth, y + boxHeight, x + boxWidth - cornerRadius, y + boxHeight);
    ctx.lineTo(x + cornerRadius, y + boxHeight);
    ctx.quadraticCurveTo(x, y + boxHeight, x, y + boxHeight - cornerRadius);
    ctx.lineTo(x, y + cornerRadius);
    ctx.quadraticCurveTo(x, y, x + cornerRadius, y);
    ctx.closePath();
    ctx.fill();

    // Subtle border
    ctx.strokeStyle = 'rgba(75, 85, 99, 0.6)'; // gray-600
    ctx.lineWidth = 2;
    ctx.stroke();

    const centerX = x + boxWidth / 2;
    const centerY = y + boxHeight * 0.42;
    const iconSize = Math.round(boxWidth * 0.22);

    // Get color based on state
    const getStateColor = (): string => {
      switch (state) {
        case 'STARTING': return '#facc15'; // yellow-400
        case 'CAPTURING': return '#22d3ee'; // cyan-400
        case 'THINKING': return '#a855f7'; // purple-500
        case 'RESPONDING': return '#3b82f6'; // blue-500
        case 'WAITING': return '#6b7280'; // gray-500
        case 'SLEEPING': return '#3b82f6'; // blue-500
        case 'SKIPPED': return '#f97316'; // orange-500
        default: return '#6b7280'; // gray-500
      }
    };

    const color = getStateColor();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = Math.max(2, iconSize * 0.12);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Draw icon based on state (Lucide-style)
    this.drawStateIcon(ctx, centerX, centerY, iconSize, state);

    // Draw pie chart for WAITING/SLEEPING states
    if ((state === 'WAITING' || state === 'SLEEPING') && progress !== undefined) {
      this.drawPieChart(ctx, centerX, centerY, iconSize * 0.9, progress, color);
    }

    // Timer text below icon
    if (timerSeconds !== undefined) {
      const fontSize = Math.round(boxWidth * 0.14);
      ctx.font = `600 ${fontSize}px -apple-system, BlinkMacSystemFont, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';

      // Format time
      const minutes = Math.floor(timerSeconds / 60);
      const seconds = timerSeconds % 60;
      const timeText = minutes > 0
        ? `${minutes}:${seconds.toString().padStart(2, '0')}`
        : `${seconds}s`;

      ctx.fillText(timeText, centerX, y + boxHeight * 0.78);
    }
  }

  private drawStateIcon(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    size: number,
    state: string
  ): void {
    const s = size / 2; // half size for easier math

    ctx.save();
    ctx.translate(cx, cy);

    switch (state) {
      case 'STARTING': // Power icon
        ctx.beginPath();
        ctx.moveTo(0, -s * 0.8);
        ctx.lineTo(0, -s * 0.2);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(0, 0, s * 0.7, -Math.PI * 0.7, Math.PI * 1.7, false);
        ctx.stroke();
        break;

      case 'CAPTURING': // Eye icon
        ctx.beginPath();
        // Eye outline
        ctx.moveTo(-s, 0);
        ctx.bezierCurveTo(-s * 0.5, -s * 0.7, s * 0.5, -s * 0.7, s, 0);
        ctx.bezierCurveTo(s * 0.5, s * 0.7, -s * 0.5, s * 0.7, -s, 0);
        ctx.stroke();
        // Pupil
        ctx.beginPath();
        ctx.arc(0, 0, s * 0.3, 0, Math.PI * 2);
        ctx.fill();
        break;

      case 'THINKING': { // CPU icon
        const r = s * 0.15; // corner radius
        // Outer rectangle with rounded corners
        ctx.beginPath();
        ctx.moveTo(-s * 0.65 + r, -s * 0.65);
        ctx.lineTo(s * 0.65 - r, -s * 0.65);
        ctx.quadraticCurveTo(s * 0.65, -s * 0.65, s * 0.65, -s * 0.65 + r);
        ctx.lineTo(s * 0.65, s * 0.65 - r);
        ctx.quadraticCurveTo(s * 0.65, s * 0.65, s * 0.65 - r, s * 0.65);
        ctx.lineTo(-s * 0.65 + r, s * 0.65);
        ctx.quadraticCurveTo(-s * 0.65, s * 0.65, -s * 0.65, s * 0.65 - r);
        ctx.lineTo(-s * 0.65, -s * 0.65 + r);
        ctx.quadraticCurveTo(-s * 0.65, -s * 0.65, -s * 0.65 + r, -s * 0.65);
        ctx.stroke();
        // Inner rectangle (the "chip")
        ctx.beginPath();
        ctx.rect(-s * 0.3, -s * 0.3, s * 0.6, s * 0.6);
        ctx.stroke();
        // Top pins
        ctx.beginPath();
        ctx.moveTo(-s * 0.25, -s * 0.65);
        ctx.lineTo(-s * 0.25, -s);
        ctx.moveTo(s * 0.25, -s * 0.65);
        ctx.lineTo(s * 0.25, -s);
        ctx.stroke();
        // Bottom pins
        ctx.beginPath();
        ctx.moveTo(-s * 0.25, s * 0.65);
        ctx.lineTo(-s * 0.25, s);
        ctx.moveTo(s * 0.25, s * 0.65);
        ctx.lineTo(s * 0.25, s);
        ctx.stroke();
        // Left pins
        ctx.beginPath();
        ctx.moveTo(-s * 0.65, -s * 0.25);
        ctx.lineTo(-s, -s * 0.25);
        ctx.moveTo(-s * 0.65, s * 0.25);
        ctx.lineTo(-s, s * 0.25);
        ctx.stroke();
        // Right pins
        ctx.beginPath();
        ctx.moveTo(s * 0.65, -s * 0.25);
        ctx.lineTo(s, -s * 0.25);
        ctx.moveTo(s * 0.65, s * 0.25);
        ctx.lineTo(s, s * 0.25);
        ctx.stroke();
        break;
      }

      case 'RESPONDING': // Message/chat icon
        ctx.beginPath();
        ctx.moveTo(-s * 0.8, -s * 0.5);
        ctx.lineTo(s * 0.8, -s * 0.5);
        ctx.quadraticCurveTo(s, -s * 0.5, s, -s * 0.3);
        ctx.lineTo(s, s * 0.3);
        ctx.quadraticCurveTo(s, s * 0.5, s * 0.8, s * 0.5);
        ctx.lineTo(-s * 0.3, s * 0.5);
        ctx.lineTo(-s * 0.5, s * 0.8);
        ctx.lineTo(-s * 0.5, s * 0.5);
        ctx.lineTo(-s * 0.8, s * 0.5);
        ctx.quadraticCurveTo(-s, s * 0.5, -s, s * 0.3);
        ctx.lineTo(-s, -s * 0.3);
        ctx.quadraticCurveTo(-s, -s * 0.5, -s * 0.8, -s * 0.5);
        ctx.stroke();
        break;

      case 'SLEEPING': // Moon icon
        ctx.beginPath();
        ctx.arc(s * 0.2, 0, s * 0.7, Math.PI * 0.75, Math.PI * 2.25, false);
        ctx.arc(-s * 0.3, -s * 0.1, s * 0.5, Math.PI * 0.25, Math.PI * 1.25, true);
        ctx.stroke();
        break;

      case 'SKIPPED': // Skip forward icon
        ctx.beginPath();
        ctx.moveTo(-s * 0.6, -s * 0.6);
        ctx.lineTo(s * 0.2, 0);
        ctx.lineTo(-s * 0.6, s * 0.6);
        ctx.closePath();
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(s * 0.3, -s * 0.6);
        ctx.lineTo(s * 0.3, s * 0.6);
        ctx.stroke();
        break;

      case 'WAITING': // Clock icon (will be overlaid with pie chart)
      default:
        // Just draw circle outline - pie chart will be drawn on top
        ctx.beginPath();
        ctx.arc(0, 0, s * 0.8, 0, Math.PI * 2);
        ctx.stroke();
        // Clock hands
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(0, -s * 0.5);
        ctx.moveTo(0, 0);
        ctx.lineTo(s * 0.35, s * 0.1);
        ctx.stroke();
        break;
    }

    ctx.restore();
  }

  private drawPieChart(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    radius: number,
    progress: number,
    color: string
  ): void {
    // Progress is 0-100
    const normalizedProgress = Math.min(100, Math.max(0, progress)) / 100;

    // Draw background track
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(75, 85, 99, 0.4)';
    ctx.lineWidth = radius * 0.2;
    ctx.stroke();

    // Draw progress arc
    const startAngle = -Math.PI / 2; // Start from top
    const endAngle = startAngle + (normalizedProgress * Math.PI * 2);

    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, endAngle);
    ctx.strokeStyle = color;
    ctx.lineWidth = radius * 0.2;
    ctx.lineCap = 'round';
    ctx.stroke();
  }
}

export const StreamManager = new Manager();
