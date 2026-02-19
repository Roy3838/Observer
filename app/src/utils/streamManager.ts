import { Logger } from './logging';
import { WhisperTranscriptionService } from './whisper/WhisperTranscriptionService';
import { hasNativeScreenCapture } from './platform';
import { tauriScreenCapture } from './tauriScreenCapture';

// --- Core Type Definitions ---
export type AudioStreamType = 'screenAudio' | 'microphone' | 'allAudio';
export type PseudoStreamType = 'camera' | 'screenVideo' | AudioStreamType;
type MasterStreamType = 'display' | 'camera' | 'microphone';

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
  
  // --- Internal Management State ---
  private masterDisplayStream: MediaStream | null = null;
  private masterCameraStream: MediaStream | null = null;
  private masterMicrophoneStream: MediaStream | null = null;
  
  private userSets = new Map<PseudoStreamType, Set<string>>();
  private listeners = new Set<StreamListener>();
  private pendingMasterStreams = new Map<MasterStreamType, Promise<void>>();
  private transcriptionServices = new Map<AudioStreamType, WhisperTranscriptionService>();
  
  private audioContext: AudioContext | null = null;
  private sourceNodes = new Map<string, MediaStreamAudioSourceNode>();

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
    
    const requiredMasterStreams = new Set<MasterStreamType>();
    for (const type of requiredStreams) {
      if (type === 'camera') requiredMasterStreams.add('camera');
      if (type === 'screenVideo' || type === 'screenAudio' || type === 'allAudio') requiredMasterStreams.add('display');
      if (type === 'microphone' || type === 'allAudio') requiredMasterStreams.add('microphone');
    }

    const acquisitionPromises: Promise<void>[] = [];
    requiredMasterStreams.forEach(masterType => acquisitionPromises.push(this.ensureMasterStream(masterType)));

    try {
      await Promise.all(acquisitionPromises);

      for (const type of requiredStreams) {
        if (!this.isPseudoStreamAvailable(type)) {
          throw new Error(`Failed to acquire required stream component '${type}'. Permission may have been denied.`);
        }
      }

      // --- ADDED: Centralized Transcription & Mixer Logic ---
      // This block ensures services are only started for the streams the agent explicitly requested.
      if (requiredStreams.includes('allAudio')) {
        this.initializeAudioMixer();
      }
      if (requiredStreams.includes('microphone') && this.microphoneStream) {
        this.startTranscriptionForStream('microphone', this.microphoneStream);
      }
      if (requiredStreams.includes('screenAudio') && this.screenAudioStream) {
        this.startTranscriptionForStream('screenAudio', this.screenAudioStream);
      }
      // --- END ADDED ---

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

  public getTranscript(type: AudioStreamType): string {
    return this.transcriptionServices.get(type)?.getTranscript() ?? '';
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
        await this.ensureMasterStream('camera');
        Logger.info("StreamManager", "Camera switched successfully.");
      } catch (error) {
        Logger.error("StreamManager", "Failed to switch camera device.", error);
        throw error;
      }
    }
  }

  // --- Private Implementation ---

  private ensureMasterStream(type: MasterStreamType): Promise<void> {
    if (this.pendingMasterStreams.has(type)) {
      Logger.debug("StreamManager", `Joining pending request for '${type}' stream.`);
      return this.pendingMasterStreams.get(type)!;
    }
    
    const promise = this.acquireMasterStream(type).finally(() => {
      this.pendingMasterStreams.delete(type);
    });

    this.pendingMasterStreams.set(type, promise);
    return promise;
  }
  
  private async acquireMasterStream(type: MasterStreamType): Promise<void> {
    try {
      switch (type) {
        case 'display':
          if (this.masterDisplayStream) return;
          
          // Check if we're on tauri to use native plugin
          if (hasNativeScreenCapture()) {
            Logger.info("StreamManager", "Tauri detected - using channel-based native screen capture");

            // Set up PiP overlay drawer before starting stream
            tauriScreenCapture.setPipOverlayDrawer((ctx, width, height) => {
              if (this.pipOverlayStatus) {
                this.drawPipOverlay(ctx, width, height);
              }
            });

            // Start the channel-based capture stream
            // Frames are pushed from Rust as they're captured - no polling!
            const captureResult = await tauriScreenCapture.startCaptureStream();

            this.masterDisplayStream = captureResult.cleanStream;
            this.screenVideoStream = captureResult.cleanStream;
            this.screenVideoStreamWithPip = captureResult.pipStream;

            // Store cleanup function
            (this.masterDisplayStream as any)._mobileCleanup = captureResult.stop;

            Logger.info("StreamManager", "Channel-based screen capture initialized");

          } else {
            // Desktop: use getDisplayMedia
            Logger.info("StreamManager", "Desktop detected - using getDisplayMedia");
            const displayStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
            this.masterDisplayStream = displayStream;
            this.screenVideoStream = new MediaStream(displayStream.getVideoTracks());
            if (displayStream.getAudioTracks().length > 0) {
              const audioStream = new MediaStream(displayStream.getAudioTracks());
              this.screenAudioStream = audioStream;
            }
            displayStream.getVideoTracks()[0]?.addEventListener('ended', () => this.handleMasterStreamEnd('display'));
          }
          break;
        case 'camera':
          if (this.masterCameraStream) return;

          // Try to use preferred camera device, fallback to default
          const preferredDeviceId = this.getPreferredCameraDevice();
          let constraints: MediaStreamConstraints = { video: true };

          if (preferredDeviceId) {
            constraints = { video: { deviceId: { exact: preferredDeviceId } } };
            Logger.debug("StreamManager", `Requesting camera with deviceId: ${preferredDeviceId}`);
          }

          try {
            const cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
            this.masterCameraStream = cameraStream;
            this.cameraStream = cameraStream;
            cameraStream.getVideoTracks()[0]?.addEventListener('ended', () => this.handleMasterStreamEnd('camera'));
          } catch (error) {
            // If preferred device fails, try default camera
            if (preferredDeviceId) {
              Logger.warn("StreamManager", `Preferred camera device failed, falling back to default.`, error);
              const fallbackStream = await navigator.mediaDevices.getUserMedia({ video: true });
              this.masterCameraStream = fallbackStream;
              this.cameraStream = fallbackStream;
              fallbackStream.getVideoTracks()[0]?.addEventListener('ended', () => this.handleMasterStreamEnd('camera'));
            } else {
              throw error;
            }
          }
          break;
        case 'microphone':
          if (this.masterMicrophoneStream) return;
          const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
          this.masterMicrophoneStream = micStream;
          this.microphoneStream = micStream;
          // --- MODIFIED: Removed transcription start from here ---
          break;
      }
      Logger.info("StreamManager", `Master '${type}' stream acquired.`);
      this.notifyListeners();
    } catch (error) {
      Logger.error("StreamManager", `Acquisition of master '${type}' stream failed.`, error);
      throw error;
    }
  }
  
  private isPseudoStreamAvailable(type: PseudoStreamType): boolean {
    switch (type) {
        case 'camera': return !!this.cameraStream;
        case 'screenVideo': return !!this.screenVideoStream;
        case 'screenAudio': return !!this.screenAudioStream;
        case 'microphone': return !!this.microphoneStream;
        case 'allAudio': return !!this.screenAudioStream && !!this.microphoneStream;
    }
    return false;
  }

  private initializeAudioMixer(): void {
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
    this.startTranscriptionForStream('allAudio', this.allAudioStream);
    this.notifyListeners();
  }
  
  private checkForTeardown(): void {
    const isUsed = (type: PseudoStreamType) => (this.userSets.get(type)?.size || 0) > 0;
    // This logic is now robust because it correctly reflects the lifecycle based on user sets.
    if (!isUsed('screenVideo') && !isUsed('screenAudio') && !isUsed('allAudio')) this.teardownDisplayStream();
    if (!isUsed('microphone') && !isUsed('allAudio')) this.teardownMicrophoneStream();
    if (!isUsed('camera')) this.teardownCameraStream();
    if (!isUsed('allAudio')) this.teardownAudioMixer();
  }
  
  private handleMasterStreamEnd(type: MasterStreamType): void {
    Logger.warn("StreamManager", `Master ${type} stream ended unexpectedly.`);
    if (type === 'display') {
      this.userSets.get('screenVideo')?.clear();
      this.userSets.get('screenAudio')?.clear();
    } else {
      this.userSets.get('camera')?.clear();
    }
    this.userSets.get('allAudio')?.clear();
    this.checkForTeardown();
  }

  private teardownDisplayStream(): void {
    if (!this.masterDisplayStream) return;
    Logger.info("StreamManager", "Tearing down master display stream.");

    // Call mobile cleanup if it exists
    if ((this.masterDisplayStream as any)._mobileCleanup) {
      (this.masterDisplayStream as any)._mobileCleanup();
    }

    this.masterDisplayStream.getTracks().forEach(track => track.stop());

    // Also stop PiP stream tracks if they exist
    if (this.screenVideoStreamWithPip) {
      this.screenVideoStreamWithPip.getTracks().forEach(track => track.stop());
    }

    this.stopTranscriptionForStream('screenAudio');
    this.masterDisplayStream = null;
    this.screenVideoStream = null;
    this.screenVideoStreamWithPip = null;
    this.screenAudioStream = null;
    this.notifyListeners();
  }
  
  private teardownCameraStream(): void {
    if (!this.masterCameraStream) return;
    Logger.info("StreamManager", "Tearing down master camera stream.");
    this.masterCameraStream.getTracks().forEach(track => track.stop());
    this.masterCameraStream = null;
    this.cameraStream = null;
    this.notifyListeners();
  }

  private teardownMicrophoneStream(): void {
    if (!this.masterMicrophoneStream) return;
    Logger.info("StreamManager", "Tearing down master microphone stream.");
    this.masterMicrophoneStream.getTracks().forEach(track => track.stop());
    this.stopTranscriptionForStream('microphone'); // This correctly stops the microphone transcription
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

  private startTranscriptionForStream(type: AudioStreamType, stream: MediaStream): void {
    if (this.transcriptionServices.has(type)) return;
    Logger.info("StreamManager", `Starting transcription service for '${type}'.`);
    const newService = new WhisperTranscriptionService();
    newService.start(stream);
    this.transcriptionServices.set(type, newService);
  }

  private stopTranscriptionForStream(type: AudioStreamType): void {
    const service = this.transcriptionServices.get(type);
    if (service) {
      Logger.info("StreamManager", `Stopping transcription service for '${type}'.`);
      service.stop();
      this.transcriptionServices.delete(type);
    }
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
