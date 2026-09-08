import { TranscriptionMode } from './types';
import { UnifiedTranscriptionService } from './UnifiedTranscriptionService';
import { WhisperModelManager } from './WhisperModelManager';
import { SensorSettings } from '../settings';
import { Logger } from '../logging';
import { AudioStreamType } from '../streamManager';

// Token provider type
export type TokenProvider = () => Promise<string | undefined>;

export class TranscriptionRouter {
  private static instance: TranscriptionRouter | null = null;
  private static tokenProvider: TokenProvider | null = null;
  private mode: TranscriptionMode;
  private modeChangeListeners: Array<(mode: TranscriptionMode) => void> = [];

  // Singleton service management - one service per stream type.
  // Ownership lives in StreamManager: its userSets track which agents need a stream, and
  // releaseService is only reached once those are empty (or the master stream died). The
  // router therefore does no counting of its own - acquire is idempotent, release is final.
  private activeServices: Map<AudioStreamType, UnifiedTranscriptionService> = new Map();
  private pendingStarts: Set<AudioStreamType> = new Set();

  private constructor() {
    this.mode = SensorSettings.getTranscriptionMode();
    Logger.info('TranscriptionRouter', `Initialized with mode: ${this.mode}`);
  }

  /**
   * Set the token provider function. Call this from App.tsx during initialization.
   */
  public static setTokenProvider(provider: TokenProvider): void {
    TranscriptionRouter.tokenProvider = provider;
    Logger.info('TranscriptionRouter', 'Token provider set');
  }

  /**
   * Get a token using the registered provider.
   */
  public static async getToken(): Promise<string | undefined> {
    if (!TranscriptionRouter.tokenProvider) {
      Logger.warn('TranscriptionRouter', 'No token provider set');
      return undefined;
    }
    return TranscriptionRouter.tokenProvider();
  }

  public static getInstance(): TranscriptionRouter {
    if (!TranscriptionRouter.instance) {
      TranscriptionRouter.instance = new TranscriptionRouter();
    }
    return TranscriptionRouter.instance;
  }

  public getMode(): TranscriptionMode {
    return this.mode;
  }

  public setMode(newMode: TranscriptionMode): void {
    if (this.mode === newMode) return;

    Logger.info('TranscriptionRouter', `Switching mode from ${this.mode} to ${newMode}`);
    this.mode = newMode;
    SensorSettings.setTranscriptionMode(newMode);

    // Notify listeners about mode change
    this.modeChangeListeners.forEach(listener => listener(newMode));
  }

  public onModeChange(listener: (mode: TranscriptionMode) => void): () => void {
    this.modeChangeListeners.push(listener);
    return () => {
      this.modeChangeListeners = this.modeChangeListeners.filter(l => l !== listener);
    };
  }

  /**
   * Acquire a transcription service for a stream type.
   * Returns the existing service if one is active, otherwise creates a new one.
   * Idempotent: repeat acquisitions of a live stream type are a no-op, so an agent that
   * requests the same blueprint twice cannot strand a service that release can't reach.
   */
  public async acquireService(streamType: AudioStreamType): Promise<UnifiedTranscriptionService> {
    // Already running? Hand back the same service
    const existing = this.activeServices.get(streamType);
    if (existing) {
      Logger.info('TranscriptionRouter', `Reusing existing service for '${streamType}'`);
      return existing;
    }

    // Another call is already starting this service? Wait for it
    if (this.pendingStarts.has(streamType)) {
      Logger.debug('TranscriptionRouter', `Waiting for pending service start for '${streamType}'`);
      await this.waitForPendingStart(streamType);
      return this.acquireService(streamType); // Retry - should now exist
    }

    // We're the first - start the service
    this.pendingStarts.add(streamType);
    try {
      await this.ensureReady();

      Logger.info('TranscriptionRouter', `Creating new service for '${streamType}' (mode: ${this.mode})`);
      const service = new UnifiedTranscriptionService(this.mode);
      await service.start(streamType);

      this.activeServices.set(streamType, service);

      return service;
    } finally {
      this.pendingStarts.delete(streamType);
    }
  }

  /**
   * Release the transcription service for a stream type, stopping it.
   * Safe to call when nothing is active (the browser hands us a screen audio track whether
   * or not an agent asked for one, so teardown can run for a type that never had a service).
   */
  public releaseService(streamType: AudioStreamType): void {
    const service = this.activeServices.get(streamType);
    if (!service) {
      Logger.debug('TranscriptionRouter', `Release for '${streamType}' ignored - no active service`);
      return;
    }

    Logger.info('TranscriptionRouter', `Stopping service for '${streamType}'`);
    service.stop();
    this.activeServices.delete(streamType);
  }

  /**
   * Get the active service for a stream type without acquiring it.
   * Returns undefined if no service is active.
   */
  public getActiveService(streamType: AudioStreamType): UnifiedTranscriptionService | undefined {
    return this.activeServices.get(streamType);
  }

  /**
   * Check if a service is active for a stream type.
   */
  public hasActiveService(streamType: AudioStreamType): boolean {
    return this.activeServices.has(streamType);
  }

  private waitForPendingStart(streamType: AudioStreamType): Promise<void> {
    return new Promise((resolve) => {
      const checkInterval = setInterval(() => {
        if (!this.pendingStarts.has(streamType)) {
          clearInterval(checkInterval);
          resolve();
        }
      }, 50);
    });
  }

  /**
   * Ensures the transcription system is ready.
   * For cloud mode, always ready. For local mode, loads the model if needed.
   * For self-hosted mode, checks URL is configured.
   */
  public async ensureReady(): Promise<void> {
    switch (this.mode) {
      case 'cloud':
        // Cloud mode is always ready
        break;
      case 'self-hosted':
        const url = SensorSettings.getSelfHostedWhisperUrl();
        if (!url || url.trim().length === 0) {
          throw new Error('Self-hosted Whisper URL is not configured');
        }
        break;
      case 'local':
        const modelManager = WhisperModelManager.getInstance();
        if (!modelManager.isReady()) {
          Logger.info('TranscriptionRouter', 'Local model not loaded, loading automatically...');
          await modelManager.loadModel();
        }
        break;
    }
  }

  public isReady(): boolean {
    switch (this.mode) {
      case 'cloud':
        return true;
      case 'self-hosted':
        const url = SensorSettings.getSelfHostedWhisperUrl();
        return !!url && url.trim().length > 0;
      case 'local':
        return WhisperModelManager.getInstance().isReady();
    }
  }
}
