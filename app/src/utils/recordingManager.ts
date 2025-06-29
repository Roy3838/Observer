import { StreamManager } from './streamManager';
// FIX: Imported 'ClipMarker' type (PascalCase) instead of 'clipMarker' value.
import { saveRecordingToDb, ClipMarker } from './recordingsDB';
import { Logger } from './logging';

type RecordableStreamType = 'screen' | 'camera';
type RecordingState = 'IDLE' | 'BUFFERING' | 'RECORDING';

// FIX: Renamed class from 'Manager' to 'RecordingManager' as requested.
class RecordingManager {
  private state: RecordingState = 'IDLE';
  private recorders = new Map<RecordableStreamType, MediaRecorder>();
  private chunks = new Map<RecordableStreamType, Blob[]>();
  // FIX: This now correctly references the imported 'ClipMarker' type.
  private pendingMarkers: ClipMarker[] = [];


  // --- Public API for Agent Lifecycle ---

  public initialize(): void {
    if (this.state === 'IDLE') {
      Logger.info("RecordingManager", "Initializing and starting the first buffer.");
      this.startNewBuffer();
    }
  }

  public handleEndOfLoop(): void {
    if (this.state === 'BUFFERING') {
      Logger.debug("RecordingManager", "End of loop: In BUFFERING state. Cycling buffer.");
      this.discardCurrentBuffer();
      this.startNewBuffer();
    } else if (this.state === 'RECORDING') {
      Logger.debug("RecordingManager", "End of loop: In RECORDING state. Continuing recording.");
    }
  }

  public async forceStop(): Promise<void> {
    Logger.info("RecordingManager", `forceStop called. Current state: ${this.state}`);
    if (this.state === 'BUFFERING') {
      this.discardAndShutdown();
    } else {
      Logger.info("RecordingManager", `Saving from forceStop`);
      await this.saveAndFinishClip();
    }
    this.state = 'IDLE';
  }

  public addMarker(label: string): void {
    if (this.state === 'IDLE') {
      Logger.warn("RecordingManager", `markClip called while IDLE. Marker will be stored and attached to the next recording session.`);
    }
    // FIX: This now correctly references the imported 'ClipMarker' type.
    const marker: ClipMarker = {
      label,
      timestamp: Date.now(),
    };
    this.pendingMarkers.push(marker);
    Logger.info("RecordingManager", `Marker added: "${label}" at ${new Date(marker.timestamp).toLocaleTimeString()}`);
  }

  // --- Public API for Agent Tools ---

  public startClip(): void {
    if (this.state === 'BUFFERING') {
      this.state = 'RECORDING';
      Logger.info("RecordingManager", "startClip called. State changed to RECORDING. The current buffer will be saved.");
    } else {
      Logger.warn("RecordingManager", `startClip called in unexpected state: ${this.state}. No action taken.`);
    }
  }

  public async stopClip(): Promise<void> {
    if (this.state === 'RECORDING') {
      Logger.info("RecordingManager", "stopClip called. Saving clip and returning to BUFFERING state.");
      await this.saveAndFinishClip();
      this.startNewBuffer();
    } else {
      Logger.warn("RecordingManager", `stopClip called in unexpected state: ${this.state}. No action taken.`);
    }
  }

  // --- Private Implementation ---

  private async saveAndFinishClip(): Promise<void> {
    if (this.recorders.size === 0) {
      Logger.warn("RecordingManager", "saveAndFinishClip called but no active recorders.");
      return;
    }

    Logger.info("RecordingManager", `saveAndFinishClip called. Preparing to save ${this.recorders.size} recorders.`);
  
    const saveJobs: Promise<void>[] = [];
  
    // The original ondataavailable listener is already populating the chunks.
    // We just need to stop the recorders and wait for them to finish.
    this.recorders.forEach((recorder, type) => {
      const chunks = this.chunks.get(type) ?? [];
      Logger.debug("RecordingManager",
        `Preparing to stop '${type}' recorder. Current chunks: ${chunks.length}, state: ${recorder.state}`);
  
      const job = new Promise<void>((resolve) => {
        // This is the function that will perform the save operation.
        const finalizeAndSave = async () => {
          // It's possible the recorder was already stopped and had no data.
          if (recorder.state === 'inactive' && chunks.length === 0) {
            Logger.warn("RecordingManager", `'${type}' recorder was already inactive with no data. Nothing to save.`);
            resolve();
            return;
          }

          // The 'stop' event fires *after* the final 'dataavailable' event.
          // By this point, our original listener has already pushed the last chunk.
          const blob = new Blob(chunks, { type: recorder.mimeType });
          const filename = `${type}-clip-${Date.now()}`;
          Logger.info("RecordingManager",
            `'${type}' stopped. Saving ${chunks.length} chunks (${blob.size} bytes) as '${filename}'.`);
  
          try {
            await saveRecordingToDb(blob, this.pendingMarkers); 
            Logger.info("RecordingManager", `Saved clip for '${type}' with ${this.pendingMarkers.length} markers successfully.`);
          } catch (err) {
            Logger.error("RecordingManager", `Failed to save clip for '${type}'.`, err);
          }
          resolve();
        };

        // If the recorder is already inactive, just save what we have.
        // This fixes the bug where nothing happens.
        if (recorder.state === 'inactive') {
          finalizeAndSave();
        } else {
          // If it's still recording, set up a listener for the 'stop' event,
          // then trigger it.
          recorder.addEventListener('stop', finalizeAndSave, { once: true });
          recorder.stop();
        }
      });
  
      saveJobs.push(job);
    });
  
    // Wait for all save jobs to complete
    await Promise.all(saveJobs);

    if (this.pendingMarkers.length > 0) {
        Logger.info("RecordingManager", `All clips for this session saved. Clearing ${this.pendingMarkers.length} markers.`);
        this.pendingMarkers = [];
    }
  
    // Clean up
    this.recorders.clear();
    this.chunks.clear();
    Logger.debug("RecordingManager", "All recorders saved & cleared.");
  }
  
  private startNewBuffer(): void {
    if (this.recorders.size > 0) {
      this.discardAndShutdown();
    }


    const { 
      screenVideoStream, 
      cameraStream, 
      screenAudioStream, 
      microphoneStream 
    } = StreamManager.getCurrentState();

    const streamsToRecord = [
      { type: 'screen' as RecordableStreamType, video: screenVideoStream, audio: screenAudioStream },
      { type: 'camera' as RecordableStreamType, video: cameraStream, audio: microphoneStream }
    ];
    
    let bufferStarted = false;

    for (const { type, video, audio } of streamsToRecord) {
      // If there's no video stream, there's nothing to record for this type.
      if (!video) continue;
  
      // 3. Start with the video tracks.
      const tracks = [...video.getVideoTracks()];

      // 4. If a corresponding audio stream exists, add its tracks.
      if (audio) {
        tracks.push(...audio.getAudioTracks());
      }

      // 5. Create the combined stream and the recorder. The rest is the same.
      const combinedStream = new MediaStream(tracks);
      
      const mediaRecorder = new MediaRecorder(combinedStream, { mimeType: 'video/mp4' });
      const chunksForType: Blob[] = [];
      this.chunks.set(type, chunksForType);
  
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksForType.push(event.data);
      };
      
      this.recorders.set(type, mediaRecorder);
      mediaRecorder.start(1000);
      bufferStarted = true;
    }
  
    if (bufferStarted) {
      this.state = 'BUFFERING';
      Logger.debug("RecordingManager", "New buffer started successfully. State is now BUFFERING.");
    } else {
      this.state = 'IDLE';
      Logger.warn("RecordingManager", "startNewBuffer called, but no active streams found to record.");
    }

  }


  private discardCurrentBuffer(): void {
    this.chunks.clear();
    this.recorders.forEach(recorder => recorder.stop());
  }

  private discardAndShutdown(): void {
    this.discardCurrentBuffer();
    this.state = 'IDLE';
    Logger.info("RecordingManager", "Buffer discarded and manager is now IDLE.");
  }

  public getState(): RecordingState {
    return this.state;
  }
}

// FIX: Instantiating the renamed 'RecordingManager' class.
export const recordingManager = new RecordingManager();
