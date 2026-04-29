// src/utils/main_loop.ts

import { getAgent, getAgentCode } from './agent_database';
import { UnauthorizedError } from './ModelManager';
import { ModelManager } from './ModelManager';
import { Logger } from './logging';
import { preProcess } from './pre-processor';
import { postProcess } from './post-processor';
import { StreamManager, PseudoStreamType } from './streamManager'; // Import the new manager
import { recordingManager } from './recordingManager';
import { IterationStore } from './IterationStore';
import { detectSignificantChange, clearAgentChangeData } from './change_detector';
import { checkPhoneWhitelist } from './pre-flight';

export type TokenProvider = () => Promise<string | undefined>;

const STREAM_REQUIREMENTS_MAP: Record<string, PseudoStreamType> = {
  '$SCREEN': 'screenVideo', '$SCREEN_64': 'screenVideo', '$SCREEN_OCR': 'screenVideo',
  '$CAMERA': 'camera', '$CAMERA_OCR': 'camera',
  '$SCREEN_AUDIO': 'screenAudio', '$MICROPHONE': 'microphone', '$ALL_AUDIO': 'allAudio'
};

export function getRequiredStreamsForPrompt(systemPrompt: string): PseudoStreamType[] {
  return Object.entries(STREAM_REQUIREMENTS_MAP)
    .filter(([placeholder]) => new RegExp(`\\${placeholder}(?![A-Z_])`).test(systemPrompt))
    .map(([_, streamType]) => streamType);
}

const activeLoops: Record<string, {
  intervalId: number | null,
  isRunning: boolean,
  isExecuting: boolean,
  intervalMs: number,
  getToken?: TokenProvider;
  lastResponse?: string;
  sleepUntil?: number | null;
}> = {};

// Event constants removed - Logger now dispatches all events based on logType
// This creates a unified system where Logger is the single source of truth

// Removed legacy server address functions - now using inferenceServer.ts

/**
 * Pause agent execution for a specified duration
 */
export function pauseAgentLoop(agentId: string, durationMs: number): void {
  if (activeLoops[agentId]?.isRunning) {
    activeLoops[agentId].sleepUntil = Date.now() + durationMs;

    // Dispatch sleep start event
    Logger.info(agentId, `Agent sleeping for ${Math.round(durationMs/1000)}s`, {
      logType: 'agent-sleep-start',
      content: {
        agentId,
        durationMs
      }
    });
  }
}

/**
 * Wake agent from sleep early
 */
export function wakeAgentLoop(agentId: string): void {
  const loop = activeLoops[agentId];
  if (loop?.isRunning && loop.sleepUntil) {
    loop.sleepUntil = null;

    // Dispatch sleep end event
    Logger.info(agentId, `Agent woken up early`, {
      logType: 'agent-sleep-end',
      content: { agentId, wokeEarly: true }
    });
  }
}

export async function startAgentLoop(agentId: string, getToken?: TokenProvider, skipWhitelistCheck = false): Promise<void> {
  if (activeLoops[agentId]?.isRunning) {
    Logger.warn(agentId, `Agent is already running`);
    return;
  }

  const isFirstAgent = getRunningAgentIds().length === 0;


  try {
    const agent = await getAgent(agentId);
    if (!agent) throw new Error(`Agent ${agentId} not found`);

    // Check phone whitelist before starting (unless explicitly skipped)
    if (!skipWhitelistCheck) {
      const agentCode = await getAgentCode(agentId) || '';
      const { phoneNumbers, hasTools, channel } = await checkPhoneWhitelist(agentCode, getToken);

      // If phone tools are used and there are issues, throw error for UI to handle
      if (hasTools && (phoneNumbers.length === 0 || phoneNumbers.some(p => !p.isWhitelisted))) {
        const error: any = new Error('Phone whitelist check failed');
        error.whitelistCheck = { phoneNumbers, hasTools, channel };
        throw error;
      }
    }

    const requiredStreams = getRequiredStreamsForPrompt(agent.system_prompt);
    
    if (requiredStreams.length > 0) {
      // A single, transactional call to the StreamManager.
      await StreamManager.requestStreamsForAgent(agentId, requiredStreams);
    }
    
    if (isFirstAgent) {
      recordingManager.initialize();
    }

    // Generate sessionId and start new session
    const sessionId = `session_${new Date().toISOString()}_${Math.random().toString(36).substring(2, 9)}`;
    IterationStore.startSession(agentId, sessionId);

    // Clear any previous change detection data for fresh start
    clearAgentChangeData(agentId);

    const intervalMs = agent.loop_interval_seconds * 1000;
    activeLoops[agentId] = {
        intervalId: null,
        isRunning: true,
        isExecuting: false,
        intervalMs,
        getToken
    };

    // Logger dispatches the window event automatically
    Logger.info(agentId, 'Agent started', {
      logType: 'agent-status-changed',
      content: { agentId, status: 'running' }
    });

    // Simple execution function
    const executeIteration = async () => {
      const loop = activeLoops[agentId];
      if (!loop?.isRunning || loop.isExecuting) return;

      // Check if agent is sleeping
      if (loop.sleepUntil && Date.now() < loop.sleepUntil) {
        const remainingMs = loop.sleepUntil - Date.now();
        Logger.debug(agentId, `Skipping iteration - agent sleeping for ${Math.round(remainingMs/1000)}s more`);
        return;
      }

      // Clear sleep state when resuming
      if (loop.sleepUntil) {
        Logger.debug(agentId, `Agent sleep period ended, resuming execution`);
        loop.sleepUntil = null;
      }

      loop.isExecuting = true;
      
      try {
        await executeAgentIteration(agentId);
        
      } catch (error) {
        // Any error stops the agent cleanly
        const errorMessage = error instanceof Error ? error.message : String(error);
        Logger.error(agentId, `Agent stopped due to error: ${errorMessage}`, {
          error,
          logType: 'agent-error',
          content: { error: errorMessage }
        });
        await stopAgentLoop(agentId);
        
      } finally {
        // Always cleanup execution state
        if (activeLoops[agentId]) {
          activeLoops[agentId].isExecuting = false;
        }
      }
    };

    // Fixed-rhythm metronome timer
    activeLoops[agentId].intervalId = window.setInterval(executeIteration, intervalMs);
    
    // Start first execution immediately
    executeIteration();
  } catch (error) {

    let displayError = error;
    // Check for the specific Safari screen sharing error
    if (error instanceof Error && error.message.includes('getDisplayMedia must be called from a user gesture handler')) {
        const safariErrorMessage = "Safari is bad at screen sharing and can't start automatically, click on the Observer App Icon on the top left to enter the Sensor Permissions Menu and ask for screen sharing manually. Also note: Safari won't capture System Audio, use chrome, firefox or edge for the best experience.";
        displayError = new Error(safariErrorMessage);
    }

    Logger.error(agentId, `Failed to start agent loop: ${displayError instanceof Error ? displayError.message : String(displayError)}`, error);
    // On startup failure, ensure we release any streams that might have been requested
    StreamManager.releaseStreamsForAgent(agentId);
    // Note: Crop configs are preserved even on startup failure

    // Logger dispatches the window event automatically
    Logger.info(agentId, 'Agent stopped (startup failure)', {
      logType: 'agent-status-changed',
      content: { agentId, status: 'stopped' }
    });
    // clean up and re-throw…
    throw displayError;
  }
}

export async function stopAgentLoop(agentId: string): Promise<void> {
  const loop = activeLoops[agentId];
  if (loop?.isRunning) {
    if (loop.intervalId !== null) window.clearInterval(loop.intervalId);

    // Clear any active sleep state
    wakeAgentLoop(agentId);

    // --- SUBSCRIBER CLEANUP ---
    // Destroy all transcription subscribers for this agent
    StreamManager.destroySubscribersForAgent(agentId);

    // --- STREAM MANAGEMENT ---
    // Tell the StreamManager this agent no longer needs these streams.
    // The manager will handle stopping the hardware if it's the last user.
    Logger.debug(agentId, "Releasing all potential streams for stopping agent.");

    StreamManager.releaseStreamsForAgent(agentId);

    // Clear change detection data
    clearAgentChangeData(agentId);

    // Note: Agent crop configurations are preserved across stop/start cycles


    // End the current session and save to IndexedDB
    await IterationStore.endSession(agentId);

    // -------------------------

    activeLoops[agentId] = { ...loop, isRunning: false, intervalId: null, sleepUntil: null };

    if (getRunningAgentIds().length === 0) {
      // This was the last running agent, so shut down the recorder.
      recordingManager.forceStop();
    }

    // Logger dispatches the window event automatically
    Logger.info(agentId, 'Agent stopped', {
      logType: 'agent-status-changed',
      content: { agentId, status: 'stopped' }
    });
  } else {
    Logger.warn(agentId, `Attempted to stop agent that wasn't running`);
  }
}

/**
 * Execute a single iteration of the agent's loop
 */
export async function executeAgentIteration(agentId: string): Promise<void> {
  const loopData = activeLoops[agentId];
  if (!loopData?.isRunning) {
    // This log is outside an iteration, so no ID is needed.
    Logger.debug(agentId, `Skipping execution for stopped agent`);
    return;
  }

  // --- ITERATION START ---
  const iterationId = `iter_${new Date().toISOString()}_${Math.random().toString(36).substring(2, 9)}`;
  const iterationStartTime = Date.now();

  try {
    const agent = await getAgent(agentId);
    const agentCode = await getAgentCode(agentId) || '';
    if (!agent) throw new Error(`Agent ${agentId} not found`);

    // Logger dispatches the window event automatically
    Logger.info(agentId, `Iteration started`, {
      logType: 'iteration-start',
      iterationId,
      content: {
        model: agent.model_name,
        interval: agent.loop_interval_seconds,
        intervalMs: loopData.intervalMs,
        iterationStartTime
      }
    });

    const preprocessResult = await preProcess(agentId, agent.system_prompt, iterationId);

    // Determine response source: cached or from model
    let response: string;
    let fromCache = false;

    // Check if we should use cached response (no significant change detected)
    const shouldUseCache = agent.only_on_significant_change &&
                          !(await detectSignificantChange(agentId, preprocessResult));

    if (shouldUseCache) {
      // Use cached response
      const cachedResponse = activeLoops[agentId]?.lastResponse;
      if (!cachedResponse) {
        throw new Error("No cached response available - this shouldn't happen after first iteration");
      }
      response = cachedResponse;
      fromCache = true;
    } else {
      // Call the model
      Logger.info(agentId, `Prompt`, { logType: 'model-prompt', iterationId, content: preprocessResult });

      let token: string | undefined;
      if (loopData.getToken) {
          try {
            Logger.debug(agentId, 'Requesting fresh API token...', { iterationId });
            token = await loopData.getToken();
          } catch (error) {
            Logger.warn(agentId, `Could not retrieve auth token: ${error}. Continuing without it.`, { iterationId });
          }
      }

      Logger.debug(agentId, `Sending prompt to inference server (model: ${agent.model_name})`, { iterationId });

      // Streaming callback that logs chunks - Logger dispatches events automatically
      let isFirstChunk = true;
      const onStreamChunk = (chunk: string) => {
        try {
          if (isFirstChunk) {
            Logger.debug(agentId, 'Stream started', { logType: 'stream-start', iterationId });
            isFirstChunk = false;
          }
          Logger.debug(agentId, 'Stream chunk', { logType: 'stream-chunk', iterationId, content: { chunk } });
        } catch (error) {
          Logger.debug(agentId, `Failed to log streaming event: ${error}`, { iterationId });
        }
      };

      const onReasoningChunk = (chunk: string) => {
        try {
          Logger.debug(agentId, 'Thinking chunk', { logType: 'thinking-chunk', iterationId, content: { chunk } });
        } catch (error) {
          Logger.debug(agentId, `Failed to log thinking event: ${error}`, { iterationId });
        }
      };

      response = await ModelManager.getInstance().sendPrompt(agent.model_name, preprocessResult, token, true, onStreamChunk, onReasoningChunk);

      // Cache new response for potential reuse on next iteration
      if (activeLoops[agentId]) {
        activeLoops[agentId].lastResponse = response;
      }
    }

    // Log based on source
    if (fromCache) {
      Logger.info(agentId, `Using cached response - no significant change detected`, {
        logType: 'iteration-response',
        iterationId,
        content: { usingCache: true }
      });
    } else {
      Logger.info(agentId, `Response`, { logType: 'model-response', iterationId, content: response });
    }

    try {
      await postProcess(agentId, response, agentCode, iterationId, loopData.getToken, preprocessResult);
      Logger.debug(agentId, `postProcess completed successfully`, { iterationId });

      // Clear subscriber transcripts for next loop window
      StreamManager.clearSubscriberTranscripts(agentId);

      // Success path
      if (isAgentLoopRunning(agentId)) {
        recordingManager.handleEndOfLoop();
      }

      // Dispatch to UI that we skipped the model call and completed iteration
      if (fromCache){
        Logger.info(agentId, `Iteration completed - no significant change detected`, {
          logType: 'iteration-skipped',
          iterationId,
          content: { success: true, cached: fromCache, reason: 'same_inputs' }
        });
      }
      // Dispatch we completed with model call
      else{
        Logger.info(agentId, `Iteration completed`, {
          logType: 'iteration-end',
          iterationId,
          content: { success: true, cached: fromCache }
        });
       }
      } catch (postProcessError) {
      Logger.error(agentId, `Error in postProcess: ${postProcessError}`, { iterationId, error: postProcessError });

      // Still clear subscriber transcripts on error to prevent buildup
      StreamManager.clearSubscriberTranscripts(agentId);

      // Error path - still clean up but log failure
      if (isAgentLoopRunning(agentId)) {
        recordingManager.handleEndOfLoop();
      }
      Logger.info(agentId, `Iteration completed`, {
        logType: 'iteration-end',
        iterationId,
        content: { success: false, cached: fromCache }
      });
    }

  } catch (error) {
    // Check for transient network errors - skip iteration but don't stop agent
    const isNetworkError = error instanceof TypeError &&
      (error.message.toLowerCase().includes('failed to fetch') ||
       error.message.toLowerCase().includes('networkerror') ||
       error.message.toLowerCase().includes('network request failed') ||
       error.message.toLowerCase().includes('load failed'));

    if (isNetworkError) {
      Logger.warn(agentId, `Network error - skipping iteration: ${error.message}`, {
        logType: 'iteration-skipped',
        iterationId,
        content: { reason: 'network_error', error: error.message }
      });
      return; // Skip iteration, don't stop agent
    }

    Logger.error(agentId, `Iteration failed`, {
      logType: 'iteration-end',
      iterationId,
      content: { success: false, error: error instanceof Error ? error.message : String(error) }
    });

    if (error instanceof UnauthorizedError) {
      // Logger dispatches the window event automatically
      Logger.warn(agentId, 'Agent stopped due to quota limit (401 Unauthorized)', {
        logType: 'quota-exceeded'
      });
      stopAgentLoop(agentId);
    } else {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      Logger.error(agentId, `Error in agent iteration: ${errorMessage}`, error);
    }

    // Re-throw error so executeIteration() can handle stopping the agent
    throw error;
  }
}

/**
 * Check if an agent's loop is currently running
 */
export function isAgentLoopRunning(agentId: string): boolean {
  return activeLoops[agentId]?.isRunning === true;
}

/**
 * Get all currently running agent IDs
 */
export function getRunningAgentIds(): string[] {
  return Object.entries(activeLoops)
    .filter(([_, loop]) => loop.isRunning)
    .map(([agentId, _]) => agentId);
}

/**
 * Execute a single test iteration - this is a simplified version for testing in the UI
 */
export async function executeTestIteration(
  agentId: string,
  systemPrompt: string,
  modelName: string,
  getToken?: TokenProvider
): Promise<string> {
  try {
    Logger.debug(agentId, `Starting test iteration with model ${modelName}`);

    // Pre-process the prompt (even for tests)
    const processedPrompt = await preProcess(agentId, systemPrompt);

    let token: string | undefined;
    // Check if the getToken function was provided before trying to call it.
    if (getToken) {
        Logger.debug(agentId, 'Requesting fresh API token for test run...');
        token = await getToken();
    }

    // Send the prompt to inference server and get response
    Logger.info(agentId, `Sending prompt to inference server (model: ${modelName})`);
    const response = await ModelManager.getInstance().sendPrompt(modelName, processedPrompt, token);
    // Since this is a one-off test, we don't use the StreamManager and just stop the capture.
    // This assumes the pre-processor for tests might call startScreenCapture directly.
    // If test logic changes, this might need updating.
    // stopScreenCapture(); // Note: This might need re-evaluation based on test flow.

    return response;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    Logger.error(agentId, `Error in test iteration: ${errorMessage}`, error);
    throw error;
  }
}
