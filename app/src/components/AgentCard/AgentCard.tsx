// components/AgentCard/AgentCard.tsx
import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Zap } from 'lucide-react';
import { CompleteAgent } from '@utils/agent_database';
import { isJupyterConnected } from '@utils/handlers/JupyterConfig';
import { listModels, getInferenceAddresses, BROWSER_LOCAL_SENTINEL } from '@utils/inferenceServer';
import { Logger, LogEntry } from '@utils/logging';
import { StreamManager, StreamState } from '@utils/streamManager';
import { isIOS } from '@utils/platform';

// Import the new modular components
import AgentCardHeader from './AgentCardHeader';
import AgentCardFooter from './AgentCardFooter';
import ActiveAgentView from './ActiveAgentView';
import StaticAgentView from './StaticAgentView';

// REMOVED: getCommunicationWarnings function is no longer needed.

const QuotaErrorView: React.FC<{ onUpgradeClick: () => void }> = ({ onUpgradeClick }) => (
  <div className="mt-4 p-4 bg-orange-50 border-l-4 border-orange-400 rounded-r-lg animate-fade-in">
    <div className="flex items-start">
      <Zap className="h-6 w-6 text-orange-500 mr-3 flex-shrink-0" />
      <div>
        <h4 className="font-bold text-orange-800">Daily Limit Reached</h4>
        <p className="text-sm text-orange-700 mt-1">You've used all free cloud credits. The agent has been paused.</p>
      </div>
    </div>
    <button onClick={onUpgradeClick} className="w-full mt-4 px-4 py-2 text-sm font-bold rounded-md text-white bg-purple-600 hover:bg-purple-700">
      Upgrade to Pro
    </button>
  </div>
);

type AgentLiveStatus = 'STARTING' | 'CAPTURING' | 'THINKING' | 'RESPONDING' | 'WAITING' | 'SKIPPED' | 'SLEEPING' | 'IDLE';

interface AgentCardProps {
  agent: CompleteAgent;
  code?: string;
  isRunning: boolean;
  isStarting: boolean;
  isMemoryFlashing: boolean;
  onEdit: (agentId: string) => void;
  onDelete: (agentId: string) => void;
  onToggle: (agentId: string, isRunning: boolean) => Promise<void>;
  onMemory: (agentId: string) => void;
  onActivity: (agentId: string) => void;
  onShowJupyterModal: () => void;
  getToken: () => Promise<string | undefined>;
  isAuthenticated: boolean;
  hasQuotaError: boolean;
  onUpgradeClick: () => void;
  onSave: (agent: CompleteAgent, code: string) => Promise<void>;
  isProUser?: boolean;
  onAIEdit?: (agentId: string) => void;
  hostingContext?: 'official-web' | 'self-hosted' | 'tauri';
}


const AgentCard: React.FC<AgentCardProps> = ({
  agent, code, isRunning, isStarting, isMemoryFlashing, onEdit, onDelete, onToggle,
  onMemory, onActivity, onShowJupyterModal, getToken, hasQuotaError, onUpgradeClick, onSave, isProUser = false, onAIEdit, hostingContext
}) => {
  const [isPythonAgent, setIsPythonAgent] = useState(false);
  const [startWarning, setStartWarning] = useState<string | null>(null);
  const [isCheckingModel, setIsCheckingModel] = useState(false);
  const [liveStatus, setLiveStatus] = useState<AgentLiveStatus>('IDLE');
  const [lastResponse, setLastResponse] = useState<string>('...');
  const [responseKey, setResponseKey] = useState(0);
  const [loopProgress, setLoopProgress] = useState(0);
  const [lastProgressUpdate, setLastProgressUpdate] = useState(0);
  const [loopDurationMs, setLoopDurationMs] = useState(0);
  const [sleepProgress, setSleepProgress] = useState(0);
  const [sleepDurationMs, setSleepDurationMs] = useState(0);
  const [isSleeping, setIsSleeping] = useState(false);
  const [currentModel, setCurrentModel] = useState(agent.model_name);
  const [skipReason, setSkipReason] = useState<'same_inputs' | 'network_error' | null>(null);
  const initialModelRef = useRef(agent.model_name);
  const loopStartTimeRef = useRef(0);
  const loopDurationRef = useRef(0);
  const [isOverrun, setIsOverrun] = useState(false);

  const showStartingState = useMemo(() => isStarting || isCheckingModel, [isStarting, isCheckingModel]);
  const isLive = useMemo(() => (isRunning || showStartingState) && !hasQuotaError, [isRunning, showStartingState, hasQuotaError]);

  const [streams, setStreams] = useState<StreamState>({
    cameraStream: null, screenVideoStream: null, screenVideoStreamWithPip: null, screenAudioStream: null, microphoneStream: null, allAudioStream: null
  });

  // REMOVED: communicationWarnings memo is no longer needed.

  useEffect(() => { setIsPythonAgent(!!code && code.trim().startsWith('#python')); }, [code]);
  useEffect(() => { setCurrentModel(agent.model_name); initialModelRef.current = agent.model_name; }, [agent.model_name]);

  useEffect(() => {
    if (currentModel !== initialModelRef.current && onSave && code !== undefined) {
      onSave({ ...agent, model_name: currentModel }, code);
      initialModelRef.current = currentModel;
    }
  }, [currentModel, agent, code, onSave]);

  const handleToggleSignificantChange = async (enabled: boolean) => {
    if (onSave && code !== undefined) {
      await onSave({ ...agent, only_on_significant_change: enabled }, code);
    }
  };

  const handleSystemPromptChange = async (newPrompt: string) => {
    if (onSave && code !== undefined) {
      await onSave({ ...agent, system_prompt: newPrompt }, code);
    }
  };

  const handleCodeChange = async (newCode: string) => {
    if (onSave) {
      await onSave(agent, newCode);
    }
  };

  useEffect(() => {
    StreamManager.addListener(setStreams);
    return () => StreamManager.removeListener(setStreams);
  }, []);

  useEffect(() => {
    if (hasQuotaError) {
      setLiveStatus('IDLE');
      setLastResponse('Agent paused due to daily credit limit.');
      return;
    }
    if (isSleeping) {
      setLiveStatus('SLEEPING');
      return;
    }
    if (showStartingState && !isRunning) {
        setLiveStatus('STARTING');
        setLastResponse('Agent is preparing to start...');
        return;
    }
    if (isRunning) {
      if (liveStatus === 'STARTING' || liveStatus === 'IDLE') {
        setLiveStatus('CAPTURING');
        // Reset progress when agent starts
        setLoopProgress(0);
        setLastProgressUpdate(0);
      }
      const handleNewLog = (log: LogEntry) => {
        if (log.source !== agent.id) return;
        if (log.details?.logType === 'model-prompt') setLiveStatus('THINKING');
        else if (log.details?.logType === 'iteration-skipped') {
          setLiveStatus('SKIPPED');
          const reason = log.details?.content?.reason as 'same_inputs' | 'network_error' | undefined;
          setSkipReason(reason || 'same_inputs');
          if (reason === 'network_error') {
            setLastResponse(`Network error - iteration skipped: ${log.details?.content?.error || 'Connection failed'}`);
          } else {
            setLastResponse('No significant change detected - reused model call to save resources.');
          }
          setResponseKey(key => key + 1);
        }
        else if (log.details?.logType === 'model-response') {
          setLiveStatus('WAITING');
          setLastResponse(log.details.content as string);
          setResponseKey(key => key + 1);
        }
      };
      Logger.addListener(handleNewLog);
      return () => Logger.removeListener(handleNewLog);
    } else {
      setLiveStatus('IDLE');
    }
  }, [isRunning, showStartingState, agent.id, liveStatus, hasQuotaError, isSleeping]);

  // Set up event listener immediately on mount - separate from state changes
  useEffect(() => {
    let progressTimer: NodeJS.Timeout | null = null;

    const handleIterationStart = (event: CustomEvent) => {
      if (event.detail.agentId !== agent.id) return;

      // Clear any existing timer
      if (progressTimer) clearInterval(progressTimer);

      console.log('Event fired!! for agent: ', agent.id);

      // Store in refs so other effects can access/update
      loopStartTimeRef.current = event.detail.iterationStartTime;
      loopDurationRef.current = event.detail.intervalMs;
      setLoopProgress(0);
      setLoopDurationMs(event.detail.intervalMs);
      setIsOverrun(false); // Reset overrun flag when new iteration starts

      // Simple progress timer using refs (can be reset by updating refs)
      progressTimer = setInterval(() => {
        const elapsed = Date.now() - loopStartTimeRef.current;
        const progress = Math.min(100, (elapsed / loopDurationRef.current) * 100);
        setLoopProgress(progress);
        setLastProgressUpdate(Date.now());
      }, 50);
    };

    // Event dispatched by Logger based on logType 'iteration-start'
    window.addEventListener('agentIterationStart' as any, handleIterationStart);
    return () => {
      if (progressTimer) clearInterval(progressTimer);
      window.removeEventListener('agentIterationStart' as any, handleIterationStart);
    };
  }, []); // Empty deps - runs once on mount

  // Detect when timer expires while model is still working -> set overrun flag
  // This happens when isExecuting=true in main_loop (CAPTURING/THINKING/RESPONDING states)
  useEffect(() => {
    if (loopProgress >= 100 && loopDurationRef.current > 0) {
      const modelStillWorking = liveStatus === 'CAPTURING' || liveStatus === 'THINKING' || liveStatus === 'RESPONDING';

      if (modelStillWorking || isOverrun) {
        // Timer expired but model hasn't finished - mark as overrun and reset timer
        setIsOverrun(true);
        loopStartTimeRef.current = Date.now();
        setLoopProgress(0);
      }
    }
  }, [loopProgress, liveStatus, isOverrun]);

  // Listen for stream start to transition from THINKING to RESPONDING
  useEffect(() => {
    const handleStreamStart = (event: CustomEvent) => {
      if (event.detail.agentId !== agent.id) return;
      // Transition from THINKING to RESPONDING when first token arrives
      setLiveStatus('RESPONDING');
    };

    window.addEventListener('agentStreamStart' as any, handleStreamStart);
    return () => {
      window.removeEventListener('agentStreamStart' as any, handleStreamStart);
    };
  }, [agent.id]);

  // Listen for sleep state changes
  useEffect(() => {
    let sleepProgressTimer: NodeJS.Timeout | null = null;

    const clearSleepState = () => {
      if (sleepProgressTimer) {
        clearInterval(sleepProgressTimer);
        sleepProgressTimer = null;
      }
      setSleepProgress(0);
      setIsSleeping(false);
      setSleepDurationMs(0);
      setLiveStatus('WAITING');
    };

    const handleSleepStart = (event: CustomEvent) => {
      if (event.detail.agentId !== agent.id) return;

      // Clear any existing timer before starting a new one
      // This matches backend behavior where pauseAgentLoop() overwrites sleepUntil
      clearSleepState();

      const durationMs = event.detail.durationMs;
      const sleepEnd = Date.now() + durationMs;

      setIsSleeping(true);
      setSleepProgress(100); // Start at 100%
      setSleepDurationMs(durationMs);

      // Update sleep progress countdown
      sleepProgressTimer = setInterval(() => {
        const now = Date.now();
        const remaining = sleepEnd - now;

        if (remaining <= 0) {
          // Sleep finished - clear state and transition back to WAITING
          clearSleepState();
        } else {
          // Calculate remaining percentage (drains from 100% to 0%)
          const progress = Math.max(0, (remaining / durationMs) * 100);
          setSleepProgress(progress);
        }
      }, 50); // Update every 50ms for smooth animation
    };

    const handleSleepEnd = (event: CustomEvent) => {
      if (event.detail.agentId !== agent.id) return;
      clearSleepState();
    };

    window.addEventListener('agentSleepStart' as any, handleSleepStart);
    window.addEventListener('agentSleepEnd' as any, handleSleepEnd);

    return () => {
      if (sleepProgressTimer) clearInterval(sleepProgressTimer);
      window.removeEventListener('agentSleepStart' as any, handleSleepStart);
      window.removeEventListener('agentSleepEnd' as any, handleSleepEnd);
    };
  }, [agent.id]);

  const handleToggle = async () => {
    if (isRunning) {
      setStartWarning(null);
      onToggle(agent.id, true);
      return;
    }

    setStartWarning(null);
    setIsCheckingModel(true);

    if (!currentModel) {
      setStartWarning(`Please select a model for this agent.`);
      setIsCheckingModel(false);
      return;
    }

    try {
      const addresses = getInferenceAddresses();
      if (addresses.length === 0) throw new Error("No inference servers configured.");
      const modelsResponse = listModels();
      const foundModel = modelsResponse.models.find(m => m.name === currentModel);
      if (modelsResponse.error || !foundModel) {
        setStartWarning(`Model "${currentModel}" is not available. Check server or edit agent.`);
        setIsCheckingModel(false);
        return;
      }
      // Check if local model is loaded
      if (foundModel.server === BROWSER_LOCAL_SENTINEL && foundModel.status !== 'loaded') {
        setStartWarning(`Local model "${currentModel}" is not loaded. Load it first.`);
        setIsCheckingModel(false);
        return;
      }
    } catch (e) {
      setStartWarning(`Error verifying model: ${e instanceof Error ? e.message : String(e)}.`);
      setIsCheckingModel(false);
      return;
    }

    if (isPythonAgent && !isJupyterConnected()) {
      onShowJupyterModal();
      setIsCheckingModel(false);
      return;
    }

    setIsCheckingModel(false);
    onToggle(agent.id, false);
  };

  return (
    <div
      className="relative bg-white rounded-xl shadow-sm border border-gray-200 transition-all duration-300 flex flex-col"
      data-tutorial-agent-card={agent.id}
    >
      {isRunning && (
        <div className="absolute top-0 left-0 right-0 h-1 z-10">
          {isSleeping ? (
            // Blue sleep progress bar (drains from 100% to 0%)
            <div
              className="h-full bg-blue-500"
              style={{
                width: `${sleepProgress}%`,
                transition: 'width 0.1s linear'
              }}
            />
          ) : (Date.now() - lastProgressUpdate < 5000) ? (
            // Loop progress bar (green normally, orange when overrun)
            <div
              className={`h-full ${isOverrun ? 'bg-orange-400' : 'bg-green-500'}`}
              style={{
                width: `${loopProgress}%`,
                transition: 'width 0.1s linear'
              }}
            />
          ) : null}
        </div>
      )}

      <div className="p-5 flex-grow overflow-visible">
        <AgentCardHeader
          agentId={agent.id}
          agentName={agent.name}
          agentDescription={agent.description}
          isRunning={isRunning}
          isStarting={showStartingState}
          hasQuotaError={hasQuotaError}
          isLive={isLive}
          onToggle={handleToggle}
        />

        <div className="overflow-visible">
          {hasQuotaError ? (
            isIOS() ? (
              // On iOS, show quota error without upgrade button
              <div className="mt-4 p-4 bg-orange-50 border-l-4 border-orange-400 rounded-r-lg animate-fade-in">
                <div className="flex items-start">
                  <Zap className="h-6 w-6 text-orange-500 mr-3 flex-shrink-0" />
                  <div>
                    <h4 className="font-bold text-orange-800">Daily Limit Reached</h4>
                    <p className="text-sm text-orange-700 mt-1">You've used all free cloud credits. The agent has been paused.</p>
                  </div>
                </div>
              </div>
            ) : (
              <QuotaErrorView onUpgradeClick={onUpgradeClick} />
            )
          ) : isLive ? (
            <ActiveAgentView
              streams={streams}
              liveStatus={liveStatus}
              lastResponse={lastResponse}
              responseKey={responseKey}
              agentId={agent.id}
              agent={agent}
              code={code}
              loopProgress={loopProgress}
              sleepProgress={sleepProgress}
              loopDurationMs={loopDurationMs}
              sleepDurationMs={sleepDurationMs}
              skipReason={skipReason}
              isOverrun={isOverrun}
            />
          ) : (
            // FIX: Pass the 'code' prop down to StaticAgentView
            <StaticAgentView
              agent={agent}
              code={code}
              currentModel={currentModel}
              onModelChange={setCurrentModel}
              onToggleSignificantChange={handleToggleSignificantChange}
              onSystemPromptChange={handleSystemPromptChange}
              onCodeChange={handleCodeChange}
              startWarning={startWarning}
              isProUser={isProUser}
              hostingContext={hostingContext}
              getToken={getToken}
              // REMOVED: communicationWarnings prop is gone
            />
          )}
        </div>
      </div>

      <AgentCardFooter
        agentId={agent.id}
        isPythonAgent={isPythonAgent}
        isJupyterConnected={isJupyterConnected()}
        isMemoryFlashing={isMemoryFlashing}
        onEdit={onEdit}
        onDelete={onDelete}
        onMemory={onMemory}
        onActivity={onActivity}
        onShowJupyterModal={onShowJupyterModal}
        onAIEdit={onAIEdit}
      />
    </div>
  );
};

export default AgentCard;
