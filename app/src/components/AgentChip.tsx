import React, { useState, useEffect, useRef } from 'react';
import { CompleteAgent } from '@utils/agent_database';
import { Logger } from '@utils/logging';
import PieTimer from './AgentCard/PieTimer';
import type { AgentLiveStatus } from './AgentCard/ActiveAgentView';

interface AgentChipProps {
  agent: CompleteAgent;
  isRunning: boolean;
  isStarting: boolean;
  isMinimized: boolean;
  onRestore: () => void;
  onToggle: (agentId: string, isRunning: boolean) => Promise<void>; // kept for API compat, unused
}

const AgentChip: React.FC<AgentChipProps> = ({ agent, isRunning, isStarting, isMinimized, onRestore, onToggle }) => {
  const [liveStatus, setLiveStatus] = useState<AgentLiveStatus>('IDLE');
  const [loopProgress, setLoopProgress] = useState(0);
  const [loopDurationMs, setLoopDurationMs] = useState(0);
  const [sleepProgress, setSleepProgress] = useState(0);
  const [sleepDurationMs, setSleepDurationMs] = useState(0);
  const [isOverrun, setIsOverrun] = useState(false);
  const [isSleeping, setIsSleeping] = useState(false);
  const [isPopping, setIsPopping] = useState(false);
  const prevMinimizedRef = useRef(false);

  const loopStartTimeRef = useRef(0);
  const loopDurationRef = useRef(0);

  const agentId = agent.id;

  // Trigger pop-in animation when chip becomes visible
  useEffect(() => {
    if (isMinimized && !prevMinimizedRef.current) {
      setIsPopping(true);
      const t = setTimeout(() => setIsPopping(false), 300);
      prevMinimizedRef.current = true;
      return () => clearTimeout(t);
    }
    if (!isMinimized) prevMinimizedRef.current = false;
  }, [isMinimized]);
  // Sync running state → liveStatus
  useEffect(() => {
    if (isStarting) {
      setLiveStatus('STARTING');
    } else if (!isRunning) {
      setLiveStatus('IDLE');
      setLoopProgress(0);
      setIsSleeping(false);
      setSleepProgress(0);
    } else if (liveStatus === 'IDLE' || liveStatus === 'STARTING') {
      setLiveStatus('CAPTURING');
    }
  }, [isRunning, isStarting]);

  // Logger listener for model events
  useEffect(() => {
    if (!isRunning) return;
    const handleLog = (log: any) => {
      if (log.source !== agentId) return;
      if (log.details?.logType === 'model-prompt') setLiveStatus('THINKING');
      else if (log.details?.logType === 'model-response') setLiveStatus('WAITING');
      else if (log.details?.logType === 'iteration-skipped') setLiveStatus('SKIPPED');
    };
    Logger.addListener(handleLog);
    return () => Logger.removeListener(handleLog);
  }, [isRunning, agentId]);

  // Loop progress timer
  useEffect(() => {
    let progressTimer: ReturnType<typeof setInterval> | null = null;

    const handleIterationStart = (event: CustomEvent) => {
      if (event.detail.agentId !== agentId) return;
      if (progressTimer) clearInterval(progressTimer);
      loopStartTimeRef.current = event.detail.iterationStartTime;
      loopDurationRef.current = event.detail.intervalMs;
      setLoopProgress(0);
      setLoopDurationMs(event.detail.intervalMs);
      setIsOverrun(false);
      progressTimer = setInterval(() => {
        const elapsed = Date.now() - loopStartTimeRef.current;
        setLoopProgress(Math.min(100, (elapsed / loopDurationRef.current) * 100));
      }, 100);
    };

    window.addEventListener('agentIterationStart', handleIterationStart as EventListener);
    return () => {
      if (progressTimer) clearInterval(progressTimer);
      window.removeEventListener('agentIterationStart', handleIterationStart as EventListener);
    };
  }, [agentId]);

  // Stream start → RESPONDING
  useEffect(() => {
    const handleStreamStart = (event: CustomEvent) => {
      if (event.detail.agentId === agentId) setLiveStatus('RESPONDING');
    };
    window.addEventListener('agentStreamStart', handleStreamStart as EventListener);
    return () => window.removeEventListener('agentStreamStart', handleStreamStart as EventListener);
  }, [agentId]);

  // Overrun detection
  useEffect(() => {
    if (loopProgress >= 100 && loopDurationRef.current > 0) {
      const modelWorking = liveStatus === 'CAPTURING' || liveStatus === 'THINKING' || liveStatus === 'RESPONDING';
      if (modelWorking || isOverrun) {
        setIsOverrun(true);
        loopStartTimeRef.current = Date.now();
        setLoopProgress(0);
      }
    }
  }, [loopProgress, liveStatus, isOverrun]);

  // Sleep state
  useEffect(() => {
    let sleepTimer: ReturnType<typeof setInterval> | null = null;

    const clearSleep = () => {
      if (sleepTimer) { clearInterval(sleepTimer); sleepTimer = null; }
      setSleepProgress(0);
      setIsSleeping(false);
      setSleepDurationMs(0);
      setLiveStatus('WAITING');
    };

    const handleSleepStart = (event: CustomEvent) => {
      if (event.detail.agentId !== agentId) return;
      clearSleep();
      const durationMs = event.detail.durationMs;
      const sleepEnd = Date.now() + durationMs;
      setIsSleeping(true);
      setSleepProgress(100);
      setSleepDurationMs(durationMs);
      setLiveStatus('SLEEPING');
      sleepTimer = setInterval(() => {
        const remaining = sleepEnd - Date.now();
        if (remaining <= 0) { clearSleep(); }
        else { setSleepProgress(Math.max(0, (remaining / durationMs) * 100)); }
      }, 100);
    };

    const handleSleepEnd = (event: CustomEvent) => {
      if (event.detail.agentId === agentId) clearSleep();
    };

    window.addEventListener('agentSleepStart', handleSleepStart as EventListener);
    window.addEventListener('agentSleepEnd', handleSleepEnd as EventListener);
    return () => {
      if (sleepTimer) clearInterval(sleepTimer);
      window.removeEventListener('agentSleepStart', handleSleepStart as EventListener);
      window.removeEventListener('agentSleepEnd', handleSleepEnd as EventListener);
    };
  }, [agentId]);

  const statusColor = () => {
    if (liveStatus === 'STARTING') return 'bg-yellow-400';
    if (isSleeping || liveStatus === 'SLEEPING') return 'bg-blue-400';
    if (isOverrun) return 'bg-orange-400';
    if (isRunning) return 'bg-green-400';
    return 'bg-gray-300';
  };

  const renderStatusIndicator = () => {
    if (isSleeping || liveStatus === 'SLEEPING') {
      return <PieTimer progress={sleepProgress} color="blue" totalDurationMs={sleepDurationMs} isFilling={false} size={14} />;
    }
    if (isOverrun) {
      return <PieTimer progress={loopProgress} color="orange" totalDurationMs={loopDurationMs} isFilling={true} size={14} />;
    }
    if (isRunning) {
      return <PieTimer progress={loopProgress} color="green" totalDurationMs={loopDurationMs} isFilling={true} size={14} />;
    }
    return (
      <span className={`w-2 h-2 rounded-full flex-shrink-0 ${statusColor()} ${liveStatus === 'STARTING' ? 'animate-pulse' : ''}`} />
    );
  };

  return (
    <button
      onClick={onRestore}
      title={`${agent.name} — tap to expand`}
      className={`flex items-center gap-2 px-3 h-9 rounded-2xl border transition-all duration-200 active:scale-95 cursor-pointer select-none flex-shrink-0
        ${isRunning || isStarting
          ? 'bg-green-50/80 border-green-200 hover:border-green-300'
          : 'bg-white/80 border-gray-200 hover:border-gray-300'}
        ${isPopping ? 'animate-chip-pop-in' : ''}
      `}
    >
      {renderStatusIndicator()}
      <span className={`text-xs font-semibold max-w-[90px] truncate ${isRunning || isStarting ? 'text-green-700' : 'text-gray-700'}`}>
        {agent.name}
      </span>
    </button>
  );
};

export default AgentChip;
