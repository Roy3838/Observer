// src/components/Observer/AgentLiveStateContext.tsx
//
// Keeps each agent's live status/progress/last-word state alive independently of whether its
// card is currently rendered inline (in the chat transcript) or floating (in the overlay) —
// those are two different component instances in two different subtrees, so computing this
// state locally inside AgentLiveCard (as a first pass did) meant popping a card out — or
// docking it back — silently reset its progress ring and last-word ticker to blank, because
// the new instance started from scratch. One hidden "keeper" per relevant agent id runs the
// actual event-driven tracking (mirrors AgentCard's state machine) and reports into a shared
// map that both AgentLiveCard render modes read from, so the underlying state survives the
// swap.

import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import { Logger, LogEntry } from '@utils/logging';

export type AgentLiveStatus = 'STARTING' | 'CAPTURING' | 'THINKING' | 'RESPONDING' | 'WAITING' | 'SKIPPED' | 'SLEEPING' | 'IDLE';

export interface AgentLiveState {
  liveStatus: AgentLiveStatus;
  lastWord: string;
  progress: number;
  durationMs: number;
  isSleeping: boolean;
  sleepRemainingMs: number;
  /** The loop interval elapsed while the model was still working; the countdown restarts
   *  (following main_loop's tick grid) and the UI shows "still working" in orange. */
  isOverrun: boolean;
}

const DEFAULT_STATE: AgentLiveState = {
  liveStatus: 'IDLE', lastWord: '', progress: 0, durationMs: 0, isSleeping: false, sleepRemainingMs: 0, isOverrun: false,
};

const AgentLiveStateContext = createContext<Record<string, AgentLiveState>>({});

export function useAgentLiveStateFor(agentId: string): AgentLiveState {
  const map = useContext(AgentLiveStateContext);
  return map[agentId] ?? DEFAULT_STATE;
}

// Mirrors the status/loop/sleep/streaming state machine in AgentCard.tsx. Event-driven — no
// polling. This is the only place that actually computes the state; everything else reads it
// back out of the map above.
function useAgentLiveState(agentId: string, isRunning: boolean, isStarting: boolean): AgentLiveState {
  const [liveStatus, setLiveStatus] = useState<AgentLiveStatus>('IDLE');
  const [lastWord, setLastWord] = useState('');
  const [progress, setProgress] = useState(0);
  const [durationMs, setDurationMs] = useState(0);
  const [isSleeping, setIsSleeping] = useState(false);
  const [sleepRemainingMs, setSleepRemainingMs] = useState(0);
  const [isOverrun, setIsOverrun] = useState(false);

  const statusRef = useRef<AgentLiveStatus>('IDLE');
  statusRef.current = liveStatus;
  const startRef = useRef(0);
  const durationRef = useRef(0);
  const loopTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopLoopTimer = useCallback(() => {
    if (loopTimerRef.current) clearInterval(loopTimerRef.current);
    loopTimerRef.current = null;
  }, []);

  // Ticks progress off startRef/durationRef. If the interval elapses while the model is still
  // working, main_loop skips that tick (isExecuting), so flag an overrun and restart the
  // countdown instead of leaving it pinned at 0.
  const startLoopTimer = useCallback(() => {
    stopLoopTimer();
    loopTimerRef.current = setInterval(() => {
      const status = statusRef.current;
      let elapsed = Date.now() - startRef.current;
      if (elapsed >= durationRef.current && (status === 'CAPTURING' || status === 'THINKING' || status === 'RESPONDING')) {
        setIsOverrun(true);
        startRef.current = Date.now();
        elapsed = 0;
      }
      setProgress(Math.min(100, (elapsed / durationRef.current) * 100));
    }, 100);
  }, [stopLoopTimer]);

  // main_loop runs one fixed setInterval and sleep only makes ticks return early, so after a
  // wake the next iteration lands on the original tick grid (last start + k * interval), not
  // a full interval from now. Re-anchor to the latest grid boundary to show the real remainder;
  // the next agentIterationStart then corrects any small drift.
  const resumeLoopTimer = useCallback(() => {
    if (!startRef.current || !durationRef.current) return;
    const ticksElapsed = Math.floor((Date.now() - startRef.current) / durationRef.current);
    startRef.current += ticksElapsed * durationRef.current;
    setProgress(Math.min(100, ((Date.now() - startRef.current) / durationRef.current) * 100));
    startLoopTimer();
  }, [startLoopTimer]);

  useEffect(() => {
    if (!isRunning && !isStarting) {
      setLiveStatus('IDLE');
      return;
    }
    if (isStarting && !isRunning) {
      setLiveStatus('STARTING');
      return;
    }
    setLiveStatus(prev => (prev === 'STARTING' || prev === 'IDLE' ? 'CAPTURING' : prev));

    const handleNewLog = (log: LogEntry) => {
      if (log.source !== agentId) return;
      if (log.details?.logType === 'model-prompt') {
        setLiveStatus('THINKING');
      } else if (log.details?.logType === 'iteration-skipped') {
        setLiveStatus('SKIPPED');
      } else if (log.details?.logType === 'model-response') {
        setLiveStatus('WAITING');
        const text = (log.details.content as string) || '';
        const words = text.trim().split(/\s+/).filter(Boolean);
        if (words.length) setLastWord(words[words.length - 1]);
      }
    };
    Logger.addListener(handleNewLog);
    return () => Logger.removeListener(handleNewLog);
  }, [agentId, isRunning, isStarting]);

  useEffect(() => {
    const handleIterationStart = (event: CustomEvent) => {
      if (event.detail.agentId !== agentId) return;
      setIsSleeping(false);
      setIsOverrun(false);
      startRef.current = event.detail.iterationStartTime;
      durationRef.current = event.detail.intervalMs;
      setDurationMs(event.detail.intervalMs);
      setProgress(0);
      if (statusRef.current === 'SLEEPING' || statusRef.current === 'IDLE' || statusRef.current === 'STARTING') {
        setLiveStatus('CAPTURING');
      }
      startLoopTimer();
    };

    const handleStreamStart = (event: CustomEvent) => {
      if (event.detail.agentId !== agentId) return;
      setLiveStatus('RESPONDING');
    };

    window.addEventListener('agentIterationStart', handleIterationStart as EventListener);
    window.addEventListener('agentStreamStart', handleStreamStart as EventListener);
    return () => {
      stopLoopTimer();
      window.removeEventListener('agentIterationStart', handleIterationStart as EventListener);
      window.removeEventListener('agentStreamStart', handleStreamStart as EventListener);
    };
  }, [agentId, startLoopTimer, stopLoopTimer]);

  // Streamed response chunks -> last word ticker. Never cleared on sleep — it should keep
  // showing the last decision the agent made until a new one replaces it.
  useEffect(() => {
    let buffer = '';
    const handleChunk = (event: CustomEvent) => {
      if (event.detail.agentId !== agentId) return;
      buffer += event.detail.chunk || '';
      const words = buffer.trim().split(/\s+/).filter(Boolean);
      if (words.length) setLastWord(words[words.length - 1]);
    };
    const resetBuffer = (event: CustomEvent) => {
      if (event.detail.agentId !== agentId) return;
      buffer = '';
    };
    window.addEventListener('agentResponseChunk', handleChunk as EventListener);
    window.addEventListener('agentStreamStart', resetBuffer as EventListener);
    return () => {
      window.removeEventListener('agentResponseChunk', handleChunk as EventListener);
      window.removeEventListener('agentStreamStart', resetBuffer as EventListener);
    };
  }, [agentId]);

  useEffect(() => {
    let sleepTimer: ReturnType<typeof setInterval> | null = null;

    const handleSleepStart = (event: CustomEvent) => {
      if (event.detail.agentId !== agentId) return;
      if (sleepTimer) clearInterval(sleepTimer);
      const sleepDurationMs = event.detail.durationMs;
      const sleepEnd = Date.now() + sleepDurationMs;
      // Stop the loop ticker so it doesn't pin progress at 100 (or flag a bogus overrun)
      // for the whole sleep; resumeLoopTimer re-anchors it on wake.
      stopLoopTimer();
      setIsOverrun(false);
      setIsSleeping(true);
      setLiveStatus('SLEEPING');
      setSleepRemainingMs(sleepDurationMs);

      sleepTimer = setInterval(() => {
        const remaining = sleepEnd - Date.now();
        if (remaining <= 0) {
          setIsSleeping(false);
          setLiveStatus('WAITING');
          if (sleepTimer) clearInterval(sleepTimer);
          resumeLoopTimer();
          return;
        }
        setSleepRemainingMs(remaining);
      }, 250);
    };

    const handleSleepEnd = (event: CustomEvent) => {
      if (event.detail.agentId !== agentId) return;
      if (sleepTimer) clearInterval(sleepTimer);
      setIsSleeping(false);
      setLiveStatus('WAITING');
      resumeLoopTimer();
    };

    window.addEventListener('agentSleepStart', handleSleepStart as EventListener);
    window.addEventListener('agentSleepEnd', handleSleepEnd as EventListener);
    return () => {
      if (sleepTimer) clearInterval(sleepTimer);
      window.removeEventListener('agentSleepStart', handleSleepStart as EventListener);
      window.removeEventListener('agentSleepEnd', handleSleepEnd as EventListener);
    };
  }, [agentId, stopLoopTimer, resumeLoopTimer]);

  return { liveStatus, lastWord, progress, durationMs, isSleeping, sleepRemainingMs, isOverrun };
}

const Keeper: React.FC<{
  agentId: string;
  isRunning: boolean;
  isStarting: boolean;
  onUpdate: (id: string, state: AgentLiveState) => void;
}> = ({ agentId, isRunning, isStarting, onUpdate }) => {
  const state = useAgentLiveState(agentId, isRunning, isStarting);
  useEffect(() => {
    onUpdate(agentId, state);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentId, state.liveStatus, state.lastWord, state.progress, state.durationMs, state.isSleeping, state.sleepRemainingMs, state.isOverrun]);
  return null;
};

export const AgentLiveStateProvider: React.FC<{
  agentIds: string[];
  runningAgents: Set<string>;
  startingAgents: Set<string>;
  children: React.ReactNode;
}> = ({ agentIds, runningAgents, startingAgents, children }) => {
  const [map, setMap] = useState<Record<string, AgentLiveState>>({});

  const handleUpdate = useCallback((id: string, state: AgentLiveState) => {
    setMap(prev => ({ ...prev, [id]: state }));
  }, []);

  return (
    <AgentLiveStateContext.Provider value={map}>
      {agentIds.map(id => (
        <Keeper key={id} agentId={id} isRunning={runningAgents.has(id)} isStarting={startingAgents.has(id)} onUpdate={handleUpdate} />
      ))}
      {children}
    </AgentLiveStateContext.Provider>
  );
};
