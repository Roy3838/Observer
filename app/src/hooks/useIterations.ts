// src/hooks/useIterations.ts
// Polling-based hooks for IterationStore - simple, efficient, no subscription overhead

import { useState, useEffect, useRef } from 'react';
import {
  IterationStore,
  IterationData,
  ToolCall,
  AgentSession
} from '@utils/IterationStore';

const DEFAULT_POLL_INTERVAL = 500;

/**
 * Hook to get iterations for an agent with polling
 * Only updates state when data actually changes
 */
export function useIterations(
  agentId: string,
  pollInterval = DEFAULT_POLL_INTERVAL
): IterationData[] {
  const [iterations, setIterations] = useState<IterationData[]>([]);
  const lastLengthRef = useRef(0);
  const lastUpdateRef = useRef('');

  useEffect(() => {
    const fetchIterations = () => {
      const latest = IterationStore.getIterationsForAgent(agentId);

      // Quick check: length changed?
      if (latest.length !== lastLengthRef.current) {
        lastLengthRef.current = latest.length;
        setIterations(latest);
        return;
      }

      // Deeper check: last iteration content changed?
      if (latest.length > 0) {
        const last = latest[latest.length - 1];
        const updateKey = `${last.id}-${last.modelResponse?.length || 0}-${last.tools.length}`;

        if (updateKey !== lastUpdateRef.current) {
          lastUpdateRef.current = updateKey;
          setIterations(latest);
        }
      }
    };

    // Initial fetch
    fetchIterations();

    // Poll for updates
    const interval = setInterval(fetchIterations, pollInterval);
    return () => clearInterval(interval);
  }, [agentId, pollInterval]);

  return iterations;
}

/**
 * Hook to get tools from the last iteration
 * Polls less frequently since tool status is less critical
 */
export function useLastTools(
  agentId: string,
  pollInterval = DEFAULT_POLL_INTERVAL
): ToolCall[] {
  const [tools, setTools] = useState<ToolCall[]>([]);
  const lastCountRef = useRef(0);

  useEffect(() => {
    const fetchTools = () => {
      const latest = IterationStore.getToolsFromLastIteration(agentId);

      if (latest.length !== lastCountRef.current) {
        lastCountRef.current = latest.length;
        setTools(latest);
      }
    };

    fetchTools();
    const interval = setInterval(fetchTools, pollInterval);
    return () => clearInterval(interval);
  }, [agentId, pollInterval]);

  return tools;
}

/**
 * Hook to get the frame + prompt actually sent to the model on the last iteration.
 * Mirrors useLastTools: only holds the latest capture and only re-renders when a new
 * one lands (keyed on id + image presence, so it updates when the image arrives shortly
 * after iteration-start). Avoids useIterations, which returns every base64 image per poll.
 */
export interface LastCapture {
  image?: string;
  prompt?: string;
}

export function useLastCapture(
  agentId: string,
  pollInterval = DEFAULT_POLL_INTERVAL
): LastCapture {
  const [capture, setCapture] = useState<LastCapture>({});
  const keyRef = useRef('');

  useEffect(() => {
    const fetchCapture = () => {
      const iterations = IterationStore.getIterationsForAgent(agentId);
      const last = iterations[iterations.length - 1];
      const key = last ? `${last.id}-${last.modelImages?.length ?? 0}` : '';
      if (key === keyRef.current) return;
      keyRef.current = key;
      setCapture(last ? { image: last.modelImages?.[0], prompt: last.modelPrompt } : {});
    };

    fetchCapture();
    const interval = setInterval(fetchCapture, pollInterval);
    return () => clearInterval(interval);
  }, [agentId, pollInterval]);

  return capture;
}

/**
 * Hook to get historical sessions for an agent
 * Polls less frequently since historical data changes rarely
 */
export function useHistoricalSessions(
  agentId: string,
  pollInterval = 2000 // Historical data changes rarely
): AgentSession[] {
  const [sessions, setSessions] = useState<AgentSession[]>([]);
  const lastCountRef = useRef(0);

  useEffect(() => {
    const fetchSessions = async () => {
      // Poll text only; load the images only when the session list changed.
      const latest = await IterationStore.getPastSessions(agentId);

      if (latest.length !== lastCountRef.current) {
        lastCountRef.current = latest.length;
        setSessions(await IterationStore.getPastSessions(agentId, { images: true }));
      }
    };

    fetchSessions();
    const interval = setInterval(fetchSessions, pollInterval);
    return () => clearInterval(interval);
  }, [agentId, pollInterval]);

  return sessions;
}

/**
 * Combined hook for components that need both current and historical data
 * Useful for AgentLogViewer which displays both
 */
export function useAllIterationData(agentId: string, pollInterval = DEFAULT_POLL_INTERVAL) {
  const iterations = useIterations(agentId, pollInterval);
  const historicalSessions = useHistoricalSessions(agentId);

  return { iterations, historicalSessions };
}

/**
 * Hook to get storage usage for an agent
 * Polls infrequently since storage stats don't need real-time updates
 */
export function useStorageUsage(agentId: string) {
  const [usage, setUsage] = useState({ currentSessionMB: 0, totalHistoryMB: 0 });

  useEffect(() => {
    const fetchUsage = async () => {
      const latest = await IterationStore.getStorageUsage(agentId);
      setUsage(latest);
    };

    fetchUsage();
    const interval = setInterval(fetchUsage, 5000); // Every 5 seconds is plenty
    return () => clearInterval(interval);
  }, [agentId]);

  return usage;
}

/**
 * Hook to get iteration count - lightweight alternative when you just need the count
 */
export function useIterationCount(agentId: string, pollInterval = DEFAULT_POLL_INTERVAL): number {
  const [count, setCount] = useState(0);
  const lastCountRef = useRef(0);

  useEffect(() => {
    const fetchCount = () => {
      const iterations = IterationStore.getIterationsForAgent(agentId);
      if (iterations.length !== lastCountRef.current) {
        lastCountRef.current = iterations.length;
        setCount(iterations.length);
      }
    };

    fetchCount();
    const interval = setInterval(fetchCount, pollInterval);
    return () => clearInterval(interval);
  }, [agentId, pollInterval]);

  return count;
}
