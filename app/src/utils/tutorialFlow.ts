// src/utils/tutorialFlow.ts
//
// State machine for the first-run demo, kept OUTSIDE React so every producer and consumer can
// reach it: the demo picker closes on send, the banner lives in ObserverTab, and the
// notification arrives from the agent loop. Purely an observer: it never sits in a tool's
// call path, it only listens (Logger `tool-success`).
//
// The demo itself runs in another browser tab (see tutorialDemos.ts), so there is no signal
// for "the job finished": the agent's notification is the moment the demo is about.
//
//   idle -> building (user picked a demo) -> monitoring (agent running)
//        -> notified (agent's notification tool succeeded, or the fallback timer ran out)
//        -> idle (CTA or banner dismissed)

import { useSyncExternalStore } from 'react';
import { Logger, type LogEntry } from './logging';
import type { TutorialDemo } from './tutorialDemos';

export type TutorialPhase = 'idle' | 'building' | 'monitoring' | 'notified';

interface TutorialSnapshot {
  phase: TutorialPhase;
  agentId: string | null;
  demo: TutorialDemo | null;
}

// Tools that count as "the agent told the user".
const NOTIFY_TOOLS = new Set([
  'sendEmail', 'sendWhatsapp', 'sendSms', 'call', 'sendTelegram', 'sendDiscord',
  'sendPushover', 'sendGotify', 'system_notify', 'notify', 'message', 'overlay',
]);

// The demo finishes 30 s after the user returns to its tab (or a minute after it opens, on
// desktop). If no notification is seen this long after the agent starts, show the CTA anyway so
// the user is never stuck.
const NOTIFY_FALLBACK_MS = 90000;

let snapshot: TutorialSnapshot = { phase: 'idle', agentId: null, demo: null };
const listeners = new Set<() => void>();
let fallbackTimer: ReturnType<typeof setTimeout> | null = null;
let wired = false;

const set = (next: Partial<TutorialSnapshot>) => {
  snapshot = { ...snapshot, ...next };
  listeners.forEach(l => l());
};

const clearFallback = () => { if (fallbackTimer) { clearTimeout(fallbackTimer); fallbackTimer = null; } };

// Listeners run after the tool has already finished; errors here must never reach it.
const onLog = (entry: LogEntry) => {
  try {
    const { phase, agentId } = snapshot;
    if (phase !== 'monitoring' || entry.source !== agentId) return;
    const d = entry.details;
    if (d?.logType !== 'tool-success') return;
    if (NOTIFY_TOOLS.has(d?.content?.tool)) {
      clearFallback();
      set({ phase: 'notified' });
    }
  } catch { /* observer only */ }
};

const wire = () => {
  if (wired) return;
  wired = true;
  Logger.addListener(onLog);
};

export const tutorialFlow = {
  /** The user picked a demo; its tab is open and the prompt is on its way to the MCP. */
  start(demo: TutorialDemo) { wire(); clearFallback(); set({ phase: 'building', agentId: null, demo }); },
  /** The demo agent is running. */
  attachAgent(agentId: string) {
    if (snapshot.phase !== 'building') return;
    set({ phase: 'monitoring', agentId });
    fallbackTimer = setTimeout(() => { if (snapshot.phase === 'monitoring') set({ phase: 'notified' }); }, NOTIFY_FALLBACK_MS);
  },
  /** Stopped/dismissed. */
  reset() { clearFallback(); set({ phase: 'idle', agentId: null, demo: null }); },
  get: () => snapshot,
  subscribe(l: () => void) { listeners.add(l); return () => { listeners.delete(l); }; },
};

export const useTutorialFlow = (): TutorialSnapshot =>
  useSyncExternalStore(tutorialFlow.subscribe, tutorialFlow.get);
