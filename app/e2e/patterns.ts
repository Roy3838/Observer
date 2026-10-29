// Scenarios for the pattern suite (patterns.spec.ts): the ground truth use cases that aren't one
// alert on time (src/utils/groundTruth.ts): "then log it", "then keep count of it", "then save a
// clip", a summary when something ends, two alerts from one request. Plus a few chat checks that
// aren't use cases: requests where the right move is to ask first, answer from what the agents
// saw, or build nothing.
//
// Each scenario's demo publishes its ground truth: marks as things happen on screen (a topic
// change, an error printed, an order arriving) and `demo.truth`. Its check's `score` compares that
// with what the agents did: what they logged to memory and when, their notifications, their clips.
// Like the alert suite, it never checks which models, tools or how many agents the MCP chose.
//
// "Log" means Observer's memory, which is also what the suite can read back (it can't fill the
// ask_user_info modal for any other destination).

import type { SuiteEvent } from './fixtures';
import { GROUND_TRUTH, prompt } from '../src/utils/groundTruth';

export type PatternVerdict = 'PASS' | 'PARTIAL' | 'FAIL';

export interface Score {
  verdict: PatternVerdict;
  metrics: Record<string, unknown>;
  notes: string[];
}

/** Everything a scenario's score can look at, collected when the scenario ends. */
export interface Run {
  events: SuiteEvent[];
  /** When the suite started the demo (demo.start), on the suite's clock. */
  startT: number;
  /** The demo's end (its trigger), if it got there. */
  triggerT: number | null;
  truth: any;
  /** Every memory slot at the end. */
  memories: { id: string; memory: string }[];
  /** The agents as built (agentReport in src/mcp/e2e.ts): system_prompt, code, iterations. */
  agents: { id: string; system_prompt: string; code: string | null; memory: string }[];
  /** The MCP's tool calls over the whole conversation, in order. */
  toolCalls: { name: string; args: string }[];
  /** For scenarios with a follow-up: whether the MCP's first turn ended without an agent running. */
  asked: boolean | null;
  /** The MCP's final text for each turn, in order: the prompt, the follow-up, the after-run question. */
  replies: (string | null)[];
}

/** How a use case is run and judged. */
export interface Check {
  /** Sent as a second turn if the first one ends with no agent running (the MCP asked something). */
  followUp?: string;
  /** Asked once the run and its grace period are over, with the agents still running: its answer is
   *  the last of `replies`. */
  afterRun?: string;
  /** The right move is to build nothing: the scenario ends after the MCP's turn. */
  noAgent?: boolean;
  /** Passed to demo.start(). Demos driven by their own media ignore it and end when it does. */
  runMs: number;
  /** How long to keep watching after the demo's end, for summaries and late notes. */
  graceMs: number;
  score: (run: Run) => Score;
}

export interface PatternScenario extends Check {
  id: string;
  /** Page in e2e/demos/, with its query string. */
  demo: string;
  prompt: string;
  /** Files under e2e/demos/ the scenario needs; skipped until they all exist. */
  requires?: string[];
}

// ---- Reading the agents' work ------------------------------------------------

/** One memory write: what it added (or set) and who wrote it, on the suite's clock. */
export interface Entry { t: number; agentId: string; targetId: string; tool: string; text: string }

/** Memory writes in order. An append is what it added; a set that replaces the memory is all of it.
 *  Writes that empty a slot (a summarizer clearing its log) are left out. */
export function entries(run: Run): Entry[] {
  const last = new Map<string, string>();
  const out: Entry[] = [];
  for (const e of run.events) {
    if (e.type !== 'memory' || !e.ok) continue;
    const prev = last.get(e.targetId) ?? '';
    last.set(e.targetId, e.memory);
    const text = (e.memory.startsWith(prev) ? e.memory.slice(prev.length) : e.memory).trim();
    if (text) out.push({ t: e.t, agentId: e.agentId, targetId: e.targetId, tool: e.tool, text });
  }
  return out;
}

const SENSORS = /\$(SCREEN|CAMERA|SCREEN_AUDIO|ALL_AUDIO|MICROPHONE)\b/;
/** Agents that read memory instead of the screen or audio: summarizers, aggregators. */
export function readsOnlyMemory(run: Run, agentId: string): boolean {
  const agent = run.agents.find(a => a.id === agentId);
  return !!agent && !SENSORS.test(agent.system_prompt ?? '');
}

const lower = (s: string) => s.toLowerCase();
export const mentions = (text: string, keys: string[]) => keys.some(k => lower(text).includes(k));

/** Video timestamps ("1:23", "0:05:40") in a note, in seconds. Clock times ("3:12 pm", what the
 *  agents' time() writes) and anything past the video's end are not video timestamps. */
export function videoTimes(text: string, duration: number): number[] {
  const out: number[] = [];
  for (const m of text.matchAll(/\b(?:(\d):)?(\d{1,2}):([0-5]\d)\b(?!\s*[ap]\.?m)/gi)) {
    const s = Number(m[1] ?? 0) * 3600 + Number(m[2]) * 60 + Number(m[3]);
    if (s <= duration + 30) out.push(s);
  }
  return out;
}

const marks = (run: Run, name: string) =>
  run.events.filter((e): e is Extract<SuiteEvent, { type: 'mark' }> => e.type === 'mark' && e.name === name);

const pct = (n: number, d: number) => (d ? Math.round((100 * n) / d) : 0);

/** When the user was alerted: notifications, with sends within 3 s of each other (a notify with a
 *  sound) counted once. */
function alertTimes(run: Run): number[] {
  const out: number[] = [];
  for (const n of run.events.filter(e => e.type === 'notification')) if (!out.length || n.t - out[out.length - 1] > 3000) out.push(n.t);
  return out;
}

// ---- Shared ground truth -------------------------------------------------------

// The talk in youtube_watch.html (media/talk.js): what identifies each chapter in a note.
const CHAPTER_KEYS = [
  ['doubled', 'intro', 'why'],
  ['money goes', 'tag', 'cost explorer', 'find where', 'visibility', 'owner'],
  ['right-siz', 'rightsiz', 'right siz', 'instance size', 'downsiz'],
  ['spot'],
  ['storage', 'lifecycle', 'glacier'],
  ['result', '52', 'takeaway', 'conclusion', 'summary', 'lesson'],
];

// The meeting in google_meet_call.html (media/meeting.js).
const DECISIONS = [
  { what: 'launch moves to March 14', keys: ['march 14', 'march fourteenth', '14th of march', 'march 14th'] },
  { what: 'annual discount drops to 15%', keys: ['15%', '15 percent', 'fifteen percent', '15 %'] },
  { what: 'beta stays invite-only', keys: ['invite-only', 'invite only', 'invitation only'] },
];
const ACTIONS = [
  { what: 'Ana: pricing doc', owner: ['ana'], keys: ['pricing doc', 'pricing document'] },
  { what: 'Ben: Stripe webhook retries', owner: ['ben'], keys: ['webhook'] },
  { what: 'Priya: launch email', owner: ['priya', 'pria', 'preya'], keys: ['launch email'] },
];
// Phrases a transcript of that meeting should contain (any spelling in each group).
const MEETING_PHRASES = [
  ['march'], ['fourteenth', '14'], ['stripe'], ['webhook'], ['pricing'], ['fifteen', '15'],
  ['invite'], ['waitlist', 'wait list'], ['launch email'], ['thursday'], ['friday'], ['wednesday'],
];

/** The summary of the meeting: what memory-only agents wrote after it ended (or, when the MCP
 *  built a single agent, anything written after the end), scored on decisions and owned actions. */
function scoreMeetingSummary(run: Run): Score {
  const notes: string[] = [];
  if (run.triggerT === null) return { verdict: 'FAIL', metrics: {}, notes: ['the meeting never ended'] };
  const all = entries(run);
  const after = all.filter(e => e.t >= run.triggerT!);
  let summary = after.filter(e => readsOnlyMemory(run, e.agentId));
  if (!summary.length && after.length) {
    summary = after;
    notes.push('summary written by an agent that also reads the sensors');
  }
  const text = summary.map(e => e.text).join('\n');
  // A summary before the meeting ended summarized part of it.
  const early = all.filter(e => e.t < run.triggerT! && readsOnlyMemory(run, e.agentId)
    && DECISIONS.filter(d => mentions(e.text, d.keys)).length >= 2);
  const decisions = DECISIONS.filter(d => mentions(text, d.keys)).map(d => d.what);
  const lines = text.split('\n');
  const actions = ACTIONS.filter(a => lines.some(l => mentions(l, a.keys) && mentions(l, a.owner))).map(a => a.what);
  if (early.length) notes.push(`summary written ${early.length}× before the meeting ended`);
  const metrics = {
    summaryWrites: summary.length, decisions, actions,
    summarySecondsAfterEnd: summary.length ? Math.round((summary[0].t - run.triggerT) / 1000) : null,
    summary: text.slice(0, 1500),
  };
  if (!text) return { verdict: 'FAIL', metrics, notes: [...notes, 'nothing saved after the meeting ended'] };
  if (!early.length && decisions.length >= 2 && actions.length >= 2) return { verdict: 'PASS', metrics, notes };
  if (decisions.length + actions.length >= 2) return { verdict: 'PARTIAL', metrics, notes };
  return { verdict: 'FAIL', metrics, notes };
}

// The apps in workday.html: what names each one in a note.
const WORKDAY_APPS: Record<string, string[]> = {
  vscode: ['vs code', 'vscode', 'visual studio code', 'code editor', 'checkout.ts', 'coding', 'typescript'],
  gmail: ['gmail', 'email', 'e-mail', 'inbox', 'invoice'],
  figma: ['figma', 'onboarding', 'mockup', 'design'],
  slack: ['slack', '#launch'],
};

// ---- Scenarios ---------------------------------------------------------------

// Judges for the ground truth use cases that aren't one alert on time (src/utils/groundTruth.ts),
// keyed by use case id: what they logged, counted or clipped, and when.
const CHECKS: Record<string, Check> = {
  youtube_topic_changes: {
    // A watcher that appends every occurrence: one note per topic change, with the video's time.
    runMs: 240_000,
    graceMs: 45_000,
    score: run => {
      const { chapters, duration } = run.truth;
      const changes: any[] = chapters.slice(1);
      const markT = new Map(marks(run, 'chapter').map(m => [m.data.index, m.t]));
      const notes = entries(run).filter(e => e.t >= run.startT);
      const byTs = new Set<number>(), byAny = new Set<number>();
      let dups = 0, stray = 0;
      for (const n of notes) {
        const times = videoTimes(n.text, duration);
        const hit = changes.find(c => times.some(s => s >= c.start - 5 && s <= c.start + 40)
          || (mentions(n.text, CHAPTER_KEYS[c.index]) && n.t >= (markT.get(c.index) ?? Infinity) && n.t <= (markT.get(c.index) ?? 0) + 60_000));
        if (!hit) { if (!mentions(n.text, CHAPTER_KEYS[0])) stray++; continue; }
        if (byAny.has(hit.index)) dups++;
        byAny.add(hit.index);
        if (times.some(s => s >= hit.start - 5 && s <= hit.start + 40)) byTs.add(hit.index);
      }
      const metrics = {
        changes: changes.length, withTimestamp: byTs.size, noted: byAny.size, duplicates: dups, stray, notes: notes.length,
        memory: notes.map(n => n.text).join('\n').slice(0, 1500),
      };
      if (byTs.size >= changes.length - 1 && dups + stray <= 3) return { verdict: 'PASS', metrics, notes: [] };
      if (byAny.size >= 3) return { verdict: 'PARTIAL', metrics, notes: [] };
      return { verdict: 'FAIL', metrics, notes: [] };
    },
  },
  terminal_errors: {
    // A watcher that logs each error once, though each stays on screen for several iterations.
    runMs: 240_000,
    graceMs: 60_000,
    score: run => {
      const keys: string[][] = run.truth.errors;
      const found = new Set<number>();
      let dups = 0, stray = 0;
      const notes = entries(run).filter(e => e.t >= run.startT);
      for (const n of notes) {
        const hits = keys.map((k, i) => (mentions(n.text, k) ? i : -1)).filter(i => i >= 0);
        if (!hits.length) { stray++; continue; }
        for (const i of hits) { if (found.has(i)) dups++; found.add(i); }
      }
      const metrics = { errors: keys.length, found: found.size, duplicates: dups, stray, notes: notes.length, memory: notes.map(n => n.text).join('\n').slice(0, 1500) };
      if (found.size >= keys.length - 1 && dups <= 2 && stray <= 2) return { verdict: 'PASS', metrics, notes: [] };
      if (found.size >= 3) return { verdict: 'PARTIAL', metrics, notes: [] };
      return { verdict: 'FAIL', metrics, notes: [] };
    },
  },
  twitch_chat: {
    // A logger of a chat that scrolls: every message once, though each stays visible for ~100 s.
    runMs: 240_000,
    graceMs: 60_000,
    score: run => {
      const chat: { user: string; text: string }[] = run.truth.chat;
      const words = (s: string) => s.toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').split(/\s+/).filter(w => w.length >= 3);
      const lines = entries(run).filter(e => e.t >= run.startT).flatMap(e => e.text.split('\n')).map(l => ({ raw: l, words: new Set(words(l)) }));
      // A message is in a line when most of its words are.
      const inLine = (m: { text: string }, l: { words: Set<string> }) => {
        const w = words(m.text);
        return w.length > 0 && w.filter(x => l.words.has(x)).length >= Math.ceil(0.6 * w.length);
      };
      let found = 0, dups = 0, withUser = 0;
      for (const m of chat) {
        const hits = lines.filter(l => inLine(m, l));
        if (hits.length) found++;
        dups += Math.max(0, hits.length - 1);
        if (hits.some(l => l.raw.toLowerCase().includes(m.user.toLowerCase()))) withUser++;
      }
      const metrics = { messages: chat.length, captured: found, capturedPct: pct(found, chat.length), duplicates: dups, withUsername: withUser, memory: lines.map(l => l.raw).join('\n').slice(0, 1500) };
      if (found >= 0.8 * chat.length && dups <= 0.25 * found) return { verdict: 'PASS', metrics, notes: [] };
      if (found >= 0.5 * chat.length) return { verdict: 'PARTIAL', metrics, notes: dups > 0.25 * found ? ['many messages saved more than once'] : [] };
      return { verdict: 'FAIL', metrics, notes: [] };
    },
  },
  meeting_transcript: {
    // A logger of the call's audio. Nothing to detect: the transcript just has to be in memory.
    runMs: 240_000,
    graceMs: 90_000,
    score: run => {
      // Everything ever written, in case a later write replaced or cleared part of it.
      const text = [...entries(run).map(e => e.text), ...run.memories.map(m => m.memory)].join('\n');
      const hit = MEETING_PHRASES.filter(p => mentions(text, p));
      const metrics = { phrases: MEETING_PHRASES.length, heard: hit.length, missing: MEETING_PHRASES.filter(p => !hit.includes(p)).map(p => p[0]), memory: text.slice(0, 1500) };
      if (hit.length >= 8) return { verdict: 'PASS', metrics, notes: [] };
      if (hit.length >= 4) return { verdict: 'PARTIAL', metrics, notes: [] };
      return { verdict: 'FAIL', metrics, notes: [] };
    },
  },
  meeting_summary: {
    // Logger + one-shot summarizer: the summary comes once, after the call ends.
    runMs: 240_000,
    graceMs: 150_000,
    score: scoreMeetingSummary,
  },
  youtube_topics: {
    // Logger + one-shot summarizer on a video: every topic, after it finishes. Timestamps are counted
    // but not required: the request doesn't ask for them.
    runMs: 240_000,
    graceMs: 150_000,
    score: run => {
      const { chapters, duration } = run.truth;
      if (run.triggerT === null) return { verdict: 'FAIL', metrics: {}, notes: ['the video never ended'] };
      const all = entries(run);
      let list = all.filter(e => e.t >= run.triggerT! && readsOnlyMemory(run, e.agentId));
      const notes: string[] = [];
      if (!list.length) { list = all.filter(e => e.t >= run.triggerT!); if (list.length) notes.push('chapters written by an agent that also reads the sensors'); }
      const early = all.filter(e => e.t < run.triggerT! && readsOnlyMemory(run, e.agentId) && videoTimes(e.text, duration).length >= 3);
      if (early.length) notes.push('a chapter list was written before the video ended');
      const lines = list.map(e => e.text).join('\n').split('\n');
      const byTs = chapters.filter((c: any) => lines.some(l => videoTimes(l, duration).some(s => Math.abs(s - c.start) <= 25)));
      const byAny = chapters.filter((c: any) => byTs.includes(c) || lines.some(l => mentions(l, CHAPTER_KEYS[c.index])));
      const metrics = { chapters: chapters.length, withTimestamp: byTs.length, named: byAny.length, writes: list.length, list: lines.join('\n').slice(0, 1500) };
      if (!list.length) return { verdict: 'FAIL', metrics, notes: [...notes, 'nothing saved after the video ended'] };
      if (!early.length && byAny.length >= chapters.length - 1) return { verdict: 'PASS', metrics, notes };
      if (byAny.length >= chapters.length - 2) return { verdict: 'PARTIAL', metrics, notes };
      return { verdict: 'FAIL', metrics, notes };
    },
  },
  workday_notes: {
    // Logger + periodic summarizer: a note every 2 minutes naming what was on screen in that window.
    runMs: 390_000,
    graceMs: 30_000,
    score: run => {
      const APP_KEYS = WORKDAY_APPS;
      // Who was on screen when: VS Code until the run starts, then the demo's switches.
      const switches = [{ t: 0, name: 'vscode' }, ...marks(run, 'app').map(m => ({ t: m.t, name: m.data.name as string }))];
      const onScreen = (from: number, to: number) => {
        const secs = new Map<string, number>();
        switches.forEach((s, i) => {
          const a = Math.max(s.t, from), b = Math.min(switches[i + 1]?.t ?? Infinity, to);
          if (b > a) secs.set(s.name, (secs.get(s.name) ?? 0) + (b - a) / 1000);
        });
        return [...secs].filter(([, s]) => s >= 15).map(([n]) => n);
      };
      const end = (run.triggerT ?? run.startT + 390_000) + 30_000;
      const inRun = entries(run).filter(e => e.t >= run.startT && e.t <= end);
      // Notes from a summarizer when there is one; otherwise whatever the MCP built wrote.
      const summarized = inRun.filter(e => readsOnlyMemory(run, e.agentId));
      const digests = summarized.length ? summarized : inRun;
      let expected = 0, named = 0, right = 0;
      const each = digests.map(d => {
        const exp = onScreen(d.t - 120_000, d.t);
        const said = Object.keys(APP_KEYS).filter(a => mentions(d.text, APP_KEYS[a]));
        expected += exp.length; named += said.length; right += said.filter(a => exp.includes(a)).length;
        return { at: Math.round((d.t - run.startT) / 1000), expected: exp, said };
      });
      const gaps = digests.slice(1).map((d, i) => Math.round((d.t - digests[i].t) / 1000));
      const metrics = { digests: digests.length, gapsSeconds: gaps, recallPct: pct(right, expected), precisionPct: pct(right, named), each, memory: digests.map(d => d.text).join('\n---\n').slice(0, 1500) };
      const regular = gaps.every(g => g >= 80 && g <= 170);
      const notes = summarized.length ? [] : ['notes written by an agent that reads the screen itself'];
      if (digests.length >= 2 && regular && right >= 0.7 * expected && right >= 0.7 * named) return { verdict: 'PASS', metrics, notes };
      if (digests.length >= 1 && right >= 0.5 * expected) return { verdict: 'PARTIAL', metrics, notes: regular ? notes : [...notes, 'digests not every ~2 min'] };
      return { verdict: 'FAIL', metrics, notes };
    },
  },
  shopify_count: {
    // An aggregator: the count lives in memory, and it's the new orders, not today's total.
    runMs: 240_000,
    graceMs: 45_000,
    score: run => {
      const { added, before } = run.truth;
      // The slot written last holds the current count: its last number that isn't an order number,
      // a price, a time or a date.
      const writes = entries(run);
      const lastSlot = writes.length ? writes[writes.length - 1].targetId : null;
      const memory = run.memories.find(m => m.id === lastSlot)?.memory ?? '';
      const cleaned = memory
        .replace(/#\s?\d+/g, ' ').replace(/\$\s?[\d,]+(\.\d+)?/g, ' ')
        .replace(/\b\d{1,2}:\d{2}(:\d{2})?\s*([ap]\.?m\.?)?/gi, ' ').replace(/\b\d{4}-\d{2}-\d{2}\b/g, ' ');
      const numbers = [...cleaned.matchAll(/\b\d+\b/g)].map(m => Number(m[0]));
      const count = numbers.length ? numbers[numbers.length - 1] : null;
      const metrics = { expected: added, counted: count, todayTotal: before + added, memory: memory.slice(-800) };
      if (count === added) return { verdict: 'PASS', metrics, notes: [] };
      if (count !== null && Math.abs(count - added) <= 1) return { verdict: 'PARTIAL', metrics, notes: [] };
      return { verdict: 'FAIL', metrics, notes: count === before + added ? ["counted today's orders, not the new ones"] : [] };
    },
  },
  ticketmaster_two_alerts: {
    // Two alerts from one request: out of the queue, then 2 minutes left on the ticket hold.
    runMs: 120_000,
    graceMs: 145_000,
    score: run => {
      const alerts = alertTimes(run);
      const two = marks(run, 'two_minutes_left')[0]?.t ?? null;
      if (run.triggerT === null || two === null) return { verdict: 'FAIL', metrics: {}, notes: ['the demo never reached its end or the 2:00 mark'] };
      const early = alerts.filter(t => t < run.triggerT!);
      const first = alerts.find(t => t >= run.triggerT! && t <= run.triggerT! + 60_000);
      const second = alerts.find(t => t >= two - 20_000 && t <= two + 45_000);
      const extra = alerts.length - early.length - (first ? 1 : 0) - (second ? 1 : 0);
      const at = (t?: number, from?: number) => (t && from ? Math.round((t - from) / 1000) : null);
      const metrics = { alerts: alerts.length, beforeQueueEnded: early.length, outOfQueueAfterS: at(first, run.triggerT), twoMinutesAfterS: at(second, two), extra };
      if (!early.length && first && second && extra <= 1) return { verdict: 'PASS', metrics, notes: [] };
      if (!early.length && (first || second)) return { verdict: 'PARTIAL', metrics, notes: [] };
      return { verdict: 'FAIL', metrics, notes: [] };
    },
  },
  minecraft_clips: {
    // Clips: a clip started or marked right after each death.
    runMs: 600_000,
    graceMs: 40_000,
    score: run => {
      const deaths = marks(run, 'death').map(m => m.t);
      const clips = run.events.filter(e => e.type === 'clip' && e.ok && (e.tool === 'startClip' || e.tool === 'markClip')).map(e => e.t);
      const covered = deaths.filter(d => clips.some(c => c >= d - 3000 && c <= d + 40_000));
      const stray = clips.filter(c => !deaths.some(d => c >= d - 3000 && c <= d + 40_000)).length;
      const metrics = { deaths: deaths.length, covered: covered.length, clipCalls: clips.length, stray, stopClips: run.events.filter(e => e.type === 'clip' && e.tool === 'stopClip').length };
      if (deaths.length && covered.length === deaths.length && stray <= 1) return { verdict: 'PASS', metrics, notes: [] };
      if (covered.length >= deaths.length / 2 && covered.length > 0) return { verdict: 'PARTIAL', metrics, notes: [] };
      return { verdict: 'FAIL', metrics, notes: [] };
    },
  },
  minecraft_deaths: {
    // Alerts, not clips: one notification right after each death, none in between.
    runMs: 600_000,
    graceMs: 40_000,
    score: run => {
      const deaths = marks(run, 'death').map(m => m.t);
      const alerts = alertTimes(run);
      const near = (a: number, d: number) => a >= d - 3000 && a <= d + 40_000;
      const covered = deaths.filter(d => alerts.some(a => near(a, d)));
      const stray = alerts.filter(a => !deaths.some(d => near(a, d))).length;
      const metrics = { deaths: deaths.length, covered: covered.length, alerts: alerts.length, stray };
      if (deaths.length && covered.length === deaths.length && stray <= 1) return { verdict: 'PASS', metrics, notes: [] };
      if (covered.length > 0 && covered.length >= deaths.length / 2) return { verdict: 'PARTIAL', metrics, notes: [] };
      return { verdict: 'FAIL', metrics, notes: [] };
    },
  },
  workday_calls: {
    // A transcript of the call in the middle of a workday, and nothing from the rest of the day.
    runMs: 390_000,
    graceMs: 60_000,
    score: run => {
      const text = [...entries(run).map(e => e.text), ...run.memories.map(m => m.memory)].join('\n');
      const heard = MEETING_PHRASES.filter(p => mentions(text, p));
      // Logged before the call started: the day, not the call.
      const callT = marks(run, 'app').find(m => m.data.name === 'meet')?.t ?? Infinity;
      const early = entries(run).filter(e => e.t < callT).length;
      const metrics = { heard: heard.length, phrases: MEETING_PHRASES.length, loggedBeforeCall: early, memory: text.slice(0, 1500) };
      const notes = early ? [`${early} notes logged before the call started`] : [];
      if (heard.length >= 6 && early <= 1) return { verdict: 'PASS', metrics, notes };
      if (heard.length >= 3) return { verdict: 'PARTIAL', metrics, notes };
      return { verdict: 'FAIL', metrics, notes };
    },
  },
  pet_food: {
    // A camera logger for one moment: the dog walking away mid-bowl isn't it.
    runMs: 180_000,
    graceMs: 60_000,
    score: run => {
      const finished = marks(run, 'finished')[0]?.t;
      if (!finished) return { verdict: 'FAIL', metrics: {}, notes: ['the clip never reached the empty bowl'] };
      // A note that says the bowl is done, sentence by sentence, skipping "not finished yet".
      const claims = entries(run).filter(e => e.text.split(/[.!?\n]/).some(l =>
        mentions(l, ['finished', 'empty', 'done eating', 'ate all', 'all gone', 'cleaned the bowl']) && !/\b(not|no longer|yet)\b|n't/i.test(l)));
      const good = claims.filter(e => e.t >= finished - 5000 && e.t <= finished + 60_000);
      const early = claims.filter(e => e.t < finished - 5000);
      const metrics = { notes: entries(run).length, claims: claims.length, early: early.length, onTimeAfterS: good.length ? Math.round((good[0].t - finished) / 1000) : null };
      if (good.length && !early.length) return { verdict: 'PASS', metrics, notes: [] };
      if (good.length) return { verdict: 'PARTIAL', metrics, notes: ['said it was finished before it was'] };
      return { verdict: 'FAIL', metrics, notes: early.length ? ['said it was finished before it was'] : [] };
    },
  },
  people_log: {
    // A camera log of comings and goings: a note near each one, not one every loop.
    runMs: 240_000,
    graceMs: 45_000,
    score: run => {
      const KEYS: Record<string, string[]> = {
        enter: ['enter', 'arriv', 'came in', 'comes in', 'walked in', 'walks in', 'appear', 'showed up', 'shows up', 'came home'],
        leave: ['left', 'leav', 'exit', 'walked out', 'walks out', 'went out', 'departed', 'gone'],
      };
      const events = [...marks(run, 'enter'), ...marks(run, 'leave')];
      const near = (t: number, e: { t: number }) => t >= e.t - 5000 && t <= e.t + 30_000;
      const notes = entries(run);
      const covered = events.filter(e => notes.some(n => near(n.t, e) && mentions(n.text, KEYS[e.name])));
      const stray = notes.filter(n => !events.some(e => near(n.t, e))).length;
      const metrics = { events: events.length, covered: covered.length, notes: notes.length, stray };
      if (!events.length) return { verdict: 'FAIL', metrics, notes: ['the clip has no comings or goings marked'] };
      if (covered.length >= 0.8 * events.length && stray <= events.length) return { verdict: 'PASS', metrics, notes: [] };
      if (covered.length >= 0.5 * events.length) return { verdict: 'PARTIAL', metrics, notes: stray > events.length ? [`${stray} notes with nobody coming or going`] : [] };
      return { verdict: 'FAIL', metrics, notes: [] };
    },
  },
};

// Chat checks: not use cases, so not in the ground truth. Requests where the right move is to ask
// first, to answer from what the agents saw, or to build nothing.
const CHAT: PatternScenario[] = [
  {
    // The same, but the user didn't say where the summary goes: the MCP should ask first.
    id: 'meeting_summary_ask #1',
    demo: 'google_meet_call.html',
    prompt: 'Summarize this meeting when it ends.',
    followUp: 'Save it to memory, please.',
    runMs: 240_000,
    graceMs: 150_000,
    score: run => {
      const s = scoreMeetingSummary(run);
      const invented = run.toolCalls.some(c => /create_agent|edit_agent/.test(c.name) && /sendEmail\(\s*["'`][^"'`]+@/.test(c.args));
      s.metrics.asked = run.asked;
      if (invented) return { ...s, verdict: 'FAIL', notes: [...s.notes, 'wrote an email address the user never gave'] };
      if (!run.asked && s.verdict === 'PASS') return { ...s, verdict: 'PARTIAL', notes: [...s.notes, "didn't ask where to save it"] };
      return s;
    },
  },
  {
    // Asked about the past with nothing logging it: the honest answer is that it can't know, and
    // the useful one offers to start (or starts) keeping track. Naming apps that were never on
    // screen (the demo holds in VS Code) is making it up.
    id: 'activity_recall_cold #1',
    demo: 'workday.html',
    prompt: 'What was I doing the last 4 hours?',
    noAgent: true,
    runMs: 0,
    graceMs: 0,
    score: run => {
      const reply = run.replies[0] ?? '';
      const invented = ['gmail', 'figma', 'slack', 'onboarding', 'northwind', '#launch'].filter(k => mentions(reply, [k]));
      const honest = mentions(reply, ["can't", 'cannot', "don't have", 'do not have', 'no record', 'no history', 'not able', 'unable',
        "haven't been", 'from now on', 'going forward', 'start tracking', 'start logging', 'keep track']);
      const built = run.agents.length > 0;
      const metrics = { invented, honest, built, reply: reply.slice(0, 1500) };
      if (invented.length) return { verdict: 'FAIL', metrics, notes: [`named apps it never saw: ${invented.join(', ')}`] };
      if (honest || built) return { verdict: 'PASS', metrics, notes: [] };
      return { verdict: 'PARTIAL', metrics, notes: ["didn't say it can't see the past or offer to start tracking"] };
    },
  },
  {
    // A logger now, a question later: the MCP answers from what its agents saw.
    id: 'activity_recall #1',
    demo: 'workday.html',
    prompt: "Keep track of what I'm working on so I can ask you about it later.",
    afterRun: 'What was I doing over the last few minutes?',
    runMs: 390_000,
    graceMs: 30_000,
    score: run => {
      const reply = run.replies[run.replies.length - 1] ?? '';
      const said = Object.keys(WORKDAY_APPS).filter(a => mentions(reply, WORKDAY_APPS[a]));
      const metrics = { said, apps: Object.keys(WORKDAY_APPS).length, reply: reply.slice(0, 1500) };
      if (!reply) return { verdict: 'FAIL', metrics, notes: ['no answer to the question'] };
      if (said.length >= 3) return { verdict: 'PASS', metrics, notes: [] };
      if (said.length >= 2) return { verdict: 'PARTIAL', metrics, notes: [] };
      return { verdict: 'FAIL', metrics, notes: [] };
    },
  },
  // Nothing to build: a greeting and an unrelated question.
  ...[
    ['hello #1', 'hi'],
    ['offtopic #1', "What's 2+2?"],
  ].map(([id, prompt]): PatternScenario => ({
    id,
    demo: 'blank.html',
    prompt,
    noAgent: true,
    runMs: 0,
    graceMs: 0,
    score: run => {
      const built = run.toolCalls.filter(c => /^(create_agent|edit_agent|start_agent)$/.test(c.name)).map(c => c.name);
      const metrics = { built, agents: run.agents.length };
      return { verdict: built.length || run.agents.length ? 'FAIL' : 'PASS', metrics, notes: [] };
    },
  })),
];

// Experiments: not use cases yet, so not in the ground truth.

/** camera_phone.html: one alert per notification banner, saying what it says, and no repeats
 *  while the same banner stays up. An alert counts for the banner whose words it quotes, else
 *  for the one on screen when it fired (or just before). */
function scorePhoneNotifications(run: Run): Score {
  const list: { keys: string[]; ms: number; title: string }[] = run.truth?.notifications ?? [];
  const shown = marks(run, 'notification').map(m => m.t);
  const hidden = marks(run, 'notification_hidden').map(m => m.t);
  if (shown.length < list.length) return { verdict: 'FAIL', metrics: {}, notes: ['not every notification was shown'] };
  const alerts = run.events.filter((e): e is Extract<SuiteEvent, { type: 'notification' }> => e.type === 'notification' && e.t >= run.startT);
  const per = list.map(() => 0);
  let stray = 0, quoted = 0, withImage = 0;
  for (const a of alerts) {
    const text = String(a.params?.message ?? a.params?.body ?? JSON.stringify(a.params ?? ''));
    let i = list.findIndex(n => mentions(text, n.keys));
    if (i >= 0) quoted++;
    // Not quoted: the banner up at the time, allowing for an iteration that started while it was.
    else i = shown.findIndex((s, k) => a.t >= s && a.t <= (hidden[k] ?? s + list[k].ms) + 15_000);
    if (a.params?.imageCount) withImage++;
    if (i < 0) stray++; else per[i]++;
  }
  const caught = per.filter(c => c > 0).length;
  const duplicates = per.reduce((s, c) => s + Math.max(0, c - 1), 0);
  const metrics = {
    notifications: list.length, caught, missed: list.filter((_, i) => !per[i]).map(n => n.title),
    duplicates, perNotification: per, stray, alerts: alerts.length, quoted, withImage,
    texts: alerts.map(a => String(a.params?.message ?? '').slice(0, 100)),
  };
  const notes: string[] = [];
  if (quoted < caught) notes.push(`${caught - quoted} alerts don't quote the notification`);
  if (caught >= list.length - 1 && duplicates <= 1 && stray <= 1) return { verdict: 'PASS', metrics, notes };
  if (caught >= list.length / 2) return { verdict: 'PARTIAL', metrics, notes };
  return { verdict: 'FAIL', metrics, notes };
}

const PHONE_PROMPT = [
  "Watch continuously through my phone's rear camera, pointed at the screen of another phone.",
  'This is a test to detect incoming pop-up notifications on the other phone.',
  'Whenever a new notification appears on the screen, immediately capture it, read all visible text, '
    + 'and send me a Telegram message containing the notification text and a photo of the screen. My Telegram chat_id is 123456789.',
  'Detect all notifications, regardless of their content or language.',
  'Do not send duplicate alerts for the same notification.',
  'Keep monitoring continuously, including when notifications appear for only a few seconds.',
].join('\n\n');

// The $MEMORY dedupe pattern, written by hand: the model reads the banner and compares it with the
// last one it reported, which the code saves only when it's new.
const PHONE_AGENT = {
  id: 'phone_notifications',
  model_name: 'gemma-3-12b-it',
  loop_interval_seconds: 5,
  system_prompt: [
    'You are watching the screen of a phone through a camera.',
    '$CAMERA',
    'The last notification already reported:',
    '$MEMORY',
    'If a notification banner is on the phone screen, write out all of its text exactly as shown.',
    'Then, on the last line, answer with exactly one word:',
    'NO_NOTIFICATION if there is no notification banner,',
    'SAME_NOTIFICATION if it is the notification already reported above,',
    'NEW_NOTIFICATION if it is a different notification.',
  ].join('\n'),
  code: [
    'if (response.includes("NEW_NOTIFICATION")) {',
    '  setMemory(response);',
    '  notify("New notification", response.replace("NEW_NOTIFICATION", "").trim());',
    '}',
  ].join('\n'),
};

const EXPERIMENTS: PatternScenario[] = [
  {
    // The hand-written agent above, built verbatim by the MCP (the suite has no other way to start
    // an agent with the user's token).
    id: 'phone_notifications_memory #1',
    demo: 'camera_phone.html',
    prompt: 'Create and start exactly this agent, verbatim: do not change the prompt, code, model or interval, '
      + 'and do not ask me anything.\n```json\n' + JSON.stringify(PHONE_AGENT, null, 2) + '\n```',
    runMs: 300_000,
    graceMs: 30_000,
    score: run => {
      const s = scorePhoneNotifications(run);
      const built = run.agents.find(a => (a.system_prompt ?? '').includes('$MEMORY') && (a.code ?? '').includes('NEW_NOTIFICATION'));
      s.metrics.verbatim = !!built && run.agents.length === 1;
      if (!s.metrics.verbatim) s.notes.push('the MCP changed the agent');
      return s;
    },
  },
  {
    // The request as a user wrote it: does the MCP come up with dedupe on its own?
    id: 'phone_notifications #1',
    demo: 'camera_phone.html',
    prompt: PHONE_PROMPT,
    followUp: "I can't set up Telegram right now: send them as browser notifications instead.",
    runMs: 300_000,
    graceMs: 30_000,
    score: scorePhoneNotifications,
  },
];

const stray = Object.keys(CHECKS).filter(id => !GROUND_TRUTH.some(u => u.id === id && !u.expect));
if (stray.length) throw new Error(`checks for no ground truth use case: ${stray.join(', ')}`);

export const PATTERNS: PatternScenario[] = [
  ...GROUND_TRUTH.filter(u => !u.expect).map(u => {
    const check = CHECKS[u.id];
    if (!check) throw new Error(`ground truth use case ${u.id} has no expect and no check in patterns.ts`);
    return { id: u.id, demo: u.demo, prompt: prompt(u), requires: u.requires, ...check };
  }),
  ...CHAT,
  ...EXPERIMENTS,
];
