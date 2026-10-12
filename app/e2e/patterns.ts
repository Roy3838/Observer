// Scenarios for the pattern suite (patterns.spec.ts): requests that aren't "tell me when X ends"
// but use the rest of Observer: notes in memory, a summary when something ends, a periodic
// digest, a running count, two alerts from one request, clips, and requests where the right
// move is to ask first or to build nothing.
//
// Each scenario's demo publishes its ground truth: marks as things happen on screen (a topic
// change, an error printed, an order arriving) and `demo.truth`. Its `score` compares that with
// what the agents did: what they wrote to memory and when, their notifications, their clips.
// Like the dial suite, it never checks which models, tools or how many agents the MCP chose.
//
// Prompts that save results say "save it to memory": the suite can't fill the ask_user_info
// modal, and memory is what it can read back.

import type { SuiteEvent } from './fixtures';

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
}

export interface PatternScenario {
  id: string;
  /** Page in e2e/demos/, with its query string. */
  demo: string;
  prompt: string;
  /** Sent as a second turn if the first one ends with no agent running (the MCP asked something). */
  followUp?: string;
  /** The right move is to build nothing: the scenario ends after the MCP's turn. */
  noAgent?: boolean;
  /** Passed to demo.start(). Demos driven by their own media ignore it and end when it does. */
  runMs: number;
  /** How long to keep watching after the demo's end, for summaries and late notes. */
  graceMs: number;
  /** A file under e2e/demos/ the scenario needs; skipped until it exists. */
  requires?: string;
  score: (run: Run) => Score;
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

// ---- Scenarios ---------------------------------------------------------------

export const PATTERNS: PatternScenario[] = [
  {
    // A watcher that appends every occurrence: one note per topic change, with the video's time.
    id: 'youtube_timestamps #1',
    demo: 'youtube_watch.html',
    prompt: "Timestamp this YouTube video for me: every time the topic changes, save a note to memory with the video's timestamp.",
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
  {
    // A watcher that logs each error once, though each stays on screen for several iterations.
    id: 'terminal_errors #1',
    demo: 'terminal_errors.html',
    prompt: 'Keep a log in memory of every error that shows up in my terminal.',
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
  {
    // A logger of a chat that scrolls: every message once, though each stays visible for ~100 s.
    id: 'twitch_chat #1',
    demo: 'twitch_chat.html',
    prompt: 'Save the full transcript of this Twitch chat to memory.',
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
  {
    // A logger of the call's audio. Nothing to detect: the transcript just has to be in memory.
    id: 'meeting_transcript #1',
    demo: 'google_meet_call.html',
    prompt: 'Listen to this meeting and write the transcript to memory.',
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
  {
    // Logger + one-shot summarizer: the summary comes once, after the call ends.
    id: 'meeting_summary #1',
    demo: 'google_meet_call.html',
    prompt: 'Summarize this meeting when it ends and save the summary to memory.',
    runMs: 240_000,
    graceMs: 150_000,
    score: scoreMeetingSummary,
  },
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
    // Logger + one-shot summarizer on a video: chapters with timestamps, after it finishes.
    id: 'youtube_chapters #1',
    demo: 'youtube_watch.html',
    prompt: 'Make chapters for this YouTube video when it finishes, and save them to memory.',
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
      if (!early.length && byTs.length >= chapters.length - 1) return { verdict: 'PASS', metrics, notes };
      if (byAny.length >= chapters.length - 2) return { verdict: 'PARTIAL', metrics, notes };
      return { verdict: 'FAIL', metrics, notes };
    },
  },
  {
    // Logger + periodic summarizer: a note every 2 minutes naming what was on screen in that window.
    id: 'workday_digest #1',
    demo: 'workday.html',
    prompt: 'Every 2 minutes, save a note to memory of what I worked on.',
    runMs: 390_000,
    graceMs: 30_000,
    score: run => {
      const APP_KEYS: Record<string, string[]> = {
        vscode: ['vs code', 'vscode', 'visual studio code', 'code editor', 'checkout.ts', 'coding', 'typescript'],
        gmail: ['gmail', 'email', 'e-mail', 'inbox', 'invoice'],
        figma: ['figma', 'onboarding', 'mockup', 'design'],
        slack: ['slack', '#launch'],
      };
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
  {
    // An aggregator: the count lives in memory, and it's the new orders, not today's total.
    id: 'shopify_count #1',
    demo: 'shopify_orders.html',
    prompt: 'Count how many orders come in on my Shopify dashboard and keep the total in memory.',
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
  {
    // Two alerts from one request: out of the queue, then 2 minutes left on the ticket hold.
    id: 'ticketmaster_two_alerts #1',
    demo: 'ticketmaster_queue.html?outcome=success&timer=200',
    prompt: "Send me a browser notification when I'm out of the Ticketmaster queue, and another one when I have 2 minutes left to pick my tickets.",
    runMs: 120_000,
    graceMs: 145_000,
    score: run => {
      // Sends within 3 s of each other are one alert (a notify with a sound).
      const alerts: number[] = [];
      for (const n of run.events.filter(e => e.type === 'notification')) if (!alerts.length || n.t - alerts[alerts.length - 1] > 3000) alerts.push(n.t);
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
  {
    // Clips: a clip started or marked right after each death.
    id: 'minecraft_clips #1',
    demo: 'minecraft.html',
    prompt: 'Save a clip every time my character dies in this game.',
    requires: 'media/minecraft.mp4',
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
