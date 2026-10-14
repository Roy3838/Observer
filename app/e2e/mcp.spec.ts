// MCP end-to-end suite: one prompt per scenario, judged only on WHEN the user is notified.
//
//   SETUP    demo holds in progress; the MCP gets the prompt once and must leave an agent
//            running (no follow-up questions: anything else is NO_AGENT / ASKED_USER_INFO)
//   PRE      the job keeps going for a random 2–6 min; any notification → FALSE_POSITIVE
//   TRIGGER  the demo ends (one of its outcomes) and reports the instant it's on screen
//   WINDOW   expect notify: a notification within 60 s → PASS, none → MISSED
//            expect silent: silent for 3 min → PASS, any → FALSE_POSITIVE
//            expect either: PASS after 60 s, notified or not
//
// Which model, tools or notification channel the MCP picks is never checked.

import { expect } from '@playwright/test';
import fs from 'node:fs';
import { pathToFileURL } from 'node:url';
import { test, openObserver, E2E_DIR, type SuiteEvent } from './fixtures';
import { SCENARIOS, type Scenario } from './scenarios';

const env = (name: string, fallback: number) => Number(process.env[name] ?? fallback);
const SETUP_MS = env('E2E_SETUP_MS', 15 * 60_000);
const PRE_MIN_MS = env('E2E_PRE_MIN_MS', 2 * 60_000);
const PRE_MAX_MS = env('E2E_PRE_MAX_MS', 6 * 60_000);
const WINDOW_MS = env('E2E_WINDOW_MS', 60_000);
const QUIET_MS = env('E2E_QUIET_MS', 3 * 60_000);

const RESULTS_DIR = `${E2E_DIR}results/${process.env.E2E_RUN_ID}`;

type Verdict = 'PASS' | 'MISSED' | 'FALSE_POSITIVE' | 'NO_AGENT' | 'ASKED_USER_INFO' | 'SETUP_TIMEOUT' | 'DEMO_BROKEN';

const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms));

async function waitUntil(cond: () => boolean, timeoutMs: number): Promise<boolean> {
  const end = Date.now() + timeoutMs;
  while (!cond() && Date.now() < end) await sleep(250);
  return cond();
}

function judge(s: Scenario, events: SuiteEvent[]): { verdict: Verdict; latencyMs?: number } {
  const trigger = events.find(e => e.type === 'trigger');
  const notes = events.filter(e => e.type === 'notification');
  if (notes.some(n => !trigger || n.t < trigger.t)) return { verdict: 'FALSE_POSITIVE' };
  if (!trigger) return { verdict: 'DEMO_BROKEN' };
  const first = notes[0];
  const latencyMs = first ? first.t - trigger.t : undefined;
  if (s.expect === 'silent') return { verdict: first ? 'FALSE_POSITIVE' : 'PASS' };
  if (s.expect === 'either') return { verdict: 'PASS', latencyMs };
  if (first && first.t - trigger.t <= WINDOW_MS) return { verdict: 'PASS', latencyMs };
  return { verdict: 'MISSED' };
}

for (const s of SCENARIOS) {
  test(s.id, async ({ observer }, testInfo) => {
    test.setTimeout(SETUP_MS + PRE_MAX_MS + Math.max(WINDOW_MS, QUIET_MS) + 5 * 60_000);
    const { context, events } = observer;
    events.length = 0;
    const has = (type: SuiteEvent['type']) => events.some(e => e.type === type);

    // A clean Observer: no agents from earlier scenarios, then a reload for a fresh
    // conversation and stream state.
    let app = await openObserver(context);
    if (!await app.evaluate(() => (window as any).__observerE2E.isAuthenticated())) {
      throw new Error('Not logged in to Observer: run `pnpm e2e:login` first.');
    }
    await app.evaluate(() => (window as any).__observerE2E.reset());
    app = await openObserver(context);

    const demo = await context.newPage();
    await demo.goto(`${pathToFileURL(`${E2E_DIR}demos/${s.demo}`).href}?outcome=${s.param}`);
    // Chrome only lets a focused tab call getDisplayMedia, so Observer stays in front, like
    // a user chatting with it. The captured demo tab keeps rendering in the background.
    await app.bringToFront();

    let verdict: Verdict | undefined;
    let latencyMs: number | undefined;
    let preMs: number | undefined;
    let mcpReply: string | null = null;
    let agents: string[] = [];
    const setupStart = Date.now();
    try {
      // SETUP: one prompt, no follow-ups.
      const send = app.evaluate(p => (window as any).__observerE2E.send(p) as Promise<string>, s.prompt + (process.env.E2E_PROMPT_SUFFIX ?? ''));
      mcpReply = await Promise.race([send, sleep(SETUP_MS).then(() => null)]);
      if (mcpReply === null) {
        verdict = 'SETUP_TIMEOUT';
        await app.evaluate(() => (window as any).__observerE2E.stop());
        mcpReply = await Promise.race([send, sleep(10_000).then(() => '')]);
      }
      agents = await app.evaluate(() => (window as any).__observerE2E.runningAgentIds());
      if (!verdict && has('ask_user_info')) verdict = 'ASKED_USER_INFO';
      if (!verdict && agents.length === 0) verdict = 'NO_AGENT';

      if (!verdict) {
        // PRE: the job keeps going, then ends. Bail out early if the agent fires first.
        preMs = PRE_MIN_MS + Math.round(Math.random() * (PRE_MAX_MS - PRE_MIN_MS));
        await demo.evaluate(ms => (window as any).demo.start(ms), preMs);
        await waitUntil(() => has('trigger') || has('notification'), preMs + 30_000);

        // WINDOW: wait out the window (or quiet period) from the trigger, unless already decided.
        const trigger = events.find(e => e.type === 'trigger');
        if (trigger && !has('notification')) {
          // `either` keeps watching the window, so a late notification still gets a latency.
          const waitMs = s.expect === 'silent' ? QUIET_MS : WINDOW_MS;
          await waitUntil(() => has('notification'), trigger.t + waitMs - Date.now());
        }
        ({ verdict, latencyMs } = judge(s, events));
      }
    } finally {
      const trigger = events.find(e => e.type === 'trigger');
      const transcript = await app.evaluate(t => (window as any).__observerE2E.transcript(t), setupStart).catch(() => null);
      // What each agent was built as and saw on every iteration: what explains a MISSED.
      const agentReport: any[] = await app.evaluate(t => (window as any).__observerE2E.agentReport(t), setupStart).catch(() => []);
      await testInfo.attach('demo-end.png', { body: await demo.screenshot().catch(() => Buffer.alloc(0)), contentType: 'image/png' });
      await demo.close();
      await app.evaluate(() => (window as any).__observerE2E.reset()).catch(() => {});

      const result = {
        id: s.id, demo: s.demo, outcome: s.outcome, prompt: s.prompt, expect: s.expect,
        verdict: verdict ?? 'ERROR',
        latencyMs,
        setupMs: Date.now() - setupStart,
        preMs,
        agents,
        mcpReply,
        // Times relative to the trigger (negative = before the job ended).
        notifications: events
          .filter(e => e.type === 'notification')
          .map(n => ({ ...n, t: undefined, sinceTriggerMs: trigger ? n.t - trigger.t : null })),
      };
      fs.mkdirSync(RESULTS_DIR, { recursive: true });
      fs.appendFileSync(`${RESULTS_DIR}/results.jsonl`, JSON.stringify(result) + '\n');
      const base = `${RESULTS_DIR}/${s.id.replace(/[^\w-]+/g, '_')}_r${testInfo.repeatEachIndex}`;
      fs.writeFileSync(`${base}.transcript.json`, JSON.stringify(transcript, null, 2));
      // Same prompt and frame as iteration 1, sent again: if the answer moves, the model call is what's unstable.
      for (const a of agentReport) {
        const it1 = a.iterations?.[0];
        if (!it1?.image || !it1.prompt) continue;
        a.replay = await app.evaluate(([m, p, img]) => (window as any).__observerE2E.replay(m, p, img, 6), [a.model_name, it1.prompt, it1.image] as const).catch((e: any) => String(e));
      }
      agentReport.forEach((a, i) => {
        if (a.lastImage) fs.writeFileSync(`${base}.agent${i + 1}-last.png`, Buffer.from(String(a.lastImage).replace(/^data:[^,]*,/, ''), 'base64'));
        // Every iteration's frame, to tell a stale or wrong first frame from a wrong answer.
        a.iterations.forEach(it => {
          if (it.image) fs.writeFileSync(`${base}.agent${i + 1}-it${it.n}.png`, Buffer.from(String(it.image).replace(/^data:[^,]*,/, ''), 'base64'));
        });
      });
      fs.writeFileSync(`${base}.agents.json`, JSON.stringify(agentReport.map(a => ({ ...a, lastImage: a.lastImage ? '(saved as png)' : undefined, iterations: a.iterations.map(it => ({ ...it, image: it.image ? '(saved as png)' : undefined })) })), null, 2));
      await testInfo.attach('result.json', { body: JSON.stringify({ ...result, transcript }, null, 2), contentType: 'application/json' });
      console.log(`${s.id}: ${result.verdict}${latencyMs !== undefined ? ` (${Math.round(latencyMs / 1000)} s after trigger)` : ''}`);
    }

    expect(verdict, `MCP replied: ${mcpReply}`).toBe('PASS');
  });
}
