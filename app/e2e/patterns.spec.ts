// Pattern suite: requests that use more of Observer than "tell me when X ends" (patterns.ts).
//
//   SETUP    the demo holds; the MCP gets the prompt. If the scenario has a follow-up and the turn
//            ends with no agent running (the MCP asked something), the follow-up is the answer.
//   RUN      the suite starts the demo, which plays out (a video, a meeting, a workday, orders
//            arriving) and marks what happens as it happens, then triggers at its end
//   GRACE    the suite keeps watching for summaries and late notes, then scores the run
//
// Verdicts: PASS / PARTIAL / FAIL from the scenario's score, or NO_AGENT, SETUP_TIMEOUT,
// DEMO_BROKEN. Results go to e2e/results/<run>/patterns.jsonl.

import fs from 'node:fs';
import { pathToFileURL } from 'node:url';
import { expect } from '@playwright/test';
import { test, openObserver, E2E_DIR } from './fixtures';
import { PATTERNS, type PatternScenario, type Run, type Score } from './patterns';

const SETUP_MS = Number(process.env.E2E_SETUP_MS ?? 15 * 60_000);
const RESULTS_DIR = `${E2E_DIR}results/${process.env.E2E_RUN_ID}`;

type Verdict = Score['verdict'] | 'NO_AGENT' | 'SETUP_TIMEOUT' | 'DEMO_BROKEN';

const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms));
async function waitUntil(cond: () => boolean, timeoutMs: number): Promise<boolean> {
  const end = Date.now() + timeoutMs;
  while (!cond() && Date.now() < end) await sleep(250);
  return cond();
}

function demoUrl(s: PatternScenario): string {
  const [file, query] = s.demo.split('?');
  return `${pathToFileURL(`${E2E_DIR}demos/${file}`).href}?${query ?? 'outcome=success'}`;
}

for (const s of PATTERNS) {
  test(s.id, async ({ observer }, testInfo) => {
    test.skip(!!s.requires && !fs.existsSync(`${E2E_DIR}demos/${s.requires}`), `needs e2e/demos/${s.requires}`);
    test.setTimeout(2 * SETUP_MS + s.runMs + s.graceMs + 10 * 60_000);
    const { context, events } = observer;
    events.length = 0;

    let app = await openObserver(context);
    if (!await app.evaluate(() => (window as any).__observerE2E.isAuthenticated())) {
      throw new Error('Not logged in to Observer: run `pnpm e2e:login` first.');
    }
    await app.evaluate(() => (window as any).__observerE2E.reset());
    app = await openObserver(context);
    const demo = await context.newPage();
    await demo.goto(demoUrl(s));
    await app.bringToFront();

    const send = async (prompt: string): Promise<string | null> => {
      const reply = app.evaluate(p => (window as any).__observerE2E.send(p) as Promise<string>, prompt);
      const done = await Promise.race([reply, sleep(SETUP_MS).then(() => null)]);
      if (done !== null) return done;
      await app.evaluate(() => (window as any).__observerE2E.stop());
      await Promise.race([reply, sleep(10_000)]);
      return null;
    };
    const running = () => app.evaluate(() => (window as any).__observerE2E.runningAgentIds() as string[]);

    let verdict: Verdict | undefined;
    let score: Score | undefined;
    const replies: (string | null)[] = [];
    let asked: boolean | null = null;
    let startT = 0;
    const setupStart = Date.now();
    try {
      replies.push(await send(s.prompt));
      if (replies[0] === null) verdict = 'SETUP_TIMEOUT';
      if (!verdict && s.followUp) {
        asked = (await running()).length === 0;
        if (asked) {
          replies.push(await send(s.followUp));
          if (replies[1] === null) verdict = 'SETUP_TIMEOUT';
        }
      }
      if (!verdict && !s.noAgent) {
        if ((await running()).length === 0) verdict = 'NO_AGENT';
        else {
          startT = Date.now();
          await demo.evaluate(ms => (window as any).demo.start(ms), s.runMs);
          // Demos that play their own media end when it does; give them room past runMs.
          const ended = await waitUntil(() => events.some(e => e.type === 'trigger'), s.runMs + 4 * 60_000);
          if (!ended) verdict = 'DEMO_BROKEN';
          else await sleep(s.graceMs);
        }
      }
    } finally {
      const transcript = await app.evaluate(t => (window as any).__observerE2E.transcript(t), setupStart).catch(() => null);
      const agents: any[] = await app.evaluate(t => (window as any).__observerE2E.agentReport(t), setupStart).catch(() => []);
      const memories = await app.evaluate(() => (window as any).__observerE2E.memories()).catch(() => []);
      const truth = await demo.evaluate(() => (window as any).demo.truth ?? null).catch(() => null);
      await testInfo.attach('demo-end.png', { body: await demo.screenshot().catch(() => Buffer.alloc(0)), contentType: 'image/png' });
      await demo.close();
      await app.evaluate(() => (window as any).__observerE2E.reset()).catch(() => {});

      const toolCalls = (transcript?.messages ?? []).flatMap((m: any) => m.toolCalls ?? []);
      if (!verdict) {
        const run: Run = {
          events: [...events], startT: startT || Date.now(), triggerT: events.find(e => e.type === 'trigger')?.t ?? null,
          truth, memories, agents, toolCalls, asked,
        };
        try {
          score = s.score(run);
          verdict = score.verdict;
        } catch (e) {
          score = { verdict: 'FAIL', metrics: {}, notes: [`score threw: ${e instanceof Error ? e.message : String(e)}`] };
          verdict = 'FAIL';
        }
      }

      const result = {
        id: s.id, demo: s.demo, prompt: s.prompt, verdict,
        metrics: score?.metrics, notes: score?.notes, asked,
        setupMs: (startT || Date.now()) - setupStart,
        agents: agents.map(a => a.id),
        replies,
        notifications: events.filter(e => e.type === 'notification').map(n => ({ ...n, sinceStartMs: startT ? n.t - startT : null, t: undefined })),
      };
      fs.mkdirSync(RESULTS_DIR, { recursive: true });
      fs.appendFileSync(`${RESULTS_DIR}/patterns.jsonl`, JSON.stringify(result) + '\n');
      const base = `${RESULTS_DIR}/${s.id.replace(/[^\w-]+/g, '_')}`;
      fs.writeFileSync(`${base}.transcript.json`, JSON.stringify(transcript, null, 2));
      fs.writeFileSync(`${base}.agents.json`, JSON.stringify(agents.map(a => ({ ...a, lastImage: a.lastImage ? '(saved as png)' : undefined })), null, 2));
      agents.forEach((a, i) => {
        if (a.lastImage) fs.writeFileSync(`${base}.agent${i + 1}-last.png`, Buffer.from(String(a.lastImage).replace(/^data:[^,]*,/, ''), 'base64'));
      });
      fs.writeFileSync(`${base}.truth.json`, JSON.stringify(truth, null, 2));
      fs.writeFileSync(`${base}.memories.json`, JSON.stringify(memories, null, 2));
      fs.writeFileSync(`${base}.events.json`, JSON.stringify(events.map(e => ({ ...e, sinceStartMs: startT ? e.t - startT : null })), null, 2));
      await testInfo.attach('result.json', { body: JSON.stringify({ ...result, transcript }, null, 2), contentType: 'application/json' });
      console.log(`${s.id}: ${verdict}${score?.notes?.length ? ` (${score.notes.join('; ')})` : ''}`);
    }

    expect(verdict, `MCP replied: ${replies.join(' / ')}`).toBe('PASS');
  });
}
