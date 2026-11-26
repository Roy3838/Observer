// src/mcp/ontology/selfcheck.ts
//
// Self-check for the ontology: every recipe the system prompt teaches must pass the rules, and one
// broken variant per rule must fail with that rule's id. Also checks the engine fixes the ontology
// relies on (JSON-safe sensor text, run trends on JSON and choice answers). No browser needed:
//
//   cd app && npx esbuild src/mcp/ontology/selfcheck.ts --bundle --platform=node --format=esm \
//     --alias:@utils=./src/utils --log-level=warning | node --input-type=module
//
// Exits non-zero on any failure.

import { runRules, type RuleContext, type Stage } from './rules';
import { strategyOf } from './derive';
import { analyzeDecisionPrompt } from './prompt';
import { parseClassification, type AgentClassification } from './schema';
import { withNoulTrend, resetNoulTrend, type Decision, type NoulAnswer, type ChoiceAnswer } from '@utils/localLlm/systemOne';
import type { IterationData } from '@utils/IterationStore';

const MODELS: Record<string, RuleContext['model']> = {
  'clef-flash': { name: 'clef-flash', decision: true, local: false },
  'gemma-3-12b-it': { name: 'gemma-3-12b-it', decision: false, local: false },
  'onejev-local': { name: 'onejev-local', decision: true, local: true },
  'Skip Model Call': { name: 'Skip Model Call', decision: false, local: true },
};

interface Case {
  name: string;
  model: string;
  prompt: string;
  code: string;
  cls: AgentClassification;
  interval?: number;
  stage?: Stage;
  crop?: RuleContext['crop'];
  agents?: Record<string, AgentClassification>;
  iteration?: Partial<IterationData>;
}

function run(c: Case) {
  const parsed = parseClassification(c.cls);
  if (!parsed.value) throw new Error(`${c.name}: bad classification: ${parsed.error}`);
  const cls = parsed.value;
  const strategy = strategyOf(cls);
  const ctx: RuleContext = {
    agent: { id: 'test', model_name: c.model, system_prompt: c.prompt, loop_interval_seconds: c.interval ?? 5 },
    code: c.code,
    cls,
    strategy,
    model: MODELS[c.model],
    prompt: cls.task === 'decide' && strategy === 'decision' ? analyzeDecisionPrompt(c.prompt) : {},
    crop: c.crop,
    findAgent: c.agents ? id => (c.agents![id] ? { classification: c.agents![id] } : null) : undefined,
    iteration: c.iteration ? { sessionIterationNumber: 1, tools: [], ...c.iteration } as IterationData : undefined,
    sounds: ['ping', 'bell'],
  };
  return runRules(c.stage ?? 'save', ctx);
}

let failures = 0;
const fail = (msg: string) => { failures++; console.log(`FAIL  ${msg}`); };
const pass = (msg: string) => console.log(`ok    ${msg}`);

// ---- Recipes: must pass with no errors -------------------------------------------

const watcher = (over: Partial<AgentClassification>): AgentClassification => ({ role: 'watcher', task: 'decide', sensor: 'screen', scope: 'region', ...over });

const RECIPES: Case[] = [
  {
    name: 'Finish Line (dial, noul, now yes, once, voice)',
    model: 'clef-flash',
    prompt: 'Is there a download percentage below 100% on screen?\n$SCREEN',
    code: `if (decision.noul < 0.25 || decision.dropped) {
  await call("tree-book-shower-golden", "Your Steam download stopped: it finished or failed.");
  await stopAgent();
}`,
    cls: watcher({ target: 'dial', shape: 'event', cardinality: 'once', question: 'noul', now: 'yes', channel: 'voice' }),
  },
  {
    name: 'Carrier Pigeon (state, noul with context, now no, once)',
    model: 'clef-flash',
    prompt: 'The newest message in this WhatsApp chat is the user\'s own: "Can you send it to me?". Is there a newer message from Maria below it?\n$SCREEN',
    code: `if (decision.noul > 0.75) {
  notify("Maria replied", "You have a new WhatsApp message from Maria.");
  await stopAgent();
}`,
    cls: watcher({ target: 'state', shape: 'event', cardinality: 'once', question: 'noul', now: 'no', channel: 'browser' }),
  },
  {
    name: 'Deadline (no reply in 5 minutes)',
    model: 'clef-flash',
    prompt: 'The newest message in this WhatsApp chat is the user\'s own: "Are we still on for tonight?". Is there a newer message from Maria below it?\n$SCREEN',
    code: `const state = JSON.parse((await getMemory()) || "{}");
if (!state.start) { state.start = Date.now(); await setMemory(JSON.stringify(state)); }
if (decision.noul > 0.75) {
  await stopAgent();
} else if (Date.now() - state.start > 5 * 60 * 1000) {
  notify("No reply", "Maria hasn't replied in 5 minutes.");
  await stopAgent();
}`,
    cls: watcher({ target: 'state', shape: 'deadline', cardinality: 'once', question: 'noul', now: 'no', channel: 'browser' }),
  },
  {
    name: 'Nag (social media, every 2 minutes)',
    model: 'clef-flash',
    prompt: 'Is a social media feed on screen?\n$SCREEN',
    code: `const state = JSON.parse((await getMemory()) || "{}");
if (decision.noul > 0.75) {
  if (!state.last || Date.now() - state.last > 2 * 60 * 1000) {
    notify("Back to work", "You're on social media.");
    state.last = Date.now();
  }
} else {
  state.last = 0;
}
await setMemory(JSON.stringify(state));`,
    cls: watcher({ target: 'state', shape: 'nag', cardinality: 'every', question: 'noul', now: 'no', channel: 'browser', scope: 'anywhere', scope_reason: 'a feed can be in any tab or window' }),
  },
  {
    name: 'Sequence (CI: tests, then deploy, choice)',
    model: 'clef-flash',
    prompt: `{"questions": {"stage": {"type": "choice", "instructions": "Which stage does the pipeline show?", "criteria": {"testing": "tests are running", "deploying": "tests passed and the deploy is running", "deployed": "the deploy finished", "failed": "a step failed"}}}}
$SCREEN`,
    code: `const seen = await getMemory();
if (decision.stage.probabilities.failed > 0.75) {
  await stopAgent();
} else if (decision.stage.probabilities.deploying > 0.75 && !seen) {
  await setMemory("deploying");
} else if (decision.stage.probabilities.deployed > 0.75 && seen) {
  await sendTelegram("tree-book-shower-golden", "Tests passed and the deploy finished.", screen);
  await stopAgent();
}`,
    cls: watcher({ target: 'state', shape: 'sequence', cardinality: 'once', question: 'choice', now: 'testing', channel: 'telegram' }),
  },
  {
    name: 'Upload Courier (dial, how it ended, choice)',
    model: 'clef-flash',
    prompt: `{"questions": {"upload": {"type": "choice", "instructions": "What does the upload status say?", "criteria": {"uploading": "still uploading, with items or time left", "done": "the upload finished", "failed": "the upload failed or was cancelled"}}}}
$SCREEN`,
    code: `if (decision.upload.probabilities.done > 0.75) {
  await sendTelegram("tree-book-shower-golden", "Your upload finished.", screen);
  await stopAgent();
} else if (decision.upload.probabilities.failed > 0.75 || decision.upload.left) {
  await sendTelegram("tree-book-shower-golden", "Your upload stopped: it failed or was cancelled.", screen);
  await stopAgent();
}`,
    cls: watcher({ target: 'dial', shape: 'event', cardinality: 'once', question: 'choice', now: 'uploading', channel: 'telegram' }),
  },
  {
    name: 'Studio Coach (pictures compared, every, paid on a 300 s loop)',
    model: 'clef-flash',
    interval: 300,
    prompt: 'Are these two images the same drawing, without a single new stroke?\n$SCREEN\n$IMEMORY',
    code: `const before = await getImageMemory();
if (before.length && decision.noul > 0.75) {
  await sendWhatsapp("tree-book-shower-golden", "No progress on your drawing in 5 minutes. Get back to it!", screen);
}
await setImageMemory(screen);`,
    cls: watcher({ target: 'image', shape: 'change', cardinality: 'every', question: 'noul', now: 'yes', channel: 'whatsapp' }),
  },
  {
    name: 'Bitcoin drop (extract VALUE)',
    model: 'gemma-3-12b-it',
    interval: 15,
    prompt: '1. **Description**: Read the Bitcoin price shown on this chart.\n2. **Decision**: Answer with only VALUE: <number>, or VALUE: NONE if you can\'t see it.\n$SCREEN',
    code: `const m = response.match(/VALUE:\\**\\s*\\$?\\s*([\\d.,]+)/);
if (m) {
  const v = parseFloat(m[1].replace(/,/g, "")), now = Date.now();
  const hist = JSON.parse((await getMemory()) || "[]").filter(([t]) => now - t < 5 * 60 * 1000);
  hist.push([now, v]);
  await setMemory(JSON.stringify(hist));
  const high = Math.max(...hist.map(([, x]) => x));
  if (v < high * 0.98) {
    notify("Bitcoin is dropping", "Down " + ((1 - v / high) * 100).toFixed(1) + "%: " + v);
    await stopAgent();
  }
}`,
    cls: { role: 'watcher', task: 'extract', sensor: 'screen', target: 'value', shape: 'change', cardinality: 'once', scope: 'region', channel: 'browser' },
  },
  {
    name: 'Door gate (decides, saves the frame, starts its scribe)',
    model: 'clef-flash',
    prompt: 'Is there a person in frame?\n$CAMERA',
    code: `if (decision.noul > 0.75 && !(await getMemory())) {
  await setImageMemory(camera);
  await setMemory("seen");
  await startAgent("door_scribe");
} else if (decision.noul < 0.25) {
  await setMemory("");
}`,
    cls: { role: 'gate', task: 'decide', sensor: 'camera', target: 'object', shape: 'event', cardinality: 'every', question: 'noul', now: 'no', scope: 'anywhere', scope_reason: 'a person can be anywhere in the doorway view', triggers: 'door_scribe' },
  },
  {
    name: 'Door scribe (describes what the gate saved, stops)',
    model: 'gemma-3-12b-it',
    prompt: 'Describe the person in this picture in one sentence.\n$IMEMORY@door_gate',
    code: `await appendMemory("[" + time() + "] " + response);
await stopAgent();`,
    cls: { role: 'scribe', task: 'describe', reads_images_from: 'door_gate' },
  },
  {
    name: 'Activity tracker (describe every iteration)',
    model: 'gemma-3-12b-it',
    interval: 60,
    prompt: 'Describe in one sentence what the user is working on.\n$SCREEN',
    code: 'await appendMemory("[" + time() + "] " + response);',
    cls: { role: 'watcher', task: 'describe', sensor: 'screen', cardinality: 'every', scope: 'anywhere', scope_reason: 'the work can be in any window' },
  },
  {
    name: 'Instagram lock (mobile, arms after the first no)',
    model: 'clef-flash',
    prompt: 'Is a photo feed with round story bubbles at the top on screen?\n$SCREEN',
    code: `if (!(await getMemory())) {
  if (decision.noul < 0.25) await setMemory("armed");
} else if (decision.noul > 0.75) {
  await lockPhone("You opened Instagram");
}`,
    cls: watcher({ target: 'state', shape: 'event', cardinality: 'every', question: 'noul', now: 'no', scope: 'anywhere', scope_reason: 'the whole phone screen is the app' }),
  },
  {
    name: 'Scout',
    model: 'clef-flash',
    prompt: 'Is a subscriber count on screen?\n$SCREEN',
    code: `if (decision.noul > 0.75 && !(await getMemory())) {
  await setMemory("woke");
  await mcp("Subscriber count on screen");
}`,
    cls: { role: 'scout', task: 'decide', sensor: 'screen', question: 'noul', now: 'no', scope: 'anywhere', scope_reason: 'waiting for the app to open' },
  },
  {
    name: 'Scratchpad',
    model: 'Skip Model Call',
    prompt: '$MEMORY',
    code: 'await unlockPhone();\nawait stopAgent();',
    cls: { role: 'scratchpad', task: 'describe' },
  },
  {
    name: 'JSON choice with $MEMORY in the state (E1)',
    model: 'onejev-local',
    prompt: `{"state": "Already logged: $MEMORY", "questions": {"who": {"type": "choice", "instructions": "Who is at the door?", "criteria": {"nobody": "the doorway is empty", "courier": "a courier with a package", "other": "someone else"}}}}
$CAMERA`,
    code: `if (decision.who.probabilities.courier > 0.75 && !(await getMemory()).includes("courier")) {
  await appendMemory("courier " + time());
  await sendTelegram("tree-book-shower-golden", "A courier is at the door", camera);
}`,
    cls: { role: 'watcher', task: 'decide', sensor: 'camera', target: 'object', shape: 'event', cardinality: 'every', question: 'choice', now: 'nobody', scope: 'anywhere', scope_reason: 'the whole doorway', channel: 'telegram' },
  },
];

for (const c of RECIPES) {
  const { errors, warnings } = run(c);
  if (errors.length) fail(`${c.name}: ${errors.map(e => `${e.id} ${e.message}`).join(' | ')}`);
  else pass(`${c.name}${warnings.length ? `  (warnings: ${warnings.map(w => w.id).join(', ')})` : ''}`);
}

// ---- Broken variants: each must fail with its rule -------------------------------

const [finish, pigeon, , nag, sequence, , , , gate] = RECIPES;
const variant = (base: Case, name: string, over: Partial<Case>): Case => ({ ...base, ...over, name, cls: { ...base.cls, ...(over.cls ?? {}) } });

const BROKEN: [string, Case][] = [
  ['C1', variant(nag, 'nag with cardinality once', { cls: { cardinality: 'once' } as AgentClassification })],
  ['S1', variant(finish, 'screen agent without $SCREEN', { prompt: 'Is there a download percentage below 100%?\n$CAMERA' })],
  ['M1', variant(finish, 'decide on an LLM', { model: 'gemma-3-12b-it' })],
  ['X1', { name: 'old LLM keyword pattern', model: 'gemma-3-12b-it', prompt: 'Describe the chat, then say NEW_MESSAGE or CONTINUE.\n$SCREEN',
    code: 'if (response.includes("NEW_MESSAGE")) { notify("Maria replied", response); await stopAgent(); }',
    cls: { role: 'watcher', task: 'describe', sensor: 'screen', cardinality: 'once', scope: 'region' } }],
  ['X2', variant(finish, 'decide reads response', { code: 'if (response.includes("no")) { await call("tree-book-shower-golden", "done"); await stopAgent(); }' })],
  ['Q1', variant(finish, 'JSON with $MEMORY outside a string', { prompt: '{"state": $MEMORY, "questions": {"a": {"type": "noul", "instructions": "Is it done?"}}}\n$SCREEN' })],
  ['Q2', variant(finish, 'one noul question as JSON', { prompt: '{"questions": {"done": {"type": "noul", "instructions": "Is there a download percentage below 100%?"}}}\n$SCREEN' })],
  ['Q3', variant(sequence, 'reads a question that does not exist', { code: sequence.code.replace('const seen = await getMemory();', 'const seen = await getMemory(); const x = decision.stagez;') })],
  ['T1', variant(finish, 'moved threshold', { code: finish.code.replace('< 0.25', '< 0.3') })],
  ['T2', variant(sequence, 'branches on .choice', { code: 'if (decision.stage.choice === "failed") { await stopAgent(); }' })],
  ['T4', variant(nag, 'nag that stops', { code: nag.code + '\nawait stopAgent();' })],
  ['K1', variant(pigeon, 'once without stopAgent', { code: 'if (decision.noul > 0.75) { notify("Maria replied", "New message"); }' })],
  ['K3', variant(finish, 'paid every-time on a fast loop', { cls: { cardinality: 'every' } as AgentClassification, code: 'if (decision.noul < 0.25 || decision.dropped) { await call("tree-book-shower-golden", "stopped"); await setMemory("x"); }' })],
  ['N1', variant(finish, 'channel voice delivered by notify', { code: 'if (decision.noul < 0.25 || decision.dropped) { notify("Done", "stopped"); await stopAgent(); }' })],
  ['N2', variant(finish, 'unawaited call', { code: 'if (decision.noul < 0.25 || decision.dropped) { call("tree-book-shower-golden", "stopped"); await stopAgent(); }' })],
  ['N5', variant(finish, 'made-up sound name', { code: 'if (decision.noul < 0.25 || decision.dropped) { sound("alarm"); await call("tree-book-shower-golden", "stopped"); await stopAgent(); }' })],
  ['N3', variant(sequence, '[screen]', { code: sequence.code.replace(', screen)', ', [screen])') })],
  ['G1', variant(gate, 'gate that never starts its scribe', { code: gate.code.replace('await startAgent("door_scribe");', '') })],
  ['P1', variant(finish, 'region agent started without a crop', { stage: 'start', crop: { screen: false, camera: false } })],
  ['G1', variant(gate, 'gate whose scribe does not exist', { stage: 'start', agents: {} })],
  ['V1', variant(finish, 'dial that reads low on iteration 1', { stage: 'verify', iteration: { decision: { type: 'noul', noul: 0.1 } as NoulAnswer } })],
  ['V2', variant(finish, 'once agent that acted on iteration 1', { stage: 'verify', iteration: { decision: { type: 'noul', noul: 0.97 } as NoulAnswer, tools: [{ name: 'stopAgent', status: 'success', timestamp: '' }] } })],
];

for (const [id, c] of BROKEN) {
  const { errors, warnings } = run(c);
  const hit = [...errors, ...warnings].filter(v => v.id === id);
  if (hit.length) pass(`${id} catches: ${c.name}`);
  else fail(`${id} should catch "${c.name}"; got ${[...errors, ...warnings].map(v => v.id).join(', ') || 'nothing'}`);
}

// Sound names that exist (N5 must stay quiet).
for (const call of ['sound()', 'sound("ping")', "sound('bell', 0.8)", 'sound("https://example.com/a.wav")', 'sound(name)', 'sound(`${x}`)']) {
  const c = variant(finish, call, { code: `if (decision.noul < 0.25 || decision.dropped) { ${call}; await call("tree-book-shower-golden", "stopped"); await stopAgent(); }` });
  const hit = run(c).errors.filter(v => v.id === 'N5');
  hit.length ? fail(`N5 should allow ${call}: ${hit[0].message}`) : pass(`N5 allows ${call}`);
}

// ---- Engine: run trends on every answer (E2) -------------------------------------

resetNoulTrend('t');
let last: Decision | undefined;
for (const n of [0.95, 0.94, 0.96, 0.6, 0.6, 0.6]) last = withNoulTrend('t', { a: { type: 'noul', noul: n } });
const jsonNoul = (last as Record<string, NoulAnswer>).a;
jsonNoul.dropped && jsonNoul.start === 0.95 ? pass('E2 JSON noul gets start + dropped') : fail(`E2 JSON noul: ${JSON.stringify(jsonNoul)}`);

resetNoulTrend('t');
const choice = (running: number) => ({ s: { type: 'choice', choice: running > 0.5 ? 'running' : 'done', probabilities: { running, done: 1 - running }, confidence: 0 } as ChoiceAnswer });
for (const p of [0.9, 0.92, 0.88, 0.55, 0.55, 0.55]) last = withNoulTrend('t', choice(p));
const left = (last as Record<string, ChoiceAnswer>).s;
left.left && left.start === 'running' ? pass('E2 choice gets start + left (even while running still wins)') : fail(`E2 choice: ${JSON.stringify(left)}`);

resetNoulTrend('t');
for (const n of [0.95, 0.95, 0.95, 0.9]) last = withNoulTrend('t', { type: 'noul', noul: n } as NoulAnswer);
!(last as NoulAnswer).dropped ? pass('E2 plain noul unchanged: a small dip is not dropped') : fail('E2 plain noul dropped too early');

if (failures) { console.log(`\n${failures} failure(s)`); process.exit(1); }
console.log('\nall checks passed');
