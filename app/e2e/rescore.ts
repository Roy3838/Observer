// Rescores a pattern run from its saved files, with the current patterns.ts, so runs scored by
// older scoring can be compared: npx tsx e2e/rescore.ts e2e/results/<run>
// Writes <run>/patterns.rescored.jsonl next to patterns.jsonl.

import fs from 'node:fs';
import { PATTERNS, type Run } from './patterns';

const dir = process.argv[2];
const original = fs.readFileSync(`${dir}/patterns.jsonl`, 'utf8').trim().split('\n').map(l => JSON.parse(l));
const out: string[] = [];
for (const r of original) {
  const s = PATTERNS.find(p => p.id === r.id);
  const base = `${dir}/${r.id.replace(/[^\w-]+/g, '_')}`;
  if (!s || !r.metrics || !fs.existsSync(`${base}.events.json`)) { out.push(JSON.stringify(r)); continue; }
  const saved = JSON.parse(fs.readFileSync(`${base}.events.json`, 'utf8'));
  const agents = JSON.parse(fs.readFileSync(`${base}.agents.json`, 'utf8'));
  const transcript = JSON.parse(fs.readFileSync(`${base}.transcript.json`, 'utf8'));
  // Saved events carry times relative to the demo's start; rebuild a clock from them.
  const events = saved.map((e: any) => ({ ...e, t: e.sinceStartMs ?? 0 }));
  const truth = JSON.parse(fs.readFileSync(`${base}.truth.json`, 'utf8'));
  const run: Run = {
    events, startT: 0, triggerT: events.find((e: any) => e.type === 'trigger')?.t ?? null, truth,
    memories: fs.existsSync(`${base}.memories.json`)
      ? JSON.parse(fs.readFileSync(`${base}.memories.json`, 'utf8'))
      : agents.map((a: any) => ({ id: a.id, memory: a.memory ?? '' })),
    agents,
    toolCalls: (transcript?.messages ?? []).flatMap((m: any) => m.toolCalls ?? []), asked: r.asked,
  };
  const score = s.score(run);
  if (score.verdict !== r.verdict) console.log(`${r.id}: ${r.verdict} → ${score.verdict}`);
  out.push(JSON.stringify({ ...r, verdict: score.verdict, metrics: score.metrics, notes: score.notes }));
}
fs.writeFileSync(`${dir}/patterns.rescored.jsonl`, out.join('\n') + '\n');
