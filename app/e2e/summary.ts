// Prints the run's verdicts and pass rate once the suite finishes (Playwright globalTeardown).

import fs from 'node:fs';
import { E2E_DIR } from './fixtures';

export default function summary() {
  patternSummary();
  const file = `${E2E_DIR}results/${process.env.E2E_RUN_ID}/results.jsonl`;
  if (!fs.existsSync(file)) return;
  const results = fs.readFileSync(file, 'utf8').trim().split('\n').map(line => JSON.parse(line));
  const passed = results.filter(r => r.verdict === 'PASS').length;
  console.log(`\nMCP e2e: ${passed}/${results.length} passed (${Math.round(100 * passed / results.length)}%)`);
  for (const r of results) {
    const latency = r.latencyMs !== undefined ? ` +${Math.round(r.latencyMs / 1000)}s` : '';
    console.log(`  ${r.verdict.padEnd(16)} ${r.id}${latency}`);
  }
  console.log(`Results: ${file}\n`);
}

function patternSummary() {
  const file = `${E2E_DIR}results/${process.env.E2E_RUN_ID}/patterns.jsonl`;
  if (!fs.existsSync(file)) return;
  const results = fs.readFileSync(file, 'utf8').trim().split('\n').map(line => JSON.parse(line));
  const passed = results.filter(r => r.verdict === 'PASS').length;
  console.log(`\nPattern e2e: ${passed}/${results.length} passed`);
  for (const r of results) console.log(`  ${r.verdict.padEnd(14)} ${r.id}${r.notes?.length ? ` (${r.notes.join('; ')})` : ''}`);
  console.log(`Results: ${file}\n`);
}
