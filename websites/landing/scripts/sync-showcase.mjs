// Builds the landing page's When / Then wheels from the ground truth (app/src/utils/groundTruth.ts),
// the same list the MCP e2e suite tests: one pair per demo, the row marked `show`, running against
// the same demo page, copied here from app/e2e/demos/.
//
// Writes (both gitignored): src/generated/showcase.json and public/demos/.
// Fails if a demo has no shown pair or more than one, or a shown pair is missing what the site needs.

import { buildSync } from 'esbuild';
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const landing = resolve(here, '..');
const app = resolve(landing, '../../app');
const demos = join(app, 'e2e/demos');

const outfile = join(tmpdir(), `observer-ground-truth-${process.pid}.mjs`);
buildSync({ entryPoints: [join(app, 'src/utils/groundTruth.ts')], bundle: true, format: 'esm', platform: 'node', outfile, logLevel: 'error' });
const { GROUND_TRUTH, CHANNELS, NOTIFY } = await import(pathToFileURL(outfile).href);
rmSync(outfile, { force: true });

const file = u => u.demo.split('?')[0];
const problems = [];
for (const f of new Set(GROUND_TRUTH.map(file))) {
  const shown = GROUND_TRUTH.filter(u => file(u) === f && u.show);
  if (shown.length !== 1) problems.push(`demo ${f} has ${shown.length} use cases with \`show\` (needs exactly one)`);
  if (!existsSync(join(demos, f))) problems.push(`demo ${f} isn't in app/e2e/demos/`);
}
// Opened without ?outcome= (here and in the app's tutorial), a demo ends by its data-outcome on
// demo-contract.js, else success: that ending has to be one its shown pair alerts on.
const JOB = { success: 'success', fail: 'fail' };
for (const u of GROUND_TRUTH.filter(u => u.show)) {
  if (u.then === NOTIFY && !CHANNELS[u.show.channel]) problems.push(`${u.id} is an alert but its show has no channel`);
  const [f, query = ''] = u.demo.split('?');
  if (!u.expect || new URLSearchParams(query).has('outcome') || !existsSync(join(demos, f))) continue;
  const html = readFileSync(join(demos, f), 'utf8');
  const param = html.match(/<script[^>]*demo-contract\.js[^>]*data-outcome="([^"]+)"/)?.[1] ?? 'success';
  const name = Object.entries(u.outcomes ?? JOB).find(([, p]) => p === param)?.[0];
  if (u.expect[name] !== 'notify') problems.push(`${f} ends by default as "${param}", but ${u.id} doesn't alert on it: set data-outcome on its demo-contract.js tag`);
}
if (problems.length) {
  console.error(problems.map(p => `  ✗ ${p}`).join('\n'));
  process.exit(1);
}

// The suite asks for a browser notification; the site shows the channel's phrase instead.
const items = GROUND_TRUTH
  .filter(u => u.show && !u.requires?.some(f => !existsSync(join(demos, f)))) // recordings not added yet
  .map(u => {
    const [f, query = ''] = u.demo.split('?');
    const channel = u.then === NOTIFY ? { id: u.show.channel, label: CHANNELS[u.show.channel].label } : null;
    return { id: u.id, file: f, query, title: u.show.title, when: u.when, then: channel ? CHANNELS[u.show.channel].phrase : u.then, channel, alert: u.show.alert };
  });
mkdirSync(join(landing, 'src/generated'), { recursive: true });
writeFileSync(join(landing, 'src/generated/showcase.json'), JSON.stringify({ items }, null, 2));

const dest = join(landing, 'public/demos');
rmSync(dest, { recursive: true, force: true });
cpSync(demos, dest, { recursive: true, filter: src => !src.endsWith('.DS_Store') && !src.endsWith('.py') });

console.log(`showcase: ${items.length} When / Then pairs`);
