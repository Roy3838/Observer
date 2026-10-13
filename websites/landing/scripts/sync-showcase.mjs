// Builds the landing page's use-case showcase from the MCP e2e suite, so app/e2e stays the
// single source of truth: prompts come from scenarios.ts / patterns.ts, presentation from
// showcase.ts, and the demo pages themselves are copied from app/e2e/demos/.
//
// Writes (both gitignored): src/generated/showcase.json and public/demos/.
// Fails if a demo used by a scenario has no showcase entry, or the reverse.

import { buildSync } from 'esbuild';
import { cpSync, existsSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const landing = resolve(here, '..');
const e2e = resolve(landing, '../../app/e2e');

const scratch = join(tmpdir(), `observer-showcase-${process.pid}`);
mkdirSync(scratch, { recursive: true });
async function load(name) {
  const outfile = join(scratch, `${name}.mjs`);
  buildSync({ entryPoints: [join(e2e, `${name}.ts`)], bundle: true, format: 'esm', platform: 'node', outfile, logLevel: 'error' });
  return import(pathToFileURL(outfile).href);
}
const { DEMOS } = await load('scenarios');
const { PATTERNS } = await load('patterns');
const { SHOWCASE, CATEGORIES, CHANNELS } = await load('showcase');
rmSync(scratch, { recursive: true, force: true });

const useCases = new Map(); // demo file -> { prompts: [] }
const add = (demoWithQuery, prompt, kind, notifyOn) => {
  const [file, query = ''] = demoWithQuery.split('?');
  if (!useCases.has(file)) useCases.set(file, { prompts: [] });
  const list = useCases.get(file).prompts;
  if (!list.some(p => p.prompt === prompt)) list.push({ prompt, kind, query, notifyOn });
};

// notifyOn: the demo's `?outcome=` values the prompt must notify on.
for (const d of DEMOS) for (const p of d.prompts) {
  const outcomes = d.outcomes ?? { success: 'success', fail: 'fail' };
  add(d.demo, p.prompt, 'alert', Object.keys(p.expect).filter(o => p.expect[o] === 'notify').map(o => outcomes[o]));
}
for (const s of PATTERNS) {
  if (s.noAgent || s.demo.startsWith('blank')) continue;
  if (s.requires && !existsSync(join(e2e, 'demos', s.requires))) continue; // media not recorded yet
  add(s.demo, s.prompt, 'pattern', []);
}

const problems = [];
for (const file of useCases.keys()) if (!SHOWCASE[file]) problems.push(`demo ${file} is in the e2e scenarios but has no entry in app/e2e/showcase.ts`);
for (const file of Object.keys(SHOWCASE)) {
  if (!existsSync(join(e2e, 'demos', file))) problems.push(`showcase entry ${file} has no file in app/e2e/demos/`);
}
if (problems.length) {
  console.error(problems.map(p => `  ✗ ${p}`).join('\n'));
  process.exit(1);
}

// The e2e prompts all ask for a browser notification; show the demo's delivery channel instead.
const withChannel = (prompt, channel) => {
  if (!channel) return prompt;
  const { phrase } = CHANNELS[channel];
  return prompt.replace(/^send me a browser notification/i, m => (phrase[0].toUpperCase() + phrase.slice(1)))
    .replace(/send me a browser notification/gi, phrase)
    .replace(/another one/, 'another');
};
// Listed in showcase.ts order, not scenario order: the site's order is presentation.
const order = Object.keys(SHOWCASE);
const items = [...useCases].sort(([a], [b]) => order.indexOf(a) - order.indexOf(b)).map(([file, { prompts }]) => {
  const entry = SHOWCASE[file];
  const channel = entry.channel && { id: entry.channel, label: CHANNELS[entry.channel].label };
  return { id: file.replace(/\.html$/, ''), file, ...entry, channel, prompts: prompts.map(p => ({ ...p, prompt: withChannel(p.prompt, entry.channel) })) };
});
mkdirSync(join(landing, 'src/generated'), { recursive: true });
writeFileSync(join(landing, 'src/generated/showcase.json'), JSON.stringify({ categories: CATEGORIES, items }, null, 2));

const dest = join(landing, 'public/demos');
rmSync(dest, { recursive: true, force: true });
cpSync(join(e2e, 'demos'), dest, { recursive: true, filter: src => !src.endsWith('.DS_Store') && !src.endsWith('.py') });

console.log(`showcase: ${items.length} use cases, ${items.reduce((n, i) => n + i.prompts.length, 0)} prompts`);
