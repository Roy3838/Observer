// src/mcp/ontology/rules.ts
//
// The rule table: what an agent's prompt, code and model must look like for its classification.
// Each rule is data (id, stage, severity, when it applies, the check), so the table can be read,
// tested and later rendered into the system prompt. The checks read the code as text (regex,
// comments stripped): the system prompt dictates exact shapes, so narrow patterns are enough.
//
// Stages: 'save' runs in create_agent / edit_agent, 'start' in start_agent, 'verify' on the
// iterations get_iteration returns.

import type { IterationData } from '@utils/IterationStore';
import type { Question } from '@utils/localLlm/systemOne';
import type { AgentClassification } from './schema';
import { classificationProblems, noulTrigger, type Strategy } from './derive';

export type Stage = 'save' | 'start' | 'verify';
export type Severity = 'error' | 'warning';

export interface PromptAnalysis {
  /** Set when the decision prompt fails the dry-run parse (rule Q1). */
  error?: string;
  single?: boolean;
  questions?: Record<string, Question>;
}

export interface RuleContext {
  agent: { id: string; model_name: string; system_prompt: string; loop_interval_seconds?: number };
  code: string;
  cls: AgentClassification;
  strategy: Strategy;
  /** undefined when the model isn't in list_models. */
  model?: { name: string; decision: boolean; local: boolean };
  /** decide agents only. */
  prompt: PromptAnalysis;
  /** start stage. */
  crop?: { screen: boolean; camera: boolean };
  /** start stage: another agent's classification, null when it doesn't exist. */
  findAgent?: (id: string) => { classification?: AgentClassification } | null;
  /** verify stage. */
  iteration?: IterationData;
  /** Names sound() accepts (bundled + aliases). Unset: sound names aren't checked (rule N5). */
  sounds?: string[];
}

export interface Rule {
  id: string;
  stage: Stage;
  severity: Severity;
  /** Also runs on role 'custom', which otherwise skips the rules. */
  onCustom?: boolean;
  applies: (ctx: RuleContext) => boolean;
  /** null / [] = ok; otherwise what's wrong and how to fix it. */
  check: (ctx: RuleContext) => string | string[] | null;
}

export interface Violation { id: string; message: string }
export interface CheckResult { errors: Violation[]; warnings: Violation[] }

/** Run every rule of `stage` that applies. A custom agent only gets the rules marked onCustom at save. */
export function runRules(stage: Stage, ctx: RuleContext): CheckResult {
  const result: CheckResult = { errors: [], warnings: [] };
  for (const rule of RULES) {
    if (rule.stage !== stage) continue;
    if (ctx.cls.role === 'custom' && !rule.onCustom && stage === 'save') continue;
    if (!rule.applies(ctx)) continue;
    const found = rule.check(ctx);
    const messages = found === null ? [] : Array.isArray(found) ? found : [found];
    for (const message of messages) {
      (rule.severity === 'error' ? result.errors : result.warnings).push({ id: rule.id, message });
    }
  }
  return result;
}

// ---- Reading code and prompts ------------------------------------------------

/** Code without comments, so a commented-out call doesn't count. */
export function stripComments(code: string): string {
  return code.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:\\'"`])\/\/.*$/gm, '$1');
}
const squash = (s: string) => s.replace(/\s+/g, '');
const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** A call to `fn(` that isn't a method of something else (`call(` but not `x.call(`). */
const callRe = (fn: string, flags = '') => new RegExp(`(?<![\\w.$])${esc(fn)}\\s*\\(`, flags);
const calls = (code: string, fn: string) => callRe(fn).test(code);

/** Every literal `sleep(ms)` duration in the code, in ms. */
const sleepsMs = (code: string) => [...code.matchAll(/(?<![\w.$])sleep\s*\(\s*([\d_]+)/g)].map(m => Number(m[1].replace(/_/g, '')));

const has = (prompt: string, sensor: string) => new RegExp(`\\$${sensor}(?![A-Z_])`).test(prompt);

/** The argument text of every `fn(...)` call, parentheses balanced. */
function callArgs(code: string, fn: string): string[] {
  const out: string[] = [];
  for (const m of code.matchAll(callRe(fn, 'g'))) {
    let depth = 1, i = m.index! + m[0].length;
    const start = i;
    while (i < code.length && depth > 0) {
      if (code[i] === '(') depth++;
      else if (code[i] === ')') depth--;
      i++;
    }
    out.push(code.slice(start, i - 1));
  }
  return out;
}

const NETWORK_SENDS = ['sendEmail', 'sendTelegram', 'sendWhatsapp', 'sendSms', 'call', 'sendDiscord', 'sendPushover'];
const IMAGE_SENDS = NETWORK_SENDS.filter(fn => fn !== 'call');
const PAID_SENDS = ['call', 'sendSms', 'sendWhatsapp'];
const ALL_SENDS = [...NETWORK_SENDS, 'notify', 'system_notify'];

const CHANNEL_FN: Record<string, string> = {
  browser: 'notify', system: 'system_notify', email: 'sendEmail', whatsapp: 'sendWhatsapp', sms: 'sendSms',
  voice: 'call', telegram: 'sendTelegram', discord: 'sendDiscord', pushover: 'sendPushover',
};

/** Every `decision[.id].noul <op> <n>` comparison, read from whitespace-free code. */
function noulComparisons(code: string): { id?: string; op: string; value: number; text: string }[] {
  return [...squash(code).matchAll(/decision(?:\.(\w+))?\.noul(<=?|>=?)(\d*\.?\d+)/g)]
    .map(m => ({ id: m[1], op: m[2], value: Number(m[3]), text: m[0] }));
}

const canonicalNoul = (op: string, value: number) => (op === '<' && value === 0.25) || (op === '>' && value === 0.75);

/** The questions of a given kind in the analysed prompt. */
function questionsOf(ctx: RuleContext, kind: Question['type']): [string, Question][] {
  return Object.entries(ctx.prompt.questions ?? {}).filter(([, q]) => q.type === kind);
}

const labelsOf = (q: Question): string[] =>
  q.type === 'noul' ? ['yes', 'no'] : q.type === 'choice' ? Object.keys(q.criteria) : q.criteria.map((_, i) => String(i));

const isRole = (ctx: RuleContext, ...roles: AgentClassification['role'][]) => roles.includes(ctx.cls.role);
const decides = (ctx: RuleContext) => ctx.cls.task === 'decide' && ctx.strategy === 'decision';
const promptOk = (ctx: RuleContext) => decides(ctx) && !ctx.prompt.error && !!ctx.prompt.questions;

// ---- The table ---------------------------------------------------------------

export const RULES: Rule[] = [
  // Classification
  {
    id: 'C1', stage: 'save', severity: 'error', onCustom: true,
    applies: () => true,
    check: ctx => classificationProblems(ctx.cls),
  },
  {
    id: 'S1', stage: 'save', severity: 'error',
    applies: ctx => isRole(ctx, 'watcher', 'gate', 'scout') && !!ctx.cls.sensor,
    check: ctx => {
      const sensor = ctx.cls.sensor === 'screen' ? 'SCREEN' : 'CAMERA';
      return has(ctx.agent.system_prompt, sensor) ? null : `sensor ${ctx.cls.sensor} needs $${sensor} in the system_prompt.`;
    },
  },

  // Model / task
  {
    id: 'M0', stage: 'save', severity: 'warning',
    applies: ctx => ctx.strategy !== 'none' && !ctx.model,
    check: ctx => `model '${ctx.agent.model_name}' isn't in list_models; the rules couldn't check it's the right kind.`,
  },
  {
    id: 'M1', stage: 'save', severity: 'error',
    applies: ctx => ctx.strategy !== 'none' && !!ctx.model,
    check: ctx => {
      if (ctx.strategy === 'decision' && !ctx.model!.decision) {
        return `task ${ctx.cls.task} is a decision: model_name must be a decision model (clef-flash, or the local model download_model model='onejev' returned), not '${ctx.model!.name}'.`;
      }
      if (ctx.strategy === 'llm' && (ctx.model!.decision || ctx.model!.name === 'Skip Model Call')) {
        return `task ${ctx.cls.task} needs an LLM to write text: model_name must be gemma-3-12b-it or the local model download_model model='default' returned, not '${ctx.model!.name}'.`;
      }
      return null;
    },
  },
  {
    id: 'M2', stage: 'save', severity: 'warning',
    applies: ctx => ctx.strategy !== 'none' && !!ctx.model && !ctx.model.local,
    check: ctx => ['clef-flash', 'gemma-3-12b-it'].includes(ctx.model!.name)
      ? null : `cloud model '${ctx.model!.name}': use clef-flash (decide) or gemma-3-12b-it (describe / extract).`,
  },

  // The core invariant: LLM text never drives control flow
  {
    id: 'X1', stage: 'save', severity: 'error', onCustom: true,
    applies: ctx => ctx.cls.task === 'describe' || ctx.cls.task === 'extract',
    check: ctx => {
      const code = stripComments(ctx.code);
      const branching = [
        /\bresponse\s*\.\s*(includes|startsWith|endsWith|indexOf|search|localeCompare)\s*\(/,
        /\bresponse\s*\.\s*to(Lower|Upper)Case\s*\(\s*\)\s*\.\s*(includes|startsWith|endsWith|indexOf)\s*\(/,
        /\.test\s*\(\s*response\b/,
        /\bswitch\s*\(\s*response\b/,
        /\bresponse\s*(===?|!==?)/,
        /\bif\s*\(\s*!?\s*response\s*\)/,
      ];
      return branching.some(re => re.test(code))
        ? `task ${ctx.cls.task}: the LLM's text never decides anything. Don't branch on response (includes / ===, ...); ${ctx.cls.task === 'extract' ? 'parse it with response.match(/VALUE:.../) and compare the number in code' : 'put it in memory or a message as it is'}, and let a decision model (a gate) decide.`
        : null;
    },
  },
  {
    id: 'X2', stage: 'save', severity: 'error', onCustom: true,
    applies: ctx => ctx.strategy !== 'none',
    check: ctx => {
      const code = stripComments(ctx.code);
      if (ctx.strategy === 'decision' && /(?<![\w.$'"`])response(?=\s*(\.|\)|;|,|\?|&&|\|\||[=!]==?|\+|\]))/.test(code)) {
        return 'a decision model writes no text: response is null. Read decision instead.';
      }
      if (ctx.strategy === 'llm' && /(?<![\w.$'"`])decision\s*\./.test(code)) {
        return 'an LLM agent has no decision: read response.';
      }
      return null;
    },
  },
  {
    id: 'X3', stage: 'save', severity: 'error',
    applies: ctx => ctx.cls.task === 'describe' && ctx.strategy === 'llm',
    check: ctx => {
      const code = stripComments(ctx.code);
      const sinks = ['appendMemory', 'setMemory', ...ALL_SENDS];
      return sinks.some(fn => callArgs(code, fn).some(a => /\bresponse\b/.test(a)))
        ? null : 'task describe: the description has to go somewhere. Pass response to appendMemory(...) or a message.';
    },
  },

  // Decision prompts
  {
    id: 'Q1', stage: 'save', severity: 'error',
    applies: decides,
    check: ctx => ctx.prompt.error
      ? `the decision prompt won't parse once sensors fill in: ${ctx.prompt.error}. Write one yes/no question as plain text, or a JSON System One request; text sensors ($MEMORY) go inside a JSON string, $SCREEN / $CAMERA / $IMEMORY after the closing }.`
      : null,
  },
  {
    id: 'Q2', stage: 'save', severity: 'error',
    applies: promptOk,
    check: ctx => {
      const qs = Object.values(ctx.prompt.questions!);
      const out: string[] = [];
      if (!ctx.prompt.single && qs.length === 1 && qs[0].type === 'noul') {
        out.push('one yes/no question goes in a plain-text prompt, not JSON (that keeps decision.dropped). Put any context into the question itself.');
      }
      if (ctx.cls.question && !qs.some(q => q.type === ctx.cls.question)) {
        out.push(`classification.question is ${ctx.cls.question} but the prompt has no ${ctx.cls.question} question.`);
      }
      return out;
    },
  },
  {
    id: 'Q3', stage: 'save', severity: 'error',
    applies: promptOk,
    check: ctx => {
      const code = stripComments(ctx.code);
      const questions = ctx.prompt.questions!;
      const out: string[] = [];
      if (!ctx.prompt.single) {
        const ids = new Set([...code.matchAll(/(?<![\w.$])decision\s*\.\s*(\w+)/g)].map(m => m[1]));
        for (const id of ids) if (!questions[id]) out.push(`the code reads decision.${id} but the prompt has no question '${id}' (questions: ${Object.keys(questions).join(', ')}).`);
      }
      const choiceLabels = new Set(questionsOf(ctx, 'choice').flatMap(([, q]) => labelsOf(q)));
      const used = [...squash(code).matchAll(/probabilities(?:\.(\w+)|\[["'](\w+)["']\])/g)].map(m => m[1] ?? m[2]);
      for (const label of new Set(used)) if (!choiceLabels.has(label)) out.push(`the code reads probabilities.${label} but no choice question has the label '${label}'.`);
      if (ctx.cls.now && ctx.cls.question && ctx.cls.question !== 'noul') {
        const labels = questionsOf(ctx, ctx.cls.question).flatMap(([, q]) => labelsOf(q));
        if (!labels.includes(ctx.cls.now)) out.push(`classification.now '${ctx.cls.now}' isn't a label of the ${ctx.cls.question} question (${labels.join(', ')}).`);
      }
      return out;
    },
  },
  {
    id: 'Q4', stage: 'save', severity: 'warning',
    applies: ctx => promptOk(ctx) && !!ctx.model?.local,
    check: ctx => Object.keys(ctx.prompt.questions!).length > 2
      ? 'every question is a full pass over the images on a local model: more than 2 make each iteration slow. Prefer one choice question.' : null,
  },
  {
    id: 'Q5', stage: 'save', severity: 'error',
    applies: decides,
    check: ctx => /<image:\d+>/.test(ctx.agent.system_prompt) && has(ctx.agent.system_prompt, 'IMEMORY')
      ? '<image:N> markers can\'t be used with $IMEMORY: its image count changes. Drop the markers (images go at the top of the state, screen → camera → imemory).' : null,
  },

  // Triggers
  {
    id: 'T1', stage: 'save', severity: 'error',
    applies: ctx => promptOk(ctx) && ctx.cls.question === 'noul' && ctx.cls.target !== 'image' && isRole(ctx, 'watcher', 'gate', 'scout'),
    check: ctx => {
      const comparisons = noulComparisons(stripComments(ctx.code));
      const out: string[] = [];
      for (const c of comparisons) {
        if (!canonicalNoul(c.op, c.value)) out.push(`noul thresholds are fixed: < 0.25 or > 0.75. Found \`${c.text}\`.`);
      }
      const ids = ctx.prompt.single ? [undefined] : questionsOf(ctx, 'noul').map(([id]) => id);
      const required = ids.map(id => noulTrigger(ctx.cls.now, id)).filter((t): t is string => !!t);
      if (required.length && !required.some(t => squash(stripComments(ctx.code)).includes(squash(t)))) {
        out.push(`the answer is '${ctx.cls.now}' now, so the agent acts when it leaves: trigger on exactly \`${required[0]}\`.`);
      }
      return out;
    },
  },
  {
    id: 'T2', stage: 'save', severity: 'error',
    applies: ctx => promptOk(ctx) && ctx.cls.question === 'choice',
    check: ctx => {
      const code = squash(stripComments(ctx.code));
      const out: string[] = [];
      if (/\.choice[=!]==?/.test(code)) out.push('don\'t branch on .choice (it wins even at 40%): act on `decision.<id>.probabilities.<label> > 0.75`, or `.left`.');
      for (const m of code.matchAll(/probabilities(?:\.\w+|\[["']\w+["']\])(<=?|>=?)(\d*\.?\d+)/g)) {
        if (!(m[1] === '>' && Number(m[2]) === 0.75)) out.push(`choice thresholds are fixed: probabilities.<label> > 0.75. Found \`${m[0]}\`.`);
      }
      if (!/probabilities(?:\.\w+|\[["']\w+["']\])>0\.75/.test(code) && !/\.left\b/.test(code)) {
        out.push('act on `decision.<id>.probabilities.<label> > 0.75` (or `.left`, when the answer leaves where it started).');
      }
      return out;
    },
  },
  {
    id: 'T3', stage: 'save', severity: 'error',
    applies: ctx => ctx.cls.shape === 'deadline',
    check: ctx => {
      const code = stripComments(ctx.code);
      return /Date\.now\s*\(/.test(code) && calls(code, 'setMemory')
        ? null : 'a deadline keeps its clock in memory: start it with Date.now() and setMemory on the first iteration.';
    },
  },
  {
    id: 'T4', stage: 'save', severity: 'error',
    applies: ctx => ctx.cls.shape === 'nag',
    check: ctx => {
      const code = stripComments(ctx.code);
      const out: string[] = [];
      if (calls(code, 'stopAgent')) out.push('a nag never stops itself: remove stopAgent().');
      if (!/Date\.now\s*\(/.test(code)) out.push('a nag throttles its alerts: keep the last alert time (Date.now()) in memory.');
      return out;
    },
  },
  {
    id: 'T5', stage: 'save', severity: 'error',
    applies: ctx => promptOk(ctx) && ctx.cls.shape === 'sequence',
    check: ctx => {
      const out: string[] = [];
      if (!questionsOf(ctx, 'choice').some(([, q]) => labelsOf(q).length >= 3)) {
        out.push('a sequence needs a choice question with a label per stage plus failure (e.g. running, passed, failed).');
      }
      const code = stripComments(ctx.code);
      if (!calls(code, 'setMemory') && !calls(code, 'appendMemory')) out.push('a sequence remembers the first stage: store it with setMemory.');
      return out;
    },
  },
  {
    id: 'T6', stage: 'save', severity: 'error',
    applies: ctx => ctx.cls.target === 'image',
    check: ctx => {
      const code = stripComments(ctx.code);
      const out: string[] = [];
      if (!has(ctx.agent.system_prompt, 'IMEMORY')) out.push('comparing pictures needs $IMEMORY in the prompt, after $SCREEN / $CAMERA.');
      if (!/(?<![\w.$])setImageMemory\s*\(\s*(screen|camera)\s*\)/.test(code)) out.push('save the frame every iteration: await setImageMemory(screen) (or camera).');
      if (!calls(code, 'getImageMemory')) out.push('the first iteration has nothing to compare: guard on (await getImageMemory()).length.');
      for (const c of noulComparisons(code)) {
        if (!canonicalNoul(c.op, c.value)) out.push(`noul thresholds are fixed: > 0.75 (no change) or < 0.25 (changed). Found \`${c.text}\`.`);
      }
      return out;
    },
  },
  {
    id: 'T7', stage: 'save', severity: 'error',
    applies: ctx => ctx.cls.target === 'value',
    check: ctx => {
      const code = stripComments(ctx.code);
      const out: string[] = [];
      if (!/VALUE:/.test(ctx.agent.system_prompt)) out.push('ask the LLM for the number alone: "VALUE: <number>".');
      if (!/response\s*\.\s*match\s*\(/.test(code)) out.push('parse the number with response.match(/VALUE:.../).');
      if (!calls(code, 'setMemory')) out.push('keep the history of values in memory (setMemory with JSON).');
      return out;
    },
  },
  {
    id: 'T8', stage: 'save', severity: 'warning',
    applies: ctx => ['deadline', 'change', 'nag'].includes(ctx.cls.shape ?? '') && ctx.cls.target !== 'image',
    check: ctx => /\$MEMORY(?![A-Z_])/.test(ctx.agent.system_prompt)
      ? 'timers and histories in memory shouldn\'t reach the model: remove $MEMORY from the prompt.' : null,
  },

  // Cardinality & delivery
  {
    id: 'K1', stage: 'save', severity: 'error',
    applies: ctx => isRole(ctx, 'watcher', 'gate') && ctx.cls.cardinality === 'once',
    check: ctx => calls(stripComments(ctx.code), 'stopAgent') ? null : 'cardinality once: await stopAgent() after acting.',
  },
  {
    id: 'K2', stage: 'save', severity: 'error',
    applies: ctx => isRole(ctx, 'watcher', 'gate') && ctx.cls.cardinality === 'every',
    check: ctx => {
      const code = stripComments(ctx.code);
      const out: string[] = [];
      if (calls(code, 'stopAgent')) out.push('cardinality every keeps running: remove stopAgent().');
      if (ctx.cls.role === 'watcher' && ctx.cls.shape === 'event' && ctx.cls.task === 'decide' && !calls(code, 'appendMemory') && !calls(code, 'setMemory')) {
        out.push('cardinality every: note what you already acted on in memory (appendMemory / setMemory) so it isn\'t reported twice.');
      }
      return out;
    },
  },
  {
    id: 'K3', stage: 'save', severity: 'error',
    // A loop of a minute or more already spaces the sends out.
    applies: ctx => ctx.cls.cardinality === 'every' && (ctx.agent.loop_interval_seconds ?? 30) < 60
      && PAID_SENDS.some(fn => calls(stripComments(ctx.code), fn)),
    check: ctx => {
      return sleepsMs(stripComments(ctx.code)).some(ms => ms >= 60000) ? null : 'call / sendSms / sendWhatsapp cost money: await sleep(60000) after them on an every-time agent.';
    },
  },
  {
    id: 'N1', stage: 'save', severity: 'error',
    applies: ctx => !!ctx.cls.channel && isRole(ctx, 'watcher', 'gate'),
    check: ctx => {
      const fn = CHANNEL_FN[ctx.cls.channel!];
      return calls(stripComments(ctx.code), fn) ? null : `channel ${ctx.cls.channel}: the code must deliver with ${fn}(...), the \`use\` ask_user_info returned.`;
    },
  },
  {
    id: 'N2', stage: 'save', severity: 'error', onCustom: true,
    applies: () => true,
    check: ctx => {
      const code = stripComments(ctx.code);
      const out: string[] = [];
      for (const fn of NETWORK_SENDS) {
        for (const m of code.matchAll(callRe(fn, 'g'))) {
          if (!/await\s*$/.test(code.slice(0, m.index))) { out.push(`await every ${fn}(...): an unawaited send can be cut off by stopAgent().`); break; }
        }
      }
      for (const m of code.matchAll(/(?<![\w.$])stopAgent\s*\(\s*\)\s*;?([^}]*)/g)) {
        if (ALL_SENDS.some(fn => calls(m[1], fn))) { out.push('nothing runs after stopAgent(): send first, then stop.'); break; }
      }
      return out;
    },
  },
  {
    id: 'N3', stage: 'save', severity: 'error', onCustom: true,
    applies: () => true,
    check: ctx => /\[\s*(screen|camera)\s*\]/.test(stripComments(ctx.code))
      ? 'screen and camera are already arrays of images: pass them as they are, never [screen].' : null,
  },
  {
    id: 'N4', stage: 'save', severity: 'warning',
    applies: ctx => isRole(ctx, 'watcher', 'gate'),
    check: ctx => {
      const code = stripComments(ctx.code);
      const bare = IMAGE_SENDS.filter(fn => callArgs(code, fn).some(a => !/\b(screen|camera|imemory)\b/.test(a)));
      return bare.length ? `attach the frame to ${bare.join(', ')}: pass screen (or camera) as the images argument.` : null;
    },
  },
  {
    // A made-up name ("alarm", "chime") only fails when the agent fires, so it's caught here.
    id: 'N5', stage: 'save', severity: 'error', onCustom: true,
    applies: ctx => !!ctx.sounds && calls(stripComments(ctx.code), 'sound'),
    check: ctx => {
      const bad = new Set<string>();
      for (const args of callArgs(stripComments(ctx.code), 'sound')) {
        const m = args.match(/^\s*(["'`])((?:(?!\1)[^\\]|\\.)*)\1\s*(?:,|$)/);
        if (!m || (m[1] === '`' && m[2].includes('${'))) continue; // no name, or not a literal
        const name = m[2];
        if (!ctx.sounds!.includes(name) && !/^(https?:|data:|blob:|\/)/.test(name)) bad.add(name);
      }
      return [...bad].map(name =>
        `sound("${name}"): there is no sound named "${name}". Use one of ${ctx.sounds!.map(s => `"${s}"`).join(', ')}, leave the name out (sound()), or pass an audio URL the user gave you.`);
    },
  },

  // Relations between agents
  {
    id: 'G1', stage: 'save', severity: 'error',
    applies: ctx => ctx.cls.role === 'gate' && !!ctx.cls.triggers,
    check: ctx => {
      const code = stripComments(ctx.code);
      const out: string[] = [];
      if (!new RegExp(`(?<![\\w.$])startAgent\\s*\\(\\s*["'\`]${esc(ctx.cls.triggers!)}["'\`]`).test(code)) out.push(`a gate starts its scribe: await startAgent("${ctx.cls.triggers}").`);
      if (!/(?<![\w.$])(append|set)ImageMemory\s*\(\s*(screen|camera)\s*\)/.test(code)) out.push('a gate saves what its scribe will describe: await appendImageMemory(screen) (or camera) before startAgent.');
      return out;
    },
  },
  {
    id: 'G1', stage: 'start', severity: 'error',
    applies: ctx => ctx.cls.role === 'gate' && !!ctx.cls.triggers && !!ctx.findAgent,
    check: ctx => {
      const target = ctx.findAgent!(ctx.cls.triggers!);
      if (!target) return `the scribe '${ctx.cls.triggers}' doesn't exist: create it before starting its gate.`;
      return target.classification?.role === 'scribe' ? null : `'${ctx.cls.triggers}' isn't classified as a scribe.`;
    },
  },
  {
    id: 'G2', stage: 'save', severity: 'error',
    applies: ctx => ctx.cls.role === 'scribe' && !!ctx.cls.reads_images_from,
    check: ctx => {
      const out: string[] = [];
      if (!ctx.agent.system_prompt.includes(`$IMEMORY@${ctx.cls.reads_images_from}`)) out.push(`a scribe reads its gate's frames: put $IMEMORY@${ctx.cls.reads_images_from} in the prompt.`);
      if (!calls(stripComments(ctx.code), 'stopAgent')) out.push('a scribe describes once per trigger: end with await stopAgent().');
      return out;
    },
  },

  // Roles
  {
    id: 'R1', stage: 'save', severity: 'error',
    applies: ctx => ctx.cls.role === 'scout',
    check: ctx => {
      const code = stripComments(ctx.code);
      const out: string[] = [];
      if (ctx.agent.model_name !== 'clef-flash') out.push('a scout runs on clef-flash.');
      if (calls(code, 'stopAgent')) out.push('a scout never stops itself: the screen share would end with it.');
      if (!calls(code, 'mcp')) out.push('a scout wakes you with await mcp("...").');
      // No memory flag: if you haven't replaced the scout a minute after it woke you, it wakes you again.
      if (!sleepsMs(code).some(ms => ms >= 60000)) out.push('a scout wakes you at most once a minute: await sleep(60000) right after mcp().');
      return out;
    },
  },
  {
    id: 'R2', stage: 'save', severity: 'error',
    applies: ctx => ctx.cls.role === 'scratchpad',
    check: ctx => {
      const out: string[] = [];
      if (ctx.agent.model_name !== 'Skip Model Call') out.push('a scratchpad uses model_name Skip Model Call.');
      if (!calls(stripComments(ctx.code), 'stopAgent')) out.push('a scratchpad runs once: end with await stopAgent().');
      return out;
    },
  },

  // Start stage
  {
    id: 'P1', stage: 'start', severity: 'error',
    applies: ctx => isRole(ctx, 'watcher', 'gate') && !!ctx.cls.sensor && (ctx.cls.scope ?? 'region') === 'region' && !!ctx.crop,
    check: ctx => {
      const stream = ctx.cls.sensor!;
      return ctx.crop![stream]
        ? null : `scope region: crop it first with set_${stream}_crop, then capture_${stream} with the agent_id to check the crop (or classify scope anywhere with a scope_reason).`;
    },
  },

  // Verify stage: the first iterations should show what `now` says
  {
    id: 'V1', stage: 'verify', severity: 'warning',
    applies: ctx => !!ctx.iteration?.decision && ctx.iteration.sessionIterationNumber <= 3
      && ctx.cls.task === 'decide' && ctx.cls.target !== 'image' && !!ctx.cls.now,
    check: ctx => {
      const d = ctx.iteration!.decision!;
      const answers = 'type' in d && typeof d.type === 'string' ? [d] : Object.values(d);
      const now = ctx.cls.now!;
      const wrong = (msg: string) => `${msg} The crop or the question is wrong: stop_agent, fix it, start_agent again.`;
      const noul = answers.find(a => a.type === 'noul');
      if (ctx.cls.question === 'noul' && noul && noul.type === 'noul') {
        if (now === 'yes' && noul.noul < 0.75) return wrong(`the answer should be 'yes' now, but noul is ${noul.noul.toFixed(2)}.`);
        if (now === 'no' && noul.noul > 0.25) return wrong(`the answer should be 'no' now, but noul is ${noul.noul.toFixed(2)}.`);
      }
      const choice = answers.find(a => a.type === 'choice');
      if (ctx.cls.question === 'choice' && choice && choice.type === 'choice' && (choice.probabilities[now] ?? 0) < 0.5) {
        return wrong(`the answer should be '${now}' now, but it's ${choice.choice} (${now}: ${(choice.probabilities[now] ?? 0).toFixed(2)}).`);
      }
      return null;
    },
  },
  {
    id: 'V2', stage: 'verify', severity: 'warning',
    applies: ctx => !!ctx.iteration && ctx.iteration.sessionIterationNumber <= 2 && ctx.cls.cardinality === 'once' && isRole(ctx, 'watcher', 'gate'),
    check: ctx => ctx.iteration!.tools.some(t => ALL_SENDS.includes(t.name) || t.name === 'stopAgent')
      ? `PrematureFire: a once agent acted on iteration ${ctx.iteration!.sessionIterationNumber}, before anything could have happened. Its condition is true already: fix the crop, question or now.` : null,
  },
];
