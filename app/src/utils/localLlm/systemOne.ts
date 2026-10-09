// systemOne.ts - System One requests and answers for decision models (OneJev)
//
// Decision models don't generate text: they score a state against typed questions and return a
// probability for every option. The request/answer types follow TypeSafe's System One API
// (POST /v1/systemone), which qev serves for OneJev, so agent code sees the same answer shapes
// any System One backend returns.
//
// An agent's system_prompt (after sensor substitution) is read in one of two forms:
//   - plain text: one noul (yes/no) question; the sensor images are the state.
//       "Is there a person in the room? $CAMERA"            → decision = NoulAnswer
//   - JSON: a System One request; images go where the state says <image:N>
//     (qev's convention), or at the top of the state when it has no markers.
//       {"state": {...}, "questions": {"person": {...}}}    → decision = { person: Answer, ... }
//
// Prompt rendering and answer assembly are ports of qev/prompt.py and qev/answers.py (prompt
// version "qev-labels-v2"). The rendered text must match the training prompts byte for byte.

import type { LocalLlmContentPart, LocalLlmMessage } from './types';

// ============================================================================
// System One types
// ============================================================================

export type Entry = string | Record<string, unknown> | unknown[] | null;

export interface NoulQuestion {
  type: 'noul';
  instructions?: Entry;
  criteria?: { true?: Entry; false?: Entry } | null;
}

export interface ChoiceQuestion {
  type: 'choice';
  instructions?: Entry;
  criteria: Record<string, Entry>;   // label → description
}

export interface ScoreQuestion {
  type: 'score';
  instructions?: Entry;
  criteria: Entry[];                  // level descriptions, lowest first
}

export type Question = NoulQuestion | ChoiceQuestion | ScoreQuestion;

export interface SystemOneRequest {
  state?: unknown;
  questions: Record<string, Question>;
}

export interface NoulAnswer {
  type: 'noul';
  noul: number;                       // P(yes)
}

export interface ChoiceAnswer {
  type: 'choice';
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
}

export interface ScoreAnswer {
  type: 'score';
  score: number;                      // expected level
  legend: Record<string, Entry>;
  probabilities: Record<string, number>;
  confidence: number;
}

export type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer;

// What agent code receives as `decision`: one answer for a plain-text prompt,
// an answers map keyed by question id for a JSON prompt.
export type Decision = Answer | Record<string, Answer>;

// ============================================================================
// Prompt rendering (port of qev/prompt.py, default PromptStyle)
// ============================================================================

const SYSTEM_PROMPT =
  'Apply the question to the state. Choose exactly one of the listed options. ' +
  'Respond with only its uppercase letter, with no explanation or reasoning.';

// qev also has two-letter labels past Z; we stop at 26 options until those are verified here.
export const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const MAX_SCORE_LEVELS = 10;

const NOUL_DEFAULT_INSTRUCTIONS = 'Is the statement true, or is the answer to the question yes?';
const NOUL_DEFAULT_TRUE = 'the statement is true / the answer is yes';
const NOUL_DEFAULT_FALSE = 'the statement is false / the answer is no';

const IMAGE_PLACEHOLDER = /<image:(\d+)>/g;
const HAS_IMAGE_PLACEHOLDER = /<image:\d+>/;

function renderEntry(entry: unknown): string {
  if (entry === null || entry === undefined) return '';
  if (typeof entry === 'string') return entry.trim();
  return JSON.stringify(entry, null, 2);
}

function renderState(state: unknown): string {
  if (state === null || state === undefined) return '';
  if (typeof state === 'string') return state;
  return JSON.stringify(state, null, 2);
}

interface RenderedQuestion {
  kind: Question['type'];
  labels: string[];
  descriptions: Entry[];
  suffix: string;
}

function renderQuestion(q: Question): RenderedQuestion {
  let instructions: string;
  let labels: string[];
  let names: string[];
  let descs: Entry[];

  if (q.type === 'noul') {
    instructions = renderEntry(q.instructions) || NOUL_DEFAULT_INSTRUCTIONS;
    const crit = q.criteria;
    const t = crit && crit.true !== undefined && crit.true !== null ? renderEntry(crit.true) : NOUL_DEFAULT_TRUE;
    const f = crit && crit.false !== undefined && crit.false !== null ? renderEntry(crit.false) : NOUL_DEFAULT_FALSE;
    labels = ['yes', 'no'];
    names = ['yes', 'no'];
    descs = [t, f];
  } else if (q.type === 'choice') {
    instructions = renderEntry(q.instructions) || 'Which option applies to the state?';
    labels = Object.keys(q.criteria);
    names = labels;
    descs = labels.map(k => q.criteria[k]);
  } else {
    instructions = renderEntry(q.instructions) || 'Which level describes the state?';
    labels = q.criteria.map((_, i) => String(i));
    names = q.criteria.map((_, i) => `level ${i}`);
    descs = q.criteria;
  }

  const letters = LETTERS.slice(0, labels.length);
  const options = letters.map((letter, i) => {
    const text = renderEntry(descs[i]);
    return text ? `${letter}. ${names[i]}: ${text}` : `${letter}. ${names[i]}`;
  }).join('\n');
  const header = q.type === 'score' ? 'Rate the state:' : 'Question:';
  const suffix =
    `${header} ${instructions}\n\n` +
    `Options:\n${options}\n\n` +
    `Answer with one letter: ${letters.join(', ')}.`;

  return { kind: q.type, labels, descriptions: descs, suffix };
}

// Split the user turn at <image:N> into chat content parts (qev/media.py content_parts).
// Each image must be referenced exactly once, in order.
function contentParts(text: string, images: string[]): string | LocalLlmContentPart[] {
  if (images.length === 0) {
    if (HAS_IMAGE_PLACEHOLDER.test(text)) {
      throw new Error('Decision prompt references <image:N> but no image sensor ($SCREEN, $CAMERA, $IMEMORY) captured an image');
    }
    return text;
  }

  const parts: LocalLlmContentPart[] = [];
  const seen: number[] = [];
  let pos = 0;
  for (const m of text.matchAll(IMAGE_PLACEHOLDER)) {
    const n = Number(m[1]);
    if (n < 1 || n > images.length) {
      throw new Error(`Decision prompt references ${m[0]} but only ${images.length} image(s) were captured`);
    }
    if (m.index! > pos) parts.push({ type: 'text', text: text.slice(pos, m.index) });
    parts.push({ type: 'image', image: images[n - 1] });
    seen.push(n);
    pos = m.index! + m[0].length;
  }
  if (pos < text.length) parts.push({ type: 'text', text: text.slice(pos) });

  if (seen.length !== images.length || seen.some((n, i) => n !== i + 1)) {
    throw new Error(`Each captured image must appear once, in order, as <image:1>..<image:${images.length}> in the state; saw ${seen.map(n => `<image:${n}>`).join(', ') || 'none'}`);
  }
  return parts;
}

// ============================================================================
// Agent prompt → System One request
// ============================================================================

function validateQuestion(id: string, q: any): Question {
  if (!q || typeof q !== 'object') throw new Error(`Question '${id}' must be an object`);
  switch (q.type) {
    case 'noul':
      return q as NoulQuestion;
    case 'choice': {
      const n = q.criteria && typeof q.criteria === 'object' && !Array.isArray(q.criteria) ? Object.keys(q.criteria).length : 0;
      if (n === 0) throw new Error(`Question '${id}': choice needs criteria, an object of label → description`);
      if (n > LETTERS.length) throw new Error(`Question '${id}': choice has ${n} options, maximum is ${LETTERS.length}`);
      return q as ChoiceQuestion;
    }
    case 'score': {
      if (!Array.isArray(q.criteria) || q.criteria.length === 0) throw new Error(`Question '${id}': score needs criteria, an array of level descriptions`);
      if (q.criteria.length > MAX_SCORE_LEVELS) throw new Error(`Question '${id}': score has ${q.criteria.length} levels, maximum is ${MAX_SCORE_LEVELS}`);
      return q as ScoreQuestion;
    }
    default:
      throw new Error(`Question '${id}': type must be 'noul', 'choice' or 'score'`);
  }
}

function parseRequest(prompt: string): SystemOneRequest {
  let raw: any;
  try {
    raw = JSON.parse(prompt);
  } catch (e) {
    throw new Error(`Decision prompt starts with '{' but is not valid JSON: ${e instanceof Error ? e.message : String(e)}`);
  }
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('Decision prompt JSON must be an object');
  const questions = raw.questions;
  if (!questions || typeof questions !== 'object' || Array.isArray(questions) || Object.keys(questions).length === 0) {
    throw new Error('Decision prompt JSON needs "questions": { id: { type, instructions, criteria } }');
  }
  const validated: Record<string, Question> = {};
  for (const [id, q] of Object.entries(questions)) validated[id] = validateQuestion(id, q);
  return { state: raw.state, questions: validated };
}

// ============================================================================
// Prepared request: one branch (forward pass) per question
// ============================================================================

export interface DecisionBranch {
  messages: LocalLlmMessage[];
  nSlots: number;                     // option letters to read: A..
}

export interface PreparedDecision {
  single: boolean;                    // plain-text prompt → decision is one answer
  ids: string[];
  rendered: RenderedQuestion[];
  branches: DecisionBranch[];
}

/**
 * Turn a pre-processed agent prompt and its captured images (data URLs) into
 * one chat request per question, in the exact format OneJev was trained on.
 */
export function prepareDecision(prompt: string, images: string[]): PreparedDecision {
  const trimmed = prompt.trim();
  const single = !trimmed.startsWith('{');

  let request: SystemOneRequest;
  if (single) {
    if (!trimmed) throw new Error('Decision models need a question: write it in the system prompt, e.g. "Is there a person? $CAMERA"');
    request = { state: null, questions: { decision: { type: 'noul', instructions: trimmed } } };
  } else {
    request = parseRequest(trimmed);
  }

  // Images the state doesn't place explicitly go at the top of the state block
  let stateText = renderState(request.state);
  if (images.length > 0 && !HAS_IMAGE_PLACEHOLDER.test(stateText)) {
    const markers = images.map((_, i) => `<image:${i + 1}>`).join('\n');
    stateText = stateText ? `${markers}\n${stateText}` : markers;
  }
  const stateBlock = `<state>\n${stateText}\n</state>\n\n`;

  const ids = Object.keys(request.questions);
  const rendered = ids.map(id => renderQuestion(request.questions[id]));
  const branches = rendered.map(r => ({
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: contentParts(stateBlock + r.suffix, images) },
    ],
    nSlots: r.labels.length,
  }));

  return { single, ids, rendered, branches };
}

// ============================================================================
// Answer assembly (port of qev/answers.py + calibrate.py softmax)
// ============================================================================

// Temperatures keyed "kind:K" or "kind" (qev Calibration); missing keys use 1.0.
// Temperature never changes the winning option, only how calibrated the probabilities are.
const CALIBRATION: Record<string, number> = {};

function temperature(kind: string, k: number): number {
  return CALIBRATION[`${kind}:${k}`] ?? CALIBRATION[kind] ?? 1.0;
}

function softmax(logits: number[], temp: number): number[] {
  const z = logits.map(x => x / temp);
  const m = Math.max(...z);
  const w = z.map(x => Math.exp(x - m));
  const s = w.reduce((a, b) => a + b, 0);
  return w.map(x => x / s);
}

function normalize(probs: number[]): number[] {
  const total = probs.reduce((a, b) => a + b, 0);
  if (total <= 0) return probs.map(() => 1 / probs.length);
  return probs.map(p => p / total);
}

// Peak probability rescaled from uniform (0) to certainty (1)
function choiceConfidence(p: number[]): number {
  if (p.length === 1) return 1;
  const uniform = 1 / p.length;
  return (Math.max(...p) - uniform) / (1 - uniform);
}

// One minus the probability-weighted mean absolute deviation from the modal level,
// normalised by that of the uniform distribution over the levels
function scoreConfidence(p: number[]): number {
  if (p.length === 1) return 1;
  const k = p.length;
  const mode = p.indexOf(Math.max(...p));
  const distance = p.reduce((acc, pi, i) => acc + pi * Math.abs(i - mode), 0);
  const center = (k - 1) / 2;
  let uniformMad = 0;
  for (let i = 0; i < k; i++) uniformMad += Math.abs(i - center);
  uniformMad /= k;
  return Math.max(0, 1 - distance / uniformMad);
}

const round6 = (x: number) => Math.round(x * 1e6) / 1e6;

function makeAnswer(r: RenderedQuestion, probs: number[]): Answer {
  const p = normalize(probs);
  if (r.kind === 'noul') {
    return { type: 'noul', noul: round6(Math.min(1, Math.max(0, p[0]))) };
  }
  const probabilities = Object.fromEntries(r.labels.map((l, i) => [l, round6(p[i])]));
  if (r.kind === 'choice') {
    const best = p.indexOf(Math.max(...p));
    return { type: 'choice', choice: r.labels[best], probabilities, confidence: round6(choiceConfidence(p)) };
  }
  return {
    type: 'score',
    score: round6(p.reduce((acc, pi, i) => acc + i * pi, 0)),
    legend: Object.fromEntries(r.descriptions.map((d, i) => [String(i), d])),
    probabilities,
    confidence: round6(scoreConfidence(p)),
  };
}

/** Turn the option-letter logits of each branch into the agent's `decision`. */
export function assembleDecision(prepared: PreparedDecision, logits: number[][]): Decision {
  const answers = prepared.rendered.map((r, i) =>
    makeAnswer(r, softmax(logits[i], temperature(r.kind, r.labels.length))));
  if (prepared.single) return answers[0];
  return Object.fromEntries(prepared.ids.map((id, i) => [id, answers[i]]));
}

// ============================================================================
// Display
// ============================================================================

function formatAnswer(a: Answer): string {
  if (a.type === 'noul') return `noul ${a.noul.toFixed(3)}`;
  if (a.type === 'choice') return `choice ${a.choice} (${a.probabilities[a.choice].toFixed(3)})`;
  return `score ${a.score.toFixed(2)}`;
}

function isAnswer(d: Decision): d is Answer {
  return typeof (d as Answer).type === 'string' && ['noul', 'choice', 'score'].includes((d as Answer).type);
}

/** One-line summary for logs, iteration history and get_runs (e.g. "person: noul 0.943 | pet: choice dog (0.812)"). */
export function formatDecision(decision: Decision): string {
  if (isAnswer(decision)) return formatAnswer(decision);
  return Object.entries(decision).map(([id, a]) => `${id}: ${formatAnswer(a)}`).join(' | ');
}
