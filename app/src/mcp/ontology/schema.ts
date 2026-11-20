// src/mcp/ontology/schema.ts
//
// The agent ontology: the checklist in systemPrompt.ts as data. The MCP (the neural half) classifies
// each request into these classes and passes the result to create_agent / edit_agent as
// `classification`; derive.ts turns it into a strategy and rules.ts checks the agent against it
// (the symbolic half). Decisions are always made by a decision model; LLM agents only describe
// what they see or extract a value, and their text never drives control flow.

import type { NotifyMethod } from '../types';

export const ROLES = ['watcher', 'gate', 'scribe', 'scout', 'scratchpad', 'custom'] as const;
export const TASKS = ['decide', 'describe', 'extract'] as const;
export const SENSORS = ['screen', 'camera'] as const;
export const TARGETS = ['dial', 'object', 'state', 'image', 'value'] as const;
export const SHAPES = ['event', 'deadline', 'change', 'nag', 'sequence'] as const;
export const CARDINALITIES = ['once', 'every'] as const;
export const SCOPES = ['region', 'anywhere'] as const;
export const QUESTIONS = ['noul', 'choice', 'score'] as const;
export const CHANNELS: NotifyMethod[] = ['browser', 'system', 'email', 'whatsapp', 'sms', 'voice', 'telegram', 'discord', 'pushover'];

export type Role = typeof ROLES[number];
export type Task = typeof TASKS[number];
export type Sensor = typeof SENSORS[number];
export type Target = typeof TARGETS[number];
export type Shape = typeof SHAPES[number];
export type Cardinality = typeof CARDINALITIES[number];
export type Scope = typeof SCOPES[number];
export type QuestionKind = typeof QUESTIONS[number];

export interface AgentClassification {
  role: Role;
  task: Task;
  sensor?: Sensor;
  target?: Target;
  shape?: Shape;
  cardinality?: Cardinality;
  scope?: Scope;
  scope_reason?: string;
  /** decide: the decision model's question type. */
  question?: QuestionKind;
  /** decide: the answer that is true NOW ('yes' / 'no', or a choice label). The agent acts when it moves away. */
  now?: string;
  channel?: NotifyMethod;
  /** gate → the scribe agent it starts. */
  triggers?: string;
  /** scribe → the gate whose image memory it reads ($IMEMORY@gate). */
  reads_images_from?: string;
  custom_reason?: string;
}

/** The JSON Schema of create_agent / edit_agent's `classification` parameter. */
export const CLASSIFICATION_PARAM = {
  type: 'object',
  description:
    'The agent\'s place in the ontology: your answers to checklist steps 1–6, as data. The agent is checked against it ' +
    'and create_agent / edit_agent fail, listing every broken rule, when the prompt, code or model don\'t fit it.',
  properties: {
    role: { type: 'string', enum: [...ROLES], description: 'watcher (the usual agent), gate (decides, then starts a scribe), scribe (describes what its gate saved, then stops), scout, scratchpad, or custom (nothing else fits: give custom_reason).' },
    task: { type: 'string', enum: [...TASKS], description: 'decide (a decision model answers a question about the frame), describe (an LLM writes a description that goes into memory or a message), or extract (an LLM reads one value, VALUE: <n>, that the code compares).' },
    sensor: { type: 'string', enum: [...SENSORS] },
    target: { type: 'string', enum: [...TARGETS], description: 'dial, object, state, image (pictures compared over time), value (a number read over time).' },
    shape: { type: 'string', enum: [...SHAPES] },
    cardinality: { type: 'string', enum: [...CARDINALITIES] },
    scope: { type: 'string', enum: [...SCOPES], description: 'region (cropped, the default) or anywhere (uncropped: give scope_reason).' },
    scope_reason: { type: 'string' },
    question: { type: 'string', enum: [...QUESTIONS], description: 'decide only: noul (yes/no), choice (one label of several) or score.' },
    now: { type: 'string', description: 'decide only: the answer that is true right now, in the frame you captured: "yes", "no", or a choice label. The agent acts when the answer moves away from it.' },
    channel: { type: 'string', enum: CHANNELS, description: 'The channel ask_user_info kind=\'method\' returned.' },
    triggers: { type: 'string', description: 'gate only: the id of the scribe it starts.' },
    reads_images_from: { type: 'string', description: 'scribe only: the id of the gate whose image memory it reads.' },
    custom_reason: { type: 'string' },
  },
  required: ['role', 'task'],
};

const oneOf = <T extends string>(values: readonly T[], v: unknown): v is T =>
  typeof v === 'string' && (values as readonly string[]).includes(v);

/** Check the shape and enums of a raw `classification` argument. Consistency between fields is rule C1. */
export function parseClassification(raw: unknown): { value?: AgentClassification; error?: string } {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return { error: 'classification is required: an object with at least role and task (see the checklist).' };
  }
  const r = raw as Record<string, unknown>;
  const enums: [string, readonly string[]][] = [
    ['role', ROLES], ['task', TASKS], ['sensor', SENSORS], ['target', TARGETS], ['shape', SHAPES],
    ['cardinality', CARDINALITIES], ['scope', SCOPES], ['question', QUESTIONS], ['channel', CHANNELS],
  ];
  for (const [key, values] of enums) {
    if (r[key] !== undefined && !oneOf(values, r[key])) {
      return { error: `classification.${key} must be one of ${values.join(', ')}; got ${JSON.stringify(r[key])}.` };
    }
  }
  if (!r.role || !r.task) return { error: 'classification needs role and task.' };
  for (const key of ['now', 'scope_reason', 'triggers', 'reads_images_from', 'custom_reason']) {
    if (r[key] !== undefined && typeof r[key] !== 'string') return { error: `classification.${key} must be a string.` };
  }

  const value: AgentClassification = { role: r.role as Role, task: r.task as Task };
  for (const key of Object.keys(CLASSIFICATION_PARAM.properties) as (keyof AgentClassification)[]) {
    if (r[key] !== undefined && r[key] !== '') (value as any)[key] = r[key];
  }
  return { value };
}
