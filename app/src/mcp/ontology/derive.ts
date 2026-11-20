// src/mcp/ontology/derive.ts
//
// What follows from a classification, with no model involved: which kind of model runs the agent,
// and which trigger its code must use. rules.ts checks the agent against these.

import type { AgentClassification } from './schema';

export type Strategy = 'decision' | 'llm' | 'none';

/** decide → a decision model; describe / extract → an LLM; a scratchpad calls no model. */
export function strategyOf(c: AgentClassification): Strategy {
  if (c.role === 'scratchpad') return 'none';
  return c.task === 'decide' ? 'decision' : 'llm';
}

/** Combinations that make sense together (rule C1). Each problem is one sentence. */
export function classificationProblems(c: AgentClassification): string[] {
  const p: string[] = [];
  const watching = c.role === 'watcher' || c.role === 'gate';

  if (watching) {
    // A describe watcher (an activity tracker) has no trigger, so no target or shape.
    const keys = c.task === 'describe' ? (['sensor', 'cardinality'] as const) : (['sensor', 'target', 'shape', 'cardinality'] as const);
    for (const key of keys) {
      if (!c[key]) p.push(`a ${c.role} needs ${key}.`);
    }
  }
  if (c.role === 'gate' && c.task !== 'decide') p.push('a gate decides: task must be decide.');
  if (c.role === 'scribe' && c.task !== 'describe') p.push('a scribe describes: task must be describe.');
  if (c.role === 'scout' && c.task !== 'decide') p.push('a scout decides: task must be decide.');

  if (c.task === 'decide') {
    if (!c.question) p.push('task decide needs question (noul, choice or score).');
    if (!c.now) p.push('task decide needs now: the answer that is true in the frame right now.');
    if (c.question === 'noul' && c.now && c.now !== 'yes' && c.now !== 'no') p.push(`a noul question's now is "yes" or "no", not "${c.now}".`);
    if (c.target === 'value') p.push('a value is read, not decided: target value needs task extract.');
  }
  if (c.task === 'extract' && c.target && c.target !== 'value') p.push('task extract reads a number: target must be value.');

  if (c.shape === 'change' && c.target && c.target !== 'image' && c.target !== 'value') {
    p.push('shape change compares over time: target must be image (pictures) or value (a number).');
  }
  if (c.shape && c.shape !== 'change' && (c.target === 'image' || c.target === 'value')) {
    p.push(`target ${c.target} is only compared over time: shape must be change.`);
  }
  if (c.target === 'image' && c.task !== 'decide') p.push('pictures are compared by a decision model: target image needs task decide.');
  if (c.shape === 'nag' && c.cardinality === 'once') p.push('a nag repeats until the condition clears: cardinality must be every.');
  if (c.shape === 'sequence' && c.question && c.question !== 'choice') p.push('a sequence tells its stages apart: question must be choice.');

  if (c.scope === 'anywhere' && !c.scope_reason) p.push('scope anywhere needs scope_reason: why the answer can be anywhere on screen.');
  if (c.role === 'custom' && !c.custom_reason) p.push('role custom needs custom_reason: what the ontology can\'t express.');
  if (c.role === 'gate' && !c.triggers) p.push('a gate needs triggers: the id of the scribe it starts.');
  if (c.role === 'scribe' && !c.reads_images_from) p.push('a scribe needs reads_images_from: the id of its gate.');
  return p;
}

/**
 * The exact trigger a noul agent's code must use, from `now`: it acts when the answer leaves what's
 * true now. Pictures compared over time (target image) pick their own direction; see rule T6.
 */
export function noulTrigger(now: string | undefined, questionId?: string): string | null {
  const d = questionId ? `decision.${questionId}` : 'decision';
  if (now === 'yes') return `${d}.noul < 0.25 || ${d}.dropped`;
  if (now === 'no') return `${d}.noul > 0.75`;
  return null;
}

/** One line for error messages: "decide / state / event / once → noul, now: no". */
export function describeClassification(c: AgentClassification): string {
  const parts = [c.role !== 'watcher' ? c.role : null, c.task, c.target, c.shape, c.cardinality].filter(Boolean).join(' / ');
  const q = c.task === 'decide' && c.question ? ` → ${c.question}${c.now ? `, now: ${c.now}` : ''}` : '';
  return parts + q;
}
