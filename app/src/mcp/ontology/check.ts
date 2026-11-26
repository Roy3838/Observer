// src/mcp/ontology/check.ts
//
// Runs the rule table (rules.ts) for one stage and formats what failed. registry.ts calls it from
// create_agent / edit_agent ('save'), start_agent ('start') and get_iteration ('verify'). An error
// blocks the tool call and goes back to the MCP model, which repairs the agent and calls again:
// the LLM proposes, the rules check.

import { ModelManager } from '@utils/ModelManager';
import { KNOWN_SOUNDS } from '@utils/handlers/utils';
import type { IterationData } from '@utils/IterationStore';
import type { AgentClassification } from './schema';
import { strategyOf, describeClassification } from './derive';
import { analyzeDecisionPrompt } from './prompt';
import { runRules, type CheckResult, type RuleContext, type Stage, type Violation } from './rules';

export interface CheckInput {
  agent: { id: string; model_name: string; system_prompt: string; loop_interval_seconds?: number };
  code: string;
  classification: AgentClassification;
  crop?: RuleContext['crop'];
  findAgent?: RuleContext['findAgent'];
  iteration?: IterationData;
}

function lookupModel(name: string): RuleContext['model'] {
  const manager = ModelManager.getInstance();
  const model = manager.listModels().models.find(m => m.name === name);
  if (!model) return undefined;
  return { name: model.name, decision: !!model.decision, local: manager.isLocalModel(model.server) };
}

export function checkAgent(stage: Stage, input: CheckInput): CheckResult {
  const cls = input.classification;
  const strategy = strategyOf(cls);
  const ctx: RuleContext = {
    agent: input.agent,
    code: input.code,
    cls,
    strategy,
    model: lookupModel(input.agent.model_name),
    prompt: cls.task === 'decide' && strategy === 'decision' ? analyzeDecisionPrompt(input.agent.system_prompt) : {},
    crop: input.crop,
    findAgent: input.findAgent,
    iteration: input.iteration,
    sounds: KNOWN_SOUNDS,
  };
  return runRules(stage, ctx);
}

/** The tool error the MCP model sees: every broken rule at once, so it repairs in one turn. */
export function formatRejection(tool: string, cls: AgentClassification, errors: Violation[]): string {
  const n = errors.length;
  return `${tool} rejected by ${n} ontology rule${n === 1 ? '' : 's'} (${describeClassification(cls)}):\n` +
    errors.map(v => `- ${v.id}: ${v.message}`).join('\n') +
    `\nFix ${n === 1 ? 'it' : 'them'} and call ${tool} again. If the classification itself is wrong, fix the classification.`;
}

export const formatWarnings = (warnings: Violation[]) => warnings.map(v => `${v.id}: ${v.message}`);
