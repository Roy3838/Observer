// src/mcp/ontology/prompt.ts
//
// Parse a decision prompt the way an iteration will, before it ever runs (rule Q1).

import { validateDecisionPrompt, isJsonPrompt, jsonEscape } from '@utils/localLlm/systemOne';
import type { PromptAnalysis } from './rules';

// Text a sensor could splice in, with every character that breaks a careless JSON prompt.
const SAMPLE_TEXT = 'she said "on my way"\nsee C:\\files';

/**
 * Image sensors become '' and count as images; text sensors become SAMPLE_TEXT, escaped in a JSON
 * prompt exactly as preProcess does. Then the real parser runs on the result.
 */
export function analyzeDecisionPrompt(prompt: string): PromptAnalysis {
  const json = isJsonPrompt(prompt);
  let images = 0;
  const filled = prompt
    .replace(/\$(SCREEN|CAMERA)(?![A-Z_])/g, () => { images++; return ''; })
    .replace(/\$IMEMORY(?![A-Z_])(?:@\w+)?/g, '')
    .replace(/\$[A-Z][A-Z_]*(?:@\w+)?/g, () => (json ? jsonEscape(SAMPLE_TEXT) : SAMPLE_TEXT));
  try {
    const { single, questions } = validateDecisionPrompt(filled, images);
    return { single, questions };
  } catch (e) {
    return { error: e instanceof Error ? e.message : String(e) };
  }
}
