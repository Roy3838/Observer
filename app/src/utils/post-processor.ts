// src/utils/post-processor.ts

import { Logger } from './logging';
import { executeJavaScript } from './handlers/javascript';
import type { TokenProvider } from './main_loop'; // Import the type for clarity
import type { PreProcessorResult } from './pre-processor';
import type { Decision } from './localLlm/systemOne';

/**
 * Process response using the JavaScript or Python handler
 * based on the presence of a '#python' marker
 */
export async function postProcess(
    agentId: string,
    response: string | null, // null when the agent runs a decision model
    code: string,
    iterationId: string, // <-- New parameter
    getToken?: TokenProvider,
    preprocessResult?: PreProcessorResult,
    decision: Decision | null = null // System One answers when the agent runs a decision model
): Promise<boolean> {
  Logger.debug(agentId, 'Starting response post-processing', { iterationId });

  if (code.trim().startsWith('#python')) {
    Logger.debug(agentId, 'Detected Python code, using Python handler', { iterationId });
    // Lazy load Python handler - only loads when Python code is executed!
    const { executePython } = await import('./handlers/python');
    return await executePython(response ?? '', agentId, code);
  } else {
    Logger.debug(agentId, 'Using JavaScript handler', { iterationId });
    // Pass iterationId, getToken, and preprocessResult to JavaScript handler
    return await executeJavaScript(response, agentId, code, iterationId, getToken, preprocessResult, decision);
  }
  // No catch - let errors bubble up naturally to main_loop
}
