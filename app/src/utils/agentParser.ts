// src/utils/agentParser.ts
import { CompleteAgent, getAgent, getAgentCode, getAgentMemory, getAllAgentIds } from './agent_database';
import { IterationStore, IterationData } from './IterationStore';

// ===================================================================================
//  AGENT REFERENCE TYPES
// ===================================================================================

export interface AgentReference {
  agentId: string;
  runCount: number; // default 3, 0 means config only
}

export interface AgentReferenceData {
  reference: AgentReference;
  agent: CompleteAgent | null;
  code: string | null;
  memory: string;
  recentRuns: IterationData[];
}

// ===================================================================================
//  DETECTION MODE TEMPLATE TYPES
// ===================================================================================

export interface DetectionModeTemplate {
  categories: string[];                           // ["spaghetti", "printing", "finished"]
  categorization: Record<string, number[]>;       // {"spaghetti": [1,4,5], "printing": [2,3,...]}
}

// ===================================================================================
//  EXISTING PARSER FUNCTIONS
// ===================================================================================

/**
 * Extracts the raw configuration block from a string.
 * @param text The text possibly containing an agent config.
 * @returns The content inside the $$$ block, or null if not found.
 */
export function extractAgentConfig(text: string): string | null {
  const agentBlockRegex = /\$\$\$\s*\n?([\s\S]*?)\n?\$\$\$/;
  const match = text.match(agentBlockRegex);
  return match && match[1] ? match[1].trim() : null;
}

/**
 * Extracts all agent configuration blocks from a string.
 * @param text The text possibly containing multiple agent configs.
 * @returns An array of content inside $$$ blocks, or empty array if none found.
 */
export function extractMultipleAgentConfigs(text: string): string[] {
  const agentBlockRegex = /\$\$\$\s*\n?([\s\S]*?)\n?\$\$\$/g;
  const matches = Array.from(text.matchAll(agentBlockRegex));
  return matches.map(match => match[1] ? match[1].trim() : '').filter(config => config.length > 0);
}

/**
 * Extracts the image request text from a string.
 * @param text The text possibly containing an image request.
 * @returns The content inside the %%% block, or null if not found.
 */
export function extractImageRequest(text: string): string | null {
  const imageRequestRegex = /%%%\s*\n?([\s\S]*?)\n?%%%/;
  const match = text.match(imageRequestRegex);
  return match && match[1] ? match[1].trim() : null;
}

/**
 * Extracts the detection mode template from a string.
 * @param text The text possibly containing a detection mode template.
 * @returns The parsed DetectionModeTemplate, or null if not found or invalid.
 */
export function extractDetectionModeTemplate(text: string): DetectionModeTemplate | null {
  // Use \s* to handle any whitespace (spaces, newlines, tabs) around the content
  const detectionModeRegex = /&&&\s*([\s\S]*?)\s*&&&/;
  const match = text.match(detectionModeRegex);

  if (!match || !match[1]) {
    return null;
  }

  try {
    const parsed = JSON.parse(match[1].trim());

    // Validate the structure
    if (!parsed.categories || !Array.isArray(parsed.categories) || parsed.categories.length < 2) {
      console.error('Detection mode template must have at least 2 categories');
      return null;
    }

    if (!parsed.categorization || typeof parsed.categorization !== 'object') {
      console.error('Detection mode template must have a categorization object');
      return null;
    }

    // Validate all categories are strings
    if (!parsed.categories.every((cat: unknown) => typeof cat === 'string')) {
      console.error('All categories must be strings');
      return null;
    }

    // Validate all categorization values are arrays of numbers
    for (const [key, value] of Object.entries(parsed.categorization)) {
      if (!Array.isArray(value) || !value.every((v: unknown) => typeof v === 'number')) {
        console.error(`Categorization for "${key}" must be an array of numbers`);
        return null;
      }
    }

    return {
      categories: parsed.categories,
      categorization: parsed.categorization
    };
  } catch (error) {
    console.error('Failed to parse detection mode template JSON:', error);
    return null;
  }
}

/**
 * Parses a raw agent configuration string into a structured agent object and code.
 * @param configText The raw agent configuration (content from within $$$).
 * @returns An object containing the agent and code, or null on failure.
 */
export function parseAgentResponse(configText: string): { agent: CompleteAgent, code: string } | null {
  try {
    const codeMatch = configText.match(/code:\s*\|\s*\n([\s\S]*?)(?=\nmemory:|$)/);
    const systemPromptMatch = configText.match(/system_prompt:\s*\|\s*\n([\s\S]*?)(?=\ncode:)/);
    
    if (!codeMatch || !codeMatch[1] || !systemPromptMatch || !systemPromptMatch[1]) {
      console.error("Parsing Error: Could not find system_prompt or code sections.");
      return null;
    }

    const getField = (field: string): string => {
      const match = configText.match(new RegExp(`^${field}:\\s*([^\\n]+)`, 'm'));
      return match && match[1] ? match[1].trim() : '';
    };

    const agent: CompleteAgent = {
      id: getField('id') || `agent_${Date.now()}`, // Fallback to ensure an ID exists
      name: getField('name') || 'Untitled Agent',
      description: getField('description') || 'No description.',
      model_name: getField('model_name'),
      system_prompt: systemPromptMatch[1].trimEnd(),
      loop_interval_seconds: parseFloat(getField('loop_interval_seconds')) || 60,
    };
    
    // Basic validation
    if (!agent.model_name) {
      console.error("Parsing Error: model_name is missing.");
      return null;
    }

    return { agent, code: codeMatch[1] };
  } catch (error) {
    console.error("Fatal error parsing agent response:", error);
    return null;
  }
}

// ===================================================================================
//  AGENT REFERENCE PARSER FUNCTIONS
// ===================================================================================

/**
 * Extracts all @agent references from text
 * @param text Input text containing @references
 * @returns Array of parsed agent references
 */
export function extractAgentReferences(text: string): AgentReference[] {
  const regex = /@(\w+)(?:#(\d+))?/g;
  const references: AgentReference[] = [];
  let match;

  while ((match = regex.exec(text)) !== null) {
    references.push({
      agentId: match[1],
      runCount: match[2] ? parseInt(match[2]) : 10 // default to 10 runs
    });
  }

  return references;
}

/**
 * Extracts agent references with their positions for inline rendering
 * @param text Input text containing @references
 * @returns Array of references with position info
 */
export function extractAgentReferencesWithPositions(text: string): Array<{
  reference: AgentReference;
  start: number;
  end: number;
  fullMatch: string;
}> {
  const regex = /@(\w+)(?:#(\d+))?/g;
  const references: Array<{
    reference: AgentReference;
    start: number;
    end: number;
    fullMatch: string;
  }> = [];
  let match;

  while ((match = regex.exec(text)) !== null) {
    references.push({
      reference: {
        agentId: match[1],
        runCount: match[2] ? parseInt(match[2]) : 3
      },
      start: match.index,
      end: match.index + match[0].length,
      fullMatch: match[0]
    });
  }

  return references;
}

/**
 * Removes @references from text, leaving clean user message
 * @param text Input text with @references
 * @returns Text with @references stripped out
 */
export function stripAgentReferences(text: string): string {
  return text.replace(/@\w+(?:#\d+)?/g, '').trim();
}

/**
 * Gets recent runs for an agent from IterationStore
 * @param agentId Agent ID
 * @param count Number of recent runs to fetch
 * @returns Array of recent iteration data
 */
async function getRecentAgentRuns(agentId: string, count: number): Promise<IterationData[]> {
  // Get current session runs
  const currentRuns = IterationStore.getIterationsForAgent(agentId);

  // Get historical runs if we need more
  if (currentRuns.length >= count) {
    return currentRuns.slice(-count); // Get last N runs
  }

  // Need to get historical data
  const historicalSessions = await IterationStore.getHistoricalSessions(agentId);
  const allRuns: IterationData[] = [...currentRuns];

  // Add runs from most recent sessions until we have enough
  for (const session of historicalSessions) {
    const needed = count - allRuns.length;
    const sessionRuns = session.iterations.slice(-needed); // Take LAST needed iterations
    allRuns.unshift(...sessionRuns);

    if (allRuns.length >= count) break;
  }

  return allRuns.slice(-count); // Return last N runs
}

/**
 * Fetches complete agent reference data
 * @param references Array of agent references to fetch
 * @returns Array of complete agent reference data
 */
export async function fetchAgentReferenceData(
  references: AgentReference[]
): Promise<AgentReferenceData[]> {
  const results: AgentReferenceData[] = [];

  for (const reference of references) {
    try {
      // Fetch agent config and code from agent_database
      const agent = await getAgent(reference.agentId);
      const code = agent ? await getAgentCode(reference.agentId) : null;
      const memory = agent ? await getAgentMemory(reference.agentId) : '';

      // Fetch recent runs from IterationStore
      let recentRuns: IterationData[] = [];
      if (reference.runCount > 0) {
        recentRuns = await getRecentAgentRuns(reference.agentId, reference.runCount);
      }

      results.push({
        reference,
        agent,
        code,
        memory,
        recentRuns
      });
    } catch (error) {
      console.error(`Failed to fetch data for agent ${reference.agentId}:`, error);
      // Add placeholder entry for failed fetch
      results.push({
        reference,
        agent: null,
        code: null,
        memory: '',
        recentRuns: []
      });
    }
  }

  return results;
}

/**
 * Extract unique agent references from all messages in conversation
 * @param messages Array of conversation messages
 * @returns Array of unique agent references
 */
export function extractUniqueAgentReferencesFromConversation(messages: string[]): AgentReference[] {
  const allReferences: AgentReference[] = [];
  const seenAgentIds = new Set<string>();

  // Extract references from all messages
  messages.forEach(message => {
    const refs = extractAgentReferences(message);
    refs.forEach(ref => {
      if (!seenAgentIds.has(ref.agentId)) {
        seenAgentIds.add(ref.agentId);
        allReferences.push(ref);
      }
    });
  });

  return allReferences;
}

/**
 * Validates if agent references exist in the database
 * @param text Input text containing @references
 * @param validAgentIds Array of valid agent IDs
 * @returns Array of references with position info, only for valid agents
 */
export async function extractValidAgentReferencesWithPositions(text: string): Promise<Array<{
  reference: AgentReference;
  start: number;
  end: number;
  fullMatch: string;
}>> {
  const validAgentIds = await getAllAgentIds();
  const validIdSet = new Set(validAgentIds);

  const allRefs = extractAgentReferencesWithPositions(text);
  return allRefs.filter(ref => validIdSet.has(ref.reference.agentId));
}

/**
 * Detects partial @agent typing for suggestions
 * @param text Input text
 * @param cursorPosition Current cursor position
 * @returns Object with suggestion info or null
 */
export function detectPartialAgentTyping(text: string, cursorPosition: number): {
  partialMatch: string;
  start: number;
  end: number;
} | null {
  // Look for @word pattern before cursor
  const beforeCursor = text.substring(0, cursorPosition);
  const match = beforeCursor.match(/@(\w*)$/);

  if (match) {
    return {
      partialMatch: match[1], // The part after @
      start: match.index! + 1, // Position after @
      end: cursorPosition
    };
  }

  return null;
}

/**
 * Formats agent reference context for appending to a message
 * Uses the official Observer agent file format ($$$...$$$)
 * @param referenceData Array of fetched agent reference data
 * @returns Formatted agent context string
 */
export function formatAgentReferenceContext(
  referenceData: AgentReferenceData[]
): string {
  if (referenceData.length === 0) return '';

  let context = "\n\n=== REFERENCED AGENTS ===\n";
  context += "The user has referenced the following agents. Here are their current configurations:\n\n";

  referenceData.forEach(data => {
    const { reference, agent, code, memory, recentRuns } = data;

    if (!agent) {
      context += `@${reference.agentId} - AGENT NOT FOUND\n`;
      context += `The user referenced @${reference.agentId} but this agent doesn't exist.\n\n`;
      return;
    }

    // Format in official Observer agent file format
    context += `$$$\n`;
    context += `id: ${agent.id}\n`;
    context += `name: ${agent.name}\n`;
    context += `description: ${agent.description}\n`;
    context += `model_name: ${agent.model_name}\n`;
    context += `loop_interval_seconds: ${agent.loop_interval_seconds}\n`;
    context += `system_prompt: |\n`;

    // Indent system prompt lines
    const systemPromptLines = agent.system_prompt.split('\n');
    systemPromptLines.forEach(line => {
      context += `  ${line}\n`;
    });

    context += `code: |\n`;

    // Indent code lines if available
    if (code) {
      const codeLines = code.split('\n');
      codeLines.forEach(line => {
        context += `  ${line}\n`;
      });
    }

    // Add memory (truncate if too long)
    if (memory.trim()) {
      const memoryContent = memory.length > 500 ? memory.substring(0, 500) + '...' : memory;
      context += `memory: |\n`;
      const memoryLines = memoryContent.split('\n');
      memoryLines.forEach(line => {
        context += `  ${line}\n`;
      });
    } else {
      context += `memory: ""\n`;
    }

    context += `$$$\n\n`;

    // Add recent performance data after the agent config
    if (recentRuns.length > 0) {
      context += `Recent Performance for @${agent.id} (last ${recentRuns.length} runs):\n`;

      const successCount = recentRuns.filter(run => !run.hasError).length;
      context += `Success Rate: ${successCount}/${recentRuns.length}\n`;

      if (recentRuns.some(run => run.duration)) {
        const avgDuration = recentRuns
          .filter(run => run.duration)
          .reduce((sum, run) => sum + (run.duration || 0), 0) / recentRuns.length;
        context += `Average Duration: ${avgDuration.toFixed(1)}s\n`;
      }

      // Add summary of recent runs
      recentRuns.forEach((run, i) => {
        context += `\nRun ${i + 1}: ${run.hasError ? 'FAILED' : 'SUCCESS'}`;
        if (run.hasError && run.tools.some(tool => tool.status === 'error')) {
          const errorTools = run.tools.filter(tool => tool.status === 'error');
          context += ` - Errors:\n`;
          errorTools.forEach(tool => {
            context += `  - ${tool.name}: ${tool.error}`;
            if (tool.params) {
              context += ` (params: ${JSON.stringify(tool.params)})`;
            }
            context += '\n';
          });
        }
        context += '\n';
      });
      context += "\n";
    }
  });

  return context;
}

/**
 * Builds enhanced system prompt with agent reference context
 * @param baseSystemPrompt Base system prompt
 * @param referenceData Array of fetched agent reference data
 * @returns Enhanced system prompt with agent context
 */
export function buildSystemPromptWithAgentContext(
  baseSystemPrompt: string,
  referenceData: AgentReferenceData[]
): string {
  let systemPrompt = baseSystemPrompt;

  if (referenceData.length > 0) {
    systemPrompt += "\n\n=== REFERENCED AGENTS ===\n";
    systemPrompt += "The user has referenced the following agents for editing/improvement:\n\n";

    referenceData.forEach(data => {
      const { reference, agent, code, memory, recentRuns } = data;

      if (!agent) {
        systemPrompt += `## @${reference.agentId} - AGENT NOT FOUND\n`;
        systemPrompt += `The user referenced @${reference.agentId} but this agent doesn't exist.\n\n`;
        return;
      }

      systemPrompt += `## @${reference.agentId} - ${agent.name}\n`;
      systemPrompt += `Description: ${agent.description}\n`;
      systemPrompt += `Model: ${agent.model_name}\n`;
      systemPrompt += `Loop Interval: ${agent.loop_interval_seconds}s\n\n`;

      // Add system prompt
      systemPrompt += `### System Prompt:\n`;
      systemPrompt += `${agent.system_prompt}\n\n`;

      // Add code if available
      if (code) {
        systemPrompt += `### Code:\n`;
        systemPrompt += `\`\`\`python\n${code}\n\`\`\`\n\n`;
      }

      // Add memory if not empty
      if (memory.trim()) {
        systemPrompt += `### Memory:\n`;
        systemPrompt += `${memory.substring(0, 500)}${memory.length > 500 ? '...' : ''}\n\n`;
      }

      // Add recent performance data
      if (recentRuns.length > 0) {
        systemPrompt += `### Recent Performance (last ${recentRuns.length} runs):\n`;

        const successCount = recentRuns.filter(run => !run.hasError).length;
        systemPrompt += `Success Rate: ${successCount}/${recentRuns.length}\n`;

        if (recentRuns.some(run => run.duration)) {
          const avgDuration = recentRuns
            .filter(run => run.duration)
            .reduce((sum, run) => sum + (run.duration || 0), 0) / recentRuns.length;
          systemPrompt += `Average Duration: ${avgDuration.toFixed(1)}s\n`;
        }

        // Add detailed run information
        recentRuns.forEach((run, i) => {
          systemPrompt += `\n#### Run ${i + 1} (${new Date(run.startTime).toLocaleString()}):\n`;
          systemPrompt += `Status: ${run.hasError ? 'FAILED' : 'SUCCESS'}\n`;

          if (run.modelPrompt) {
            const truncatedPrompt = run.modelPrompt.substring(0, 200);
            systemPrompt += `Input: ${truncatedPrompt}${run.modelPrompt.length > 200 ? '...' : ''}\n`;
          }

          if (run.modelResponse) {
            const truncatedResponse = run.modelResponse.substring(0, 200);
            systemPrompt += `Output: ${truncatedResponse}${run.modelResponse.length > 200 ? '...' : ''}\n`;
          }

          if (run.hasError && run.tools.some(tool => tool.status === 'error')) {
            const errorTools = run.tools.filter(tool => tool.status === 'error');
            systemPrompt += `Errors:\n`;
            errorTools.forEach(tool => {
              systemPrompt += `  - ${tool.name}: ${tool.error}`;
              if (tool.params) {
                systemPrompt += ` (params: ${JSON.stringify(tool.params)})`;
              }
              systemPrompt += '\n';
            });
          }

          if (run.tools.length > 0) {
            const toolNames = run.tools.map(tool => `${tool.name}(${tool.status})`).join(', ');
            systemPrompt += `Tools Used: ${toolNames}\n`;
          }
        });
        systemPrompt += "\n";
      }

      systemPrompt += "---\n\n";
    });

    systemPrompt += "When editing these agents, consider:\n";
    systemPrompt += "- Their current performance patterns and success rates\n";
    systemPrompt += "- Recent errors and how to address them\n";
    systemPrompt += "- User feedback and requested improvements\n";
    systemPrompt += "- Maintaining compatibility with existing functionality\n\n";
  }

  return systemPrompt;
}
