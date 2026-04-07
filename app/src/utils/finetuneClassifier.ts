// src/utils/finetuneClassifier.ts
// Core logic for finetuning classification prompts against ground truth images

import { fetchResponse } from './sendApi';
import { listModels } from './inferenceServer';

export interface TestCase {
  imageData: string;
  expectedLabel: string;
}

export interface TestResult {
  imageIndex: number;
  expected: string;
  actual: string;
  passed: boolean;
}

export interface FinetuneConfig {
  testModel: string;           // vision model (from source agent)
  finetunerModel: string;      // AI model for prompt improvement
  serverAddress: string;
  token?: string;
  maxIterations: number;
  categories: string[];        // list of valid category labels
}

export interface FinetuneProgress {
  phase: 'testing' | 'improving';
  iteration: number;
  currentTestIndex?: number;
  totalTests?: number;
  latestResult?: TestResult;  // The most recent test result (for real-time updates)
  allResults?: TestResult[];  // All results so far in this iteration
}

export interface IterationHistory {
  iteration: number;
  prompt: string;
  failedTests: TestResult[];
  totalTests: number;
  passedCount: number;
}

export interface FinetuneResult {
  success: boolean;
  finalPrompt: string;
  finalResults: TestResult[];
  iterations: number;
}

/**
 * Test a single image against the prompt using the vision model
 */
async function testImage(
  prompt: string,
  imageData: string,
  config: FinetuneConfig
): Promise<string> {
  const messages = [
    {
      role: 'user',
      content: [
        { type: 'text', text: prompt },
        {
          type: 'image_url',
          image_url: {
            url: `data:image/png;base64,${imageData}`
          }
        }
      ]
    }
  ];

  const response = await fetchResponse(
    config.serverAddress,
    messages,
    config.testModel,
    config.token,
    false // no streaming for tests
  );

  return response.trim();
}

/**
 * Normalize a classification response to match expected labels
 */
function normalizeResponse(response: string, categories: string[]): string {
  const upperResponse = response.toUpperCase().trim();

  // Try exact match first
  for (const cat of categories) {
    if (upperResponse === cat.toUpperCase()) {
      return cat;
    }
  }

  // Try contains match
  for (const cat of categories) {
    if (upperResponse.includes(cat.toUpperCase())) {
      return cat;
    }
  }

  // Return original if no match
  return response;
}

/**
 * Run all test cases against the current prompt
 */
export async function runTestSuite(
  prompt: string,
  testCases: TestCase[],
  config: FinetuneConfig,
  onProgress?: (index: number, result: TestResult) => void,
  abortSignal?: AbortSignal
): Promise<TestResult[]> {
  const results: TestResult[] = [];

  for (let i = 0; i < testCases.length; i++) {
    // Check for abort
    if (abortSignal?.aborted) {
      throw new Error('Aborted');
    }

    const testCase = testCases[i];

    try {
      const rawResponse = await testImage(prompt, testCase.imageData, config);
      const actual = normalizeResponse(rawResponse, config.categories);
      const passed = actual.toUpperCase() === testCase.expectedLabel.toUpperCase();

      const result: TestResult = {
        imageIndex: i,
        expected: testCase.expectedLabel,
        actual,
        passed
      };

      results.push(result);

      if (onProgress) {
        onProgress(i, result);
      }
    } catch (error) {
      // If aborted, rethrow
      if (error instanceof Error && error.message === 'Aborted') {
        throw error;
      }

      // On error, mark as failed with error message
      const result: TestResult = {
        imageIndex: i,
        expected: testCase.expectedLabel,
        actual: `ERROR: ${error instanceof Error ? error.message : 'Unknown error'}`,
        passed: false
      };
      results.push(result);

      if (onProgress) {
        onProgress(i, result);
      }
    }
  }

  return results;
}

/**
 * Ask AI to improve the prompt based on failures, showing the actual failed images
 * and full history of previous iterations for context
 */
export async function improvePrompt(
  currentPrompt: string,
  failedTests: TestResult[],
  testCases: TestCase[],
  iterationHistory: IterationHistory[],
  config: FinetuneConfig,
  abortSignal?: AbortSignal
): Promise<string> {
  if (abortSignal?.aborted) {
    throw new Error('Aborted');
  }

  const finetunerSystemPrompt = `You are improving a visual classification prompt through iterative refinement.

THE VALID OUTPUT CATEGORIES ARE EXACTLY: ${config.categories.join(', ')}
The model MUST output one of these exact category labels.

You will be shown the FULL HISTORY of prompt iterations, including:
- Each prompt that was tried
- Which images failed for each prompt (with the actual images)
- How the results changed between iterations

Use this history to understand:
- What changes improved classification accuracy
- What changes made things worse or had no effect
- Which images are persistently difficult to classify correctly

YOUR TASK:
1. Analyze the progression of prompts and results across all iterations.
2. LOOK at the failed images carefully and understand their visual characteristics.
3. Identify patterns: Are the same images failing repeatedly? Did a change fix some but break others?
4. Make targeted improvements based on what's actually working or not working.

CRITICAL RULES:
- Output ONLY the complete improved system prompt - no explanations, no markdown, no "here is the improved prompt"
- The prompt must instruct the model to output EXACTLY one of: ${config.categories.join(', ')}
- Keep any $CAMERA, $SCREENSHOT, or $SCREEN placeholder at the end
- Do NOT add "CATEGORIES:" lines or any meta-information - just the prompt itself
- The output format should remain: describe what you see, then output one category label
- Learn from the history - don't repeat changes that didn't work`;

  // Build user message with full iteration history
  const userContent: Array<{ type: string; text?: string; image_url?: { url: string } }> = [];

  // Add history of previous iterations
  for (const hist of iterationHistory) {
    userContent.push({
      type: 'text',
      text: `\n${'='.repeat(60)}\n=== ITERATION ${hist.iteration} === (${hist.passedCount}/${hist.totalTests} passed)\n${'='.repeat(60)}\n\nPROMPT:\n---\n${hist.prompt}\n---\n`
    });

    if (hist.failedTests.length > 0) {
      userContent.push({
        type: 'text',
        text: `\nFAILED IMAGES (${hist.failedTests.length}):`
      });

      for (const failedTest of hist.failedTests) {
        const testCase = testCases[failedTest.imageIndex];
        if (testCase) {
          userContent.push({
            type: 'text',
            text: `\n• Image ${failedTest.imageIndex + 1}: Expected "${failedTest.expected}" → Model output "${failedTest.actual}"`
          });
          userContent.push({
            type: 'image_url',
            image_url: {
              url: `data:image/png;base64,${testCase.imageData}`
            }
          });
        }
      }
    }
  }

  // Add current iteration (the one we're trying to improve)
  const currentIteration = iterationHistory.length + 1;
  const currentPassedCount = testCases.length - failedTests.length;

  userContent.push({
    type: 'text',
    text: `\n${'='.repeat(60)}\n=== CURRENT ITERATION ${currentIteration} === (${currentPassedCount}/${testCases.length} passed)\n${'='.repeat(60)}\n\nCURRENT PROMPT:\n---\n${currentPrompt}\n---\n`
  });

  if (failedTests.length > 0) {
    userContent.push({
      type: 'text',
      text: `\nCURRENT FAILURES (${failedTests.length}):`
    });

    for (const failedTest of failedTests) {
      const testCase = testCases[failedTest.imageIndex];
      if (testCase) {
        userContent.push({
          type: 'text',
          text: `\n• Image ${failedTest.imageIndex + 1}: Expected "${failedTest.expected}" → Model output "${failedTest.actual}"`
        });
        userContent.push({
          type: 'image_url',
          image_url: {
            url: `data:image/png;base64,${testCase.imageData}`
          }
        });
      }
    }
  }

  userContent.push({
    type: 'text',
    text: `\n\n${'='.repeat(60)}\nBased on the full history above, please output an improved prompt that fixes the current failures while maintaining what's already working.`
  });

  const messages = [
    { role: 'system', content: finetunerSystemPrompt },
    { role: 'user', content: userContent }
  ];

  const response = await fetchResponse(
    config.serverAddress,
    messages,
    config.finetunerModel,
    config.token,
    false
  );

  // Clean up the response - remove any markdown formatting
  let improvedPrompt = response.trim();

  // Remove markdown code blocks if present
  if (improvedPrompt.startsWith('```')) {
    improvedPrompt = improvedPrompt.replace(/^```[\w]*\n?/, '').replace(/\n?```$/, '');
  }

  return improvedPrompt.trim();
}

/**
 * Main finetuning loop - runs test/improve cycles until all pass or max iterations
 */
export async function finetuneLoop(
  initialPrompt: string,
  testCases: TestCase[],
  config: FinetuneConfig,
  onIterationComplete: (iteration: number, results: TestResult[], newPrompt: string, phase: 'testing' | 'improving') => void,
  onProgress?: (progress: FinetuneProgress) => void,
  abortSignal?: AbortSignal
): Promise<FinetuneResult> {
  let currentPrompt = initialPrompt;
  let iteration = 0;
  let lastResults: TestResult[] = [];
  const iterationHistory: IterationHistory[] = [];

  while (iteration < config.maxIterations) {
    iteration++;

    // Check for abort before testing
    if (abortSignal?.aborted) {
      return {
        success: false,
        finalPrompt: currentPrompt,
        finalResults: lastResults,
        iterations: iteration - 1
      };
    }

    // Report testing phase
    if (onProgress) {
      onProgress({
        phase: 'testing',
        iteration,
        currentTestIndex: 0,
        totalTests: testCases.length
      });
    }

    // Run test suite - accumulate results in real-time
    const iterationResults: TestResult[] = [];
    try {
      lastResults = await runTestSuite(
        currentPrompt,
        testCases,
        config,
        (index, result) => {
          iterationResults.push(result);
          if (onProgress) {
            onProgress({
              phase: 'testing',
              iteration,
              currentTestIndex: index + 1,
              totalTests: testCases.length,
              latestResult: result,
              allResults: [...iterationResults]
            });
          }
        },
        abortSignal
      );
    } catch (error) {
      if (error instanceof Error && error.message === 'Aborted') {
        return {
          success: false,
          finalPrompt: currentPrompt,
          finalResults: lastResults,
          iterations: iteration - 1
        };
      }
      throw error;
    }

    // Check if all passed
    const failedTests = lastResults.filter(r => !r.passed);
    const passedCount = lastResults.filter(r => r.passed).length;

    if (failedTests.length === 0) {
      // All tests passed!
      onIterationComplete(iteration, lastResults, currentPrompt, 'testing');
      return {
        success: true,
        finalPrompt: currentPrompt,
        finalResults: lastResults,
        iterations: iteration
      };
    }

    // Report improvement phase
    onIterationComplete(iteration, lastResults, currentPrompt, 'testing');

    if (onProgress) {
      onProgress({
        phase: 'improving',
        iteration
      });
    }

    // If not the last iteration, improve the prompt
    if (iteration < config.maxIterations) {
      try {
        // Save the prompt before improvement so we can add to history
        const promptBeforeImprovement = currentPrompt;

        // Pass full iteration history to improvePrompt for context
        currentPrompt = await improvePrompt(
          currentPrompt,
          failedTests,
          testCases,
          iterationHistory,
          config,
          abortSignal
        );
        onIterationComplete(iteration, lastResults, currentPrompt, 'improving');

        // Add this iteration to history AFTER improving (using the prompt that was tested)
        iterationHistory.push({
          iteration,
          prompt: promptBeforeImprovement,
          failedTests: [...failedTests],
          totalTests: testCases.length,
          passedCount
        });
      } catch (error) {
        if (error instanceof Error && error.message === 'Aborted') {
          return {
            success: false,
            finalPrompt: currentPrompt,
            finalResults: lastResults,
            iterations: iteration
          };
        }
        throw error;
      }
    }
  }

  // Max iterations reached
  return {
    success: false,
    finalPrompt: currentPrompt,
    finalResults: lastResults,
    iterations: iteration
  };
}

/**
 * Get the server address for a model
 */
export function getServerForModel(modelName: string, isUsingObServer: boolean): string {
  if (isUsingObServer) {
    return 'https://api.observer-ai.com:443';
  }

  const modelsResponse = listModels();
  const model = modelsResponse.models.find(m => m.name === modelName);

  if (model) {
    return model.server;
  }

  // Default to localhost if not found
  return 'http://127.0.0.1:11434';
}
