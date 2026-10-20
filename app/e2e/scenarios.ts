// Alert scenarios for the MCP end-to-end suite (mcp.spec.ts), from the ground truth
// (src/utils/groundTruth.ts): every use case judged on timing, run once per ending of its demo.
//
// The prompt is always `When …, then send me a browser notification.` and each use case says what
// an Observer agent should do on EVERY ending of the demo:
//   notify  the user asked for this: a notification within the window
//   silent  a notification here is a false alarm (the wrong person replied, the wrong price level)
//   either  the wait ended some other way: notifying is fine, so is staying quiet ("if my order
//           gets canceled" and it arrives; "if my build fails" and it passes)
// A notification before the ending is shown is always a false positive.
//
// The rest of the ground truth (logs, counts, clips, two alerts from one request) is judged by
// patterns.ts.

import { GROUND_TRUTH, prompt, type Expect } from '../src/utils/groundTruth';

export type { Expect };

export interface Scenario {
  id: string;
  demo: string;
  /** The outcome's name; `param` is what the demo is opened with. */
  outcome: string;
  param: string;
  prompt: string;
  expect: Expect;
  requires?: string[];
}

const JOB = { success: 'success', fail: 'fail' };

export const SCENARIOS: Scenario[] = GROUND_TRUTH.filter(u => u.expect).flatMap(u => {
  const outcomes = u.outcomes ?? JOB;
  const unknown = Object.keys(u.expect!).filter(o => !(o in outcomes));
  if (unknown.length) throw new Error(`${u.id} expects unknown outcomes: ${unknown.join(', ')}`);
  return Object.entries(outcomes).map(([outcome, param]) => {
    const expect = u.expect![outcome];
    if (!expect) throw new Error(`${u.id} has no expectation for outcome "${outcome}"`);
    return { id: `${u.id} ${outcome}`, demo: u.demo, outcome, param, prompt: prompt(u), expect, requires: u.requires };
  });
});
