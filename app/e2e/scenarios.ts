// Scenarios for the MCP end-to-end suite.
//
// Each demo lists the ways it can end (its outcomes) and the prompts a user might type while
// it's on screen. Each prompt says what an Observer agent should do on EVERY outcome, and the
// suite runs it against each one:
//   notify  the user asked for this: a notification within the window
//   silent  a notification here is a false alarm (the wrong person replied, the wrong price level)
//   either  the wait ended some other way: notifying is fine, so is staying quiet ("if my order
//           gets canceled" and it arrives; "if my build fails" and it passes)
// A notification before the outcome is shown is always a false positive.
//
// Prompts must be answerable without contact info (no Telegram/SMS/email/...): the suite
// can't fill the ask_user_info modal, so a prompt that needs it measures nothing.

export type Expect = 'notify' | 'silent' | 'either';

export interface DemoSpec {
  /** File in e2e/demos/, implementing demo-contract.js. */
  demo: string;
  /** How the demo can end: name (used in scenario ids) → its `?outcome=` value. Default: success, fail. */
  outcomes?: Record<string, string>;
  /** `expect` names every outcome of the demo. */
  prompts: { prompt: string; expect: Record<string, Expect> }[];
}

export interface Scenario {
  id: string;
  demo: string;
  /** The outcome's name; `param` is what the demo is opened with. */
  outcome: string;
  param: string;
  prompt: string;
  expect: Expect;
}

const JOB = { success: 'success', fail: 'fail' };
// A job that ends either way: "when it's done" wants both endings; "if it fails" wants the
// failure and may also mention a success (the moment is right, the user still hears it ended).
const DONE: Record<string, Expect> = { success: 'notify', fail: 'notify' };
const IF_FAILS: Record<string, Expect> = { success: 'either', fail: 'notify' };

export const DEMOS: DemoSpec[] = [
  {
    demo: 'fake_download.html',
    prompts: [
      // A failed download is also "done" from the user's side: they want to know it stopped.
      { prompt: 'Send me a browser notification when my download is done.', expect: DONE },
      { prompt: 'Send me a browser notification if my download fails.', expect: IF_FAILS },
    ],
  },
  {
    demo: 'fake_simulation.html',
    prompts: [
      { prompt: 'Send me a browser notification when my simulation is done.', expect: DONE },
      { prompt: 'Send me a browser notification if my simulation crashes.', expect: IF_FAILS },
    ],
  },
  {
    demo: 'fake_github_workflow.html',
    prompts: [
      { prompt: 'Send me a browser notification when my GitHub workflow finishes.', expect: DONE },
      { prompt: 'Send me a browser notification if my GitHub workflow fails.', expect: IF_FAILS },
    ],
  },
  {
    demo: 'youtube_studio.html',
    prompts: [
      { prompt: 'Send me a browser notification when my YouTube upload finishes.', expect: DONE },
      { prompt: 'Send me a browser notification if my YouTube upload fails.', expect: IF_FAILS },
    ],
  },
  {
    // fail = Claude stops on a permission prompt instead of finishing.
    demo: 'claude_code.html',
    prompts: [
      { prompt: 'Send me a browser notification when Claude Code finishes or needs my input.', expect: DONE },
      { prompt: 'Send me a browser notification if Claude Code asks for my permission.', expect: IF_FAILS },
    ],
  },
  {
    demo: 'vercel_deploy.html',
    prompts: [
      { prompt: 'Send me a browser notification when my Vercel deployment finishes.', expect: DONE },
      { prompt: 'Send me a browser notification if my Vercel deployment fails.', expect: IF_FAILS },
    ],
  },
  {
    demo: 'colab_training.html',
    prompts: [
      { prompt: 'Send me a browser notification when my Colab training finishes.', expect: DONE },
      { prompt: 'Send me a browser notification if my Colab training crashes.', expect: IF_FAILS },
    ],
  },
  {
    // Warnings print either way (a build-script one while holding, an unused variable at the end).
    demo: 'cargo_build.html',
    prompts: [
      { prompt: 'Send me a browser notification when my build finishes compiling.', expect: DONE },
      { prompt: 'Send me a browser notification if my Rust build fails.', expect: IF_FAILS },
    ],
  },
  {
    // An IDE instead of a terminal. Kotlin deprecation warnings print either way (yellow, under the task).
    demo: 'androidstudio_build.html',
    prompts: [
      { prompt: 'Send me a browser notification when my Android Studio build finishes.', expect: DONE },
      { prompt: 'Send me a browser notification if my Android build fails.', expect: IF_FAILS },
    ],
  },
  // Waiting for an event rather than a job. Group messages keep arriving in the WhatsApp demo the whole time.
  {
    demo: 'whatsapp_web.html',
    outcomes: { maria_replies: 'success', carlos_messages: 'fail' },  // Maria replies in her open chat, or Carlos messages instead
    prompts: [
      { prompt: 'Send me a browser notification when Maria replies to me on WhatsApp.', expect: { maria_replies: 'notify', carlos_messages: 'silent' } },
      { prompt: 'Send me a browser notification when Maria or Carlos message me on WhatsApp.', expect: { maria_replies: 'notify', carlos_messages: 'notify' } },
    ],
  },
  {
    demo: 'ticketmaster_queue.html',
    outcomes: { my_turn: 'success', sold_out: 'fail' },  // the event sells out while you're in line
    prompts: [
      { prompt: "Send me a browser notification when I'm out of the Ticketmaster queue.", expect: { my_turn: 'notify', sold_out: 'notify' } },
      { prompt: "Send me a browser notification when it's my turn to buy tickets.", expect: { my_turn: 'notify', sold_out: 'either' } },
      { prompt: 'Send me a browser notification if the tickets sell out before I get in.', expect: { my_turn: 'either', sold_out: 'notify' } },
    ],
  },
  {
    // "Picking up" and "Heading your way" (with a falling ETA) come first and aren't "almost here".
    // late: still heading over, but stuck in traffic and the ETA jumps ~15 min. Not over yet, so
    // "almost here" or "canceled" alerts on it are false alarms.
    demo: 'ubereats_tracking.html',
    outcomes: { almost_here: 'success', late: 'late', canceled: 'fail' },
    prompts: [
      { prompt: 'Send me a browser notification when my Uber Eats driver is almost here.', expect: { almost_here: 'notify', late: 'silent', canceled: 'either' } },
      { prompt: 'Monitor my Uber Eats order and send me a browser notification.', expect: { almost_here: 'notify', late: 'either', canceled: 'notify' } },
      { prompt: 'Keep an eye on my Uber Eats order and send me a browser notification if anything goes wrong.', expect: { almost_here: 'either', late: 'notify', canceled: 'notify' } },
      { prompt: 'Send me a browser notification if my Uber Eats order gets canceled.', expect: { almost_here: 'either', late: 'silent', canceled: 'notify' } },
    ],
  },
  {
    // Dana Whitfield's thread was already in the inbox, read. A different Dana (Dana Kim) emails mid-run.
    demo: 'gmail_inbox.html',
    outcomes: { dana_replies: 'success', bounced: 'fail' },  // bounced: the follow-up to her bounces, threaded into the conversation
    prompts: [
      { prompt: 'Send me a browser notification when Dana Whitfield replies to my email.', expect: { dana_replies: 'notify', bounced: 'either' } },
      { prompt: 'Send me a browser notification when Dana Whitfield replies, or if my email to her bounces.', expect: { dana_replies: 'notify', bounced: 'notify' } },
      { prompt: 'Send me a browser notification if my email to Dana Whitfield bounces.', expect: { dana_replies: 'either', bounced: 'notify' } },
    ],
  },
  {
    // A price level rather than an end state. Two near misses (~80,120, ~80,040) come first:
    // notifying on those is a false positive, and so is alerting on the level the user didn't ask about.
    demo: 'tradingview_btc.html',
    outcomes: { below_80k: 'success', above_82_5k: 'fail' },
    prompts: [
      { prompt: 'Send me a browser notification if Bitcoin drops below $80,000.', expect: { below_80k: 'notify', above_82_5k: 'silent' } },
      { prompt: 'Send me a browser notification if Bitcoin goes below $80,000 or above $82,500.', expect: { below_80k: 'notify', above_82_5k: 'notify' } },
      { prompt: 'Send me a browser notification if Bitcoin breaks above $82,500.', expect: { below_80k: 'silent', above_82_5k: 'notify' } },
    ],
  },
  {
    demo: 'google_meet.html',
    outcomes: { let_in: 'success', denied: 'fail' },  // your request to join is denied
    prompts: [
      { prompt: "Send me a browser notification when I'm let into my Google Meet call.", expect: { let_in: 'notify', denied: 'either' } },
      { prompt: 'Send me a browser notification when someone answers my request to join the Meet call.', expect: { let_in: 'notify', denied: 'notify' } },
    ],
  },
  // Long-running dashboards, one plain request each: the user wants to hear about either ending.
  ...[
    ['bigquery_query.html', 'my BigQuery query'],
    ['cloudformation_stack.html', 'my CloudFormation stack update'],
    ['gitlab_pipeline.html', 'my GitLab pipeline'],
    ['wandb_run.html', 'my Weights & Biases training run'],
    ['fusion360_assembly.html', 'my Fusion 360 assembly while it opens'],
    ['argocd_sync.html', 'my Kubernetes rollout in Argo CD'],
  ].map(([demo, what]): DemoSpec => ({
    demo,
    prompts: [{ prompt: `Please watch ${what} and send me a browser notification.`, expect: DONE }],
  })),
];

export const SCENARIOS: Scenario[] = DEMOS.flatMap(d => {
  const outcomes = d.outcomes ?? JOB;
  return d.prompts.flatMap((p, i) => {
    const unknown = Object.keys(p.expect).filter(o => !(o in outcomes));
    if (unknown.length) throw new Error(`${d.demo} prompt #${i + 1} expects unknown outcomes: ${unknown.join(', ')}`);
    return Object.entries(outcomes).map(([outcome, param]) => {
      const expect = p.expect[outcome];
      if (!expect) throw new Error(`${d.demo} prompt #${i + 1} has no expectation for outcome "${outcome}"`);
      return { id: `${d.demo.replace(/\.html$/, '')} #${i + 1} ${outcome}`, demo: d.demo, outcome, param, prompt: p.prompt, expect };
    });
  });
});
