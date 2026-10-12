// Scenarios for the MCP end-to-end suite.
//
// Each demo lists the prompts a user might type while it's on screen. Each prompt says on
// which outcomes the user expects to be notified; the suite runs it against every outcome,
// so "only if it fails" also gets a success run where staying silent is the pass.
//
// Prompts must be answerable without contact info (no Telegram/SMS/email/...): the suite
// can't fill the ask_user_info modal, so a prompt that needs it measures nothing.

export type Outcome = 'success' | 'fail';

export interface DemoSpec {
  /** File in e2e/demos/, implementing demo-contract.js. */
  demo: string;
  prompts: { prompt: string; notifyOn: Outcome[] }[];
}

export interface Scenario {
  id: string;
  demo: string;
  outcome: Outcome;
  prompt: string;
  shouldNotify: boolean;
}

export const DEMOS: DemoSpec[] = [
  {
    demo: 'fake_download.html',
    prompts: [
      // A failed download is also "done" from the user's side: they want to know it stopped.
      { prompt: 'Send me a browser notification when my download is done.', notifyOn: ['success', 'fail'] },
      { prompt: 'Send me a browser notification if my download fails.', notifyOn: ['fail'] },
    ],
  },
  {
    demo: 'fake_simulation.html',
    prompts: [
      { prompt: 'Send me a browser notification when my simulation is done.', notifyOn: ['success', 'fail'] },
      { prompt: 'Send me a browser notification if my simulation crashes.', notifyOn: ['fail'] },
    ],
  },
  {
    demo: 'fake_github_workflow.html',
    prompts: [
      { prompt: 'Send me a browser notification when my GitHub workflow finishes.', notifyOn: ['success', 'fail'] },
      { prompt: 'Send me a browser notification if my GitHub workflow fails.', notifyOn: ['fail'] },
    ],
  },
  {
    demo: 'youtube_studio.html',
    prompts: [
      { prompt: 'Send me a browser notification when my YouTube upload finishes.', notifyOn: ['success', 'fail'] },
      { prompt: 'Send me a browser notification if my YouTube upload fails.', notifyOn: ['fail'] },
    ],
  },
  {
    // fail = Claude stops on a permission prompt instead of finishing.
    demo: 'claude_code.html',
    prompts: [
      { prompt: 'Send me a browser notification when Claude Code finishes or needs my input.', notifyOn: ['success', 'fail'] },
      { prompt: 'Send me a browser notification if Claude Code asks for my permission.', notifyOn: ['fail'] },
    ],
  },
  {
    demo: 'vercel_deploy.html',
    prompts: [
      { prompt: 'Send me a browser notification when my Vercel deployment finishes.', notifyOn: ['success', 'fail'] },
      { prompt: 'Send me a browser notification if my Vercel deployment fails.', notifyOn: ['fail'] },
    ],
  },
  {
    demo: 'colab_training.html',
    prompts: [
      { prompt: 'Send me a browser notification when my Colab training finishes.', notifyOn: ['success', 'fail'] },
      { prompt: 'Send me a browser notification if my Colab training crashes.', notifyOn: ['fail'] },
    ],
  },
  {
    // Warnings print either way (a build-script one while holding, an unused variable at the end).
    demo: 'cargo_build.html',
    prompts: [
      { prompt: 'Send me a browser notification when my build finishes compiling.', notifyOn: ['success', 'fail'] },
      { prompt: 'Send me a browser notification if my Rust build fails.', notifyOn: ['fail'] },
    ],
  },
  {
    // An IDE instead of a terminal. Kotlin deprecation warnings print either way (yellow, under the task).
    demo: 'androidstudio_build.html',
    prompts: [
      { prompt: 'Send me a browser notification when my Android Studio build finishes.', notifyOn: ['success', 'fail'] },
      { prompt: 'Send me a browser notification if my Android build fails.', notifyOn: ['fail'] },
    ],
  },
  // Waiting for an event rather than a job: `fail` is the other way the wait ends, which only
  // some prompts care about. Group messages keep arriving in the WhatsApp demo the whole time.
  {
    // success = Maria replies in her open chat; fail = Carlos messages instead.
    demo: 'whatsapp_web.html',
    prompts: [
      { prompt: 'Send me a browser notification when Maria replies to me on WhatsApp.', notifyOn: ['success'] },
      { prompt: 'Send me a browser notification when Maria or Carlos message me on WhatsApp.', notifyOn: ['success', 'fail'] },
    ],
  },
  {
    // success = it's your turn to pick tickets; fail = the event sells out while in line.
    demo: 'ticketmaster_queue.html',
    prompts: [
      { prompt: "Send me a browser notification when I'm out of the Ticketmaster queue.", notifyOn: ['success', 'fail'] },
      { prompt: "Send me a browser notification when it's my turn to buy tickets.", notifyOn: ['success'] },
    ],
  },
  {
    // success = "Almost here" (the courier is a block or two out); fail = the order is canceled mid-delivery.
    // "Picking up" and "Heading your way" (with a falling ETA) come first and aren't "almost here".
    demo: 'ubereats_tracking.html',
    prompts: [
      { prompt: 'Send me a browser notification when my Uber Eats driver is almost here.', notifyOn: ['success'] },
      { prompt: 'Send me a browser notification when my Uber Eats driver is almost here, or if my order gets canceled.', notifyOn: ['success', 'fail'] },
    ],
  },
  {
    // success = Dana Whitfield replies (her thread was already in the inbox, read); fail = the follow-up to
    // her bounces, threaded into the same conversation. A different Dana (Dana Kim) emails mid-run.
    demo: 'gmail_inbox.html',
    prompts: [
      { prompt: 'Send me a browser notification when Dana Whitfield replies to my email.', notifyOn: ['success'] },
      { prompt: 'Send me a browser notification when Dana Whitfield replies, or if my email to her bounces.', notifyOn: ['success', 'fail'] },
    ],
  },
  {
    // A price level rather than an end state. success = BTC breaks below 80,000; fail = it rallies
    // through 82,500 instead. Two near misses (~80,120, ~80,040) come first: notifying on those is a false positive.
    demo: 'tradingview_btc.html',
    prompts: [
      { prompt: 'Send me a browser notification if Bitcoin drops below $80,000.', notifyOn: ['success'] },
      { prompt: 'Send me a browser notification if Bitcoin goes below $80,000 or above $82,500.', notifyOn: ['success', 'fail'] },
    ],
  },
  {
    // success = someone lets you in; fail = your request to join is denied.
    demo: 'google_meet.html',
    prompts: [
      { prompt: "Send me a browser notification when I'm let into my Google Meet call.", notifyOn: ['success'] },
      { prompt: 'Send me a browser notification when someone answers my request to join the Meet call.', notifyOn: ['success', 'fail'] },
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
    prompts: [{ prompt: `Please watch ${what} and send me a browser notification.`, notifyOn: ['success', 'fail'] }],
  })),
];

const OUTCOMES: Outcome[] = ['success', 'fail'];

export const SCENARIOS: Scenario[] = DEMOS.flatMap(d =>
  d.prompts.flatMap((p, i) =>
    OUTCOMES.map(outcome => ({
      id: `${d.demo.replace(/\.html$/, '')} #${i + 1} ${outcome}`,
      demo: d.demo,
      outcome,
      prompt: p.prompt,
      shouldNotify: p.notifyOn.includes(outcome),
    })),
  ),
);
