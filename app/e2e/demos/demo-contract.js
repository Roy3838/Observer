// demo-contract.js: the contract between a demo page and the MCP end-to-end suite.
//
// A demo loads HOLDING: its job is visibly in progress and stays that way however long the
// MCP takes to build and start its agent. The suite then calls `demo.start(durationMs)`.
// The job keeps going for durationMs, then ends the way `?outcome=success|fail` says, and
// the demo calls `demo.trigger()` once that end state is on screen. That instant is the
// ground truth the agent's notification is timed against.
//
// A demo implements `demo.onStart = durationMs => { ... }` and calls `demo.trigger()`.
// Opened by hand (no suite), `?autostart=10000` skips holding and ends the job 10 s after load.

// The suite's browser auto-selects the tab with exactly this title in the screen-share picker.
// The app's first-run tutorial opens demos with `?title=GitHub Actions` instead, so the user
// can find the tab in the picker (and the desktop MCP can find the window) by name.
const tourTitle = new URLSearchParams(location.search).get('title');
document.title = tourTitle ? `${tourTitle} · Observer demo` : 'Observer E2E Demo';

window.demo = {
  // `success` or `fail`, or a demo's own ending (`late` in ubereats_tracking). Defaults to success.
  outcome: new URLSearchParams(location.search).get('outcome') || 'success',
  startedAt: null,
  triggeredAt: null,
  onStart: null,

  start(durationMs) {
    if (this.startedAt) return;
    this.startedAt = Date.now();
    this.onStart(durationMs);
  },

  // Ground truth along the way, for the pattern scenarios (patterns.spec.ts): a topic change,
  // an error printed, an order arriving. Stamped with the suite's clock, like the trigger.
  mark(name, data) {
    setTimeout(() => {
      console.log(`[demo] mark: ${name}`);
      window.__e2eReport?.({ type: 'mark', name, data });
    }, 50);
  },

  trigger() {
    if (this.triggeredAt) return;
    // Let the end state paint before reporting it.
    setTimeout(() => {
      this.triggeredAt = Date.now();
      console.log(`[demo] trigger: ${this.outcome}`);
      window.__e2eReport?.({ type: 'trigger', outcome: this.outcome });
    }, 50);
  },
};

window.addEventListener('load', () => {
  const autostart = Number(new URLSearchParams(location.search).get('autostart'));
  if (autostart) window.demo.start(autostart);
});
