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
// Opened by the app's first-run tutorial, `?tutorial=30000` holds until the user comes back to
// the tab, then ends the job 30 s later (see the bottom of this file).

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

// Tutorial mode. Opening the demo brings this tab to the front, but the user has to go back to
// Observer to share it, and Chrome brings them back here the moment they do. So the job holds
// until that return and the user watches it finish while Observer watches with them. A card
// says what to do meanwhile; it hides as soon as they leave, so it's never in what Observer sees.
const tutorialMs = Number(new URLSearchParams(location.search).get('tutorial'));
if (tutorialMs) {
  // The desktop app captures the browser window natively, so nobody ever comes back to the tab
  // to notice: start anyway once there's been time to build the agent.
  const FALLBACK_MS = 60000;
  const start = () => window.demo.start(tutorialMs);
  const card = tutorialCard(tourTitle);
  let left = false;
  const leave = () => { left = true; card.remove(); };
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) leave();
    else if (left) start();
  });
  // Switching to another app (the desktop app) blurs the window without hiding the tab.
  window.addEventListener('blur', leave);
  setTimeout(start, FALLBACK_MS);
}

function tutorialCard(title) {
  const host = document.createElement('div');
  host.style.cssText = 'position:fixed;top:20px;left:50%;transform:translateX(-50%);z-index:2147483647;';
  // Shadow DOM so the demo page's own styles can't reach the card.
  const root = host.attachShadow({ mode: 'open' });
  const style = document.createElement('style');
  style.textContent = `
    .card { all: initial; display: block; box-sizing: border-box; max-width: min(440px, calc(100vw - 32px));
      padding: 14px 18px; border-radius: 16px; background: #0f172a; color: #fff;
      font: 14px/1.45 system-ui, -apple-system, 'Segoe UI', sans-serif; text-align: center;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.35); }
    strong { font-weight: 600; }
    span { color: rgba(255, 255, 255, 0.75); }`;
  const card = document.createElement('div');
  card.className = 'card';
  // Text only: the title comes from the URL.
  const head = document.createElement('strong');
  head.textContent = title ? `This is your ${title} demo.` : 'This is your demo.';
  const body = document.createElement('span');
  body.textContent = ' Go back to Observer and share this tab. It starts running when you come back.';
  card.append(head, body);
  root.append(style, card);
  const mount = () => { if (host.isConnected === false && !host.dataset.gone) document.body.append(host); };
  if (document.body) mount(); else document.addEventListener('DOMContentLoaded', mount);
  return { remove: () => { host.dataset.gone = '1'; host.remove(); } };
}
