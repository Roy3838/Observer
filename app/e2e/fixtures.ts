// Shared browser for the suite: one persistent Chromium profile (e2e/.profile) that keeps the
// Auth0 session from `pnpm e2e:login` and the on-device models' download cache between runs.
// A persistent profile can't be opened twice, so the suite runs on one worker.

import { test as base, chromium, type BrowserContext, type Page } from '@playwright/test';
import { fileURLToPath } from 'node:url';

export const APP_URL = `http://localhost:${process.env.E2E_PORT ?? '3001'}/`;
export const DEMO_TITLE = 'Observer E2E Demo'; // set by demos/demo-contract.js
export const E2E_DIR = fileURLToPath(new URL('.', import.meta.url));
const PROFILE_DIR = `${E2E_DIR}.profile`;

export type SuiteEvent =
  | { type: 'trigger'; t: number; outcome: string }
  | { type: 'notification'; t: number; agentId: string; tool: string; ok: boolean; params: any }
  | { type: 'ask_user_info'; t: number; message: string }
  | { type: 'mark'; t: number; name: string; data?: any }
  | { type: 'memory'; t: number; agentId: string; tool: string; ok: boolean; targetId: string; memory: string }
  | { type: 'clip'; t: number; agentId: string; tool: string; ok: boolean; params: any };

export interface Observer {
  context: BrowserContext;
  /** Every event reported by the app or a demo, stamped with Node's clock. Cleared per test. */
  events: SuiteEvent[];
}

export const test = base.extend<{}, { observer: Observer }>({
  observer: [async ({}, use) => {
    const context = await chromium.launchPersistentContext(PROFILE_DIR, {
      headless: false,
      channel: process.env.E2E_CHANNEL || undefined, // e.g. 'chrome' to use the installed Chrome
      viewport: { width: 1280, height: 800 },
      // Meetings are heard through $ALL_AUDIO, which also opens the mic: a fake one that hears silence.
      permissions: ['notifications', 'microphone'],
      args: [
        // capture_screen's getDisplayMedia picks the demo tab with no one clicking the picker.
        `--auto-select-tab-capture-source-by-title=${DEMO_TITLE}`,
        // Observer's tab sits in the background while agents run; keep its timers at full speed.
        '--disable-background-timer-throttling',
        '--disable-renderer-backgrounding',
        '--disable-backgrounding-occluded-windows',
        '--enable-unsafe-webgpu',
        // Demos with speech start playing without a click. It plays out loud: muting Chrome
        // (--mute-audio) would also silence the tab capture the agents listen to.
        '--autoplay-policy=no-user-gesture-required',
        '--use-fake-device-for-media-stream',
        `--use-file-for-fake-audio-capture=${E2E_DIR}demos/media/silence.wav`,
      ],
    });
    const events: SuiteEvent[] = [];
    await context.exposeFunction('__e2eReport', (event: any) => {
      events.push({ ...event, t: Date.now() });
    });
    await use({ context, events });
    await context.close();
  }, { scope: 'worker' }],
});

/** Opens Observer with the e2e hooks installed and returns its page. */
export async function openObserver(context: BrowserContext): Promise<Page> {
  const app = context.pages()[0] ?? await context.newPage();
  await app.goto(APP_URL);
  await app.waitForFunction(() => !!(window as any).__observerE2E, null, { timeout: 60_000 });
  return app;
}
