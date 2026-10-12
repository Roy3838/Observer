// `pnpm e2e:login`: opens Observer in the suite's profile so you can log in once by hand
// (and finish any onboarding). The session is kept in e2e/.profile for every later run.

import { test, openObserver } from './fixtures';

test('log in to Observer', async ({ observer }) => {
  test.setTimeout(15 * 60_000);
  const app = await openObserver(observer.context);
  console.log('Log in to Observer in the opened browser. Waiting up to 15 minutes...');
  await app.waitForFunction(
    () => (window as any).__observerE2E.isAuthenticated(),
    null,
    { timeout: 15 * 60_000, polling: 2_000 },
  );
  console.log('Logged in. Finish any onboarding, then close the browser window (or wait 30 s).');
  await app.waitForEvent('close', { timeout: 30_000 }).catch(() => {});
});
