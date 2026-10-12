// MCP end-to-end suite. `pnpm e2e:login` once, then `pnpm e2e`. See e2e/README.md.

import { defineConfig } from '@playwright/test';

// One id per run, shared with the worker through the environment: results land in e2e/results/<id>/.
process.env.E2E_RUN_ID ??= new Date().toISOString().replace(/[:.]/g, '-');
// Auth0 only redirects back to the usual dev port, so change it only for runs that don't log in.
const port = process.env.E2E_PORT ?? '3001';

export default defineConfig({
  testDir: '.',
  outputDir: './test-results',
  // One persistent browser profile (fixtures.ts), and one screen stream at a time.
  workers: 1,
  fullyParallel: false,
  reporter: [['list'], ['html', { outputFolder: './report', open: 'never' }]],
  globalTeardown: './summary.ts',
  webServer: {
    command: `pnpm vite --port ${port} --strictPort`,
    cwd: '..',
    url: `http://localhost:${port}`,
    env: { VITE_E2E: '1' },
    // A dev server that's already up wasn't started with VITE_E2E=1, so don't reuse it.
    reuseExistingServer: false,
    timeout: 120_000,
  },
  projects: [
    { name: 'login', testMatch: /login\.spec\.ts/ },
    { name: 'mcp', testMatch: /mcp\.spec\.ts/ },
    { name: 'patterns', testMatch: /patterns\.spec\.ts/ },
  ],
});
