// Screenshots the first-run tutorial's demos (src/utils/tutorialDemos.ts) for the picker's
// preview. Run from app/ after adding or changing a demo:
//
//   node e2e/previews.ts
//
// Each demo is shot from e2e/demos/ in its holding state (job visibly in progress, see
// demo-contract.js) at the suite's 1280×800 window size: the preview shows at up to 512 CSS px
// wide, so this stays sharp on 2x screens. Writes public/demo-previews/<demo>.jpg, which ships
// with the app (and works offline on desktop); the picker fetches them a few rows ahead.

import { chromium } from '@playwright/test';
import { mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { TUTORIAL_DEMOS } from '../src/utils/tutorialDemos.ts';

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, '../public/demo-previews');
mkdirSync(out, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
for (const d of TUTORIAL_DEMOS) {
  await page.goto(pathToFileURL(join(here, 'demos', d.file)).href, { waitUntil: 'load' });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(1500); // let the holding animation get going
  await page.screenshot({ path: join(out, d.preview.split('/').pop()!), type: 'jpeg', quality: 85 });
  console.log(`  ✓ ${d.file}`);
}
await browser.close();
