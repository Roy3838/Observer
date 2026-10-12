// How each e2e demo is presented on the landing page (websites/landing).
//
// The e2e suite stays the ground truth for what Observer is useful for: the landing page shows
// exactly the prompts in scenarios.ts / patterns.ts, running against exactly these demo pages.
// This file only adds presentation: a name, a category and a one-line pitch per demo.
//
// websites/landing/scripts/sync-showcase.mjs reads it at build time and fails if a demo used by
// a scenario has no entry here (or an entry points at a demo no scenario uses), so a new demo
// can't ship without showing up on the site.

export const CATEGORIES = [
  { id: 'ship', label: 'Builds & deploys' },
  { id: 'data', label: 'Data & ML' },
  { id: 'wait', label: 'Alerts & arrivals' },
  { id: 'media', label: 'Meetings & media' },
] as const;

// How the landing page shows each alert being delivered. The e2e prompts all ask for a browser
// notification (the one channel the suite can observe); the site swaps in `phrase` so visitors
// see Observer's real integrations. Display only: nothing here changes what the suite tests.
export const CHANNELS = {
  call: { label: 'Voice call', phrase: 'call me' },
  whatsapp: { label: 'WhatsApp', phrase: 'send me a WhatsApp' },
  sms: { label: 'SMS', phrase: 'text me' },
  email: { label: 'Email', phrase: 'email me' },
  discord: { label: 'Discord', phrase: 'ping me on Discord' },
  telegram: { label: 'Telegram', phrase: 'send me a Telegram' },
} as const;

export type ChannelId = keyof typeof CHANNELS;

export type CategoryId = (typeof CATEGORIES)[number]['id'];

export interface ShowcaseEntry {
  /** What the thing is called on the site: "Vercel deployment". */
  title: string;
  category: CategoryId;
  /** One line on what Observer does here. */
  pitch: string;
  /** Delivery channel shown on the site for alert prompts. */
  channel?: ChannelId;
}

/** Keyed by demo file name in e2e/demos/ (no query string). The order here is the order on the site. */
export const SHOWCASE: Record<string, ShowcaseEntry> = {
  'fake_github_workflow.html': { title: 'GitHub Actions', category: 'ship', pitch: 'Know the second a workflow finishes or breaks.', channel: 'telegram' },
  'claude_code.html': { title: 'Claude Code', category: 'ship', pitch: 'Walk away while your coding agent works. Come back when it needs you.', channel: 'whatsapp' },
  'vercel_deploy.html': { title: 'Vercel deployment', category: 'ship', pitch: 'Hear about a failed deploy before your users do.', channel: 'discord' },
  'cargo_build.html': { title: 'Rust build', category: 'ship', pitch: 'Go get coffee while 700 crates compile. Hear if it breaks.', channel: 'telegram' },
  'androidstudio_build.html': { title: 'Android Studio build', category: 'ship', pitch: 'Gradle takes its time. You don\'t have to watch it.', channel: 'discord' },
  'cloudformation_stack.html': { title: 'CloudFormation stack', category: 'ship', pitch: 'Stack updates take forever. Get told when yours is done.', channel: 'email' },
  'gitlab_pipeline.html': { title: 'GitLab pipeline', category: 'ship', pitch: 'Stop refreshing the pipeline page.', channel: 'email' },
  'argocd_sync.html': { title: 'Argo CD rollout', category: 'ship', pitch: 'Watch a Kubernetes rollout settle without staring at it.', channel: 'discord' },
  'terminal_errors.html': { title: 'Terminal errors', category: 'ship', pitch: 'A running log of every error your dev server prints.' },

  'fake_simulation.html': { title: 'Simulation solver', category: 'data', pitch: 'Catch a diverged or crashed solver the moment it happens.', channel: 'call' },
  'colab_training.html': { title: 'Colab training run', category: 'data', pitch: 'Know when a training run finishes or crashes.', channel: 'whatsapp' },
  'bigquery_query.html': { title: 'BigQuery query', category: 'data', pitch: 'Long query, short attention span. Get pinged when it lands.', channel: 'sms' },
  'wandb_run.html': { title: 'Weights & Biases run', category: 'data', pitch: 'Keep an eye on a long experiment from anywhere.', channel: 'telegram' },
  'fusion360_assembly.html': { title: 'Fusion 360 assembly', category: 'data', pitch: 'Heavy CAD files, light on your time.', channel: 'email' },

  'whatsapp_web.html': { title: 'WhatsApp reply', category: 'wait', pitch: 'Ignore the group chat. Hear about the one person who matters.', channel: 'sms' },
  'ticketmaster_queue.html': { title: 'Ticketmaster queue', category: 'wait', pitch: 'Never miss your turn, and never miss the sell-out.', channel: 'call' },
  'ubereats_tracking.html': { title: 'Uber Eats delivery', category: 'wait', pitch: 'Head downstairs when the driver is close, not before.', channel: 'sms' },
  'gmail_inbox.html': { title: 'Gmail reply', category: 'wait', pitch: 'Stop refreshing your inbox for the one email you\'re waiting on.', channel: 'whatsapp' },
  'tradingview_btc.html': { title: 'TradingView price alert', category: 'wait', pitch: 'Name your price. Observer watches the chart so you don\'t have to.', channel: 'call' },
  'google_meet.html': { title: 'Google Meet lobby', category: 'wait', pitch: 'Get told the moment the host lets you in.', channel: 'whatsapp' },
  'shopify_orders.html': { title: 'Shopify orders', category: 'wait', pitch: 'A running count of orders as they arrive.' },
  'fake_download.html': { title: 'Download', category: 'wait', pitch: 'Know when a big download is done, or when it died.', channel: 'sms' },
  'youtube_studio.html': { title: 'YouTube upload', category: 'wait', pitch: 'Uploads and processing finish on their own clock.', channel: 'telegram' },

  'youtube_watch.html': { title: 'YouTube video', category: 'media', pitch: 'Timestamps and chapters for anything you watch.' },
  'twitch_chat.html': { title: 'Twitch chat', category: 'media', pitch: 'Capture a fast-moving chat in full.' },
  'google_meet_call.html': { title: 'Meeting notes', category: 'media', pitch: 'Live transcript and a summary when the call ends.' },
  'workday.html': { title: 'Your workday', category: 'media', pitch: 'A periodic note of what you actually worked on.' },
  'minecraft.html': { title: 'Minecraft', category: 'media', pitch: 'Clip every moment worth keeping.' },
};
