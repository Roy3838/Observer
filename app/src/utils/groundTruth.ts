// src/utils/groundTruth.ts
//
// The ground truth: what Observer is for, as "When …, then …" pairs, each one running against a
// demo page in app/e2e/demos/. Two lists:
//
//   GROUND_TRUTH  the use cases Observer is measured to be good at, one per demo, in the order the
//                 landing page's When / Then wheels and the app's tutorial and Recipe Wheel show
//                 them (websites/landing via scripts/sync-showcase.mjs, tutorialDemos.ts,
//                 RecipeSplash TRIGGERS)
//   TEST_CASES    what the MCP end-to-end suite (app/e2e) also runs and nobody is shown: the other
//                 endings of a demo, near misses where staying quiet is right, and agent shapes
//                 that aren't on the wheels
//
// The suite runs both: it sends `When ${when}, then ${then}.` and judges the agent it builds, rows
// with `expect` on WHEN they notify (mcp.spec.ts), the rest with their check in e2e/patterns.ts.
//
// Data only: the app imports this, so no imports and no functions beyond the prompt.

export type Expect = 'notify' | 'silent' | 'either';

/** The suite can't fill in contact details, so every alert it tests is a browser notification. */
export const NOTIFY = 'send me a browser notification';

/** How the landing page shows an alert being delivered: the phrase replaces NOTIFY on its wheel. */
export const CHANNELS = {
  call: { label: 'Voice call', phrase: 'call me' },
  whatsapp: { label: 'WhatsApp', phrase: 'send me a WhatsApp' },
  sms: { label: 'SMS', phrase: 'send me an SMS' },
  email: { label: 'Email', phrase: 'send me an email' },
  discord: { label: 'Discord', phrase: 'ping my Discord' },
  telegram: { label: 'Telegram', phrase: 'send me a Telegram' },
} as const;
export type ChannelId = keyof typeof CHANNELS;

/** What kind of agent a use case needs, for the e2e suite's per-group pass rates (not marketing):
 *   alert     something happens or ends: tell me
 *   on_track  nag me while I'm off task or stuck
 *   change    compare against what was there before (a baseline, a deadline)
 *   record    log, count, summarize or save what happens */
export type Group = 'alert' | 'on_track' | 'change' | 'record';

export interface UseCase {
  /** Names it in test results: 'github_fails'. */
  id: string;
  group: Group;
  /** Page in app/e2e/demos/, with a query string if it needs one. */
  demo: string;
  when: string;
  /** NOTIFY, or what the agent records instead: "log it", "keep count of it", "save a clip". */
  then: string;
  /** How the demo can end: name (used in test names) → its `?outcome=` value. Default: success, fail. */
  outcomes?: Record<string, string>;
  /** What the agent should do on each outcome, judged on timing. Rows without it are judged by
   *  their check in e2e/patterns.ts. */
  expect?: Record<string, Expect>;
  /** Recordings under e2e/demos/ the demo needs: skipped everywhere until they exist. */
  requires?: string[];
  /** On GROUND_TRUTH rows: on the wheels. `channel` is how an alert is delivered on the landing page;
   *  the demo shows the rest. */
  show?: { channel?: ChannelId };
}

export const prompt = (u: Pick<UseCase, 'when' | 'then'>): string => `When ${u.when}, then ${u.then}.`;

// A job that ends either way: "when it's done" wants both endings; "if it fails" wants the
// failure and may also mention a success (the moment is right, the user still hears it ended).
const DONE: Record<string, Expect> = { success: 'notify', fail: 'notify' };
const IF_FAILS: Record<string, Expect> = { success: 'either', fail: 'notify' };

const RENDER = { success: 'success', fail: 'fail', stalled: 'stalled' };
const GMAIL = { dana_replies: 'success', bounced: 'fail', invoice_overdue: 'invoice', lead: 'lead' };

// In the order the wheels show them.
export const GROUND_TRUTH: UseCase[] = [
  // Loss falls epoch by epoch; a crash ("Your session crashed") comes partway through the rest.
  { id: 'colab_crashes', group: 'alert', demo: 'colab_training.html', when: 'my GPU training crashes', then: NOTIFY, expect: IF_FAILS,
    show: { channel: 'call' } },

  // sold_out: the event sells out while you're in line.
  { id: 'ticketmaster_out', group: 'alert', demo: 'ticketmaster_queue.html', when: "I'm out of the Ticketmaster queue", then: NOTIFY,
    outcomes: { my_turn: 'success', sold_out: 'fail' }, expect: { my_turn: 'notify', sold_out: 'notify' },
    show: { channel: 'call' } },

  // Quick work detours to Slack and MDN come first. work_video: a coding tutorial on YouTube.
  { id: 'distracted', group: 'on_track', demo: 'focus_session.html', when: 'I get distracted', then: NOTIFY,
    outcomes: { social: 'success', work_video: 'video', focused: 'fail' }, expect: { social: 'notify', work_video: 'either', focused: 'silent' },
    show: { channel: 'whatsapp' } },

  // CapCut exporting 4K. stalled: stuck at 62%, spinner still spinning (it hasn't failed, yet).
  { id: 'render_fails', group: 'alert', demo: 'capcut_render.html', when: 'my overnight render fails', then: NOTIFY,
    outcomes: RENDER, expect: { success: 'either', fail: 'notify', stalled: 'either' },
    show: { channel: 'sms' } },

  // Krita on a Cintiq: the artist draws for 5 minutes, then stops (the pen still drifts). Judged
  // on the stop by e2e/patterns.ts; no_progress_drawing is the run where they never stop.
  { id: 'no_progress', group: 'on_track', demo: 'drawing_progress.html', when: "I haven't made progress on my drawing", then: NOTIFY,
    show: { channel: 'whatsapp' } },

  { id: 'printer_fails', group: 'alert', demo: 'camera_printer.html', when: 'my 3D print fails', then: NOTIFY,
    outcomes: { spaghetti: 'success', finished: 'fail' }, expect: { spaghetti: 'notify', finished: 'either' },
    requires: ['media/camera/printer-spaghetti.mp4', 'media/camera/printer-finished.mp4'],
    show: { channel: 'telegram' } },

  // Dana Whitfield's thread was already in the inbox, read. A different Dana (Dana Kim) emails mid-run,
  // and so does a receipt for an invoice that's already paid. bounced: the follow-up to her bounces,
  // threaded into the conversation. invoice_overdue: an overdue invoice arrives. lead: a new client
  // asks about a $12k redesign.
  { id: 'gmail_lead', group: 'alert', demo: 'gmail_inbox.html', when: 'a new lead emails me', then: NOTIFY,
    outcomes: GMAIL, expect: { dana_replies: 'silent', bounced: 'silent', invoice_overdue: 'silent', lead: 'notify' },
    show: { channel: 'call' } },

  // "No appointments available" at every office, reloaded every ~15 s.
  // still_full: nothing opens during the run.
  { id: 'dmv_slot', group: 'alert', demo: 'dmv_appointments.html', when: 'an earlier appointment opens up', then: NOTIFY,
    outcomes: { slot_opens: 'success', still_full: 'fail' }, expect: { slot_opens: 'notify', still_full: 'silent' },
    show: { channel: 'whatsapp' } },

  // Reloads itself. Other cards on the page are in stock the whole time; they aren't the item.
  { id: 'restock', group: 'alert', demo: 'restock.html', when: 'the item is back in stock', then: NOTIFY,
    outcomes: { in_stock: 'success', still_out: 'fail' }, expect: { in_stock: 'notify', still_out: 'silent' },
    show: { channel: 'telegram' } },

  // A release workflow building the desktop app on four platforms.
  { id: 'github_fails', group: 'alert', demo: 'fake_github_workflow.html', when: 'my GitHub release fails', then: NOTIFY, expect: IF_FAILS,
    show: { channel: 'whatsapp' } },

  // Airbnb's own messages (payouts, review reminders) keep arriving and aren't guests.
  { id: 'airbnb_guest', group: 'alert', demo: 'airbnb_inbox.html', when: 'a guest messages me on Airbnb', then: NOTIFY,
    outcomes: { guest: 'success', system_only: 'fail' }, expect: { guest: 'notify', system_only: 'silent' },
    show: { channel: 'whatsapp' } },

  // Six model runs, a blue dot each until a green check. Three finish along the way, so "5 of 6"
  // comes before the end and isn't it. fail: the last run errors out instead.
  { id: 'annotation_done', group: 'alert', demo: 'dataannotation_runs.html', when: 'my tasks finish on DataAnnotation', then: NOTIFY, expect: DONE,
    show: { channel: 'sms' } },

  // The download reaches 100% and then Steam installs it: downloaded isn't ready to play.
  { id: 'steam_ready', group: 'alert', demo: 'steam_download.html', when: 'my Steam game is ready to play', then: NOTIFY, expect: DONE,
    show: { channel: 'email' } },

  // Other bidders bid twice under your max along the way, so you stay the high bidder.
  { id: 'ebay_outbid', group: 'alert', demo: 'ebay_auction.html', when: "I'm outbid on eBay", then: NOTIFY,
    outcomes: { outbid: 'success', still_winning: 'fail' }, expect: { outbid: 'notify', still_winning: 'silent' },
    show: { channel: 'telegram' } },

  // A move against where the price was, not a level. dip: a 1.4% scare, under the 2%.
  { id: 'btc_drop', group: 'change', demo: 'tradingview_btc.html', when: 'Bitcoin drops more than 2% in 5 minutes', then: NOTIFY,
    outcomes: { drop: 'drop', dip: 'dip' }, expect: { drop: 'notify', dip: 'silent' },
    show: { channel: 'sms' } },

  // A waiting room: a doctor's video visit or any call. denied: your request to join is denied.
  { id: 'meet_let_in', group: 'alert', demo: 'google_meet.html', when: "I'm let in from the waiting room", then: NOTIFY,
    outcomes: { let_in: 'success', denied: 'fail' }, expect: { let_in: 'notify', denied: 'either' },
    show: { channel: 'sms' } },

  { id: 'fusion360_done', group: 'alert', demo: 'fusion360_assembly.html', when: 'my Fusion 360 assembly finishes loading', then: NOTIFY, expect: DONE,
    show: { channel: 'email' } },

  // "Picking up" and "Heading your way" (with a falling ETA) come first and aren't "almost here".
  // late: still heading over, but stuck in traffic and the ETA jumps ~15 min.
  { id: 'ubereats_canceled', group: 'alert', demo: 'ubereats_tracking.html', when: 'my Uber Eats order gets canceled', then: NOTIFY,
    outcomes: { almost_here: 'success', late: 'late', canceled: 'fail' }, expect: { almost_here: 'either', late: 'silent', canceled: 'notify' },
    show: { channel: 'whatsapp' } },

  // fail = Claude stops on a permission prompt instead of finishing.
  { id: 'claude_code_permission', group: 'alert', demo: 'claude_code.html', when: 'Claude Code asks for my permission', then: NOTIFY, expect: IF_FAILS,
    show: { channel: 'whatsapp' } },

  // 48 files, each one finishing along the way; fail: the connection drops with a quarter to go.
  { id: 'drive_upload_done', group: 'alert', demo: 'drive_upload.html', when: 'my Drive upload finishes', then: NOTIFY, expect: DONE,
    show: { channel: 'sms' } },

  // Other players brag about their shinies in chat along the way; those aren't on this screen.
  { id: 'shiny', group: 'alert', demo: 'pokemon_battle.html', when: 'a shiny Pokémon spawns', then: NOTIFY,
    outcomes: { shiny: 'success', disconnected: 'fail' }, expect: { shiny: 'notify', disconnected: 'either' },
    show: { channel: 'call' } },

  // A party invite and a "queue times are longer than usual" notice pop up first; neither is the match.
  { id: 'valorant_match', group: 'alert', demo: 'valorant_queue.html', when: 'my Valorant match is found', then: NOTIFY,
    outcomes: { match_found: 'success', queue_error: 'fail' }, expect: { match_found: 'notify', queue_error: 'either' },
    show: { channel: 'call' } },

  // Group messages keep arriving the whole time. carlos_messages: Carlos messages instead of Maria.
  { id: 'whatsapp_maria', group: 'alert', demo: 'whatsapp_web.html', when: 'Maria replies to me on WhatsApp', then: NOTIFY,
    outcomes: { maria_replies: 'success', carlos_messages: 'fail' }, expect: { maria_replies: 'notify', carlos_messages: 'silent' },
    show: { channel: 'sms' } },

  // The summary comes once, after the call ends.
  { id: 'meeting_summary', group: 'record', demo: 'google_meet_call.html', when: 'my meeting ends', then: 'log a summary of the transcript',
    show: {} },

  // A news app's banner ad rotates every 10 s while the feed scrolls; the Acme ad gets one slot.
  { id: 'acme_ad', group: 'change', demo: 'ad_banner.html', when: 'the Acme ad shows up in my app',
    then: "save an image of it to memory, and if it hasn't shown up after 5 minutes, send me a browser notification",
    show: {} },

  // The new orders, not today's total.
  { id: 'shopify_count', group: 'record', demo: 'shopify_orders.html', when: 'a new Shopify order comes in', then: 'keep count of it',
    show: {} },

  // A note every 2 minutes naming what was on screen in that window.
  { id: 'workday_notes', group: 'record', demo: 'workday.html', when: 'two minutes pass', then: 'log what I worked on',
    show: {} },

  // Logger + one-shot summarizer: the topics, after the video ends.
  { id: 'youtube_topics', group: 'record', demo: 'youtube_watch.html', when: 'this YouTube video ends', then: 'log all the topics',
    show: {} },

  // Every message once, though each stays visible for ~100 s.
  { id: 'twitch_chat', group: 'record', demo: 'twitch_chat.html', when: 'five seconds pass', then: 'log new Twitch chat, no duplicates',
    show: {} },

  // Warnings print either way (a build-script one while holding, an unused variable at the end).
  { id: 'rust_done', group: 'alert', demo: 'cargo_build.html', when: 'my Rust build finishes', then: NOTIFY, expect: DONE,
    show: { channel: 'telegram' } },

  { id: 'linux_done', group: 'alert', demo: 'linux_update.html', when: 'my Linux update finishes', then: NOTIFY, expect: DONE,
    show: { channel: 'email' } },

  // The menu bar battery drains; plugged_in: the charger goes in at 31%, so it never got low.
  { id: 'battery_low', group: 'alert', demo: 'macos_battery.html', when: 'my battery is low', then: NOTIFY,
    outcomes: { low: 'success', plugged_in: 'fail' }, expect: { low: 'notify', plugged_in: 'silent' },
    show: { channel: 'telegram' } },

  // Your store, reloading itself. The banner and a "people shopping" counter change; that's not down.
  { id: 'site_down', group: 'alert', demo: 'site_down.html', when: 'my website goes down', then: NOTIFY,
    outcomes: { down: 'success', up: 'fail' }, expect: { down: 'notify', up: 'silent' },
    show: { channel: 'sms' } },
];

// What the suite runs besides GROUND_TRUTH: agent shapes the wheels don't show. Not rewordings of a
// shown row: the suite tests the shape of the agent the MCP builds, not how well it reads a phrase.
export const TEST_CASES: UseCase[] = [
  // Two alerts from one request: out of the queue, then 2 minutes left on the ticket hold.
  { id: 'ticketmaster_two_alerts', group: 'alert', demo: 'ticketmaster_queue.html?outcome=success&timer=200', when: "I'm out of the Ticketmaster queue, and again when I have 2 minutes left to pick my tickets", then: NOTIFY },
  { id: 'gmail_dana_or_bounce', group: 'alert', demo: 'gmail_inbox.html', when: 'Dana Whitfield replies or my email to her bounces', then: NOTIFY,
    outcomes: GMAIL, expect: { dana_replies: 'notify', bounced: 'notify', invoice_overdue: 'silent', lead: 'silent' } },
  // acme_ad_missing: it never shows, so the 5-minute notification is the only thing to do.
  { id: 'acme_ad_missing', group: 'change', demo: 'ad_banner.html?outcome=fail', when: 'the Acme ad shows up in my app',
    then: "save an image of it to memory, and if it hasn't shown up after 5 minutes, send me a browser notification" },
  // Each error stays on screen for several iterations: logged once each.
  { id: 'terminal_errors', group: 'record', demo: 'terminal_errors.html', when: "there's an error in my terminal", then: 'log it' },
  // The shapes from the e2e matrix that aren't on the wheels.

  // Nag while it holds: on X at the end, back to work 5 minutes later. At least two alerts while
  // away, none once back.
  { id: 'social_nag', group: 'on_track', demo: 'focus_session.html?outcome=returns', when: "I'm on social media",
    then: 'send me a browser notification every 2 minutes until I get back to work' },
  // no_progress, where the artist never stops: no alert at all.
  { id: 'no_progress_drawing', group: 'on_track', demo: 'drawing_progress.html?outcome=drawing', when: "I haven't made progress on my drawing", then: NOTIFY },

  // A deadline, kept in the agent's code or memory. Carlos messages 3 minutes in and Maria never
  // does: an alert around the 5-minute mark. maria_deadline_replied: she replies at 3 minutes, so never.
  { id: 'maria_deadline', group: 'change', demo: 'whatsapp_web.html?outcome=fail', when: "Maria hasn't replied to me on WhatsApp in 5 minutes", then: NOTIFY },
  { id: 'maria_deadline_replied', group: 'change', demo: 'whatsapp_web.html?outcome=success', when: "Maria hasn't replied to me on WhatsApp in 5 minutes", then: NOTIFY },
  // A stall: progress that stops moving. Progress creeps for 3 minutes, then sticks at 62% for good.
  { id: 'render_stalled', group: 'change', demo: 'capcut_render.html?outcome=stalled', when: "my render hasn't moved in 3 minutes", then: NOTIFY },

  // A sequence: tests pass, then the deploy fails. Tests failing (deploy skipped) and an all-green
  // pipeline are both silence.
  { id: 'deploy_after_tests', group: 'alert', demo: 'gitlab_pipeline.html', when: 'my pipeline passes its tests and then the deploy fails', then: NOTIFY,
    outcomes: { deploy_failed: 'deploy_failed', test_failed: 'fail', all_green: 'success' }, expect: { deploy_failed: 'notify', test_failed: 'silent', all_green: 'silent' } },

  // Capture on a level: a clip of the person, started when they appear.
  { id: 'person_clip', group: 'record', demo: 'camera_room.html?outcome=success', when: "there's a person on my camera", then: 'record a clip of it',
    requires: ['media/camera/room-person.mp4', 'media/camera/room-cat.mp4'] },

  // A clip started or marked right after each death.
  { id: 'minecraft_clips', group: 'record', demo: 'minecraft.html', when: 'my character dies', then: 'save a clip', requires: ['media/minecraft.mp4'] },
];

/** Everything the e2e suite runs. */
export const ALL_CASES: UseCase[] = [...GROUND_TRUTH, ...TEST_CASES];

/** The demo page's file name, without the query string. */
export const demoFile = (u: UseCase): string => u.demo.split('?')[0];

/** Demos that are a camera feed rather than a screen. */
export const isCameraDemo = (u: UseCase): boolean => demoFile(u).startsWith('camera_');

/** The one shown pair per demo, in order. */
export const SHOWN = GROUND_TRUTH.filter(u => u.show);
