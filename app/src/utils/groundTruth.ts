// src/utils/groundTruth.ts
//
// The ground truth: everything Observer is for, as "When …, then …" pairs, each one running against
// a demo page in app/e2e/demos/. One list feeds every place that shows or tests a use case:
//
//   - the MCP end-to-end suite (app/e2e) sends `When ${when}, then ${then}.` and judges the agent
//     it builds: rows with `expect` on WHEN they notify (mcp.spec.ts), the rest with their check in
//     e2e/patterns.ts (what they logged, counted or clipped, and when)
//   - the landing page's When / Then wheels (websites/landing, via scripts/sync-showcase.mjs)
//   - the app's first-run tutorial (tutorialDemos.ts) and Recipe Wheel (RecipeSplash TRIGGERS)
//
// Every demo has exactly one row with `show`: the pair the landing page shows and the When the
// app's wheels offer. The other rows on the same demo are what the suite also tests: the other
// endings, and the near misses where staying quiet is the right answer.
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

export interface UseCase {
  /** Names it in test results: 'docker_built'. */
  id: string;
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
  /** The demo's one shown pair. `channel` is how its alert is delivered on the landing page;
   *  `alert` is what that notification (or log line) says. */
  show?: { title: string; channel?: ChannelId; alert: string };
}

export const prompt = (u: Pick<UseCase, 'when' | 'then'>): string => `When ${u.when}, then ${u.then}.`;

// A job that ends either way: "when it's done" wants both endings; "if it fails" wants the
// failure and may also mention a success (the moment is right, the user still hears it ended).
const DONE: Record<string, Expect> = { success: 'notify', fail: 'notify' };
const IF_FAILS: Record<string, Expect> = { success: 'either', fail: 'notify' };

const GMAIL = { dana_replies: 'success', bounced: 'fail', invoice_overdue: 'invoice', lead: 'lead' };

// The order here is the order on the landing page's wheels and in the app's: what a miss costs.
// First the alerts where missing the moment costs money or a deadline, then real pain that comes
// up less often, then conveniences and logs. Pages that only change when reloaded (an appointment
// opening, a restock) reload themselves in the demo; for real, the agent click()s refresh.
export const GROUND_TRUTH: UseCase[] = [
  // ── A miss costs money or a deadline ─────────────────────────────────────────

  // Loss falls epoch by epoch; a crash ("Your session crashed") comes partway through the rest.
  { id: 'colab_crashes', demo: 'colab_training.html', when: 'my GPU training crashes', then: NOTIFY, expect: IF_FAILS,
    show: { title: 'Google Colab', channel: 'call', alert: 'Your training crashed. The GPU is still billing.' } },
  { id: 'colab_done', demo: 'colab_training.html', when: 'my Colab training finishes', then: NOTIFY, expect: DONE },

  // "No appointments available" at every office, reloaded every ~15 s.
  // still_full: nothing opens during the run.
  { id: 'dmv_slot', demo: 'dmv_appointments.html', when: 'an appointment is available', then: NOTIFY,
    outcomes: { slot_opens: 'success', still_full: 'fail' }, expect: { slot_opens: 'notify', still_full: 'silent' },
    show: { title: 'DMV', channel: 'whatsapp', alert: 'A DMV appointment just opened up. Book it now!' } },

  // sold_out: the event sells out while you're in line.
  { id: 'ticketmaster_out', demo: 'ticketmaster_queue.html', when: "I'm out of the Ticketmaster queue", then: NOTIFY,
    outcomes: { my_turn: 'success', sold_out: 'fail' }, expect: { my_turn: 'notify', sold_out: 'notify' },
    show: { title: 'Ticketmaster', channel: 'call', alert: "You're out of the queue. Pick your tickets!" } },
  { id: 'ticketmaster_turn', demo: 'ticketmaster_queue.html', when: "it's my turn to buy tickets", then: NOTIFY,
    outcomes: { my_turn: 'success', sold_out: 'fail' }, expect: { my_turn: 'notify', sold_out: 'either' } },
  { id: 'ticketmaster_sold_out', demo: 'ticketmaster_queue.html', when: 'the tickets sell out before I get in', then: NOTIFY,
    outcomes: { my_turn: 'success', sold_out: 'fail' }, expect: { my_turn: 'either', sold_out: 'notify' } },
  // Two alerts from one request: out of the queue, then 2 minutes left on the ticket hold.
  { id: 'ticketmaster_two_alerts', demo: 'ticketmaster_queue.html?outcome=success&timer=200', when: "I'm out of the Ticketmaster queue, and again when I have 2 minutes left to pick my tickets", then: NOTIFY },

  // Rendered, not recorded: she potters about and bends down to pick things up while holding.
  // lies_down: she lies down on the couch on purpose and pulls a blanket over.
  { id: 'fall', demo: 'camera_fall.html', when: 'someone falls in my living room', then: NOTIFY,
    outcomes: { falls: 'success', lies_down: 'fail' }, expect: { falls: 'notify', lies_down: 'silent' },
    show: { title: 'Home camera', channel: 'call', alert: "Grandma fell in the living room and hasn't gotten up." } },
  { id: 'fall_grandma', demo: 'camera_fall.html', when: 'my grandma falls', then: NOTIFY,
    outcomes: { falls: 'success', lies_down: 'fail' }, expect: { falls: 'notify', lies_down: 'silent' } },

  // Premiere Pro exporting a sequence.
  { id: 'render_fails', demo: 'fake_render.html', when: 'my overnight render fails', then: NOTIFY, expect: IF_FAILS,
    show: { title: 'Premiere Pro', channel: 'sms', alert: 'Your render failed at 62%. The deadline is in 6 hours.' } },
  { id: 'render_done', demo: 'fake_render.html', when: 'my render finishes', then: NOTIFY, expect: DONE },

  // Dana Whitfield's thread was already in the inbox, read. A different Dana (Dana Kim) emails mid-run,
  // and so does a receipt for an invoice that's already paid. bounced: the follow-up to her bounces,
  // threaded into the conversation. invoice_overdue: an overdue invoice arrives. lead: a new client
  // asks about a $12k redesign.
  { id: 'gmail_lead', demo: 'gmail_inbox.html', when: 'a new lead emails me', then: NOTIFY,
    outcomes: GMAIL, expect: { dana_replies: 'silent', bounced: 'silent', invoice_overdue: 'silent', lead: 'notify' },
    show: { title: 'Gmail', channel: 'call', alert: 'A new lead just emailed: a $12k website redesign.' } },
  { id: 'gmail_dana', demo: 'gmail_inbox.html', when: 'Dana Whitfield replies to my email', then: NOTIFY,
    outcomes: GMAIL, expect: { dana_replies: 'notify', bounced: 'either', invoice_overdue: 'silent', lead: 'silent' } },
  { id: 'gmail_dana_or_bounce', demo: 'gmail_inbox.html', when: 'Dana Whitfield replies or my email to her bounces', then: NOTIFY,
    outcomes: GMAIL, expect: { dana_replies: 'notify', bounced: 'notify', invoice_overdue: 'silent', lead: 'silent' } },
  { id: 'gmail_bounce', demo: 'gmail_inbox.html', when: 'my email to Dana Whitfield bounces', then: NOTIFY,
    outcomes: GMAIL, expect: { dana_replies: 'either', bounced: 'notify', invoice_overdue: 'silent', lead: 'silent' } },
  { id: 'gmail_invoice', demo: 'gmail_inbox.html', when: 'I get an email about an overdue invoice', then: NOTIFY,
    outcomes: GMAIL, expect: { dana_replies: 'silent', bounced: 'silent', invoice_overdue: 'notify', lead: 'silent' } },

  // Behind a login, reloaded every ~20 s. Partway through, a slot LATER than the current appointment
  // opens: not an earlier one. logged_out: the session expires and the page asks to sign in again.
  { id: 'visa_slot', demo: 'visa_portal.html', when: 'an earlier appointment opens up', then: NOTIFY,
    outcomes: { slot_opens: 'success', still_full: 'fail', logged_out: 'expired' }, expect: { slot_opens: 'notify', still_full: 'silent', logged_out: 'either' },
    show: { title: 'Visa appointment', channel: 'call', alert: 'An earlier visa appointment opened: Nov 20. Reschedule now!' } },
  { id: 'visa_logged_out', demo: 'visa_portal.html', when: 'I get logged out of the visa portal', then: NOTIFY,
    outcomes: { slot_opens: 'success', still_full: 'fail', logged_out: 'expired' }, expect: { slot_opens: 'silent', still_full: 'silent', logged_out: 'notify' } },

  // Airbnb's own messages (payouts, review reminders) keep arriving and aren't guests.
  { id: 'airbnb_guest', demo: 'airbnb_inbox.html', when: 'a guest messages me on Airbnb', then: NOTIFY,
    outcomes: { guest: 'success', system_only: 'fail' }, expect: { guest: 'notify', system_only: 'silent' },
    show: { title: 'Airbnb', channel: 'whatsapp', alert: 'A guest messaged you: the lockbox code isn\'t working.' } },

  // Reloads itself. Other cards on the page are in stock the whole time; they aren't the item.
  { id: 'restock', demo: 'restock.html', when: 'the item is back in stock', then: NOTIFY,
    outcomes: { in_stock: 'success', still_out: 'fail' }, expect: { in_stock: 'notify', still_out: 'silent' },
    show: { title: 'Restock', channel: 'telegram', alert: 'The RTX 5090 is back in stock. Go!' } },

  // Your store, reloading itself. The banner and a "people shopping" counter change; that's not down.
  { id: 'site_down', demo: 'site_down.html', when: 'my website goes down', then: NOTIFY,
    outcomes: { down: 'success', up: 'fail' }, expect: { down: 'notify', up: 'silent' },
    show: { title: 'Your website', channel: 'sms', alert: 'lumencoffee.co is down: 502 Bad Gateway.' } },

  // A release workflow building the desktop app on four platforms.
  { id: 'github_fails', demo: 'fake_github_workflow.html', when: 'my GitHub release fails', then: NOTIFY, expect: IF_FAILS,
    show: { title: 'GitHub Actions', channel: 'whatsapp', alert: 'Your GitHub release build failed.' } },
  { id: 'github_done', demo: 'fake_github_workflow.html', when: 'my GitHub workflow finishes', then: NOTIFY, expect: DONE },

  { id: 'printer_fails', demo: 'camera_printer.html', when: 'my 3D print fails', then: NOTIFY,
    outcomes: { spaghetti: 'success', finished: 'fail' }, expect: { spaghetti: 'notify', finished: 'either' },
    requires: ['media/camera/printer-spaghetti.mp4', 'media/camera/printer-finished.mp4'],
    show: { title: '3D printer', channel: 'telegram', alert: 'Your 3D print just failed.' } },
  { id: 'printer_done', demo: 'camera_printer.html', when: 'my 3D print is done or fails', then: NOTIFY,
    outcomes: { spaghetti: 'success', finished: 'fail' }, expect: { spaghetti: 'notify', finished: 'notify' },
    requires: ['media/camera/printer-spaghetti.mp4', 'media/camera/printer-finished.mp4'] },

  // ── Real pain, less often ─────────────────────────────────────────────────────

  // Six model runs, a blue dot each until a green check. Three finish along the way, so "5 of 6"
  // comes before the end and isn't it. fail: the last run errors out instead.
  { id: 'annotation_done', demo: 'dataannotation_runs.html', when: 'my tasks finish on DataAnnotation', then: NOTIFY, expect: DONE,
    show: { title: 'DataAnnotation', channel: 'sms', alert: 'All 6 runs are done. Time to rate them.' } },
  { id: 'annotation_error', demo: 'dataannotation_runs.html', when: 'a run errors out on DataAnnotation', then: NOTIFY, expect: IF_FAILS },

  // Other bidders bid twice under your max along the way, so you stay the high bidder.
  { id: 'ebay_outbid', demo: 'ebay_auction.html', when: "I'm outbid on eBay", then: NOTIFY,
    outcomes: { outbid: 'success', still_winning: 'fail' }, expect: { outbid: 'notify', still_winning: 'silent' },
    show: { title: 'eBay', channel: 'telegram', alert: "You've been outbid on the Leica M6." } },

  // A price level rather than an end state. Two near misses (~80,120, ~80,040) come first.
  { id: 'btc_below', demo: 'tradingview_btc.html', when: 'Bitcoin drops below my $80,000 stop', then: NOTIFY,
    outcomes: { below_80k: 'success', above_82_5k: 'fail' }, expect: { below_80k: 'notify', above_82_5k: 'silent' },
    show: { title: 'TradingView', channel: 'sms', alert: 'Bitcoin hit your $80,000 stop.' } },
  { id: 'btc_either', demo: 'tradingview_btc.html', when: 'Bitcoin goes below $80,000 or above $82,500', then: NOTIFY,
    outcomes: { below_80k: 'success', above_82_5k: 'fail' }, expect: { below_80k: 'notify', above_82_5k: 'notify' } },
  { id: 'btc_above', demo: 'tradingview_btc.html', when: 'Bitcoin breaks above $82,500', then: NOTIFY,
    outcomes: { below_80k: 'success', above_82_5k: 'fail' }, expect: { below_80k: 'silent', above_82_5k: 'notify' } },

  // Reloads itself; only its "last updated" time moves until the status does.
  { id: 'passport_status', demo: 'passport_status.html', when: 'my passport status changes', then: NOTIFY,
    outcomes: { approved: 'success', still_in_process: 'fail' }, expect: { approved: 'notify', still_in_process: 'silent' },
    show: { title: 'Passport status', channel: 'email', alert: 'Your passport was approved.' } },

  // A waiting room: a doctor's video visit or any call. denied: your request to join is denied.
  { id: 'meet_let_in', demo: 'google_meet.html', when: "I'm let in from the waiting room", then: NOTIFY,
    outcomes: { let_in: 'success', denied: 'fail' }, expect: { let_in: 'notify', denied: 'either' },
    show: { title: 'Waiting room', channel: 'sms', alert: "You've been let in. Join now." } },
  { id: 'meet_answered', demo: 'google_meet.html', when: 'someone answers my request to join the Meet call', then: NOTIFY,
    outcomes: { let_in: 'success', denied: 'fail' }, expect: { let_in: 'notify', denied: 'notify' } },

  // "Picking up" and "Heading your way" (with a falling ETA) come first and aren't "almost here".
  // late: still heading over, but stuck in traffic and the ETA jumps ~15 min.
  { id: 'ubereats_canceled', demo: 'ubereats_tracking.html', when: 'my Uber Eats order gets canceled', then: NOTIFY,
    outcomes: { almost_here: 'success', late: 'late', canceled: 'fail' }, expect: { almost_here: 'either', late: 'silent', canceled: 'notify' },
    show: { title: 'Uber Eats', channel: 'whatsapp', alert: 'Your Uber Eats order was canceled.' } },
  { id: 'ubereats_close', demo: 'ubereats_tracking.html', when: 'my Uber Eats driver is almost here', then: NOTIFY,
    outcomes: { almost_here: 'success', late: 'late', canceled: 'fail' }, expect: { almost_here: 'notify', late: 'silent', canceled: 'either' } },
  { id: 'ubereats_close_or_canceled', demo: 'ubereats_tracking.html', when: 'my Uber Eats driver is almost here or my order gets canceled', then: NOTIFY,
    outcomes: { almost_here: 'success', late: 'late', canceled: 'fail' }, expect: { almost_here: 'notify', late: 'either', canceled: 'notify' } },
  { id: 'ubereats_wrong', demo: 'ubereats_tracking.html', when: 'anything goes wrong with my Uber Eats order', then: NOTIFY,
    outcomes: { almost_here: 'success', late: 'late', canceled: 'fail' }, expect: { almost_here: 'either', late: 'notify', canceled: 'notify' } },

  { id: 'coyote', demo: 'camera_yard.html', when: 'a coyote is in my yard', then: NOTIFY,
    outcomes: { coyote: 'success', dog: 'fail' }, expect: { coyote: 'notify', dog: 'silent' },
    requires: ['media/camera/yard-coyote.mp4', 'media/camera/yard-dog.mp4'],
    show: { title: 'Security camera', channel: 'telegram', alert: 'A coyote is in the backyard.' } },
  { id: 'coyote_or_dog', demo: 'camera_yard.html', when: 'a dog or a coyote shows up on my security camera', then: NOTIFY,
    outcomes: { coyote: 'success', dog: 'fail' }, expect: { coyote: 'notify', dog: 'notify' },
    requires: ['media/camera/yard-coyote.mp4', 'media/camera/yard-dog.mp4'] },

  // fail = Claude stops on a permission prompt instead of finishing.
  { id: 'claude_code_permission', demo: 'claude_code.html', when: 'Claude Code asks for my permission', then: NOTIFY, expect: IF_FAILS,
    show: { title: 'Claude Code', channel: 'whatsapp', alert: 'Claude Code is waiting for your permission.' } },
  { id: 'claude_code_done', demo: 'claude_code.html', when: 'Claude Code finishes or needs me', then: NOTIFY, expect: DONE },

  // Quick work detours to Slack and MDN come first. work_video: a coding tutorial on YouTube.
  { id: 'distracted', demo: 'focus_session.html', when: 'I get distracted', then: NOTIFY,
    outcomes: { social: 'success', work_video: 'video', focused: 'fail' }, expect: { social: 'notify', work_video: 'either', focused: 'silent' },
    show: { title: 'Focus', channel: 'whatsapp', alert: 'You drifted to X. Back to work?' } },
  // No work_video ending: YouTube is arguably social media, so neither answer would be wrong.
  { id: 'social_media', demo: 'focus_session.html', when: 'I go on social media', then: NOTIFY,
    outcomes: { social: 'success', focused: 'fail' }, expect: { social: 'notify', focused: 'silent' } },
  { id: 'opens_youtube', demo: 'focus_session.html', when: 'I open YouTube', then: NOTIFY,
    outcomes: { social: 'success', work_video: 'video', focused: 'fail' }, expect: { social: 'silent', work_video: 'notify', focused: 'silent' } },

  // An alert right after each death; and a clip started or marked right after each one.
  { id: 'minecraft_deaths', demo: 'minecraft.html', when: 'my Minecraft character dies', then: NOTIFY, requires: ['media/minecraft.mp4'],
    show: { title: 'Minecraft', channel: 'telegram', alert: 'You died in Minecraft.' } },
  { id: 'minecraft_clips', demo: 'minecraft.html', when: 'my character dies', then: 'save a clip', requires: ['media/minecraft.mp4'] },

  // ── Conveniences and logs ──────────────────────────────────────────────────────

  // Each plan step ticks done along the way, and the spreadsheet is created at step 4 of 5.
  { id: 'cowork_approval', demo: 'claude_cowork.html', when: 'Claude Cowork needs my approval', then: NOTIFY,
    outcomes: { done: 'success', needs_approval: 'fail' }, expect: { done: 'either', needs_approval: 'notify' },
    show: { title: 'Claude Cowork', channel: 'telegram', alert: 'Claude Cowork needs your approval.' } },
  { id: 'cowork_done', demo: 'claude_cowork.html', when: 'Claude Cowork finishes or needs me', then: NOTIFY,
    outcomes: { done: 'success', needs_approval: 'fail' }, expect: { done: 'notify', needs_approval: 'notify' } },

  // Plain progress prints "Pull complete" per layer and "DONE 0.4s" per build step all the way through.
  // pull_failed: the image download times out before anything is built.
  { id: 'docker_fails', demo: 'docker_build.html', when: 'my Docker build fails', then: NOTIFY,
    outcomes: { built: 'success', build_failed: 'fail', pull_failed: 'pull' }, expect: { built: 'either', build_failed: 'notify', pull_failed: 'notify' },
    show: { title: 'Docker', channel: 'discord', alert: 'Your Docker build failed.' } },
  { id: 'docker_built', demo: 'docker_build.html', when: 'my Docker image is built', then: NOTIFY,
    outcomes: { built: 'success', build_failed: 'fail', pull_failed: 'pull' }, expect: { built: 'notify', build_failed: 'notify', pull_failed: 'notify' } },

  { id: 'vercel_fails', demo: 'vercel_deploy.html', when: 'my Vercel deployment fails', then: NOTIFY, expect: IF_FAILS,
    show: { title: 'Vercel', channel: 'discord', alert: 'Your Vercel deployment failed.' } },
  { id: 'vercel_done', demo: 'vercel_deploy.html', when: 'my Vercel deployment finishes', then: NOTIFY, expect: DONE },

  { id: 'simulation_crashes', demo: 'fake_simulation.html', when: 'my simulation crashes', then: NOTIFY, expect: IF_FAILS,
    show: { title: 'Simulation', channel: 'telegram', alert: 'Your simulation crashed.' } },
  { id: 'simulation_done', demo: 'fake_simulation.html', when: 'my simulation finishes', then: NOTIFY, expect: DONE },

  { id: 'youtube_upload_fails', demo: 'youtube_studio.html', when: 'my YouTube upload fails', then: NOTIFY, expect: IF_FAILS,
    show: { title: 'YouTube Studio', channel: 'telegram', alert: 'Your YouTube upload failed.' } },
  { id: 'youtube_upload_done', demo: 'youtube_studio.html', when: 'my YouTube upload finishes', then: NOTIFY, expect: DONE },

  // Wild battles keep coming, and a plain Tyrogue shows up twice before the end.
  { id: 'tyrogue', demo: 'pokemon_battle.html', when: 'Cloned Tyrogue shows up in my battle list', then: NOTIFY,
    outcomes: { cloned: 'success', disconnected: 'fail' }, expect: { cloned: 'notify', disconnected: 'either' },
    show: { title: 'Padventures', channel: 'call', alert: 'Cloned Tyrogue just spawned!' } },
  { id: 'tyrogue_or_disconnect', demo: 'pokemon_battle.html', when: 'Cloned Tyrogue spawns or I get disconnected', then: NOTIFY,
    outcomes: { cloned: 'success', disconnected: 'fail' }, expect: { cloned: 'notify', disconnected: 'notify' } },

  // A party invite and a "queue times are longer than usual" notice pop up first; neither is the match.
  { id: 'valorant_match', demo: 'valorant_queue.html', when: 'my Valorant match is found', then: NOTIFY,
    outcomes: { match_found: 'success', queue_error: 'fail' }, expect: { match_found: 'notify', queue_error: 'either' },
    show: { title: 'Valorant', channel: 'call', alert: 'Match found! Agent select is starting.' } },
  { id: 'valorant_kicked', demo: 'valorant_queue.html', when: 'I get kicked out of the Valorant queue', then: NOTIFY,
    outcomes: { match_found: 'success', queue_error: 'fail' }, expect: { match_found: 'silent', queue_error: 'notify' } },

  // Waiting for an event rather than a job. Group messages keep arriving the whole time.
  // carlos_messages: Carlos messages instead of Maria.
  { id: 'whatsapp_maria', demo: 'whatsapp_web.html', when: 'Maria replies to me on WhatsApp', then: NOTIFY,
    outcomes: { maria_replies: 'success', carlos_messages: 'fail' }, expect: { maria_replies: 'notify', carlos_messages: 'silent' },
    show: { title: 'WhatsApp', channel: 'sms', alert: 'Maria replied on WhatsApp.' } },
  { id: 'whatsapp_either', demo: 'whatsapp_web.html', when: 'Maria or Carlos messages me on WhatsApp', then: NOTIFY,
    outcomes: { maria_replies: 'success', carlos_messages: 'fail' }, expect: { maria_replies: 'notify', carlos_messages: 'notify' } },

  { id: 'person', demo: 'camera_room.html', when: 'a person walks into my living room', then: NOTIFY,
    outcomes: { person: 'success', cat_only: 'fail' }, expect: { person: 'notify', cat_only: 'silent' },
    requires: ['media/camera/room-person.mp4', 'media/camera/room-cat.mp4'],
    show: { title: 'Home camera', channel: 'telegram', alert: 'Someone just walked into the living room.' } },
  { id: 'anything_moves', demo: 'camera_room.html', when: 'anything moves on my camera', then: NOTIFY,
    outcomes: { person: 'success', cat_only: 'fail' }, expect: { person: 'notify', cat_only: 'notify' },
    requires: ['media/camera/room-person.mp4', 'media/camera/room-cat.mp4'] },

  // The summary comes once, after the call ends.
  { id: 'meeting_summary', demo: 'google_meet_call.html', when: 'my meeting ends', then: 'log a summary of the transcript',
    show: { title: 'Google Meet', alert: 'Logged the meeting summary: 3 decisions, 3 action items.' } },
  { id: 'meeting_transcript', demo: 'google_meet_call.html', when: 'someone speaks in my meeting', then: 'log what they said' },

  // Each error stays on screen for several iterations: logged once each.
  { id: 'terminal_errors', demo: 'terminal_errors.html', when: "there's an error in my terminal", then: 'log it',
    show: { title: 'Terminal', alert: 'Logged: TypeError in checkout.ts.' } },

  // A news app's banner ad rotates every 10 s while the feed scrolls; the Acme ad gets one slot.
  // acme_ad_missing: it never shows, so the 5-minute notification is the only thing to do.
  { id: 'acme_ad', demo: 'ad_banner.html', when: 'the Acme ad shows up in my app',
    then: "save an image of it to memory, and if it hasn't shown up after 5 minutes, send me a browser notification",
    show: { title: 'Ad check', alert: 'Saved: the Acme banner ad, seen at 9:43.' } },
  { id: 'acme_ad_missing', demo: 'ad_banner.html?outcome=fail', when: 'the Acme ad shows up in my app',
    then: "save an image of it to memory, and if it hasn't shown up after 5 minutes, send me a browser notification" },

  // The new orders, not today's total.
  { id: 'shopify_count', demo: 'shopify_orders.html', when: 'a new Shopify order comes in', then: 'keep count of it',
    show: { title: 'Shopify', alert: 'Logged: 7 new orders so far.' } },

  // A note every 2 minutes naming what was on screen in that window.
  { id: 'workday_notes', demo: 'workday.html', when: 'two minutes pass', then: 'log what I worked on',
    show: { title: 'Your workday', alert: 'Logged: 9:41–9:43, VS Code, checkout.ts tax fix.' } },
  { id: 'workday_calls', demo: 'workday.html?call=1', when: "I'm on a call", then: 'log the transcript' },

  // Logger + one-shot summarizer: the topics, after the video ends.
  { id: 'youtube_topics', demo: 'youtube_watch.html', when: 'this YouTube video ends', then: 'log all the topics',
    show: { title: 'YouTube', alert: 'Logged 6 topics, from "Why our AWS bill doubled" to "Results".' } },
  { id: 'youtube_topic_changes', demo: 'youtube_watch.html', when: 'the topic of this YouTube video changes', then: "log the new topic with the video's timestamp" },

  // Every message once, though each stays visible for ~100 s.
  { id: 'twitch_chat', demo: 'twitch_chat.html', when: 'five seconds pass', then: 'log new Twitch chat, no duplicates',
    show: { title: 'Twitch', alert: 'Logged 14 new chat messages.' } },

  // The dog walking away mid-bowl isn't it.
  { id: 'pet_food', demo: 'camera_bowl.html', when: 'my dog finishes their food', then: 'log it', requires: ['media/camera/bowl.mp4'],
    show: { title: 'Pet camera', alert: 'Logged: the bowl is empty.' } },

  // A note near each coming and going, not one every loop.
  { id: 'people_log', demo: 'camera_hallway.html', when: 'someone comes or goes on my camera', then: 'log who it was', requires: ['media/camera/hallway.mp4'],
    show: { title: 'Front hall', alert: 'Logged: someone in a red jacket came in.' } },

  // Warnings print either way (a build-script one while holding, an unused variable at the end).
  { id: 'rust_done', demo: 'cargo_build.html', when: 'my Rust build finishes', then: NOTIFY, expect: DONE,
    show: { title: 'Rust', channel: 'telegram', alert: 'Your Rust build finished.' } },
  { id: 'rust_fails', demo: 'cargo_build.html', when: 'my Rust build fails', then: NOTIFY, expect: IF_FAILS },

  // An IDE instead of a terminal. Kotlin deprecation warnings print either way.
  { id: 'android_done', demo: 'androidstudio_build.html', when: 'my Android Studio build finishes', then: NOTIFY, expect: DONE,
    show: { title: 'Android Studio', channel: 'discord', alert: 'Your Android build finished.' } },
  { id: 'android_fails', demo: 'androidstudio_build.html', when: 'my Android build fails', then: NOTIFY, expect: IF_FAILS },

  { id: 'bigquery_done', demo: 'bigquery_query.html', when: 'my BigQuery query finishes', then: NOTIFY, expect: DONE,
    show: { title: 'BigQuery', channel: 'sms', alert: 'Your BigQuery query finished.' } },

  // Long-running dashboards.
  { id: 'cloudformation_done', demo: 'cloudformation_stack.html', when: 'my CloudFormation stack is done', then: NOTIFY, expect: DONE,
    show: { title: 'CloudFormation', channel: 'email', alert: 'Your CloudFormation stack finished updating.' } },
  { id: 'gitlab_done', demo: 'gitlab_pipeline.html', when: 'my GitLab pipeline finishes', then: NOTIFY, expect: DONE,
    show: { title: 'GitLab', channel: 'email', alert: 'Your GitLab pipeline finished.' } },
  { id: 'wandb_done', demo: 'wandb_run.html', when: 'my W&B training run finishes', then: NOTIFY, expect: DONE,
    show: { title: 'Weights & Biases', channel: 'telegram', alert: 'Your W&B run finished.' } },
  { id: 'argocd_done', demo: 'argocd_sync.html', when: 'my Argo CD rollout is synced', then: NOTIFY, expect: DONE,
    show: { title: 'Argo CD', channel: 'discord', alert: 'Your Argo CD rollout is synced.' } },
  { id: 'fusion360_done', demo: 'fusion360_assembly.html', when: 'my Fusion 360 assembly finishes loading', then: NOTIFY, expect: DONE,
    show: { title: 'Fusion 360', channel: 'email', alert: 'Your Fusion 360 assembly is ready.' } },

  // The download reaches 100% and then Steam installs it: downloaded isn't ready to play.
  { id: 'steam_ready', demo: 'steam_download.html', when: 'my Steam game is ready to play', then: NOTIFY, expect: DONE,
    show: { title: 'Steam', channel: 'email', alert: 'Cyberpunk 2077 is ready to play.' } },
  { id: 'steam_fails', demo: 'steam_download.html', when: 'my Steam download fails', then: NOTIFY, expect: IF_FAILS },

  { id: 'linux_done', demo: 'linux_update.html', when: 'my Linux update finishes', then: NOTIFY, expect: DONE,
    show: { title: 'Ubuntu', channel: 'email', alert: 'Your updates are installed. Restart when ready.' } },
  { id: 'linux_fails', demo: 'linux_update.html', when: 'my Linux update fails', then: NOTIFY, expect: IF_FAILS },

  // The menu bar battery drains; plugged_in: the charger goes in at 31%, so it never got low.
  { id: 'battery_low', demo: 'macos_battery.html', when: 'my battery is low', then: NOTIFY,
    outcomes: { low: 'success', plugged_in: 'fail' }, expect: { low: 'notify', plugged_in: 'silent' },
    show: { title: 'MacBook', channel: 'telegram', alert: 'Your MacBook is at 10%. Plug it in.' } },
  { id: 'battery_dying', demo: 'macos_battery.html', when: 'my MacBook is about to die', then: NOTIFY,
    outcomes: { low: 'success', plugged_in: 'fail' }, expect: { low: 'notify', plugged_in: 'silent' } },

  { id: 'download_done', demo: 'fake_download.html', when: 'my download finishes', then: NOTIFY, expect: DONE,
    show: { title: 'Download', channel: 'sms', alert: 'Your download finished.' } },
  // A failed download is also "done" from the user's side: they want to know it stopped.
  { id: 'download_fails', demo: 'fake_download.html', when: 'my download fails', then: NOTIFY, expect: IF_FAILS },
];

/** The demo page's file name, without the query string. */
export const demoFile = (u: UseCase): string => u.demo.split('?')[0];

/** Demos that are a camera feed rather than a screen. */
export const isCameraDemo = (u: UseCase): boolean => demoFile(u).startsWith('camera_');

/** The one shown pair per demo, in order. */
export const SHOWN = GROUND_TRUTH.filter(u => u.show);
