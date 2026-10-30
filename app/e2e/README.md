# MCP end-to-end suite

A black-box test of the MCP against the **ground truth**, `src/utils/groundTruth.ts`: everything Observer is for, as **When … then …** pairs, each with a demo page. The MCP gets one prompt, always `When ${when}, then ${then}.`, and the suite judges only what the agent it built does: **when** it notifies the user, or what it logs, counts or clips, and when. It never checks which model, tools or channel the MCP picks.

The same list drives the landing page's When / Then wheels and the app's tutorial and Recipe Wheel, so a use case added there is tested, shown and offered at once. Use cases with `expect` are judged on timing (`pnpm e2e`, below); the rest by their check in `patterns.ts` (`pnpm e2e:patterns`).

Each alert scenario is a demo page, a When / Then pair, one of the demo's outcomes (`success` / `fail` for a job; its own names for a wait, like `almost_here` / `late` / `canceled`), and what an Observer agent should do on it: `notify`, stay `silent` (a notification is a false alarm), or `either` (the wait ended another way; notifying is fine and so is staying quiet):

| Phase | What happens | Verdict |
|---|---|---|
| SETUP | The demo holds in progress; the MCP gets the prompt once | `NO_AGENT` if no agent is running when the MCP's turn ends (it asked a question, errored…); `ASKED_USER_INFO`; `SETUP_TIMEOUT` (15 min) |
| PRE | The job keeps going for a random 2–6 min | every notification is recorded; none ends the test |
| TRIGGER | The demo ends and reports the instant the end state is on screen | |
| WINDOW | notify: within 60 s → right, else `MISSED`; silent: quiet for 3 min → right, else `FALSE_POSITIVE`; either: right after 60 s. Ends at the first notification after the trigger | |

"Right" is `PASS` if nothing fired before the trigger (one shot) and `PARTIAL` if the agent fired early but behaved correctly from the trigger on. `PARTIAL` still passes the Playwright test; the summary counts one-shot and partial separately.

A notification is any user-facing agent-code tool (`notify`, `sendTelegram`, `call`, `sound`, …), caught by the app's logger in `src/mcp/e2e.ts`. Failed sends count too.

## Running it

```sh
pnpm e2e:login   # once per machine: log in to Observer in the browser that opens
pnpm e2e         # all scenarios; add --grep "fake_download #2" for some of them
pnpm e2e --repeat-each 5   # pass rates, since the MCP isn't deterministic
caffeinate -dimsu pnpm e2e # unattended on a Mac: no sleep (keep it on power, lid open)
```

- Both start their own dev server on port 3001 with `VITE_E2E=1`, so **stop your own `pnpm dev` first**. (`E2E_PORT` changes the port, but Auth0 only accepts logins on the usual one.)
- The browser runs headed: Chrome only lets the focused Observer tab share a screen, and the demo is captured in a background tab, selected by its title.
- The profile in `e2e/.profile` keeps the login and the on-device model cache (OneJev downloads once).
- Results: `e2e/results/<run>/results.jsonl` (one line per scenario) and a transcript of each MCP conversation. HTML report in `e2e/report/`.
- `E2E_MODEL_CHOICE` answers the MCP's local-vs-cloud question (`ask_user_info` kind `model`): `cloud` (default), `local` (downloads the model, which can take 10+ minutes), or `ask` (leave it unanswered, so the scenario ends as `ASKED_USER_INFO`).
- Timing knobs (ms): `E2E_SETUP_MS`, `E2E_PRE_MIN_MS`, `E2E_PRE_MAX_MS`, `E2E_WINDOW_MS`, `E2E_QUIET_MS`. For a quick try: `E2E_PRE_MIN_MS=30000 E2E_PRE_MAX_MS=60000 E2E_QUIET_MS=60000 pnpm e2e`.

## Adding a demo

1. Put the page in `e2e/demos/` and load `demo-contract.js` in its `<head>`.
2. It loads already **in progress** and stays there indefinitely. On `demo.onStart = durationMs => …` it runs to its end over `durationMs`, ends by `demo.outcome`, and calls `demo.trigger()` the moment the end state is shown.
3. Check it by hand: `demos/my_demo.html?outcome=fail&autostart=10000` ends after 10 s.
   Keep everything local so runs don't depend on the network: fonts the demos share are in `demos/fonts/` (Google Sans, Roboto, Geist, Material Symbols, YouTube Sans…), a demo's own images in `demos/<name>_assets/`. Use the real site's look and copy wherever possible; that's what the agent will see.
   A demo can also be a wait for an event (a reply, your turn in a queue). Give it the endings a real one has (`?outcome=late` in `ubereats_tracking.html`) and name them in the use case's `outcomes`.
4. Add its use cases to `src/utils/groundTruth.ts`: a `when` and a `then`, and for an alert an `expect` for every outcome (`then: NOTIFY`: the suite can't fill the `ask_user_info` modal, so alerts it tests are browser notifications). A use case without `expect` needs a check in `patterns.ts` under its id.
5. Mark exactly one of the demo's use cases with `show` (title, the channel the landing page shows, what the alert says): that pair is on the landing page's wheels and in the app's tutorial and Recipe Wheel. `websites/landing` fails to build if a demo has none.

Native apps sit on a real-looking desktop: `desktop_assets/macos.js` (menu bar with battery and clock, dock, windows, notification banners; `Mac.mount()`, `Mac.setBattery()`, `Mac.notify()`) and `desktop_assets/ubuntu.js` (GNOME top bar and dock). `macos_battery.html`, `docker_build.html`, `steam_download.html` and `linux_update.html` show how.

### Camera demos

`camera_*.html` are recordings, not pages: `camera_assets/camera.js` plays `demos/media/camera/<name>-<outcome>.mp4` under a security-camera overlay, with each clip's timings (the quiet stretch it loops while holding, the event, marks along the way) in `media/camera/<name>.js`. The suite hands the demo tab to the app as its camera (`useTabAsCamera` in `fixtures.ts`), so `$CAMERA` sees the clip. For any other demo the camera stays Chrome's fake device.

Their scenarios list the clips in `requires` and are skipped (and left off the landing page) until every file exists. To add the recordings, drop them in `demos/media/camera/` and fill in the timings; the comments in each timings file say what to record.

## Pattern scenarios

`pnpm e2e:patterns` runs the use cases that aren't one alert on time (`patterns.ts`, run by `patterns.spec.ts`): "then log it", "then keep count of it", "then save a clip", a summary when a meeting or video ends, a note every two minutes, two alerts from one request. "Log" means Observer's memory, which is also what the suite reads back. It also runs a few chat checks that aren't use cases and stay out of the ground truth: a greeting, an unrelated question, "summarize this meeting" with nowhere to put it, and "what was I doing?".

- Each demo marks what happens as it happens (`demo.mark(name, data)`: a topic change, an error printed, an order arriving) and can publish `demo.truth`. Media-driven demos (the meeting, the video, Minecraft) end on their own and ignore `durationMs`.
- Each scenario's `score` compares that with what the agents did: their memory writes (with timestamps), notifications and clips, all reported by `src/mcp/e2e.ts`. Verdicts are `PASS` / `PARTIAL` / `FAIL`, plus `NO_AGENT`, `SETUP_TIMEOUT`, `DEMO_BROKEN`; the numbers behind each go in `results/<run>/patterns.jsonl`.
- A scenario with `followUp` answers the MCP's question with it when the first turn ends with no agent running (e.g. "Save it to memory, please.").
- A scenario with `afterRun` asks it once the run and grace period are over, with the agents still running ("What was I doing over the last few minutes?"); its score reads the answer from `run.replies`.
- The meeting and video demos play speech (`demos/media/`, made with macOS `say` by `make_audio.py`) **out loud**: agents hear it through the tab capture, which muting Chrome would also silence. The browser's microphone is a fake one that hears silence.
- `minecraft_clips` and `minecraft_deaths` need a recording at `demos/media/minecraft.mp4` with its timings in `demos/media/minecraft.js`; they're skipped until then. So are `pet_food` and `people_log` until their camera clips exist.

