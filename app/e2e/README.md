# MCP end-to-end suite

A black-box test of the MCP: it gets **one prompt**, and the test only checks **when the agent it built notifies the user**. It never checks which model, tools or channel the MCP picks.

Each scenario is a demo page, a prompt, one of the demo's outcomes (`success` / `fail` for a job; its own names for a wait, like `almost_here` / `late` / `canceled`), and what an Observer agent should do on it: `notify`, stay `silent` (a notification is a false alarm), or `either` (the wait ended another way; notifying is fine and so is staying quiet):

| Phase | What happens | Verdict |
|---|---|---|
| SETUP | The demo holds in progress; the MCP gets the prompt once | `NO_AGENT` if no agent is running when the MCP's turn ends (it asked a question, errored…); `ASKED_USER_INFO`; `SETUP_TIMEOUT` (15 min) |
| PRE | The job keeps going for a random 2–6 min | any notification → `FALSE_POSITIVE` |
| TRIGGER | The demo ends and reports the instant the end state is on screen | |
| WINDOW | notify: within 60 s → `PASS`, else `MISSED`; silent: quiet for 3 min → `PASS`, else `FALSE_POSITIVE`; either: `PASS` after 60 s | |

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
- Timing knobs (ms): `E2E_SETUP_MS`, `E2E_PRE_MIN_MS`, `E2E_PRE_MAX_MS`, `E2E_WINDOW_MS`, `E2E_QUIET_MS`. For a quick try: `E2E_PRE_MIN_MS=30000 E2E_PRE_MAX_MS=60000 E2E_QUIET_MS=60000 pnpm e2e`.

## Adding a demo

1. Put the page in `e2e/demos/` and load `demo-contract.js` in its `<head>`.
2. It loads already **in progress** and stays there indefinitely. On `demo.onStart = durationMs => …` it runs to its end over `durationMs`, ends by `demo.outcome`, and calls `demo.trigger()` the moment the end state is shown.
3. Check it by hand: `demos/my_demo.html?outcome=fail&autostart=10000` ends after 10 s.
   Keep everything local so runs don't depend on the network: fonts the demos share are in `demos/fonts/` (Google Sans, Roboto, Geist, Material Symbols, YouTube Sans…), a demo's own images in `demos/<name>_assets/`. Use the real site's look and copy wherever possible; that's what the agent will see.
   A demo can also be a wait for an event (a reply, your turn in a queue). Give it the endings a real one has (`?outcome=late` in `ubereats_tracking.html`) and name them in `outcomes` in `scenarios.ts`.
4. Add it with its prompts to `scenarios.ts`, each with an `expect` for every outcome. Prompts must not need contact info (no Telegram/SMS/email): the suite can't fill the `ask_user_info` modal.

## Pattern scenarios

`pnpm e2e:patterns` runs the requests that aren't "tell me when X ends" (`patterns.ts`, run by `patterns.spec.ts`): notes in memory, a summary when a meeting or video ends, a periodic digest, a running count, two alerts from one request, clips, and requests where the right move is to ask first or build nothing.

- Each demo marks what happens as it happens (`demo.mark(name, data)`: a topic change, an error printed, an order arriving) and can publish `demo.truth`. Media-driven demos (the meeting, the video, Minecraft) end on their own and ignore `durationMs`.
- Each scenario's `score` compares that with what the agents did: their memory writes (with timestamps), notifications and clips, all reported by `src/mcp/e2e.ts`. Verdicts are `PASS` / `PARTIAL` / `FAIL`, plus `NO_AGENT`, `SETUP_TIMEOUT`, `DEMO_BROKEN`; the numbers behind each go in `results/<run>/patterns.jsonl`.
- A scenario with `followUp` answers the MCP's question with it when the first turn ends with no agent running (e.g. "Save it to memory, please.").
- The meeting and video demos play speech (`demos/media/`, made with macOS `say` by `make_audio.py`) **out loud**: agents hear it through the tab capture, which muting Chrome would also silence. The browser's microphone is a fake one that hears silence.
- `minecraft_clips` needs a recording at `demos/media/minecraft.mp4` with its timings in `demos/media/minecraft.js`; it's skipped until then.

