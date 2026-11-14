# Adding a New Tool to Observer AI (Framework Tool Integration Guide)

This guide is a checklist for adding a new **framework tool** — a function the agent's
sandboxed JavaScript can call after every model run (`sendEmail()`, `overlay()`, `click()`,
`call()`, `celebrate()`, `mcp()`, …). It covers wiring the executor and every UI surface that has to
know about the tool.

> **Not covered here:** the MCP creator tools (`create_agent`, `start_agent`, `get_runs`,
> `capture_screen`, …) that the AI Creator assistant calls to build and manage agents. Those
> live entirely in `src/mcp/registry.ts` and are a separate system — if you're adding one of
> those, this checklist does not apply. The only overlap is the MCP system prompt in step 8,
> which describes the framework tools *to* that assistant as plain text.

> Note on churn: the old version of this guide used `click()` and `type()` as its worked
> examples. `type()` has since been **removed**. The current reference tools are `click()`
> (now takes a `'left' | 'right'` button arg), `call()` (phone call), and `celebrate()`
> (UI animation) — this guide uses those.

---

## Integration checklist

A framework tool touches up to **9 places**. Notification-style tools (email/SMS/etc.)
skip the native-desktop steps; a desktop-only tool like `click()` needs them.

### 1. Backend: the executor

**a) HTTP / implementation function — `src/utils/handlers/utils.ts`**

Tools that do real I/O call a function exported from `utils.ts`. Notification tools POST to
the Observer API; native desktop tools (`click`, `ask`, `message`, `system_notify`,
`overlay`) POST to the local app server at `http://localhost:3838`.

```typescript
// Triggers a mouse click at the current cursor position.
export async function click(appUrl: string, button: 'left' | 'right' = 'left'): Promise<void> {
  const response = await platformFetch(`${appUrl}/click`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ button }),
  });
  // ...error handling...
}
```

Pure-UI tools skip this step: `celebrate()` dispatches a window event and `mcp()` calls
`wakeMCP()` from `src/mcp/agentWake.ts`, both directly from the context entry below.

**b) Sandbox context wrapper — `src/utils/handlers/javascript.ts`**

This is where the function name the agent actually calls is defined, inside the big
`context` object (roughly lines 300–690). Each entry wraps the `utils.ts` call with
`Logger.info` / `Logger.error` using `logType: 'tool-success'` / `'tool-error'` and a
`content: { tool: '<name>', params: {...} }` payload — the ToolStatus UI and iteration
history read those logs.

```typescript
click: async (button: 'left' | 'right' = 'left'): Promise<void> => {
  try {
    const appUrl = "http://localhost:3838";
    await utils.click(appUrl, button);
    Logger.info(agentId, `Mouse ${button} click executed`, {
      logType: 'tool-success', iterationId,
      content: { tool: 'click', params: { button }, success: true }
    });
  } catch (error) {
    Logger.error(agentId, `Failed to execute mouse ${button} click`, {
      logType: 'tool-error', iterationId,
      content: { tool: 'click', params: { button }, error: extractErrorMessage(error) }
    });
    throw error;
  }
},
```

Context variables already in scope for a tool body: `agentId`, `iterationId`, `response`,
`getToken` (async → auth token, needed for any Observer-API tool), the sensor vars
(`screen`, `camera`, `images`, `microphone`, …).

**c) Native desktop tools only — Rust handler**

The Tauri code moved out of `app/src-tauri/`. It now lives in:

- `app/desktop/src/controls.rs` — the handler (e.g. `click_handler`), plus a mobile stub
  (`#[cfg(mobile)]`) that returns "not supported".
- `app/desktop/src/lib.rs` — route registration, e.g.
  `.route("/click", axum::routing::post(controls::click_handler))` (search for the other
  `controls::` routes).
- `app/mobile/src/` — mobile has its own local server (`server.rs`); native input tools are
  desktop-only, so usually just the stub applies.

---

### 2. Capability detection & UI badges — `src/components/AgentCard/agentCapabilities.tsx`

(The file's top comment says `src/utils/agentCapabilities.ts` — stale, ignore it.)

**a) Add to `TOOL_CONFIG`** (around line 61) — keyed by a detection key, matched against
agent code by `regex`:

```typescript
export const TOOL_CONFIG: Record<string, ToolConfigEntry> = {
  // ...
  click: { label: 'Mouse Click', iconName: 'MousePointer', iconType: 'lucide',
           regex: /\bclick\s*\(/g, warning: 'Position mouse before agent runs' },
  call:  { label: 'Phone Call', iconName: 'Phone', iconType: 'lucide', regex: /\bcall\s*\(/g },
  celebrate: { label: 'Celebrate', iconName: 'PartyPopper', iconType: 'lucide', regex: /\bcelebrate\s*\(/g },
};
```

Fields: `label`, `iconName` (lucide-react name, or a key from `./icons` when
`iconType: 'custom'`), `iconType`, `regex` (use `\b` to avoid matching substrings), optional
`warning`. Icons are loaded dynamically by name — no import statement needed in this file.

**b) `webIncompatibleTools` / `mobileOnlyTools`** (around line 150, inside
`detectAgentTools`) — add the key to `webIncompatibleTools` if the tool cannot run on
`hostingContext === 'official-web'` (i.e. it needs the desktop app). Currently:
`['overlay', 'message', 'ask', 'system_notify', 'click']`. These get an `isBlocking`
"only available in the Observer App" warning. Mobile-app-only tools (`lockPhone`,
`unlockPhone`) go in `mobileOnlyTools` instead. Tools that work everywhere (`mcp`,
`celebrate`) go in neither.

---

### 3. Simple Creator UI — `src/components/EditAgent/SimpleCreatorModal.tsx`

**Optional.** Only add a tool here if it makes sense as a one-click building block for
non-technical users. Tools that only make sense in context (e.g. `mcp()`, whose message is
meant for an open MCP conversation) skip steps 3 and 4.

The wizard where users pick tools as buttons. `hostingContext` here is
`'official-web' | 'self-hosted'` (no `'tauri'`).

**a)** Import any lucide icon you need (top of file).

**b)** Add a `<button>` in the right section (roughly lines 576–660). Sections:

- **Reliable Notifications** — Discord, Email, Pushover, Telegram, **Call**
- **Logging & Recording** — Memory, Start/Label recording
- **App Specific Utils** — gated on `hostingContext === 'self-hosted'`: ask, system_notify,
  message, overlay, **click**
- **Fun** — celebrate
- **Other Notifications** — browser notification, WhatsApp, SMS

Pattern (toggles the tool in the `selectedTools` Map):

```tsx
<button type="button" onClick={() => toggleTool('click')}
  className={`group flex items-center space-x-4 p-4 border-2 rounded-lg text-left transition-all ${selectedTools.has('click') ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}>
  <MousePointer className={`h-8 w-8 ${selectedTools.has('click') ? 'text-blue-500' : 'text-gray-400 group-hover:text-gray-600'}`} />
  <div>
    <h3 className="font-semibold text-gray-900">Mouse Click</h3>
    <p className="text-sm text-gray-500">Clicks at current cursor position.</p>
  </div>
</button>
```

Tools that need a value (phone number, webhook, …) render an `<input>` when selected and
store it via `setSelectedTools(prev => { const m = new Map(prev); m.set('call', { phoneNumber }); return m; })`.
Add the matching field to `ToolData` in `agentTemplateManager.ts` (step 4).

If the tool should appear in the onboarding tutorial, also wire a `data-tutorial="..."`
attribute and a matching step in `SimpleCreatorTutorial.tsx`.

---

### 4. Code generation — `src/utils/agentTemplateManager.ts`

**a) `SimpleTool` union** (line 5): add your tool id.

**b) `TOOL_CODE_SNIPPETS`** (around line 30): each entry is `(data: ToolData, ctx: SensorContext) => string`.
`ctx` tells you whether `$SCREEN` / `$CAMERA` are in the prompt, so notification snippets
can append the captured image (`buildImageArg`):

```typescript
click: () => `
// --- MOUSE CLICK TOOL ---
// Triggers a mouse click at the current cursor position.
// IMPORTANT: Position the mouse before the agent runs.
click();
`,
call: (data: ToolData) => {
  const phoneNumber = data.phoneNumber ? JSON.stringify(data.phoneNumber) : '""';
  return `
// --- PHONE CALL TOOL ---
call(${phoneNumber}, response);
`;
},
```

Use `response` for the model output, `agentId` for the current agent. Add any new
`ToolData` field near the top of the file.

---

### 5. Tools Testing Modal — `src/components/AgentCard/ToolsModal.tsx`

Lets users click a tool call in the code editor and test-fire it.

**a)** Import the icon (line 3 block).

**b) `getAllTools(channel?)`** (line 62) returns a `ToolConfig[]` split into **testable**
(top) and **non-testable / info-only** (after the `// Non-testable tools` comment). Add an
entry:

```typescript
{
  id: 'click', name: 'click()', functionName: 'click', icon: MousePointer,
  description: 'Trigger mouse click at cursor position',
  isTestable: true, parameters: [], testMessage: '',
  warning: '⚠️ IMPORTANT: Position mouse on a button that the agent will click.'
},
```

`parameters: [{ name, description }]` drives the test input fields; `channel`
(`WhitelistChannel`) lets you vary a warning for whatsapp vs sms/voice.

**c) `handleTest` switch** (around line 1154): add a `case`. Native tools call the
`utils.*` fn with `const appUrl = 'http://localhost:3838'`; pure-UI tools do their thing
inline:

```typescript
case 'click': {
  await utils.click('http://localhost:3838');
  break;
}
case 'celebrate': {
  window.dispatchEvent(new CustomEvent('celebrateAgent', { detail: { agentId } }));
  break;
}
```

Use `testInputs[0]`, `testInputs[1]`, … for parameters, falling back to
`selectedToolConfig.testMessage`.

---

### 6. Execution-status icon — `src/components/AgentCard/ToolStatus.tsx`

(File-top comment says `src/components/shared/ToolStatus.tsx` — stale.)

**a)** Import the icon (line 3 block).
**b)** Add to `iconMap` inside `getToolIcon` (around line 12), keyed by the **function name**
the agent calls (matches `content.tool` in the logs):

```typescript
const iconMap: Record<string, React.ElementType> = {
  // ...
  call: Phone,
  click: MousePointerClick,
  celebrate: PartyPopper,
};
```

Falls back to `HelpCircle` if unmapped.

---

### 7. Sensitive-data guard — `src/utils/code_sanitizer.ts`

**Only if the tool takes PII as an argument** (phone number, email, webhook, chat id).
Add the function name to `SENSITIVE_FUNCTIONS` (line 10) and, optionally, entries in
`FUNCTION_DESCRIPTIONS` / `PLACEHOLDER_SUGGESTIONS`. This powers the "you're sharing a
phone number, replace with a placeholder?" warning before an agent is published to the
marketplace. `click` / `celebrate` are **not** listed; `call` / `sendSms` / `sendEmail` are.

Phone tools (`sendSms`, `call`, `sendWhatsapp`) additionally go through
`checkPhoneWhitelist` in `src/utils/pre-flight.ts` before an agent starts — that list is
keyed off the function names in the agent code, so a new phone-delivery tool must be added
there too.

---

### 8. MCP system prompt (so the AI knows the tool exists)

The framework tools are described to the MCP AI Creator as **text** — the AI that writes
agent code needs to know the tool exists.

| File | What it is | Where to edit |
|---|---|---|
| `src/mcp/systemPrompt.ts` | System prompt for the MCP AI Creator. | The agent-code API block ("You never call these; you write them into `code`:" — `Notifications:` / `Memory:` / `Control:` / `Images:` / `Video:` lines, ~170–174). Platform-specific tools go in the `appTools` string near the top (`Desktop only:` / `Mobile app only:`). This is the only touchpoint the MCP has with framework tools — you're just adding a name to a text list, not registering anything. |
| `src/utils/system_prompt.ts` | Deliberately restricted "copy this prompt" generator — only allows `appendMemory`/`setMemory`/`time`/`notify` | **Usually leave alone.** Only touch it if the tool belongs in that minimal safe set. |

The legacy text creators (`multi_agent_creator.ts`, `conversational_system_prompt.ts`) have
been deleted; the MCP is the only creator now.

---

### 9. README — `README.md`

Update the collapsible **Tools** list (around lines 68–100). Sections: **Memory and
agents**, **Notifications**, **Video recording**, **Desktop app**, **Mobile app**. Include
the signature and, if it isn't obvious, a short description after a colon. Mark
whitelist requirements with `†`.

```markdown
**Memory and agents** (`agentId` is optional and defaults to the running agent)
- `time()`, `sleep(ms)`
- `mcp(message)`: wakes Observer's MCP with a message, as a new turn in the open chat

**Desktop app**
- `overlay(body)`, `click('left' | 'right')`, `celebrate()`
```

---

## Verification checklist

- [ ] `utils.ts` fn + `javascript.ts` context entry (with success/error logging)
- [ ] Rust handler + `lib.rs` route + mobile stub (native tools only)
- [ ] `TOOL_CONFIG` entry, and `webIncompatibleTools` / `mobileOnlyTools` if platform-specific
- [ ] (Optional) Simple Creator button in the right section; `ToolData` field if it needs input
- [ ] (Optional, with the button) `SimpleTool` union + `TOOL_CODE_SNIPPETS` entry
- [ ] `ToolsModal`: `getAllTools` entry + `handleTest` case
- [ ] `ToolStatus` `iconMap` entry (keyed by function name)
- [ ] `code_sanitizer.ts` + `pre-flight.ts` if it takes PII / is a phone tool
- [ ] `mcp/systemPrompt.ts`
- [ ] `README.md` tool list
- [ ] Badge shows on Agent Card when the tool is used in code
- [ ] Test button fires in Tools Modal
- [ ] Simple Creator generates correct code (if added)
- [ ] Web-incompatibility / warnings display correctly

---

## Questions?

1. Implement the executor first (`utils.ts` → `javascript.ts` → Rust if native), and
   confirm an agent can actually call the tool from the Code tab.
2. Then walk the UI files in order — `click()` / `call()` / `celebrate()` are complete
   worked examples of a native tool, a phone/PII tool, and a pure-UI tool respectively.
   `mcp()` is a pure-UI tool that skips the Simple Creator.
3. Grep the codebase for an existing tool name (e.g. `celebrate`) to find every site that
   references it — that's the real checklist.
