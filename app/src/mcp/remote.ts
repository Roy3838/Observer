// src/mcp/remote.ts
//
// Remote-control transport: lets the user talk to this tab's MCP from WhatsApp or Telegram.
// Framework-free, like runner.ts. The API is only a mailbox (api/remote.py): messages from the
// user's linked phone/chat are long-polled here, and the MCP's answers are posted back.
//
// Each channel has its own session id: the code ask_user_info's QR already has the user send
// (SensorSettings.getWhitelistCode for WhatsApp, getTelegramCode for Telegram), so pairing
// needs no extra step. WhatsApp routes need the account that owns the code; Telegram routes
// need only the code, so remote control over Telegram works without signing in.

import type { TokenProvider } from '@utils/main_loop';

const API_HOST = 'https://api.observer-ai.com';

export type RemoteChannel = 'whatsapp' | 'telegram';

export interface RemoteMessage {
  code: string;
  channel: RemoteChannel;
  text: string;
}

const MAX_BACKOFF_MS = 15_000;   // stay under the server's 45s "session active" TTL
const NOT_OWNER_RECHECK_MS = 60_000;

/** Fired by the pairing UIs the moment they see a phone/chat link, so remote control starts
 *  listening right away instead of on its next status poll. */
export const REMOTE_LINKED_EVENT = 'observer:remote-linked';

export function notifyRemoteLinked(): void {
  window.dispatchEvent(new Event(REMOTE_LINKED_EVENT));
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise(resolve => {
    if (signal.aborted) return resolve();
    const timer = setTimeout(resolve, ms);
    signal.addEventListener('abort', () => { clearTimeout(timer); resolve(); }, { once: true });
  });
}

const routes: Record<RemoteChannel, string> = {
  whatsapp: '/remote',
  telegram: '/remote/telegram',
};

/** Telegram routes take no token; WhatsApp ones need the account's. */
function authHeaders(channel: RemoteChannel, token?: string): Record<string, string> {
  return channel === 'whatsapp' && token ? { Authorization: `Bearer ${token}` } : {};
}

/** The code to listen on, per channel. Only channels with a linked phone/chat belong here. */
export type RemoteSessions = Partial<Record<RemoteChannel, string>>;

/**
 * Long-poll the inboxes of `sessions` in one request until `signal` aborts. Each request also
 * tells the server those sessions are active; while no tab is polling, the phone gets "no
 * Observer session is active" instead. WhatsApp is skipped while there is no token.
 */
export async function listenInbox(opts: {
  sessions: RemoteSessions;
  getToken: TokenProvider;
  signal: AbortSignal;
  onMessage: (message: RemoteMessage) => void;
}): Promise<void> {
  const { sessions, getToken, signal, onMessage } = opts;
  let backoff = 1000;

  while (!signal.aborted) {
    const token = sessions.whatsapp ? await getToken().catch(() => undefined) : undefined;
    const params = new URLSearchParams();
    if (sessions.whatsapp && token) params.set('whatsapp', sessions.whatsapp);
    if (sessions.telegram) params.set('telegram', sessions.telegram);
    if (!params.toString()) {
      await sleep(backoff, signal);
      backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
      continue;
    }

    try {
      const response = await fetch(`${API_HOST}/remote/listen?${params}`, {
        headers: params.has('whatsapp') ? authHeaders('whatsapp', token) : {},
        signal,
      });
      // 403: the WhatsApp code isn't this account's. Nothing to receive until that changes.
      if (response.status === 403) {
        await sleep(NOT_OWNER_RECHECK_MS, signal);
        continue;
      }
      if (!response.ok) throw new Error(`Inbox poll failed: ${response.status}`);

      const data = await response.json();
      for (const m of data.messages ?? []) {
        const channel = m.channel as RemoteChannel;
        const code = sessions[channel];
        if (code) onMessage({ code, channel, text: m.text });
      }
      backoff = 1000;
    } catch {
      if (signal.aborted) return;
      await sleep(backoff, signal);
      backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
    }
  }
}

const CHANNEL_NAME: Record<RemoteChannel, string> = {
  whatsapp: 'WhatsApp',
  telegram: 'Telegram',
};

/** What the model sees for a remote message: the text plus why it should answer differently.
 *  The prefix stays on the wire (the model needs it on every replay); the UI strips it via
 *  parseRemotePrompt and shows a channel icon instead. */
export function remotePrompt(message: RemoteMessage): string {
  return (
    `[Sent from the user's phone via ${CHANNEL_NAME[message.channel]}. They are away from the computer: ` +
    `keep the answer short, and ask for anything you need in chat instead of calling ask_user_info.]\n\n` +
    message.text
  );
}

const REMOTE_PREFIX = /^\[Sent from the user's phone via (WhatsApp|Telegram)\.[^\]]*\]\n\n/;

/** Splits a remotePrompt back into its channel and the user's own text; null for normal messages. */
export function parseRemotePrompt(text: string): { channel: RemoteChannel; text: string } | null {
  const match = REMOTE_PREFIX.exec(text);
  if (!match) return null;
  return { channel: match[1] === 'WhatsApp' ? 'whatsapp' : 'telegram', text: text.slice(match[0].length) };
}

export interface RemoteStatus {
  linked: boolean;
  /** The paired chat's / phone's display name, as Telegram or WhatsApp reported it. */
  name: string | null;
  /** The phone/chat sent "stop" or paired a newer code: this one is dead for good. */
  revoked: boolean;
}

const NOT_LINKED: RemoteStatus = { linked: false, name: null, revoked: false };

/** Whether the code is paired on `channel`, or null when the server couldn't say (network error,
 *  5xx, expired token). A refusal (e.g. 403, not this account's WhatsApp code) reads as "not linked". */
export async function tryFetchStatus(channel: RemoteChannel, code: string, token?: string): Promise<RemoteStatus | null> {
  try {
    const response = await fetch(`${API_HOST}${routes[channel]}/status?code=${encodeURIComponent(code)}`, {
      headers: authHeaders(channel, token),
    });
    if (response.ok) return await response.json();
    return response.status >= 500 || response.status === 401 ? null : NOT_LINKED;
  } catch {
    return null;
  }
}

/** Whether the code is paired on `channel`. Any failure reads as "not linked". */
export async function fetchStatus(channel: RemoteChannel, code: string, token?: string): Promise<RemoteStatus> {
  return (await tryFetchStatus(channel, code, token)) ?? NOT_LINKED;
}

/** Strip a data-URL's `data:image/...;base64,` prefix — the API wants raw base64, like every
 *  other notification tool (sendSms/sendEmail/sendTelegram/...). */
function toRawBase64(dataUrl: string): string {
  const comma = dataUrl.indexOf(',');
  return comma === -1 ? dataUrl : dataUrl.slice(comma + 1);
}

/** Send the MCP's answer back to the phone/chat the message came from. `images` are data-URLs
 *  the model captured this turn (e.g. via capture_screen) — optional, best-effort on the server. */
export async function postReply(message: RemoteMessage, text: string, token: string | undefined, images?: string[]): Promise<void> {
  const response = await fetch(`${API_HOST}${routes[message.channel]}/reply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders(message.channel, token) },
    body: JSON.stringify({
      code: message.code,
      text: text.slice(0, 4000),
      images: images && images.length > 0 ? images.map(toRawBase64) : undefined,
    }),
  });
  if (!response.ok) throw new Error(`Remote reply failed: ${response.status}`);
}
