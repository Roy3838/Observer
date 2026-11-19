// src/mcp/useRemoteControl.ts
//
// Remote control: messages the user sends from their linked WhatsApp/Telegram are injected into
// whatever MCP conversation is open in this tab, and that conversation is mirrored back to every
// linked phone/chat as it happens — whoever started the turn (the phone, the desktop, an agent).
// Mounted once, by MCPProvider, so it keeps listening while the chat panel is closed —
// "watch this for me", walk away, "how's it going?".
//
// `send` ignores calls while a run is in flight, so messages wait in a queue and are drained
// one at a time whenever the MCP is idle.
//
// The dead-man's switch rides on the same long-poll: with `deadman` on, the server tells the
// linked phone/chat if this tab stops polling (device locked, app killed, offline). It implies
// remote control: turning remote control off disarms it first, so that silence isn't an alarm.

import { useCallback, useEffect, useRef, useState } from 'react';
import type { TokenProvider } from '@utils/main_loop';
import { Logger } from '@utils/logging';
import { SensorSettings } from '@utils/settings';
import { CHANNEL_NAME, disarmDeadman, listenInbox, parseRemotePrompt, type PhoneItem, remotePrompt, REMOTE_LINKED_EVENT, type RemoteChannel, type RemoteMessage, type RemoteSessions, type RemoteStatus, sendToPhone, tryFetchStatus } from './remote';
import { parseAgentPrompt } from './agentWake';
import { TOOL_IMAGES_TEXT } from './runner';
import type { WireMessage } from './types';
import type { UseMCPReturn } from './useMCP';

/** What the settings card renders. `status` is the server's view; the rest is this tab's. */
export interface RemoteControlState {
  enabled: boolean;
  setEnabled: (enabled: boolean) => void;
  /** The linked phone/chat is told if this device stops reporting. Implies `enabled`. */
  deadman: boolean;
  setDeadman: (deadman: boolean) => void;
  /** Per channel: null until its first status fetch answers (or while there is no code, or
   *  for WhatsApp no token). Telegram needs no sign-in. */
  status: Record<RemoteChannel, RemoteStatus | null>;
  /** This tab is the one holding the session open for the phone. */
  listening: boolean;
  lastMessageAt: number | null;
}

const STATUS_POLL_MS = 60_000;
const CHANNELS: RemoteChannel[] = ['whatsapp', 'telegram'];

/** Each channel listens on its own code. */
const codeFor = (channel: RemoteChannel) =>
  channel === 'telegram' ? SensorSettings.getTelegramCode() : SensorSettings.getWhitelistCode();

/** A wire message's text and images, whichever shape its content has. */
function contentParts(content: WireMessage['content']): { text: string; images: string[] } {
  if (typeof content === 'string') return { text: content, images: [] };
  if (!Array.isArray(content)) return { text: '', images: [] };
  return {
    text: content.filter(p => p?.type === 'text').map(p => p.text).join('\n'),
    images: content.filter(p => p?.type === 'image_url' && p.image_url?.url).map(p => p.image_url.url),
  };
}

/** How a user message reads on the phone, and who gets it: a message from a phone/chat isn't
 *  echoed back to the chat it came from (only to the other one, if both are linked). */
function userEcho(text: string, sessions: RemoteSessions): { to: RemoteSessions; text: string } {
  const fromPhone = parseRemotePrompt(text);
  if (fromPhone) {
    const { [fromPhone.channel]: _origin, ...others } = sessions;
    return { to: others, text: `💬 From ${CHANNEL_NAME[fromPhone.channel]}: ${fromPhone.text}` };
  }
  const fromAgent = parseAgentPrompt(text);
  if (fromAgent) return { to: sessions, text: `🤖 ${fromAgent.agentName}: ${fromAgent.text}` };
  return { to: sessions, text: `🖥️ On the computer: ${text}` };
}

export function useRemoteControl(
  mcp: Pick<UseMCPReturn, 'send' | 'isRunning' | 'subscribeTurn'>,
  getToken: TokenProvider,
): RemoteControlState {
  const getTokenRef = useRef(getToken);
  getTokenRef.current = getToken;

  const queue = useRef<RemoteMessage[]>([]);
  const draining = useRef(false);
  // The inbox loop that owns the `listening` flag.
  const listenerRef = useRef<AbortController | null>(null);
  // Bumped to re-run the drain effect when a message arrives or a remote run finishes.
  const [wake, setWake] = useState(0);

  const [enabled, setEnabledState] = useState(() => SensorSettings.isRemoteControlEnabled());
  const [deadman, setDeadmanState] = useState(() => SensorSettings.isDeadmanEnabled());
  const [status, setStatus] = useState<Record<RemoteChannel, RemoteStatus | null>>({ whatsapp: null, telegram: null });
  // The linked channels' codes, as JSON so the listener only restarts when they actually change.
  const [sessionKey, setSessionKey] = useState('{}');
  const sessionKeyRef = useRef(sessionKey);
  sessionKeyRef.current = sessionKey;
  const deadmanRef = useRef(deadman);
  deadmanRef.current = deadman;
  // The last answer per channel, and the code it was for: a failed status fetch falls back to it.
  const lastStatus = useRef<Partial<Record<RemoteChannel, { code: string; status: RemoteStatus }>>>({});
  const [listening, setListening] = useState(false);
  const [lastMessageAt, setLastMessageAt] = useState<number | null>(null);

  const setDeadman = useCallback((next: boolean) => {
    SensorSettings.setDeadmanEnabled(next);
    setDeadmanState(next);
    // Turning it off while listening needs nothing more: the listener restarts and its next
    // poll carries deadman=0.
    if (next) {
      SensorSettings.setRemoteControlEnabled(true);
      setEnabledState(true);
    }
  }, []);

  const setEnabled = useCallback((next: boolean) => {
    if (!next && deadmanRef.current) {
      // The listener is about to stop for good: disarm first, or the silence reads as an alarm.
      disarmDeadman(JSON.parse(sessionKeyRef.current) as RemoteSessions, () => getTokenRef.current());
      SensorSettings.setDeadmanEnabled(false);
      setDeadmanState(false);
    }
    SensorSettings.setRemoteControlEnabled(next);
    setEnabledState(next);
  }, []);

  // Only channels with a linked phone/chat are polled, all in one request: most tabs have nothing
  // linked and so hold no connection open at all.
  useEffect(() => {
    const sessions = JSON.parse(sessionKey) as RemoteSessions;
    if (!enabled || Object.keys(sessions).length === 0) {
      setListening(false);
      return;
    }
    const controller = new AbortController();
    listenerRef.current = controller;
    setListening(true);
    listenInbox({
      sessions,
      getToken: () => getTokenRef.current(),
      signal: controller.signal,
      deadman,
      onMessage: message => {
        Logger.info('MCP', `Remote message received via ${message.channel}`);
        setLastMessageAt(Date.now());
        queue.current.push(message);
        setWake(n => n + 1);
      },
      // Only the current loop may clear the flag. A superseded loop (React re-running this
      // effect, e.g. StrictMode's double mount) settles after its replacement started, and
      // would otherwise leave the UI reading "connecting…" while the new loop polls happily.
    }).finally(() => { if (listenerRef.current === controller) setListening(false); });
    return () => controller.abort();
  }, [enabled, sessionKey, deadman]);

  // The link itself lives on the server and changes when the user scans a QR on their phone,
  // so it is polled rather than derived from anything in this tab, and re-checked as soon as a
  // pairing UI sees a link. Runs even when disabled: the settings card still wants to say what
  // a re-enable would connect to.
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const token = await getTokenRef.current().catch(() => undefined);
      const results = await Promise.all(CHANNELS.map(async channel => {
        const code = codeFor(channel);
        if (!code || (channel === 'whatsapp' && !token)) return { channel, code, status: null };
        // A blip (network, 5xx) keeps the last answer for this code instead of reading as
        // unlinked, which would drop the listener and have the phone told no session is open.
        const prev = lastStatus.current[channel];
        const status = await tryFetchStatus(channel, code, token) ?? (prev?.code === code ? prev.status : null);
        return { channel, code, status };
      }));
      if (cancelled) return;
      const sessions: RemoteSessions = {};
      for (const { channel, code, status } of results) {
        lastStatus.current[channel] = code && status ? { code, status } : undefined;
        if (code && status?.linked) sessions[channel] = code;
      }
      setStatus(Object.fromEntries(results.map(r => [r.channel, r.status])) as Record<RemoteChannel, RemoteStatus | null>);
      setSessionKey(JSON.stringify(sessions));
    };
    check();
    const interval = window.setInterval(check, STATUS_POLL_MS);
    window.addEventListener(REMOTE_LINKED_EVENT, check);
    return () => {
      cancelled = true;
      clearInterval(interval);
      window.removeEventListener(REMOTE_LINKED_EVENT, check);
    };
  }, [enabled, lastMessageAt]);

  useEffect(() => {
    // No mcp.isRunning gate: mcp.send() now interrupts whatever's in flight (a hung tool call
    // like capture_screen's picker with nobody there to click it) rather than queuing behind
    // it, so a fresh remote message should take over immediately too.
    if (draining.current || queue.current.length === 0) return;
    const message = queue.current.shift()!;
    draining.current = true;

    // The answer reaches the phone through the mirror below, like every other turn.
    mcp.send(remotePrompt(message))
      .catch(error => Logger.error('MCP', `Remote run failed: ${error instanceof Error ? error.message : String(error)}`))
      .finally(() => {
        draining.current = false;
        setWake(n => n + 1);
      });
  }, [mcp.isRunning, mcp.send, wake]);

  // The mirror: each step of every turn goes to the linked phones/chats as it happens, one
  // request at a time so they arrive in order. Tool images wait for the next text and ride
  // with it (one per message); tool calls and raw results stay on the desktop.
  useEffect(() => {
    const sessions = JSON.parse(sessionKey) as RemoteSessions;
    if (!enabled || Object.keys(sessions).length === 0) return;
    let chain = Promise.resolve();
    let image: string | undefined;
    const forward = (to: RemoteSessions, item: PhoneItem) => {
      chain = chain
        .then(() => sendToPhone(to, item, () => getTokenRef.current()))
        .catch(error => { Logger.error('MCP', `Remote send failed: ${error instanceof Error ? error.message : String(error)}`); });
    };
    return mcp.subscribeTurn(event => {
      const { turnId } = event;
      if (event.kind === 'end') {
        image = undefined;
        forward(sessions, { turnId, done: true });
        return;
      }
      const { role, content } = event.message;
      const { text, images } = contentParts(content);
      if (role === 'assistant') {
        const prose = text.replace(/\u200B/g, '').trim();
        if (prose) {
          forward(sessions, { turnId, text: prose, image });
          image = undefined;
        }
      } else if (role === 'user' && text === TOOL_IMAGES_TEXT) {
        image = images.at(-1) ?? image;
      } else if (role === 'user') {
        const echo = userEcho(text, sessions);
        forward(echo.to, { turnId, text: echo.text, image: images.at(-1) });
      }
    });
  }, [enabled, sessionKey, mcp.subscribeTurn]);

  return { enabled, setEnabled, deadman, setDeadman, status, listening, lastMessageAt };
}
