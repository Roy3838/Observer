// src/mcp/useRemoteControl.ts
//
// Remote control, for each linked WhatsApp/Telegram the user chose to mirror to: messages sent
// from it are injected into whatever MCP conversation is open in this tab, and that conversation
// is mirrored back to it as it happens — whoever started the turn (the phone, the desktop, an
// agent). A linked phone/chat that isn't mirrored only gets what agents send.
// Mounted once, by MCPProvider, so it keeps listening while the chat panel is closed —
// "watch this for me", walk away, "how's it going?".
//
// `send` ignores calls while a run is in flight, so messages wait in a queue and are drained
// one at a time whenever the MCP is idle.
//
// The dead-man's switch is part of the mirror and rides on the same long-poll: the server tells
// the phone/chat if this tab stops polling (device locked, app killed, offline). Turning a mirror
// off disarms it first, so that silence isn't an alarm.

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
  /** Per channel: its phone/chat mirrors the conversation (once linked). */
  mirror: Record<RemoteChannel, boolean>;
  setMirror: (channel: RemoteChannel, on: boolean) => void;
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

  const [mirror, setMirrorState] = useState<Record<RemoteChannel, boolean>>(() => ({
    whatsapp: SensorSettings.getMirror('whatsapp') === 'on',
    telegram: SensorSettings.getMirror('telegram') === 'on',
  }));
  const [status, setStatus] = useState<Record<RemoteChannel, RemoteStatus | null>>({ whatsapp: null, telegram: null });
  const [linked, setLinked] = useState<RemoteSessions>({});
  const linkedRef = useRef(linked);
  linkedRef.current = linked;
  // The one gate: linked and mirrored. As JSON so the listener only restarts on a real change.
  const sessionKey = JSON.stringify(Object.fromEntries(
    CHANNELS.filter(c => linked[c] && mirror[c]).map(c => [c, linked[c]]),
  ));
  // The last answer per channel, and the code it was for: a failed status fetch falls back to it.
  const lastStatus = useRef<Partial<Record<RemoteChannel, { code: string; status: RemoteStatus }>>>({});
  const [listening, setListening] = useState(false);
  const [lastMessageAt, setLastMessageAt] = useState<number | null>(null);

  const setMirror = useCallback((channel: RemoteChannel, on: boolean) => {
    const code = linkedRef.current[channel];
    if (!on && code && SensorSettings.getMirror(channel) === 'on') {
      // Its polls are about to stop for good: disarm first, or the silence reads as an alarm.
      disarmDeadman({ [channel]: code }, () => getTokenRef.current());
    }
    SensorSettings.setMirror(channel, on);
    setMirrorState(prev => ({ ...prev, [channel]: on }));
  }, []);

  // Only mirrored phones/chats are polled, all in one request: most tabs have none and so hold
  // no connection open at all.
  useEffect(() => {
    const sessions = JSON.parse(sessionKey) as RemoteSessions;
    if (Object.keys(sessions).length === 0) {
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
  }, [sessionKey]);

  // The link itself lives on the server and changes when the user scans a QR on their phone,
  // so it is polled rather than derived from anything in this tab, and re-checked as soon as a
  // pairing UI sees a link. Covers unmirrored channels too: the settings card shows them.
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const token = await getTokenRef.current().catch(() => undefined);
      const results = await Promise.all(CHANNELS.map(async channel => {
        const code = codeFor(channel);
        if (!code) SensorSettings.seedMirror(channel, false);
        if (!code || (channel === 'whatsapp' && !token)) return { channel, code, status: null };
        // A blip (network, 5xx) keeps the last answer for this code instead of reading as
        // unlinked, which would drop the listener and have the phone told no session is open.
        const prev = lastStatus.current[channel];
        const status = await tryFetchStatus(channel, code, token) ?? (prev?.code === code ? prev.status : null);
        if (status) SensorSettings.seedMirror(channel, status.linked);
        return { channel, code, status };
      }));
      if (cancelled) return;
      const sessions: RemoteSessions = {};
      for (const { channel, code, status } of results) {
        lastStatus.current[channel] = code && status ? { code, status } : undefined;
        if (code && status?.linked) sessions[channel] = code;
      }
      setStatus(Object.fromEntries(results.map(r => [r.channel, r.status])) as Record<RemoteChannel, RemoteStatus | null>);
      setLinked(sessions);
      // Picks up a seeded mirror (or one turned on in another tab). Never turns one off:
      // that has to disarm first, see setMirror.
      setMirrorState(prev => ({
        whatsapp: prev.whatsapp || SensorSettings.getMirror('whatsapp') === 'on',
        telegram: prev.telegram || SensorSettings.getMirror('telegram') === 'on',
      }));
    };
    check();
    const interval = window.setInterval(check, STATUS_POLL_MS);
    window.addEventListener(REMOTE_LINKED_EVENT, check);
    return () => {
      cancelled = true;
      clearInterval(interval);
      window.removeEventListener(REMOTE_LINKED_EVENT, check);
    };
  }, [lastMessageAt]);

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

  // The mirror: each step of every turn goes to the mirrored phones/chats as it happens, one
  // request at a time so they arrive in order. Tool images wait for the next text and ride
  // with it (one per message); tool calls and raw results stay on the desktop.
  useEffect(() => {
    const sessions = JSON.parse(sessionKey) as RemoteSessions;
    if (Object.keys(sessions).length === 0) return;
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
  }, [sessionKey, mcp.subscribeTurn]);

  return { mirror, setMirror, status, listening, lastMessageAt };
}
