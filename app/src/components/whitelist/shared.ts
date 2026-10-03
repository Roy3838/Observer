// src/components/whitelist/shared.ts
//
// Single source of truth for the Observer SMS and WhatsApp contacts, the pairing QR payloads,
// and the background polling loop. Shared by the full WhitelistModal and the compact inline
// chip the MCP renders under a `check_whitelist` tool call, so the two surfaces can never
// drift apart.
//
// Phones are reached only through the user's 4-word codes (see api/remote.py), one per
// channel because each pairing is its own opt-in: the SMS code, texted to Observer's number,
// enables SMS and calls; the WhatsApp code, sent to the bot, enables WhatsApp and calls.
// Telegram has its own code, which needs no account: see useTelegramStatus.

import { useEffect, useRef, useState } from 'react';
import type { WhitelistChannel } from '@utils/logging';
import { openExternal } from '@utils/platform';
import { SensorSettings } from '@utils/settings';
import { fetchStatus, type RemoteStatus } from '../../mcp/remote';

export const OBSERVER_WHATSAPP = '+1 (555) 783-4727';
export const OBSERVER_WHATSAPP_PLAIN = '15557834727';

/** Opens WhatsApp with the code prefilled: sending it pairs the phone. */
export const whatsappCodeQRValue = (code: string) =>
  `https://wa.me/${OBSERVER_WHATSAPP_PLAIN}?text=${encodeURIComponent(code)}`;

export const openWhatsApp = () => openExternal(`https://wa.me/${OBSERVER_WHATSAPP_PLAIN}`);

export const OBSERVER_SMS = '+1 (863) 208-5341';
export const OBSERVER_SMS_PLAIN = '+18632085341';

/** Opens the messaging app with the code prefilled: texting it pairs the phone for SMS and calls.
 *  `?&body=` is the form both iOS and Android read. */
export const smsCodeQRValue = (code: string) =>
  `sms:${OBSERVER_SMS_PLAIN}?&body=${encodeURIComponent(code)}`;

export const openSms = (code: string) => openExternal(smsCodeQRValue(code));

/** The disclosure shown wherever the SMS code is: texting it is the user's SMS opt-in. */
export const SMS_CONSENT =
  'By texting this code you agree to receive alert texts and calls from Observer AI. ' +
  'Msg frequency varies. Msg & data rates may apply. Reply HELP for help, DISCONNECT to unlink.';

export interface PhoneEntry {
  number: string;
  isWhitelisted: boolean;
}

export type WhitelistPollStatus = 'idle' | 'checking' | 'success';

/** Check a single code against the whitelist API; resolves false on any failure. */
export async function checkNumber(
  number: string,
  token: string,
  channel?: WhitelistChannel,
): Promise<PhoneEntry> {
  try {
    const response = await fetch('https://api.observer-ai.com/tools/is-whitelisted', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({ phone_number: number, ...(channel ? { channel } : {}) }),
    });
    if (!response.ok) return { number, isWhitelisted: false };
    const data = await response.json();
    return { number, isWhitelisted: data.is_whitelisted };
  } catch (error) {
    console.error(`Error checking whitelist for ${number}:`, error);
    return { number, isWhitelisted: false };
  }
}

/**
 * Polls every number against the whitelist API on a 5s loop until all are whitelisted,
 * then stops. Returns the live statuses plus an aggregate flag the UI flips to green on.
 */
export function useWhitelistPolling(
  initial: PhoneEntry[],
  getToken: () => Promise<string | undefined>,
  channel?: WhitelistChannel,
  enabled = true,
) {
  // Keep the latest getToken without making it an effect dependency (callers often pass a
  // fresh closure each render, which would otherwise restart the interval constantly).
  const getTokenRef = useRef(getToken);
  getTokenRef.current = getToken;

  const key = initial.map(p => p.number).join(',');
  const [numbers, setNumbers] = useState<PhoneEntry[]>(initial);
  const [status, setStatus] = useState<WhitelistPollStatus>('idle');

  useEffect(() => {
    const list = key ? key.split(',') : [];
    if (!enabled || list.length === 0) return;

    let cancelled = false;
    let intervalId = 0;

    const checkAll = async () => {
      setStatus(prev => (prev === 'success' ? prev : 'checking'));
      const token = await getTokenRef.current();
      if (!token) {
        if (!cancelled) setStatus('idle');
        return;
      }
      const checks = await Promise.all(list.map(n => checkNumber(n, token, channel)));
      if (cancelled) return;
      setNumbers(checks);
      if (checks.every(p => p.isWhitelisted)) {
        setStatus('success');
        clearInterval(intervalId);
      } else {
        setStatus('idle');
      }
    };

    checkAll();
    intervalId = window.setInterval(checkAll, 5000);
    return () => { cancelled = true; clearInterval(intervalId); };
  }, [enabled, channel, key]);

  const allWhitelisted = numbers.length > 0 && numbers.every(p => p.isWhitelisted);
  return { numbers, status, allWhitelisted };
}

/**
 * The user's Telegram code and its pairing, polled every 5s until linked. No token: the
 * Telegram code is its own credential. A revoked code (the chat sent /stop, or paired a newer
 * code) is dead for good, so it is swapped for a fresh one and the QR shows that instead.
 */
export function useTelegramStatus(enabled = true) {
  const [code, setCode] = useState(() => SensorSettings.ensureTelegramCode());
  const [status, setStatus] = useState<RemoteStatus | null>(null);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let intervalId = 0;

    const check = async () => {
      const next = await fetchStatus('telegram', code);
      if (cancelled) return;
      if (next.revoked) {
        setStatus(null);
        setCode(SensorSettings.rotateTelegramCode());
        return;
      }
      setStatus(next);
      if (next.linked) clearInterval(intervalId);
    };

    check();
    intervalId = window.setInterval(check, 5000);
    return () => { cancelled = true; clearInterval(intervalId); };
  }, [enabled, code]);

  const rotate = () => {
    setStatus(null);
    setCode(SensorSettings.rotateTelegramCode());
  };

  return { code, status, linked: !!status?.linked, rotate };
}
