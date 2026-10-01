// src/utils/contactInfo.ts
//
// Validation + user-facing copy for the contact values Observer's notification tools need
// (sendSms/call/sendWhatsapp, sendEmail, sendTelegram, sendDiscord, sendPushover).
//
// Lives here rather than in a component so the `ask_user_info` modal and any other collector
// share one definition of "is this value usable" — previously this logic was inline in
// RecipeSplash and duplicated as prose in SimpleCreatorModal.

import type { UserInfoKind } from '../mcp/types';
import { normalizeWhitelistCode } from './whitelistCode';

/** The Observer Telegram bot: users pair a chat by sending it their Telegram code. */
export const TELEGRAM_BOT = 'observer_notification_bot';
export const TELEGRAM_BOT_URL = `https://t.me/${TELEGRAM_BOT}`;
/** Bot deep link that sends `/start <code>`, linking the chat to the user's whitelist code. */
export const telegramCodeLink = (code: string) => `${TELEGRAM_BOT_URL}?start=${encodeURIComponent(code)}`;

export const CONTACT_PLACEHOLDER: Record<UserInfoKind, string> = {
  phone: '+1 555 123 4567',
  email: 'you@email.com',
  telegram: 'tree-book-shower-golden',
  discord: 'https://discord.com/api/webhooks/…',
  pushover: 'Your Pushover user key',
};

export const CONTACT_LABEL: Record<UserInfoKind, string> = {
  phone: 'Your phone number',
  email: 'Your email address',
  telegram: 'Your Telegram code',
  discord: 'Discord webhook URL',
  pushover: 'Pushover user key',
};

/** Whether a value is complete enough to hand to the notification tool. */
export function contactValid(kind: UserInfoKind, value: string): boolean {
  const v = value.trim();
  switch (kind) {
    case 'phone': return /^\+?[0-9][0-9\s()-]{6,}$/.test(v);
    case 'email': return /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(v);
    // Telegram is reached only through its 4-word code, never a raw chat ID.
    case 'telegram': return normalizeWhitelistCode(v) !== null;
    case 'discord': return /^https?:\/\/(canary\.|ptb\.)?discord(app)?\.com\/api\/webhooks\/.+/.test(v);
    case 'pushover': return v.length >= 20;
  }
}

/** Why a value was rejected, for inline feedback. Null when valid or still empty. */
export function contactError(kind: UserInfoKind, value: string): string | null {
  const v = value.trim();
  if (!v || contactValid(kind, v)) return null;
  switch (kind) {
    case 'phone': return 'Include the country code, e.g. +1 555 123 4567.';
    case 'email': return "That doesn't look like an email address.";
    case 'telegram': return 'Use your 4-word Telegram code from Settings.';
    case 'discord': return 'Paste the full webhook URL starting with https://discord.com/api/webhooks/';
    case 'pushover': return 'Pushover user keys are 30 characters.';
  }
}

/** Normalize before handing to the model / persisting. */
export function normalizeContact(kind: UserInfoKind, value: string): string {
  const v = value.trim();
  if (kind === 'phone') {
    const digits = v.replace(/[^\d+]/g, '');
    return digits.startsWith('+') ? digits : `+${digits}`;
  }
  return v;
}
