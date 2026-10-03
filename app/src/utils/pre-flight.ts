// src/utils/pre-flight.ts

import type { TokenProvider } from './main_loop';
import type { WhitelistChannel } from './logging';
import { normalizeWhitelistCode } from './whitelistCode';

export interface PhoneWhitelistResult {
  /**
   * Every literal passed to a phone tool. `isCode` is false for anything that isn't a
   * 4-word whitelist code (e.g. a raw phone number): the server rejects those outright,
   * so they are never whitelisted and the fix is to swap in the user's code.
   */
  phoneNumbers: Array<{ number: string; isWhitelisted: boolean; isCode: boolean; channel: WhitelistChannel }>;
  hasTools: boolean;
  channel?: WhitelistChannel; // 'whatsapp' | 'sms' | 'voice'
}

/**
 * Check if agent code uses phone tools and verify the codes it sends to are paired
 */
export async function checkPhoneWhitelist(
  agentCode: string,
  getToken?: TokenProvider
): Promise<PhoneWhitelistResult> {
  // Check if code contains phone tools
  const hasWhatsapp = agentCode.includes('sendWhatsapp(');
  const hasSms = agentCode.includes('sendSms(');
  const hasCall = agentCode.includes('call(');
  const hasPhoneTools = hasWhatsapp || hasSms || hasCall;

  if (!hasPhoneTools) {
    return { phoneNumbers: [], hasTools: false };
  }

  // Each tool needs its own pairing: sendSms an SMS one, sendWhatsapp a WhatsApp one (with
  // WhatsApp's 24h window open), call either. So every code is checked once per tool it is
  // used with. `channel` is the most demanding one, for callers that show a single prompt.
  const channel: WhitelistChannel = hasWhatsapp ? 'whatsapp' : hasSms ? 'sms' : 'voice';

  // Extract the literal string argument passed to each phone tool call, rather than
  // guessing at phone-shaped substrings in the code.
  const argRegex = /\b(sendWhatsapp|sendSms|call)\(\s*["']([^"']+)["']/g;
  const toolChannel: Record<string, WhitelistChannel> = { sendWhatsapp: 'whatsapp', sendSms: 'sms', call: 'voice' };
  const uses = new Map<string, { number: string; channel: WhitelistChannel }>();
  for (const m of agentCode.matchAll(argRegex)) {
    const use = { number: m[2], channel: toolChannel[m[1]] };
    uses.set(`${use.channel}:${use.number}`, use);
  }

  if (uses.size === 0) {
    // Tools present but no numbers found
    return { phoneNumbers: [], hasTools: true, channel };
  }

  // Get auth token
  if (!getToken) {
    throw new Error('Authentication required to check phone whitelist');
  }

  const token = await getToken();
  if (!token) {
    throw new Error('No authentication token available');
  }

  const phoneNumbers = await Promise.all(
    [...uses.values()].map(async ({ number, channel }) => {
      if (!normalizeWhitelistCode(number)) {
        return { number, isWhitelisted: false, isCode: false, channel };
      }
      try {
        const response = await fetch('https://api.observer-ai.com/tools/is-whitelisted', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
          },
          body: JSON.stringify({ phone_number: number, channel }),
        });

        if (!response.ok) {
          return { number, isWhitelisted: false, isCode: true, channel };
        }

        const data = await response.json();
        return { number, isWhitelisted: data.is_whitelisted, isCode: true, channel };
      } catch (error) {
        console.error(`Error checking whitelist for ${number}:`, error);
        return { number, isWhitelisted: false, isCode: true, channel };
      }
    })
  );

  return { phoneNumbers, hasTools: true, channel };
}
