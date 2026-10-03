// Utility to detect sensitive personal information in agent code

import { WORDLIST } from './whitelistCode';

const WORDLIST_SET = new Set(WORDLIST.map(w => w.toLowerCase()));

// Matches the shape of a golden-path whitelist code, e.g. "tree-book-shower-golden"
const PASSPHRASE_PATTERN = /\b[a-zA-Z]+(?:-[a-zA-Z]+){3}\b/g;

export interface PassphraseRedactionResult {
  redactedCode: string;
  removedCount: number;
  lineNumbers: number[];
}

/**
 * Finds and strips any whitelist passphrase (4 hyphen-joined words drawn from
 * the golden-path wordlist) from code. Only the owner can send with a code, but
 * whoever sends it by SMS/WhatsApp/Telegram first while the owner is pairing gets
 * bound to it — so they're removed automatically rather than just flagged.
 */
export function redactPassphrases(code: string): PassphraseRedactionResult {
  if (!code || typeof code !== 'string') {
    return { redactedCode: code, removedCount: 0, lineNumbers: [] };
  }

  let removedCount = 0;
  const lineNumbers: number[] = [];

  const redactedLines = code.split('\n').map((line, index) => {
    return line.replace(PASSPHRASE_PATTERN, (match) => {
      const words = match.split('-');
      if (words.length === 4 && words.every(w => WORDLIST_SET.has(w.toLowerCase()))) {
        removedCount++;
        lineNumbers.push(index + 1);
        return '[PASSPHRASE-REMOVED]';
      }
      return match;
    });
  });

  return {
    redactedCode: redactedLines.join('\n'),
    removedCount,
    lineNumbers
  };
}

export interface SensitiveDataDetection {
  hasSensitiveData: boolean;
  detectedFunctions: string[];
  lineNumbers: Record<string, number[]>; // function name -> line numbers where it appears
}

// Notification functions that may contain sensitive information
const SENSITIVE_FUNCTIONS = [
  'sendEmail',
  'sendPushover',
  'sendDiscord',
  'sendTelegram',
  'sendWhatsapp',
  'sendSms',
  'call'
];

const FUNCTION_DESCRIPTIONS: Record<string, string> = {
  sendEmail: 'Email addresses',
  sendPushover: 'Pushover tokens',
  sendDiscord: 'Discord webhooks',
  sendTelegram: 'Telegram chat IDs',
  sendWhatsapp: 'Observer codes',
  sendSms: 'Observer codes',
  call: 'Observer codes'
};

const PLACEHOLDER_SUGGESTIONS: Record<string, string> = {
  sendEmail: '"example@email.com"',
  sendPushover: '"your_pushover_token_here"',
  sendDiscord: '"https://discord.com/api/webhooks/1234/id"',
  sendTelegram: '"123456789"',
  sendWhatsapp: '"your-observer-code"',
  sendSms: '"your-observer-code"',
  call: '"your-observer-code"'
};

/**
 * Detects if code contains notification functions that may have personal information
 * @param code The JavaScript code to analyze
 * @returns Detection result with list of found functions and their locations
 */
export function detectSensitiveFunctions(code: string): SensitiveDataDetection {
  const detectedFunctions: string[] = [];
  const lineNumbers: Record<string, number[]> = {};

  if (!code || typeof code !== 'string') {
    return {
      hasSensitiveData: false,
      detectedFunctions: [],
      lineNumbers: {}
    };
  }

  // Split code into lines for line number tracking
  const lines = code.split('\n');

  // Check each sensitive function
  for (const funcName of SENSITIVE_FUNCTIONS) {
    // Create regex to match function calls
    // Matches: funcName( with optional whitespace
    const regex = new RegExp(`\\b${funcName}\\s*\\(`, 'g');

    // Track line numbers where this function appears
    const foundLines: number[] = [];

    lines.forEach((line, index) => {
      if (regex.test(line)) {
        foundLines.push(index + 1); // Line numbers start at 1
      }
    });

    if (foundLines.length > 0) {
      detectedFunctions.push(funcName);
      lineNumbers[funcName] = foundLines;
    }
  }

  return {
    hasSensitiveData: detectedFunctions.length > 0,
    detectedFunctions,
    lineNumbers
  };
}

/**
 * Get human-readable description of what data type a function may contain
 */
export function getFunctionDescription(funcName: string): string {
  return FUNCTION_DESCRIPTIONS[funcName] || 'sensitive data';
}

/**
 * Get all detected data types as a readable list
 */
export function getDetectedDataTypes(detectedFunctions: string[]): string[] {
  return detectedFunctions.map(func => FUNCTION_DESCRIPTIONS[func] || func);
}

/**
 * Get placeholder suggestion for a function
 */
export function getPlaceholderSuggestion(funcName: string): string {
  return PLACEHOLDER_SUGGESTIONS[funcName] || '"placeholder"';
}
