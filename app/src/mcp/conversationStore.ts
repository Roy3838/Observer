// src/mcp/conversationStore.ts
//
// localStorage persistence for MCP conversations. Framework-free. Only non-system wire
// messages are stored (the system prompt is rebuilt on load so prompt updates apply to old chats).

import type { WireMessage } from './types';
import { parseRemotePrompt } from './remote';
import { parseAgentPrompt } from './agentWake';

const STORAGE_KEY = 'observer-mcp-conversations';
const MAX_CONVERSATIONS = 50;

export interface StoredConversation {
  id: string;
  title: string;
  updatedAt: number;
  messages: WireMessage[];
}

export type ConversationSummary = Pick<StoredConversation, 'id' | 'title' | 'updatedAt'>;

export const newConversationId = (): string =>
  `c_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`;

function readAll(): StoredConversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeAll(list: StoredConversation[]): void {
  // Quota errors: drop the oldest conversations until it fits.
  let items = list;
  for (;;) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
      return;
    } catch {
      if (items.length === 0) return;
      items = items.slice(0, -1);
    }
  }
}

/** Base64 screenshots would blow the ~5MB quota, so persisted image parts become a text stub. */
function stripImages(messages: WireMessage[]): WireMessage[] {
  return messages.map(m =>
    Array.isArray(m.content)
      ? {
          ...m,
          content: m.content.map((p: any) =>
            p?.type === 'image_url' ? { type: 'text', text: '[image omitted]' } : p,
          ),
        }
      : m,
  );
}

function deriveTitle(messages: WireMessage[]): string {
  const first = messages.find(m => m.role === 'user');
  let text = '';
  if (typeof first?.content === 'string') text = first.content;
  else if (Array.isArray(first?.content)) text = first!.content.find((p: any) => p?.type === 'text')?.text ?? '';
  text = (parseRemotePrompt(text)?.text ?? parseAgentPrompt(text)?.text ?? text).replace(/\s+/g, ' ').trim();
  return text ? (text.length > 40 ? `${text.slice(0, 40)}…` : text) : 'New chat';
}

export function listConversations(): ConversationSummary[] {
  return readAll()
    .map(({ id, title, updatedAt }) => ({ id, title, updatedAt }))
    .sort((a, b) => b.updatedAt - a.updatedAt);
}

export function loadConversation(id: string): StoredConversation | null {
  return readAll().find(c => c.id === id) ?? null;
}

export function saveConversation(id: string, messages: WireMessage[]): void {
  const visible = messages.filter(m => m.role !== 'system');
  if (visible.length === 0) return;
  const rest = readAll().filter(c => c.id !== id);
  const entry: StoredConversation = {
    id,
    title: deriveTitle(visible),
    updatedAt: Date.now(),
    messages: stripImages(visible),
  };
  writeAll([entry, ...rest].slice(0, MAX_CONVERSATIONS));
}

export function deleteStoredConversation(id: string): void {
  writeAll(readAll().filter(c => c.id !== id));
}
