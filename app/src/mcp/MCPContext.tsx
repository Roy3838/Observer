// src/mcp/MCPContext.tsx
//
// App-level provider for the MCP conversation. Mounting the useMCP state HERE (once, high
// in the tree) instead of inside MCP.tsx gives us two things at once:
//   1. A single shared conversation — open the MCP UI from GetStarted or the modal and it's
//      the same wire/messages, so users keep their progress as they move around the app.
//   2. A stable home for the async agentic loop — the provider never unmounts, so a run
//      keeps updating state even if the user closes the modal mid-flight.
//
// MCP.tsx is a pure consumer via useMCPContext().

import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import type { TokenProvider } from '@utils/main_loop';
import { useMCP, type UseMCPReturn } from './useMCP';
import { useRemoteControl, type RemoteControlState } from './useRemoteControl';
import UserInfoModal from '@components/AICreator/UserInfoModal';

const DEFAULT_MODEL = 'gemini-2.5-flash-lite-free';

interface MCPContextValue extends UseMCPReturn {
  modelName: string;
  setModelName: (name: string) => void;
  /** Remote control (WhatsApp/Telegram) status + on/off, rendered by the settings card. */
  remote: RemoteControlState;
}

const MCPContext = createContext<MCPContextValue | null>(null);

interface MCPProviderProps {
  getToken: TokenProvider;
  isUsingObServer: boolean;
  children: React.ReactNode;
}

export const MCPProvider: React.FC<MCPProviderProps> = ({ getToken, isUsingObServer, children }) => {
  const [modelName, setModelName] = useState(DEFAULT_MODEL);
  const mcp = useMCP({
    getToken,
    isUsingObServer,
    modelName,
  });
  // Messages from the user's linked WhatsApp/Telegram land in this same conversation.
  const remote = useRemoteControl(mcp, getToken);
  // End-to-end suite hooks (app/e2e), only when served with VITE_E2E=1. Refs so the hooks
  // always reach this render's mcp and getToken.
  const e2eRef = useRef({ mcp: { ...mcp, modelName }, getToken });
  e2eRef.current = { mcp: { ...mcp, modelName }, getToken };
  useEffect(() => {
    if (import.meta.env.VITE_E2E !== '1') return;
    import('./e2e').then(m => m.installE2EHooks({
      mcp: () => e2eRef.current.mcp,
      getToken: () => e2eRef.current.getToken,
    }));
  }, []);
  return (
    <MCPContext.Provider value={{ ...mcp, modelName, setModelName, remote }}>
      {children}
      {/* Hosted here, not in MCP.tsx: a run survives closing the chat panel, and RecipeSplash
          closes itself right after send() — so the modal must outlive both. */}
      {mcp.pendingUserInfo && (
        <UserInfoModal key={mcp.pendingUserInfo.requestId} req={mcp.pendingUserInfo} onResolve={mcp.resolveUserInfo} />
      )}
    </MCPContext.Provider>
  );
};

export function useMCPContext(): MCPContextValue {
  const ctx = useContext(MCPContext);
  if (!ctx) {
    throw new Error('useMCPContext must be used within an <MCPProvider>');
  }
  return ctx;
}
