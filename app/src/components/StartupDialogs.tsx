import React, { useState, useEffect } from 'react';
import { Analytics } from '@utils/analytics';
import { addCustomServer, checkCustomServer, removeCustomServer } from '@utils/inferenceServer';

interface StartupDialogProps {
  onDismiss: () => void;
  onSkip?: () => void;
  onLogin?: () => void;
  onToggleObServer?: () => void;
  onServerConnected?: () => void;
  isAuthenticated: boolean;
  hostingContext: 'official-web' | 'self-hosted' | 'tauri';
  hasPendingImport?: boolean;
}


const StartupDialog: React.FC<StartupDialogProps> = ({
  onLogin,
  onToggleObServer,
  onServerConnected,
  onDismiss,
  onSkip,
  isAuthenticated,
  hasPendingImport,
}) => {
  const [showServerInput, setShowServerInput] = useState(false);
  const [serverAddress, setServerAddress] = useState('');
  const [serverError, setServerError] = useState('');
  const [isChecking, setIsChecking] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);

  // After a successful server check, count down then skip sign in
  useEffect(() => {
    if (countdown === null) return;
    if (countdown === 0) {
      Analytics.startupSkip();
      onSkip?.();
      onDismiss();
      return;
    }
    const t = setTimeout(() => setCountdown(countdown - 1), 1000);
    return () => clearTimeout(t);
  }, [countdown]);

  const handleCheckServer = async () => {
    setServerError('');
    const address = serverAddress.trim();
    if (!address) { setServerError('Please enter a server address'); return; }
    if (!address.match(/^https?:\/\//)) { setServerError('URL must start with http:// or https://'); return; }
    if (address.includes('api.observer-ai.com')) { setServerError('Use Sign In for Observer Cloud'); return; }
    try {
      new URL(address);
    } catch {
      setServerError('Invalid URL format');
      return;
    }

    setIsChecking(true);
    addCustomServer(address);
    const result = await checkCustomServer(address);
    setIsChecking(false);

    if (result.status === 'online') {
      onServerConnected?.();
      setCountdown(3);
    } else {
      removeCustomServer(address);
      setServerError(result.error || 'Could not connect to server');
    }
  };

  // Don't show dialog if user is already authenticated
  if (isAuthenticated) {
    return null;
  }

  const handleSignIn = () => {
    // Set login intent in sessionStorage (persists through auth redirect, clears on tab close)
    sessionStorage.setItem('observer_login_intent', 'true');
    Analytics.startupSignIn();

    if (onLogin) {
      onLogin();
    }
    // Enable ObServer after signing in
    if (onToggleObServer) {
      onToggleObServer();
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex flex-col items-center justify-center z-[102] backdrop-blur-sm p-4">
      <div className="bg-white rounded-xl shadow-xl p-6 sm:p-8 max-w-md w-full transition-all duration-300">
        <div className="text-center">
          {/* Observer Logo/Icon */}
          <div className="flex justify-center mb-6">
            <img
              src="/eye-logo-black.svg"
              alt="Observer Logo"
              className="h-16 w-16"
            />
          </div>

          {/* Welcome Message */}
          <h1 className="text-2xl font-bold text-gray-900 mb-2">
            {hasPendingImport ? 'Sign in first!' : 'Welcome to Observer'}
          </h1>
          <p className="text-gray-600 mb-8 leading-relaxed">
            {hasPendingImport
              ? "You need an account to import agents. Sign in, then click the share link again to import it."
              : <>The agent that monitors your screen,<br />so you don't have to.</>}
          </p>

          {/* Action Buttons */}
          <div className="space-y-3">
            <button
              onClick={handleSignIn}
              className="w-full px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium shadow-sm hover:shadow-md"
            >
              {hasPendingImport ? 'Sign In to Import Agent' : 'Sign In to Start Creating Agents'}
            </button>
          </div>
        </div>
      </div>

      {/* Other options: intentionally de-emphasized, outside the white card.
          Skipping sign in requires a working v1 inference server. */}
      {onServerConnected && !hasPendingImport && (
        !showServerInput ? (
          <button
            onClick={() => setShowServerInput(true)}
            className="mt-4 text-xs text-gray-400 hover:text-gray-300 transition-colors"
          >
            Other options
          </button>
        ) : (
          <div className="mt-4 bg-white rounded-xl shadow-xl p-4 max-w-md w-full text-left">
            <p className="text-sm font-medium text-gray-800 mb-1">Use your own inference server</p>
            <p className="text-xs text-gray-500 mb-2">
              Enter your v1/chat/completions server (e.g. Ollama, LM Studio, vLLM) to skip sign in.
              The agent creator needs a capable model.
            </p>
            <div className="flex gap-2">
              <input
                type="text"
                value={serverAddress}
                onChange={(e) => { setServerAddress(e.target.value); setServerError(''); }}
                placeholder="http://localhost:11434"
                disabled={isChecking || countdown !== null}
                autoFocus
                className="flex-1 min-w-0 p-2 text-sm border border-gray-200 rounded-md"
              />
              <button
                onClick={handleCheckServer}
                disabled={isChecking || countdown !== null}
                className="px-3 py-2 text-sm bg-gray-800 text-white rounded-md hover:bg-gray-700 disabled:opacity-50"
              >
                {isChecking ? 'Checking...' : 'Check'}
              </button>
            </div>
            {serverError && <p className="text-xs text-red-500 mt-2">{serverError}</p>}
            {countdown !== null && (
              <p className="text-xs text-green-600 mt-2">Done! Skipping in {countdown}...</p>
            )}
            <div className="text-xs text-gray-500 mt-3 space-y-1">
              <p>❌ SMS, WhatsApp, Telegram, Email and Voice calling won't work without an account.</p>
              <p>✅ Discord notifications and Memory logging/recording work!</p>
            </div>
          </div>
        )
      )}
    </div>
  );
};

export default StartupDialog;
