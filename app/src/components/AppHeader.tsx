// components/AppHeader.tsx
import React, { useState, useEffect } from 'react';
import { Server, Menu, Sun, Moon, User } from 'lucide-react';
import {
  checkInferenceServer,
  addInferenceAddress,
  removeInferenceAddress,
  fetchModels,
  loadCustomServers,
  getCustomServers,
  addCustomServer,
  removeCustomServer,
  toggleCustomServer,
  checkCustomServer,
  type CustomServer
} from '@utils/inferenceServer';
import { Logger } from '@utils/logging';
import { isTauri } from '@utils/platform';
import { GemmaModelManager } from '@utils/localLlm/GemmaModelManager';
import { NativeLlmManager } from '@utils/localLlm/NativeLlmManager';
import SharingPermissionsModal from './SharingPermissionsModal';
import ModelHub from './ModelHub';
import AccountModal from './AccountModal';
import StartupDialogs from './StartupDialogs';
import type { TokenProvider } from '@utils/main_loop';

// Server address constants
const OB_SERVER_ADDRESS = 'https://api.observer-ai.com:443';
const LOCAL_SERVER_ADDRESS = 'http://localhost:3838';


// --- The rest of your component ---
type QuotaInfo = {
  used: number;
  remaining: number;
  limit: number;
  tier: string;
} | null;

interface AuthState {
  isLoading: boolean;
  isAuthenticated: boolean;
  user: any;
  loginWithRedirect: () => void;
  logout: (options?: any) => void;
}

interface AppHeaderProps {
  authState?: AuthState;
  shouldHighlightMenu?: boolean;
  isUsingObServer?: boolean;
  setIsUsingObServer?: (value: boolean) => void;
  hostingContext?: 'official-web' | 'self-hosted' | 'tauri';
  getToken: TokenProvider;
  onUpgradeClick?: () => void;
  quotaInfo: QuotaInfo;
  setQuotaInfo: React.Dispatch<React.SetStateAction<QuotaInfo>>;
  onToggleMobileMenu?: () => void;
  isDarkMode?: boolean;
  onToggleDarkMode?: () => void;
}



const AppHeader: React.FC<AppHeaderProps> = ({
  authState,
  isUsingObServer: externalIsUsingObServer,
  setIsUsingObServer: externalSetIsUsingObServer,
  hostingContext = 'self-hosted',
  getToken,
  onUpgradeClick,
  quotaInfo,
  setQuotaInfo,
  onToggleMobileMenu,
  isDarkMode,
  onToggleDarkMode,
}) => {
  const [localServerOnline, setLocalServerOnline] = useState(false);
  const [customServers, setCustomServers] = useState<CustomServer[]>([]);
  const [appInferenceUrl, setAppInferenceUrl] = useState<string | null>(null);

  const [internalIsUsingObServer, setInternalIsUsingObServer] = useState(false);
  const [isLoadingQuota, setIsLoadingQuota] = useState(false);
  const [isPermissionsModalOpen, setIsPermissionsModalOpen] = useState(false);
  const [showLoginMessage, setShowLoginMessage] = useState(false);
  const [isSessionExpired, setIsSessionExpired] = useState(false);
  const [isQuotaHovered, setIsQuotaHovered] = useState(false);
  const [has70PercentWarningBeenShown, setHas70PercentWarningBeenShown] = useState(false);

  // --- NEW --- State to control the visibility of the new settings modal
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);
  const [isStartupDialogOpen, setIsStartupDialogOpen] = useState(false);
  const [isAccountModalOpen, setIsAccountModalOpen] = useState(false);
  const [autoDownloadPreset, setAutoDownloadPreset] = useState<import('@utils/modelPresets').ModelPreset | undefined>(undefined);
  const [isModelDownloading, setIsModelDownloading] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(false);

  useEffect(() => {
    const handleOpenModelHub = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail?.autoDownloadPreset) setAutoDownloadPreset(detail.autoDownloadPreset);
      setIsSettingsModalOpen(true);
    };
    window.addEventListener('openModelHub', handleOpenModelHub);
    return () => window.removeEventListener('openModelHub', handleOpenModelHub);
  }, []);

  const [isNativeLoading, setIsNativeLoading] = useState(false);
  const [isNativeDownloading, setIsNativeDownloading] = useState(false);

  // Track Transformers.js model loading state
  useEffect(() => {
    const manager = GemmaModelManager.getInstance();
    const unsubscribe = manager.onStateChange((state) => {
      setIsModelLoading(state.status === 'loading');
    });
    const initial = manager.getState();
    setIsModelLoading(initial.status === 'loading');
    return unsubscribe;
  }, []);

  // Track llama.cpp download/loading state (Tauri only)
  useEffect(() => {
    if (!isTauri()) return;
    const manager = NativeLlmManager.getInstance();
    const unsubscribe = manager.onStateChange((state) => {
      setIsNativeDownloading(state.status === 'downloading');
      setIsNativeLoading(state.status === 'loading');
    });
    const initial = manager.getState();
    setIsNativeDownloading(initial.status === 'downloading');
    setIsNativeLoading(initial.status === 'loading');
    return unsubscribe;
  }, []);

  const anyModelDownloading = isModelDownloading || isNativeDownloading;
  const anyModelLoading = isModelLoading || isNativeLoading;

  const isUsingObServer = externalIsUsingObServer !== undefined
    ? externalIsUsingObServer
    : internalIsUsingObServer;

  const isAuthenticated = authState?.isAuthenticated ?? false;
  const user = authState?.user;

  // Calculate overall server status based on all enabled servers
  const computedServerStatus: 'unchecked' | 'online' | 'offline' = (() => {
    const enabledCustomServersOnline = customServers.some(s => s.enabled && s.status === 'online');
    const obServerOnline = isUsingObServer && isAuthenticated;

    // If ANY enabled server is online, show green
    if (localServerOnline || obServerOnline || enabledCustomServersOnline) {
      return 'online';
    }

    // If all checked servers are offline, show red
    const hasCheckedServers = customServers.some(s => s.status !== 'unchecked');
    if (!localServerOnline && (!isUsingObServer || !isAuthenticated) && (customServers.length === 0 || hasCheckedServers)) {
      return 'offline';
    }

    // Otherwise show unchecked
    return 'unchecked';
  })();

  const handleLogout = () => {
    authState?.logout();
  };

  const handleDeleteAccount = async () => {
    try {
      const token = await getToken();
      await fetch('https://api.observer-ai.com/delete-account', {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
    } catch (error) {
      console.error('Failed to delete account:', error);
    }
    // Logout after deletion (or failed deletion attempt)
    handleLogout();
  };

  const fetchQuotaInfo = async (forceObServer = false) => {
    const usingObServer = forceObServer || isUsingObServer;
    if (!usingObServer || !isAuthenticated) {
      setQuotaInfo(null);
      setIsSessionExpired(false);
      return;
    }

    try {
      setIsLoadingQuota(true);
      const token = await getToken();
      if (!token) throw new Error("Authentication token not available.");

      const headers = { 'Authorization': `Bearer ${token}` };
      const response = await fetch('https://api.observer-ai.com/quota', { headers });

      if (response.ok) {
        const data: QuotaInfo = await response.json();
        setQuotaInfo(data);
        setIsSessionExpired(false);
        if (data && typeof data.remaining === 'number') {
          localStorage.setItem('observer-quota-remaining', data.remaining.toString());

          // Trigger upgrade modal at 50% usage for non-pro users
          if (data.tier !== 'pro' && data.tier !== 'max' && data.tier !== 'plus' && typeof data.limit === 'number' && data.limit > 0) {
            const usagePercentage = ((data.limit - data.remaining) / data.limit) * 100;
            console.log(`Usage: ${usagePercentage.toFixed(1)}%, Remaining: ${data.remaining}/${data.limit}, Warning shown: ${has70PercentWarningBeenShown}`);
            if (usagePercentage >= 50 && !has70PercentWarningBeenShown && onUpgradeClick) {
              console.log('Triggering upgrade modal at 50% usage');
              setHas70PercentWarningBeenShown(true);
              onUpgradeClick();
            }
          }
        } else {
          localStorage.removeItem('observer-quota-remaining');
        }
      } else if (response.status === 401) {
        Logger.warn('AUTH', 'Session expired. Quota check failed with 401.');
        setQuotaInfo(null);
        setIsSessionExpired(true);
        localStorage.removeItem('observer-quota-remaining');
      } else {
        Logger.error('QUOTA', `Failed to fetch quota, status: ${response.status}`);
        setQuotaInfo(null);
        setIsSessionExpired(false);
        localStorage.removeItem('observer-quota-remaining');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      Logger.error('QUOTA', `Error fetching quota info: ${errorMessage}`, err);
      setQuotaInfo(null);
      setIsSessionExpired(false);
    } finally {
      setIsLoadingQuota(false);
    }
  };

  // Check for 50% usage threshold whenever quotaInfo updates
  useEffect(() => {
    if (!quotaInfo || quotaInfo.tier === 'pro' || quotaInfo.tier === 'max' || quotaInfo.tier === 'plus' || has70PercentWarningBeenShown || !onUpgradeClick) {
      return;
    }

    if (typeof quotaInfo.remaining === 'number' && typeof quotaInfo.limit === 'number' && quotaInfo.limit > 0) {
      const usagePercentage = ((quotaInfo.limit - quotaInfo.remaining) / quotaInfo.limit) * 100;
      console.log(`Real-time usage check: ${usagePercentage.toFixed(1)}%, Remaining: ${quotaInfo.remaining}/${quotaInfo.limit}`);

      if (usagePercentage >= 50) {
        console.log('Triggering upgrade modal at 50% usage (real-time)');
        setHas70PercentWarningBeenShown(true);
        onUpgradeClick();
      }
    }
  }, [quotaInfo, has70PercentWarningBeenShown, onUpgradeClick]);

  const handleToggleObServer = () => {
    const newValue = !isUsingObServer;

    if (newValue && !isAuthenticated) {
      Logger.warn('AUTH', 'User attempted to enable ObServer while not authenticated.');
      setShowLoginMessage(true);
      setTimeout(() => setShowLoginMessage(false), 3000);
      return;
    }

    // If switching FROM ObServer TO local on official web app, show warning
    if (!newValue && hostingContext === 'official-web') {
      setIsStartupDialogOpen(true);
      return;
    }

    // Update state and manage inference addresses
    if (newValue) {
      // Add ObServer immediately
      addInferenceAddress(OB_SERVER_ADDRESS);
      // Fetch models to include ObServer models
      fetchModels();
      // Check quota when turning on ObServer
      if (isAuthenticated) {
        fetchQuotaInfo(true); // Force check even though state hasn't updated yet
      }
    } else {
      // Remove ObServer
      removeInferenceAddress(OB_SERVER_ADDRESS);
      // Fetch models to remove ObServer models
      fetchModels();
    }

    if (externalSetIsUsingObServer) {
      externalSetIsUsingObServer(newValue);
    } else {
      setInternalIsUsingObServer(newValue);
    }
  };

  const checkLocalServer = async () => {
    try {
      Logger.info('SERVER', `Checking local server connection at ${LOCAL_SERVER_ADDRESS}...`);
      const result = await checkInferenceServer(LOCAL_SERVER_ADDRESS);

      if (result.status === 'online') {
        setLocalServerOnline(true);
        addInferenceAddress(LOCAL_SERVER_ADDRESS);
        Logger.info('SERVER', `Local server at ${LOCAL_SERVER_ADDRESS} is online and added to inference addresses`);
        // Update model list when server comes online
        await fetchModels();
      } else {
        setLocalServerOnline(false);
        removeInferenceAddress(LOCAL_SERVER_ADDRESS);
        Logger.warn('SERVER', `Local server at ${LOCAL_SERVER_ADDRESS} is offline: ${result.error}`);
        // Update model list when server goes offline
        await fetchModels();
      }
    } catch (err) {
      setLocalServerOnline(false);
      removeInferenceAddress(LOCAL_SERVER_ADDRESS);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      Logger.error('SERVER', `Error checking local server: ${errorMessage}`, err);
    }
  };


  const checkLocalServerOnly = async () => {
    await checkLocalServer();
  };



  useEffect(() => {
    const handleQuotaUpdate = () => {
      const storedQuota = localStorage.getItem('observer-quota-remaining');
      if (storedQuota) {
        setQuotaInfo(prev => prev ? { ...prev, remaining: parseInt(storedQuota, 10) } : null);
      }
    };

    window.addEventListener('quotaUpdated', handleQuotaUpdate);
    return () => {
      window.removeEventListener('quotaUpdated', handleQuotaUpdate);
    };
  }, []);

  // Initialize custom servers and check local server on mount
  useEffect(() => {
    // Load custom servers from localStorage
    const loaded = loadCustomServers();
    setCustomServers(loaded);

    // Check local server
    checkLocalServer();

    // Load inference URL from Tauri backend
    if (isTauri()) {
      import('@tauri-apps/api/core').then(({ invoke }) => {
        invoke<string | null>('get_ollama_url').then(url => {
          Logger.info('SETTINGS', `Loaded inference URL: ${url}`);
          setAppInferenceUrl(url);
        }).catch(err => {
          Logger.error('SETTINGS', `Failed to load inference URL: ${err}`);
        });
      });
    }
  }, []);


  // Clear quota info when switching away from ObServer
  useEffect(() => {
    if (!isUsingObServer) {
      setQuotaInfo(null);
      setIsSessionExpired(false);
    }
  }, [isUsingObServer]);


  // Handle ObServer address and models (only depends on isUsingObServer)
  useEffect(() => {
    if (isUsingObServer) {
      addInferenceAddress(OB_SERVER_ADDRESS);
      fetchModels();
    } else {
      removeInferenceAddress(OB_SERVER_ADDRESS);
      fetchModels();
    }
  }, [isUsingObServer]);

  // Handle quota fetching (depends on both isUsingObServer and isAuthenticated)
  useEffect(() => {
    if (isUsingObServer && isAuthenticated) {
      fetchQuotaInfo(true);
    }
  }, [isUsingObServer, isAuthenticated]);

  // Removed: No longer need to save server address to localStorage

  // Custom server handlers
  const handleAddCustomServer = (address: string) => {
    const updated = addCustomServer(address);
    setCustomServers(updated);
    fetchModels();
  };

  const handleRemoveCustomServer = (address: string) => {
    const updated = removeCustomServer(address);
    setCustomServers(updated);
    fetchModels();
  };

  const handleToggleCustomServer = (address: string) => {
    const updated = toggleCustomServer(address);
    setCustomServers(updated);
    fetchModels();
  };

  const handleCheckCustomServer = async (address: string) => {
    await checkCustomServer(address);
    const updated = getCustomServers();
    setCustomServers(updated);
    fetchModels();
  };

  // Handler for saving inference URL (Tauri only)
  const handleSetAppInferenceUrl = async (url: string) => {
    if (isTauri()) {
      try {
        const { invoke } = await import('@tauri-apps/api/core');
        await invoke('set_ollama_url', { newUrl: url });
        setAppInferenceUrl(url);
        Logger.info('SETTINGS', `Saved inference URL: ${url}`);
        // Re-check local server after URL change
        checkLocalServer();
      } catch (err) {
        Logger.error('SETTINGS', `Failed to save inference URL: ${err}`);
      }
    }
  };

  const renderQuotaStatus = () => {
    if (isSessionExpired) {
      return (
        <button
          type="button"
          onClick={() => authState?.loginWithRedirect()}
          className="text-red-500 font-semibold hover:underline cursor-pointer"
          title="Your session has expired. Click to log in again."
        >
          Session Expired
        </button>
      );
    }

    if (isLoadingQuota) {
      return <span className="text-gray-500">Loading...</span>;
    }

    if (quotaInfo) {
      if (quotaInfo.tier === 'max') {
        return <span className="font-semibold text-green-600">MAX unlimited</span>;
      }
      if (quotaInfo.tier === 'plus') {
        return (
          <div
            className="font-semibold text-blue-600 cursor-help"
            onMouseEnter={() => setIsQuotaHovered(true)}
            onMouseLeave={() => setIsQuotaHovered(false)}
          >
            {isQuotaHovered && typeof quotaInfo.remaining === 'number' && typeof quotaInfo.limit === 'number'
              ? `${quotaInfo.remaining} / ${quotaInfo.limit} Credits left`
              : 'Plus monitoring'
            }
          </div>
        );
      }
      if (quotaInfo.tier === 'pro') {
        return (
          <div
            className="font-semibold text-green-600 cursor-help"
            onMouseEnter={() => setIsQuotaHovered(true)}
            onMouseLeave={() => setIsQuotaHovered(false)}
          >
            {isQuotaHovered && typeof quotaInfo.remaining === 'number' && typeof quotaInfo.limit === 'number'
              ? `${quotaInfo.remaining} / ${quotaInfo.limit} Credits left`
              : 'Pro extended'
            }
          </div>
        );
      }
      if (typeof quotaInfo.remaining === 'number') {
        if (quotaInfo.remaining <= 0) {
          return (
            <span className="font-medium text-red-500">
              No credits left!
            </span>
          );
        }
        
        // Show "Limited Use" that changes to credit count on hover
        return (
          <div
            className={`font-medium cursor-help ${
              quotaInfo.remaining <= 10 ? 'text-orange-500'
              : 'text-green-600'
            }`}
            onMouseEnter={() => setIsQuotaHovered(true)}
            onMouseLeave={() => setIsQuotaHovered(false)}
          >
            {isQuotaHovered 
              ? `${quotaInfo.remaining} / ${quotaInfo.limit} Credits left`
              : 'Limited Use'
            }
          </div>
        );
      }
    }
    return <span className="text-gray-500">Quota N/A</span>;
  };

  return (
    <>
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-2 sm:px-4 py-4">
          <div className="flex justify-between items-center">
            {/* Left side */}
            <div className="flex items-center space-x-2 sm:space-x-4">
              {/* Burger Menu - Mobile Only */}
              <button
                onClick={onToggleMobileMenu}
                className="md:hidden p-2 rounded-md hover:bg-gray-100"
                aria-label="Toggle navigation menu"
              >
                <Menu className="h-5 w-5 text-gray-600" />
              </button>

              <img
                src="/eye-logo-black.svg"
                alt="Observer Logo"
                className="h-8 w-8 cursor-pointer hover:opacity-80"
                onClick={() => setIsPermissionsModalOpen(true)}
                title="Initialize screen capture"
              />
              {/* Updated Logo with conditional "pro" badge */}
              <div className="relative hidden md:block">
              {/* FIX: Wrap the text in an <a> tag instead of putting href on <h1> */}
              <a href="https://observer-ai.com" target="_blank" rel="noopener noreferrer" className="text-xl font-semibold">
                <h1>Observer</h1>
              </a>
              {quotaInfo?.tier === 'max' && (
                <span className="absolute top-0.5 -right-7 text-xs font-semibold text-black">
                  MAX
                </span>
              )}
              {quotaInfo?.tier === 'pro' && (
                <span className="absolute top-0.5 -right-5 text-xs font-semibold text-black">
                  pro
                </span>
              )}
              {quotaInfo?.tier === 'plus' && (
                <span className="absolute top-0.5 -right-7 text-xs font-semibold text-black">
                  plus
                </span>
              )}
            </div>

            </div>

            {/* Right side */}
            <div className="flex items-center space-x-1 sm:space-x-2 md:space-x-4">
              {/* Dark Mode Toggle */}
              {onToggleDarkMode && (
                <button
                  onClick={onToggleDarkMode}
                  className="p-2 rounded-md hover:bg-gray-100"
                  aria-label="Toggle dark mode"
                  title="Toggle dark mode"
                >
                  {isDarkMode ? (
                    <Sun className="h-5 w-5 text-yellow-500" />
                  ) : (
                    <Moon className="h-5 w-5 text-gray-600" />
                  )}
                </button>
              )}

              {/* Server Status and Settings Button (All screen sizes) */}
              <div className="flex items-center space-x-2">
                {/* Status Indicator Dot */}
                {anyModelDownloading ? (
                  <svg className="h-4 w-4 animate-spin text-blue-500" viewBox="0 0 24 24" fill="none" title="Downloading model…">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.25" />
                    <path d="M22 12a10 10 0 00-10-10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                  </svg>
                ) : (
                  <div className={`w-3 h-3 rounded-full
                      ${anyModelLoading ? 'bg-yellow-400 animate-pulse'
                      : computedServerStatus === 'online' ? 'bg-green-500'
                      : computedServerStatus === 'offline' ? 'bg-red-500'
                      : 'bg-orange-500 animate-pulse'}
                  `} title={anyModelLoading ? 'Loading model…' : `Status: ${computedServerStatus}`} />
                )}

                {/* Settings Button */}
                <button
                    onClick={() => setIsSettingsModalOpen(true)}
                    className="p-2 rounded-md hover:bg-gray-100"
                    aria-label="Open connection settings"
                >
                    <Server className="h-5 w-5 text-gray-600" />
                </button>
              </div>

              {/* Auth Section */}
              <div className="flex items-center space-x-1 sm:space-x-2 md:space-x-3">
                {authState ? (
                  authState.isLoading ? (
                    <div className="text-sm px-2 sm:px-3 py-2 bg-gray-100 rounded md:text-base md:px-4">Loading...</div>
                  ) : isAuthenticated ? (
                    <div className="flex items-center space-x-1 sm:space-x-2 md:space-x-3">
                      <span className="text-sm text-gray-700 hidden md:inline">
                        {user?.name || user?.email || 'User'}
                      </span>
                      <button
                        onClick={() => setIsAccountModalOpen(true)}
                        className="w-8 h-8 rounded-full cursor-pointer hover:ring-2 hover:ring-blue-300 flex items-center justify-center overflow-hidden bg-gray-200"
                        aria-label="Account settings"
                      >
                        {user?.picture ? (
                          <img
                            src={user.picture}
                            alt={user?.name || 'User avatar'}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <User className="h-5 w-5 text-gray-600" />
                        )}
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => authState.loginWithRedirect()}
                      className="bg-green-500 text-white rounded hover:bg-green-600
                                 text-sm px-2 py-2 sm:px-3 md:text-base md:px-4"
                    >
                      <span className="md:hidden">Log In</span>
                      <span className="hidden md:inline">Log In | Sign Up</span>
                    </button>
                  )
                ) : (
                  <div className="bg-yellow-100 text-yellow-800 rounded text-xs sm:text-sm px-2 py-2 sm:px-3">
                    <span className="md:hidden">Auth...</span>
                    <span className="hidden md:inline">Auth not initialized</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </header>

      <SharingPermissionsModal
        isOpen={isPermissionsModalOpen}
        onClose={() => setIsPermissionsModalOpen(false)}
      />

      {/* Model Hub - central modal for all model/server management */}
      <ModelHub
        isOpen={isSettingsModalOpen}
        onClose={() => { setIsSettingsModalOpen(false); setAutoDownloadPreset(undefined); }}
        autoDownloadPreset={autoDownloadPreset}
        isUsingObServer={isUsingObServer}
        handleToggleObServer={handleToggleObServer}
        showLoginMessage={showLoginMessage}
        isAuthenticated={isAuthenticated}
        quotaInfo={quotaInfo}
        renderQuotaStatus={renderQuotaStatus}
        localServerOnline={localServerOnline}
        checkLocalServer={checkLocalServerOnly}
        customServers={customServers}
        onAddCustomServer={handleAddCustomServer}
        onRemoveCustomServer={handleRemoveCustomServer}
        onToggleCustomServer={handleToggleCustomServer}
        onCheckCustomServer={handleCheckCustomServer}
        appInferenceUrl={appInferenceUrl}
        onSetAppInferenceUrl={handleSetAppInferenceUrl}
      />

      {isStartupDialogOpen && (
        <StartupDialogs
          onDismiss={() => setIsStartupDialogOpen(false)}
          onLogin={() => authState?.loginWithRedirect()}
          onToggleObServer={handleToggleObServer}
          isAuthenticated={isAuthenticated}
          hostingContext={hostingContext}
        />
      )}

      <AccountModal
        isOpen={isAccountModalOpen}
        onClose={() => setIsAccountModalOpen(false)}
        user={user}
        onLogout={handleLogout}
        onDeleteAccount={handleDeleteAccount}
      />
    </>
  );
};

export default AppHeader;
