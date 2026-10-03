import { datadogRum } from '@datadog/browser-rum';
import { reactPlugin } from '@datadog/browser-rum-react';
import { Analytics } from '@utils/analytics';
import { ModelManager } from '@utils/ModelManager';
import { useState, useEffect, useCallback, useMemo } from 'react';
import { Auth0Provider } from '@auth0/auth0-react';
import { platform as getPlatform } from '@tauri-apps/plugin-os';
import { BrowserRouter, Routes, Route, useSearchParams } from 'react-router-dom';
import { AuthProvider, useAuth } from '@contexts/AuthContext';
import { useIOSKeyboard } from '@hooks/useIOSKeyboard';
import { useAgentGridLayout, GRID_ROW_HEIGHT, GRID_MARGIN } from '@hooks/useAgentGridLayout';
import GridLayout from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import { isMobile, confirm, isDesktop, isTauri, getPlatformName } from '@utils/platform';
import { version as appVersion } from '../../package.json';
import type { QuotaInfo } from '@/types/quota';
import { fetchQuota, remaining as remainingOf } from '@/types/quota';
import {
  addInferenceAddress,
  removeInferenceAddress,
  fetchModels,
  loadCustomServers,
  getCustomServers,
  addCustomServer,
  removeCustomServer,
  toggleCustomServer,
  checkCustomServer,
  checkInferenceServer,
  type CustomServer,
} from '@utils/inferenceServer';
import {
  listAgents,
  getAgentCode,
  deleteAgent,
  saveAgent,
  CompleteAgent,
} from '@utils/agent_database';
import { startAgentLoop, stopAgentLoop } from '@utils/main_loop';
import { Logger, type WhitelistChannel } from '@utils/logging';
import { TranscriptionRouter } from '@utils/whisper/TranscriptionRouter';
import { MEMORY_UPDATE_EVENT } from '@components/MemoryManager';
import { IterationStore } from '@utils/IterationStore';

// Imported Components
import AppHeader from '@components/AppHeader';
import AgentCard from '@components/AgentCard/AgentCard';
import EditAgentModal from '@components/EditAgent/EditAgentModal';
import StartupDialogs from '@components/StartupDialogs';
import GlobalLogsViewer from '@components/GlobalLogsViewer';
import MemoryManager from '@components/MemoryManager';
import ErrorDisplay from '@components/ErrorDisplay';
import AgentImportHandler from '@components/AgentImportHandler';
import PersistentSidebar from '@components/PersistentSidebar';
import AvailableModels from '@components/AvailableModels';
import CommunityTab from '@components/CommunityTab';
import GetStarted from '@components/GetStarted';
import ObserverTab from '@components/Observer/ObserverTab';
import RecipeSplash from '@components/AICreator/RecipeSplash';
import { useTutorialFlow, tutorialFlow } from '@utils/tutorialFlow';
import JupyterServerModal from '@components/JupyterServerModal';
import { generateAgentFromSimpleConfig } from '@utils/agentTemplateManager';
import SimpleCreatorModal from '@components/EditAgent/SimpleCreatorModal';
import RecordingsViewer from '@components/RecordingsViewer';
import SettingsTab from '@components/SettingsTab';
import MemoryStoreTab from '@components/MemoryStoreTab';
import { UpgradeSuccessPage } from '../pages/UpgradeSuccessPage';
import { JoinOrgPage } from '../pages/JoinOrgPage';
import { TeamPage } from '../pages/TeamPage';
import AgentShareLandingPage from '@components/AgentShareLandingPage';
import { UpgradeModal } from '@components/UpgradeModal';
import { AcceptToS } from '@components/AcceptToS';
import { AttributionSplash } from '@components/AttributionSplash';
import { WelcomeModal } from '@components/WelcomeModal';
import AgentActivityModal from '@components/AgentCard/AgentActivityModal';
import FeedbackDialog from '@components/FeedbackDialog';
import { startCommandSSE, updateCommandSSEToken } from '@utils/commandSSE';
import WhitelistModal from '@components/WhitelistModal';
import LocalOnboardingTutorial from '@components/LocalOnboardingTutorial';
import AgentChip from '@components/AgentChip';
import LiveStream from '@components/LiveStream';
import { MCPProvider } from '../mcp/MCPContext';

datadogRum.init({
  applicationId: 'ed504b99-0755-4aff-b155-06eeb559c705',
  clientToken: 'pub2abb69c9ad9708fa859220211d5b26e5',
  site: 'us5.datadoghq.com',
  service: 'observer-web',
  env: import.meta.env.MODE,
  version: appVersion,
  sessionSampleRate: 100,
  sessionReplaySampleRate: 100,
  defaultPrivacyLevel: 'mask',
  trackResources: true,
  trackUserInteractions: true,
  trackLongTasks: true,
  plugins: [reactPlugin({ router: false })],
});

datadogRum.setGlobalContextProperty('platform', getPlatformName());

// Main app content - uses the unified auth hook
function AppContent() {
  const { isAuthenticated, isLoading, user, login, logout, getAccessToken } = useAuth();

  // Handle iOS keyboard - updates CSS variables when keyboard shows/hides
  useIOSKeyboard();

  const [agents, setAgents] = useState<CompleteAgent[]>([]);
  const [agentCodes, setAgentCodes] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [startingAgents, setStartingAgents] = useState<Set<string>>(new Set());
  const [runningAgents, setRunningAgents] = useState<Set<string>>(new Set());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isCreateMode, setIsCreateMode] = useState(false);
  const [showStartupDialog, setShowStartupDialog] = useState(false);
  const [showGlobalLogs, setShowGlobalLogs] = useState(false);
  const [isMemoryManagerOpen, setIsMemoryManagerOpen] = useState(false);
  const [memoryAgentId, setMemoryAgentId] = useState<string | null>(null);
  const [flashingMemories, setFlashingMemories] = useState<Set<string>>(new Set());
  const [activeTab, setActiveTab] = useState('observerChat');
  const [isUsingObServer, setIsUsingObServer] = useState(false);
  const [isJupyterModalOpen, setIsJupyterModalOpen] = useState(false);
  const [isSimpleCreatorOpen, setIsSimpleCreatorOpen] = useState(false);
  const [stagedAgentConfig, setStagedAgentConfig] = useState<{ agent: CompleteAgent, code: string } | null>(null);
  const [hasPendingImport, setHasPendingImport] = useState(false);
  const [searchParams, setSearchParams] = useSearchParams();
  const [isPermissionsModalOpen, setIsPermissionsModalOpen] = useState(false);
  const [isAccountModalOpen, setIsAccountModalOpen] = useState(false);
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(true);

  // Quota error state
  const [agentsWithQuotaError, setAgentsWithQuotaError] = useState<Set<string>>(new Set());
  const [isUpgradeModalOpen, setIsUpgradeModalOpen] = useState(false);
  const [isHalfwayWarning, setIsHalfwayWarning] = useState(false);
  const [currentQuotaType, setCurrentQuotaType] = useState<string>('monitor');

  // Activity modal state
  const [activityModalOpen, setActivityModalOpen] = useState(false);
  const [activityModalAgentId, setActivityModalAgentId] = useState<string | null>(null);

  // Quota info state
  const [quotaInfo, setQuotaInfo] = useState<QuotaInfo | null>(null);

  // Mobile UI state
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  // Feedback dialog state
  const [isFeedbackOpen, setIsFeedbackOpen] = useState(false);

  // Whitelist modal state
  const [whitelistModalInfo, setWhitelistModalInfo] = useState<{
    phoneNumbers: Array<{ number: string; isWhitelisted: boolean; channel?: WhitelistChannel }>;
    agentId?: string;
    onStartAgent?: () => void;
    channel?: WhitelistChannel;
  } | null>(null);

  // AcceptToS modal state
  const [isAcceptToSOpen, setIsAcceptToSOpen] = useState(false);
  const [isAttributionOpen, setIsAttributionOpen] = useState(false);
  const [isWelcomeUpsellOpen, setIsWelcomeUpsellOpen] = useState(false);
  const [welcomeUpsellVariant, setWelcomeUpsellVariant] = useState<'onboarding' | 'activation'>('onboarding');
  const [isRecipeSplashOpen, setIsRecipeSplashOpen] = useState(false);
  const [showLocalModeWarning, setShowLocalModeWarning] = useState(false);
  const [isLocalOnboardingActive, setIsLocalOnboardingActive] = useState(false);


  // Minimized agents — persisted to localStorage
  const [minimizedAgents, setMinimizedAgents] = useState<Set<string>>(() => {
    const saved = localStorage.getItem('observer_minimized_agents');
    return saved ? new Set(JSON.parse(saved)) : new Set();
  });

  useEffect(() => {
    localStorage.setItem('observer_minimized_agents', JSON.stringify([...minimizedAgents]));
  }, [minimizedAgents]);

  const handleMinimize = (agentId: string) =>
    setMinimizedAgents(prev => new Set([...prev, agentId]));

  const handleRestore = (agentId: string) => {
    setMinimizedAgents(prev => { const s = new Set(prev); s.delete(agentId); return s; });
    // Its old grid spot may now collide with a card that moved in while it
    // was minimized — drop it in at the end instead of letting the grid's
    // compactor shove it far down the page trying to resolve that.
    restoreGridAgentToEnd(agentId);
  };

  // Dark mode state
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('observer-dark-mode');
    return saved === 'true';
  });

  const isProUser = quotaInfo?.tier === 'pro' || quotaInfo?.tier === 'max';
  const { phase: tutorialPhase } = useTutorialFlow();

  const fetchAgents = useCallback(async () => {
    try {
      setIsRefreshing(true);
      Logger.debug('APP', 'Fetching agents from database');
      const agentsData = await listAgents();
      setAgents(agentsData);
      Logger.debug('APP', `Found ${agentsData.length} agents`);

      const codeResults = await Promise.all(
        agentsData.map(async (a) => ({ id: a.id, code: await getAgentCode(a.id) }))
      );
      const newCodes: Record<string, string> = {};
      codeResults.forEach((r) => {
        if (r.code) newCodes[r.id] = r.code;
      });
      setAgentCodes(newCodes);

      setError(null);
    } catch (err) {
      setError('Failed to fetch agents from database');
      Logger.error('APP', `Error fetching agents:`, err);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  const getToken = useCallback(async () => {
    if (isLoading) {
      Logger.warn('AUTH', 'getToken called while auth state is loading. Aborting.');
      return undefined;
    }

    if (!isAuthenticated) {
      Logger.warn('AUTH', 'getToken called, but user is not authenticated.');
      try {
        const token = await getAccessToken();
        if (token) Logger.info('AUTH', `getToken succeeded even though isAuthenticated is false.`);
        return token;
      } catch {
        Logger.warn('AUTH', `errored out trying getToken when not authenticated.`);
      }
      return undefined;
    }

    try {
      const token = await getAccessToken();
      return token;
    } catch (error) {
      Logger.error('AUTH', 'Failed to retrieve access token silently.', error);
      throw error;
    }
  }, [isAuthenticated, isLoading, getAccessToken]);

  // Set up token provider for cloud transcription
  useEffect(() => {
    TranscriptionRouter.setTokenProvider(getToken);
  }, [getToken]);

  const hostingContext = useMemo(() => {
    const { protocol, hostname } = window.location;

    if (protocol === 'https:' && hostname !== 'localhost' && hostname !== '127.0.0.1') {
      return 'official-web';
    }

    return 'self-hosted';
  }, []);

  // ── Model connectivity: custom servers, local server, Ob-Server quota/toggle ──
  // Owned here (not in AppHeader) so the header's status dot and the Models tab
  // share one source of truth instead of each keeping a partial copy.
  const [customServers, setCustomServers] = useState<CustomServer[]>([]);
  const [localServerOnline, setLocalServerOnline] = useState(false);
  const [appInferenceUrl, setAppInferenceUrl] = useState<string | null>(null);
  const [isLoadingQuota, setIsLoadingQuota] = useState(false);
  const [showLoginMessage, setShowLoginMessage] = useState(false);
  const [isSessionExpired, setIsSessionExpired] = useState(false);
  const [isQuotaHovered, setIsQuotaHovered] = useState(false);
  const [has70PercentWarningBeenShown, setHas70PercentWarningBeenShown] = useState(false);
  const [isObServerWarningOpen, setIsObServerWarningOpen] = useState(false);

  const fetchQuotaInfo = useCallback(async (forceObServer = false) => {
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

      const data = await fetchQuota(token);
      setQuotaInfo(data);
      setIsSessionExpired(false);
      if (data && data.daily) {
        localStorage.setItem('observer-quota-remaining', remainingOf(data.daily).toString());

        // Trigger upgrade modal at 50% usage for non-pro users
        if (data.tier !== 'pro' && data.tier !== 'max' && data.tier !== 'plus' && data.daily.limit > 0) {
          const usagePercentage = (data.daily.used / data.daily.limit) * 100;
          if (usagePercentage >= 50 && !has70PercentWarningBeenShown) {
            setHas70PercentWarningBeenShown(true);
            setIsHalfwayWarning(true);
            setIsUpgradeModalOpen(true);
          }
        }
      } else {
        localStorage.removeItem('observer-quota-remaining');
      }
    } catch (err) {
      if (err instanceof Error && err.message === 'unauthorized') {
        Logger.warn('AUTH', 'Session expired. Quota check failed with 401.');
        setQuotaInfo(null);
        setIsSessionExpired(true);
        localStorage.removeItem('observer-quota-remaining');
      } else {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        Logger.error('QUOTA', `Error fetching quota info: ${errorMessage}`, err);
        setQuotaInfo(null);
        setIsSessionExpired(false);
        localStorage.removeItem('observer-quota-remaining');
      }
    } finally {
      setIsLoadingQuota(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isUsingObServer, isAuthenticated, getToken, has70PercentWarningBeenShown]);

  const handleToggleObServer = useCallback(() => {
    const newValue = !isUsingObServer;

    if (newValue && !isAuthenticated) {
      Logger.warn('AUTH', 'User attempted to enable ObServer while not authenticated.');
      setShowLoginMessage(true);
      setTimeout(() => setShowLoginMessage(false), 3000);
      return;
    }

    // If switching FROM ObServer TO local on official web app, warn first
    if (!newValue && hostingContext === 'official-web') {
      setIsObServerWarningOpen(true);
      return;
    }

    if (newValue) {
      addInferenceAddress('https://api.observer-ai.com:443');
      fetchModels();
      if (isAuthenticated) fetchQuotaInfo(true);
    } else {
      removeInferenceAddress('https://api.observer-ai.com:443');
      fetchModels();
    }

    setIsUsingObServer(newValue);
  }, [isUsingObServer, isAuthenticated, hostingContext, fetchQuotaInfo]);

  const checkLocalServer = useCallback(async () => {
    const LOCAL_SERVER_ADDRESS = 'http://localhost:3838';
    try {
      Logger.info('SERVER', `Checking local server connection at ${LOCAL_SERVER_ADDRESS}...`);
      const result = await checkInferenceServer(LOCAL_SERVER_ADDRESS);

      if (result.status === 'online') {
        setLocalServerOnline(true);
        addInferenceAddress(LOCAL_SERVER_ADDRESS);
        await fetchModels();
      } else {
        setLocalServerOnline(false);
        removeInferenceAddress(LOCAL_SERVER_ADDRESS);
        await fetchModels();
      }
    } catch (err) {
      setLocalServerOnline(false);
      removeInferenceAddress(LOCAL_SERVER_ADDRESS);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      Logger.error('SERVER', `Error checking local server: ${errorMessage}`, err);
    }
  }, []);

  const handleAddCustomServer = (address: string) => { setCustomServers(addCustomServer(address)); fetchModels(); };
  const handleRemoveCustomServer = (address: string) => { setCustomServers(removeCustomServer(address)); fetchModels(); };
  const handleToggleCustomServer = (address: string) => { setCustomServers(toggleCustomServer(address)); fetchModels(); };
  const handleCheckCustomServer = async (address: string) => {
    await checkCustomServer(address);
    setCustomServers(getCustomServers());
    fetchModels();
  };

  const handleSetAppInferenceUrl = async (url: string) => {
    if (!isTauri()) return;
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      await invoke('set_ollama_url', { newUrl: url });
      setAppInferenceUrl(url);
      Logger.info('SETTINGS', `Saved inference URL: ${url}`);
      checkLocalServer();
    } catch (err) {
      Logger.error('SETTINGS', `Failed to save inference URL: ${err}`);
    }
  };

  const renderQuotaStatus = () => {
    if (isSessionExpired) {
      return (
        <button
          type="button"
          onClick={() => login()}
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
      const dailyRemaining = quotaInfo.daily ? remainingOf(quotaInfo.daily) : undefined;
      const monthlyRemaining = quotaInfo.monthly ? remainingOf(quotaInfo.monthly) : undefined;
      const hoverDetail = dailyRemaining !== undefined && monthlyRemaining !== undefined
        ? `${dailyRemaining} / ${quotaInfo.daily.limit} today · ${monthlyRemaining} / ${quotaInfo.monthly.limit} this month`
        : undefined;

      if (quotaInfo.tier === 'plus') {
        return (
          <div className="font-semibold text-blue-600 cursor-help" onMouseEnter={() => setIsQuotaHovered(true)} onMouseLeave={() => setIsQuotaHovered(false)}>
            {isQuotaHovered && hoverDetail ? hoverDetail : 'Plus monitoring'}
          </div>
        );
      }
      if (quotaInfo.tier === 'pro') {
        return (
          <div className="font-semibold text-green-600 cursor-help" onMouseEnter={() => setIsQuotaHovered(true)} onMouseLeave={() => setIsQuotaHovered(false)}>
            {isQuotaHovered && hoverDetail ? hoverDetail : 'Pro extended'}
          </div>
        );
      }
      if (dailyRemaining !== undefined) {
        if (dailyRemaining <= 0) {
          return <span className="font-medium text-red-500">No credits left!</span>;
        }
        return (
          <div
            className={`font-medium cursor-help ${dailyRemaining <= 10 ? 'text-orange-500' : 'text-green-600'}`}
            onMouseEnter={() => setIsQuotaHovered(true)}
            onMouseLeave={() => setIsQuotaHovered(false)}
          >
            {isQuotaHovered && hoverDetail ? hoverDetail : 'Limited Use'}
          </div>
        );
      }
    }
    return <span className="text-gray-500">Quota N/A</span>;
  };

  // Optimistic quota decrement — a request elsewhere in the app just consumed a credit.
  useEffect(() => {
    const handleQuotaUpdate = () => {
      const storedRemaining = localStorage.getItem('observer-quota-remaining');
      if (storedRemaining) {
        const newRemaining = parseInt(storedRemaining, 10);
        setQuotaInfo(prev => (prev && prev.daily)
          ? { ...prev, daily: { ...prev.daily, used: Math.max(0, prev.daily.limit - newRemaining) } }
          : prev);
      }
    };
    window.addEventListener('quotaUpdated', handleQuotaUpdate);
    return () => window.removeEventListener('quotaUpdated', handleQuotaUpdate);
  }, []);

  // Load custom servers + (Tauri) the saved inference URL once on mount
  useEffect(() => {
    setCustomServers(loadCustomServers());
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

  useEffect(() => {
    if (!isUsingObServer) {
      setQuotaInfo(null);
      setIsSessionExpired(false);
    }
  }, [isUsingObServer]);

  useEffect(() => {
    if (isUsingObServer) {
      addInferenceAddress('https://api.observer-ai.com:443');
      fetchModels();
    } else {
      removeInferenceAddress('https://api.observer-ai.com:443');
      fetchModels();
    }
  }, [isUsingObServer]);

  useEffect(() => {
    if (isUsingObServer && isAuthenticated) {
      fetchQuotaInfo(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isUsingObServer, isAuthenticated]);

  // Check if we're on mobile (for SSE and other platform-specific behavior)
  const [isMobileDevice, setIsMobileDevice] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => {
      try {
        setIsMobileDevice(isMobile());
      } catch {
        setIsMobileDevice(false);
      }
    }, 50);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const handleAgentStatusChange = (event: CustomEvent) => {
      const { agentId, status } = event.detail || {};
      Logger.info('APP', `agentStatusChanged:`, { agentId, status });
      setRunningAgents(prev => {
        const updated = new Set(prev);
        if (status === 'running') {
          updated.add(agentId);
        } else {
          updated.delete(agentId);
        }
        return updated;
      });

    };

    window.addEventListener('agentStatusChanged', handleAgentStatusChange as EventListener);
    return () => {
      window.removeEventListener('agentStatusChanged', handleAgentStatusChange as EventListener);
    };
  }, []);

  useEffect(() => {
    const handleQuotaExceeded = (event: CustomEvent<{ agentId: string; quotaType: string }>) => {
      const { agentId, quotaType } = event.detail;
      setCurrentQuotaType(quotaType);
      setAgentsWithQuotaError(prevSet => {
        const newSet = new Set(prevSet);
        newSet.add(agentId);
        return newSet;
      });

      setIsHalfwayWarning(false);
      setIsUpgradeModalOpen(true);
    };

    window.addEventListener('quotaExceeded', handleQuotaExceeded as EventListener);
    return () => {
      window.removeEventListener('quotaExceeded', handleQuotaExceeded as EventListener);
    };
  }, []);

  useEffect(() => {
    const handleAgentRuntimeError = (event: CustomEvent<{ agentId: string; error: string }>) => {
      const { error } = event.detail;
      setError(error);
    };

    window.addEventListener('agentRuntimeError', handleAgentRuntimeError as EventListener);
    return () => {
      window.removeEventListener('agentRuntimeError', handleAgentRuntimeError as EventListener);
    };
  }, []);

  useEffect(() => {
    const handleWhitelistRequired = (event: CustomEvent<{
      phoneNumber: string;
      toolName: string;
      channel: WhitelistChannel;
    }>) => {
      const { phoneNumber, channel } = event.detail;
      setWhitelistModalInfo({
        phoneNumbers: [{ number: phoneNumber, isWhitelisted: false }],
        channel,
      });
      Logger.info('APP', `Whitelist required: ${phoneNumber} (${channel})`);
    };

    window.addEventListener('whitelistRequired', handleWhitelistRequired as EventListener);
    return () => {
      window.removeEventListener('whitelistRequired', handleWhitelistRequired as EventListener);
    };
  }, []);

  const handleEditClick = async (agentId: string) => {
    setSelectedAgent(agentId);
    setIsCreateMode(false);
    setIsEditModalOpen(true);
    Logger.info('APP', `Opening editor for agent ${agentId}`);
  };


  const handleAddAgentClick = () => {
    setSelectedAgent(null);
    setIsCreateMode(true);
    setStagedAgentConfig(null);
    setIsSimpleCreatorOpen(true);
    Logger.info('APP', 'Opening Simple Creator to create new agent');
  };

  const handleSimpleCreatorNext = (config: Parameters<typeof generateAgentFromSimpleConfig>[0]) => {
    Logger.info('APP', `Generating agent from Simple Creator`, config);
    const { agent, code } = generateAgentFromSimpleConfig(config);

    setStagedAgentConfig({ agent, code });
    setIsSimpleCreatorOpen(false);
    setIsEditModalOpen(true);
  };

  const handleMemoryClick = (agentId: string) => {
    if (flashingMemories.has(agentId)) {
      const newFlashing = new Set(flashingMemories);
      newFlashing.delete(agentId);
      setFlashingMemories(newFlashing);
    }

    setMemoryAgentId(agentId);
    setIsMemoryManagerOpen(true);
    Logger.info('APP', `Opening memory manager for agent ${agentId}`);
  };

  const handleActivityClick = (agentId: string) => {
    setActivityModalAgentId(agentId);
    setActivityModalOpen(true);
    Logger.info('APP', `Opening activity modal for agent ${agentId}`);
  };

  const markOnboardingComplete = () => {
    if (user && 'sub' in user && user.sub) {
      localStorage.setItem(`observer_onboarding_complete_${user.sub}`, 'true');
    }
  };

  const markTutorialSeen = () => {
    if (user && 'sub' in user && user.sub) {
      localStorage.setItem(`observer_tutorial_seen_${user.sub}`, 'true');
    }
  };

  const handleDeleteClick = async (agentId: string) => {
    const agent = agents.find(a => a.id === agentId);
    if (!agent) return;

    if (!await confirm(`Are you sure you want to delete agent "${agent.name}"?`)) {
      return;
    }

    try {
      setError(null);
      Logger.info('APP', `Deleting agent "${agent.name}" (${agentId})`);

      if (runningAgents.has(agentId)) {
        Logger.info(agentId, `Stopping agent before deletion`);
        stopAgentLoop(agentId);
      }

      await IterationStore.clearAllHistory(agentId);
      Logger.info('APP', `Cleared iteration history for agent "${agent.name}"`);

      await deleteAgent(agentId);
      Logger.info('APP', `Agent "${agent.name}" deleted successfully`);
      setMinimizedAgents(prev => { const s = new Set(prev); s.delete(agentId); return s; });
      await fetchAgents();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      Logger.error('APP', `Failed to delete agent: ${errorMessage}`, err);
    }
  };

  const handleDismissStartupDialog = () => {
    setShowStartupDialog(false);
  };

  const toggleAgent = async (id: string, isCurrentlyRunning: boolean): Promise<void> => {
    if (isUsingObServer && !isAuthenticated) {
      Logger.info('AUTH', 'User attempted to use a protected feature while logged out. Redirecting to login.');
      login();
      return;
    }

    try {
      setError(null);
      const agent = agents.find(a => a.id === id);

      if (!agent) {
        throw new Error(`Agent ${id} not found`);
      }
      const isStartingUp = startingAgents.has(id);

      if (isStartingUp || isCurrentlyRunning) {
        Logger.info(id, `Stopping agent "${agent.name}"`);
        stopAgentLoop(id);
        if (isStartingUp) {
          setStartingAgents(prev => {
            const updated = new Set(prev);
            updated.delete(id);
            return updated;
          });
        }
      } else {
        Logger.info(id, `Starting agent "${agent.name}"`);
        setStartingAgents(prev => {
          const updated = new Set(prev);
          updated.add(id);
          return updated;
        });

        try {
          await startAgentLoop(id, getToken);
        } catch (err: any) {
          if (err.whitelistCheck) {
            const { phoneNumbers, channel } = err.whitelistCheck;

            setWhitelistModalInfo({
              phoneNumbers,
              agentId: id,
              channel,
              onStartAgent: () => {
                setWhitelistModalInfo(null);
                toggleAgent(id, false);
              }
            });
            setStartingAgents(prev => {
              const updated = new Set(prev);
              updated.delete(id);
              return updated;
            });
            return;
          }
          throw err;
        } finally {
          setStartingAgents(prev => {
            const updated = new Set(prev);
            updated.delete(id);
            return updated;
          });
        }
      }

      await fetchAgents();
    } catch (err) {
      setStartingAgents(prev => {
        const updated = new Set(prev);
        updated.delete(id);
        return updated;
      });

      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      Logger.error('APP', `Failed to toggle agent status: ${errorMessage}`, err);
    }
  };

  const handleSaveAgent = async (agent: CompleteAgent, code: string) => {
    try {
      setError(null);
      const isNew = !agents.some(a => a.id === agent.id);

      Logger.info('APP', isNew ? `Creating new agent "${agent.name}"` : `Updating agent "${agent.name}" (${agent.id})`);

      await saveAgent(agent, code);
      Logger.info('APP', `Agent "${agent.name}" saved successfully`);
      await fetchAgents();

      // Onboarding is now triggered on first login, not on agent save
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      Logger.error('APP', `Failed to save agent: ${errorMessage}`, err);
    }
  };

  // (Onboarding is triggered from auth useEffect below)

  useEffect(() => {
    const handleMemoryUpdate = (event: CustomEvent) => {
      const updatedAgentId = event.detail.agentId;

      if (updatedAgentId !== memoryAgentId || !isMemoryManagerOpen) {
        setFlashingMemories(prev => {
          const newSet = new Set(prev);
          newSet.add(updatedAgentId);
          return newSet;
        });

        Logger.debug('APP', `Memory updated for agent ${updatedAgentId}, setting flash indicator`);
      }
    };

    window.addEventListener(MEMORY_UPDATE_EVENT, handleMemoryUpdate as EventListener);
    return () => {
      window.removeEventListener(MEMORY_UPDATE_EVENT, handleMemoryUpdate as EventListener);
    };
  }, [memoryAgentId, isMemoryManagerOpen]);

  useEffect(() => {
    Logger.info('APP', 'Application starting');
    fetchAgents();

    if (isAuthenticated) {
      Logger.info('AUTH', `User authenticated: ${user?.name || user?.email || 'Unknown user'}`);
    } else if (!isLoading) {
      Logger.info('AUTH', 'User not authenticated');
    }

    const handleWindowError = (event: ErrorEvent) => {
      Logger.error('APP', `Uncaught error: ${event.message}`, {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        error: event.error
      });
    };

    window.addEventListener('error', handleWindowError);
    return () => {
      window.removeEventListener('error', handleWindowError);
    };
  }, [isAuthenticated, isLoading, user]);

  // Handle ?importAgent=id from marketplace share URL
  useEffect(() => {
    const importAgentId = searchParams.get('importAgent');
    if (!importAgentId) return;

    setSearchParams(prev => { prev.delete('importAgent'); return prev; }, { replace: true });

    (async () => {
      try {
        const response = await fetch(`https://api.observer-ai.com/agents/${importAgentId}`);
        if (!response.ok) return;
        const agent = await response.json();
        setStagedAgentConfig({
          agent: {
            id: agent.id,
            name: agent.name,
            description: agent.description,
            model_name: agent.model_name,
            system_prompt: agent.system_prompt,
            loop_interval_seconds: agent.loop_interval_seconds,
          },
          code: agent.code,
        });
        setIsCreateMode(true);
        setIsEditModalOpen(true);
        setHasPendingImport(true);
      } catch (err) {
        Logger.error('APP', `Failed to fetch marketplace agent for import: ${importAgentId}`, err);
      }
    })();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Deep link listener for agent sharing (Tauri only)
  useEffect(() => {
    if (!isDesktop() && !isMobile()) return;

    let cleanup: (() => void) | undefined;

    (async () => {
      try {
        const { onOpenUrl } = await import('@tauri-apps/plugin-deep-link');
        const unlisten = await onOpenUrl(async (urls) => {
          for (const rawUrl of urls) {
            let agentId: string | null = null;
            try {
              const url = new URL(rawUrl);
              if (url.protocol === 'observer:') {
                // observer://marketplace/agent-id
                agentId = url.pathname.replace(/^\//, '');
              } else if (url.pathname.startsWith('/marketplace/')) {
                // https://app.observer-ai.com/marketplace/agent-id
                agentId = url.pathname.replace('/marketplace/', '');
              }
            } catch { continue; }

            if (!agentId) continue;

            try {
              const response = await fetch(`https://api.observer-ai.com/agents/${agentId}`);
              if (!response.ok) continue;
              const agent = await response.json();
              setStagedAgentConfig({
                agent: {
                  id: agent.id,
                  name: agent.name,
                  description: agent.description,
                  model_name: agent.model_name,
                  system_prompt: agent.system_prompt,
                  loop_interval_seconds: agent.loop_interval_seconds,
                },
                code: agent.code,
              });
              setIsCreateMode(true);
              setIsEditModalOpen(true);
            } catch (err) {
              Logger.error('DEEPLINK', `Failed to fetch agent from deep link: ${rawUrl}`, err);
            }
          }
        });
        cleanup = () => unlisten();
      } catch {
        // Plugin not available in this context
      }
    })();

    return () => cleanup?.();
  }, [fetchAgents]);

  // Start command SSE for hotkey support (desktop only)
  useEffect(() => {
    if (isDesktop()) {
      startCommandSSE(getToken);
    }
  }, [hostingContext, isMobileDevice]);

  // Update token when it changes
  useEffect(() => {
    if (hostingContext === 'self-hosted' && !isMobileDevice) {
      updateCommandSSEToken(getToken);
    }
  }, [getToken, hostingContext, isMobileDevice]);

  useEffect(() => {
    if (!isLoading) {
      Logger.info('AUTH', `Auth loading complete, authenticated: ${isAuthenticated}`);
      if (isAuthenticated && !isUsingObServer) {
        Logger.info('AUTH', 'Auto-enabling ObServer for authenticated user');
        setIsUsingObServer(true);
      }
      if (!isAuthenticated) {
        const localOnboardingComplete = localStorage.getItem('observer_onboarding_complete_local');
        if (!localOnboardingComplete && !ModelManager.getInstance().hasUserServer()) {
          setShowStartupDialog(true);
          Analytics.startupShown();
        }
      }
    }
  }, [isLoading, isAuthenticated]);


  // Onboarding / welcome logic after auth resolves
  useEffect(() => {
    if (!isLoading && isAuthenticated && user && 'sub' in user && user.sub) {
      const sub = user.sub as string;
      const onboardingComplete = localStorage.getItem(`observer_onboarding_complete_${sub}`);

      if (!onboardingComplete) {
        // New user: show Privacy/ToS first, then tutorial on accept
        Logger.info('ONBOARDING', 'First-time user detected, showing Privacy/ToS then tutorial');
        setIsAcceptToSOpen(true);
        return;
      }

      // Returning user: clear any leftover login intent
      sessionStorage.removeItem('observer_login_intent');
    }
  }, [isLoading, isAuthenticated, user]);


  // Reload agents when switching to My Agents or the Observer tab
  useEffect(() => {
    if (activeTab === 'myAgents' || activeTab === 'observerChat') {
      fetchAgents();
    }
  }, [activeTab, fetchAgents]);

  // Dark mode persistence
  useEffect(() => {
    localStorage.setItem('observer-dark-mode', isDarkMode.toString());
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  const toggleDarkMode = useCallback(() => {
    setIsDarkMode(prev => !prev);
  }, []);

  // Sort agents - active ones first
  const sortedAgents = useMemo(() => {
    return [...agents].sort((a, b) => {
      const isALive = runningAgents.has(a.id) || startingAgents.has(a.id);
      const isBLive = runningAgents.has(b.id) || startingAgents.has(b.id);

      if (isALive && !isBLive) return -1;
      if (!isALive && isBLive) return 1;
      return 0;
    });
  }, [agents, runningAgents, startingAgents]);

  // Tiling grid layout for the (non-minimized) agent cards — drag to move,
  // drag the corner to resize, position/size persisted per agent.
  const visibleAgentIds = useMemo(
    () => sortedAgents.filter(a => !minimizedAgents.has(a.id)).map(a => a.id),
    [sortedAgents, minimizedAgents]
  );
  const {
    layout: gridLayout,
    onDragStop: onGridDragStop,
    onResizeStop: onGridResizeStop,
    restoreToEnd: restoreGridAgentToEnd,
    containerRef: gridContainerRef,
    width: gridWidth,
    cols: gridCols,
    mounted: gridMounted,
  } = useAgentGridLayout(visibleAgentIds);
  // The hook keeps a layout entry for every agent it's ever seen (so a
  // minimized/restored agent keeps its spot), but react-grid-layout still
  // renders a positioned box for any entry in `layout` even when it has no
  // matching child — so this has to be filtered down to what's actually
  // being rendered, or a minimized (or deleted) agent leaves an empty tile
  // sitting in the grid.
  const visibleAgentIdSet = useMemo(() => new Set(visibleAgentIds), [visibleAgentIds]);
  const visibleGridLayout = useMemo(
    () => gridLayout.filter(item => visibleAgentIdSet.has(item.i)),
    [gridLayout, visibleAgentIdSet]
  );

  // Hero (full-width MCP co-pilot) when there are no agents OR all of them are minimized.
  // Show GetStarted when no agents exist OR all are minimized
  const showGetStarted = agents.length === 0 || minimizedAgents.size === agents.length;

  return (
    <div className="app-container bg-gray-50">
     <MCPProvider getToken={getToken} isUsingObServer={isUsingObServer}>
      <style>
        {`
          @keyframes memory-flash {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
          }
          .animate-pulse {
            animation: memory-flash 1.5s ease-in-out infinite;
          }
        `}
      </style>

      <UpgradeModal
        isOpen={isUpgradeModalOpen}
        onClose={() => setIsUpgradeModalOpen(false)}
        isHalfwayWarning={isHalfwayWarning}
        quotaType={currentQuotaType}
      />

      <AcceptToS
        isOpen={isAcceptToSOpen}
        onAccept={() => {
          // Ask where they came from while it's fresh, then show the Pro trial pitch,
          // then land the user on the recipe splash (the guided one-line builder).
          setIsAcceptToSOpen(false);
          markOnboardingComplete();
          setIsAttributionOpen(true);
        }}
      />

      <AttributionSplash
        isOpen={isAttributionOpen}
        onDone={() => {
          setIsAttributionOpen(false);
          setWelcomeUpsellVariant('onboarding');
          setIsWelcomeUpsellOpen(true);
        }}
      />

      <RecipeSplash
        isOpen={isRecipeSplashOpen}
        onClose={() => { setIsRecipeSplashOpen(false); markTutorialSeen(); }}
      />

      {/* The first-run demo's notification landed: same free-trial pitch as post-ToS. */}
      <WelcomeModal
        isOpen={tutorialPhase === 'notified'}
        onClose={tutorialFlow.reset}
        onViewAllTiers={() => setActiveTab('obServer')}
        mode="upsell"
        variant="tutorial"
        isProUser={isProUser}
      />

      <WelcomeModal
        isOpen={isWelcomeUpsellOpen}
        onClose={() => {
          setIsWelcomeUpsellOpen(false);
          markOnboardingComplete();
        }}
        onViewAllTiers={() => setActiveTab('obServer')}
        mode="upsell"
        variant={welcomeUpsellVariant}
      />

      <AppHeader
        authState={{
          isLoading,
          isAuthenticated,
          user,
          loginWithRedirect: login,
          logout
        }}
        getToken={getToken}
        onToggleMobileMenu={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        isPermissionsModalOpen={isPermissionsModalOpen}
        onClosePermissionsModal={() => setIsPermissionsModalOpen(false)}
        isAccountModalOpen={isAccountModalOpen}
        onCloseAccountModal={() => setIsAccountModalOpen(false)}
      />

      {isObServerWarningOpen && (
        <StartupDialogs
          onDismiss={() => setIsObServerWarningOpen(false)}
          onLogin={login}
          onToggleObServer={handleToggleObServer}
          isAuthenticated={isAuthenticated}
          hostingContext={hostingContext}
        />
      )}

      <JupyterServerModal
        isOpen={isJupyterModalOpen}
        onClose={() => setIsJupyterModalOpen(false)}
      />

      {/* Sidebar + main content, side by side — a standard in-flow layout (the sidebar
          used to be position:fixed and overlay main, which needed main's own padding kept
          in sync with the sidebar's width by hand; a flex row does that for free). */}
      <div className="flex-1 min-h-0 flex flex-row">
        <PersistentSidebar
          activeTab={activeTab}
          onTabChange={setActiveTab}
          isMobileMenuOpen={isMobileMenuOpen}
          onCloseMobileMenu={() => setIsMobileMenuOpen(false)}
          authState={{ isLoading, isAuthenticated, user, loginWithRedirect: login }}
          quotaInfo={quotaInfo}
          onOpenAccount={() => setIsAccountModalOpen(true)}
          onOpenPermissions={() => setIsPermissionsModalOpen(true)}
          onFeedbackClick={() => setIsFeedbackOpen(true)}
          onToggleLogs={() => setShowGlobalLogs(prev => !prev)}
          isExpanded={isSidebarExpanded}
          onToggleExpanded={() => setIsSidebarExpanded(prev => !prev)}
          isUsingObServer={isUsingObServer}
          customServers={customServers}
          localServerOnline={localServerOnline}
        />

        <main className="relative w-full min-w-0 pt-4 px-2 md:px-4 wide:px-4 max-w-7xl mx-auto pb-20 md:pb-4">
        {error && <ErrorDisplay message={error} />}

        {/* My Agents Tab — kept mounted (not `hidden`/display:none) when inactive, so it still
            has real layout dimensions: GridLayout's ResizeObserver (see useAgentGridLayout.ts)
            reads 0 width under display:none and never mounts the grid, so cards would only
            render once the user actually visited this tab. Positioned off-screen (not just
            `invisible inset-0`) rather than stacked exactly on top of the other tab's content —
            `inset-0` put it in the same box as whatever tab IS visible, so a compositor glitch
            during the visibility toggle would briefly paint the grid ghosted on top of it. `w-full`
            still gives it main's real content width for the ResizeObserver. */}
        <div className={activeTab !== 'myAgents' ? 'absolute top-0 -left-[9999px] w-full overflow-hidden pointer-events-none' : ''}>
          {/* Agent grid — hidden (not unmounted) when showGetStarted so cards keep their state */}
          <div className="px-4">
            <div className={showGetStarted ? 'hidden' : ''}>
              <AgentImportHandler
                onAddAgent={handleAddAgentClick}
                agentCount={agents.length}
                activeAgentCount={runningAgents.size}
                isRefreshing={isRefreshing}
                onRefresh={fetchAgents}
              />
            </div>

            {/* Minimized agent chips — tucked away from the grid, but still reachable
                to restore. Rendered regardless of showGetStarted so minimizing every
                agent doesn't strand them behind the GetStarted screen. */}
            {minimizedAgents.size > 0 && (
              <div className="flex items-center gap-2 overflow-x-auto pb-3" style={{ scrollbarWidth: 'none' }}>
                {agents.map(a => minimizedAgents.has(a.id) ? (
                  <AgentChip
                    key={a.id}
                    agent={a}
                    isRunning={runningAgents.has(a.id)}
                    isStarting={startingAgents.has(a.id)}
                    isMinimized={true}
                    onRestore={() => handleRestore(a.id)}
                  />
                ) : null)}
              </div>
            )}

            {/* Tiling grid — drag a card by its title to move it, drag the bottom-right
                corner to resize. Position/size persist per agent (see useAgentGridLayout). */}
            <div ref={gridContainerRef as React.Ref<HTMLDivElement>} className={showGetStarted ? 'hidden overflow-x-hidden' : 'overflow-x-hidden'}>
              {gridMounted && (
                <GridLayout
                  layout={visibleGridLayout}
                  width={gridWidth}
                  gridConfig={{ cols: gridCols, rowHeight: GRID_ROW_HEIGHT, margin: GRID_MARGIN }}
                  dragConfig={{
                    enabled: !isMobile(),
                    bounded: true,
                    handle: '.agent-drag-handle',
                    cancel: 'button, a, input, textarea, select, [data-no-drag]',
                  }}
                  resizeConfig={{ enabled: !isMobile() }}
                  onDragStop={onGridDragStop}
                  onResizeStop={onGridResizeStop}
                >
                  {visibleGridLayout.map(item => {
                    const agent = sortedAgents.find(a => a.id === item.i);
                    if (!agent) return null;
                    return (
                      <div key={agent.id} className="h-full">
                        <AgentCard
                          agent={agent}
                          code={agentCodes[agent.id]}
                          isRunning={runningAgents.has(agent.id)}
                          isStarting={startingAgents.has(agent.id)}
                          isMemoryFlashing={flashingMemories.has(agent.id)}
                          onEdit={handleEditClick}
                          onDelete={handleDeleteClick}
                          onToggle={toggleAgent}
                          onMemory={handleMemoryClick}
                          onActivity={handleActivityClick}
                          onShowJupyterModal={() => setIsJupyterModalOpen(true)}
                          getToken={getToken}
                          isAuthenticated={isAuthenticated}
                          hasQuotaError={agentsWithQuotaError.has(agent.id)}
                          onUpgradeClick={() => {
                            setIsHalfwayWarning(false);
                            setIsUpgradeModalOpen(true);
                          }}
                          onSave={handleSaveAgent}
                          isProUser={isProUser}
                          hostingContext={hostingContext}
                          onMinimize={() => handleMinimize(agent.id)}
                          isMinimized={false}
                        />
                      </div>
                    );
                  })}
                </GridLayout>
              )}
            </div>

            {/* Minimized cards stay mounted (off-grid, hidden) so their agent-loop
                subscriptions and local state survive being tucked into the footer tray. */}
            <div className="hidden">
              {sortedAgents.filter(agent => minimizedAgents.has(agent.id)).map(agent => (
                <AgentCard
                  key={agent.id}
                  agent={agent}
                  code={agentCodes[agent.id]}
                  isRunning={runningAgents.has(agent.id)}
                  isStarting={startingAgents.has(agent.id)}
                  isMemoryFlashing={flashingMemories.has(agent.id)}
                  onEdit={handleEditClick}
                  onDelete={handleDeleteClick}
                  onToggle={toggleAgent}
                  onMemory={handleMemoryClick}
                  onActivity={handleActivityClick}
                  onShowJupyterModal={() => setIsJupyterModalOpen(true)}
                  getToken={getToken}
                  isAuthenticated={isAuthenticated}
                  hasQuotaError={agentsWithQuotaError.has(agent.id)}
                  onUpgradeClick={() => {
                    setIsHalfwayWarning(false);
                    setIsUpgradeModalOpen(true);
                  }}
                  onSave={handleSaveAgent}
                  isProUser={isProUser}
                  hostingContext={hostingContext}
                  onMinimize={() => handleMinimize(agent.id)}
                  isMinimized={true}
                />
              ))}
            </div>
          </div>

          {showGetStarted && (
            <GetStarted
              onExploreCommunity={() => setActiveTab('community')}
              onCreateNewAgent={handleAddAgentClick}
              getToken={getToken}
              isAuthenticated={isAuthenticated}
              isUsingObServer={isUsingObServer}
              onSignIn={login}
              onSwitchToObServer={() => setIsUsingObServer(true)}
              onUpgrade={() => {
                setActiveTab('obServer');
                setIsUsingObServer(true);
              }}
              onRefresh={fetchAgents}
              onUpgradeClick={() => {
                setIsHalfwayWarning(false);
                setIsUpgradeModalOpen(true);
              }}
              onOpenRecipe={() => setIsRecipeSplashOpen(true)}
            />
          )}
        </div>

        {/* Observer Tab — default landing view, chat-first. Stays mounted (hidden, not
            unmounted) like myAgents above, so RunningAgentsStrip's agentIterationStart/
            agentSleepStart/agentSleepEnd listeners and liveStatus don't reset on tab
            switch — that reset was what made status look stale until the next event fired. */}
        <div className={activeTab !== 'observerChat' ? 'hidden' : 'px-0 md:px-4 h-full'}>
          <ObserverTab
            getToken={getToken}
            isAuthenticated={isAuthenticated}
            isUsingObServer={isUsingObServer}
            onSignIn={login}
            onSwitchToObServer={() => setIsUsingObServer(true)}
            onUpgrade={() => {
              setActiveTab('obServer');
              setIsUsingObServer(true);
            }}
            onRefresh={fetchAgents}
            agents={agents}
            runningAgents={runningAgents}
            startingAgents={startingAgents}
            onToggleAgent={toggleAgent}
            onOpenMicroAgents={() => setActiveTab('myAgents')}
          />
        </div>

        {/* Community Tab */}
        {activeTab === 'community' && (
          <div className="px-4">
            <CommunityTab />
          </div>
        )}

        {/* Models Tab */}
        {activeTab === 'models' && (
          <div className="px-4">
            <AvailableModels
              isProUser={isProUser}
              isUsingObServer={isUsingObServer}
              handleToggleObServer={handleToggleObServer}
              showLoginMessage={showLoginMessage}
              isAuthenticated={isAuthenticated}
              quotaInfo={quotaInfo}
              renderQuotaStatus={renderQuotaStatus}
              localServerOnline={localServerOnline}
              checkLocalServer={checkLocalServer}
              customServers={customServers}
              onAddCustomServer={handleAddCustomServer}
              onRemoveCustomServer={handleRemoveCustomServer}
              onToggleCustomServer={handleToggleCustomServer}
              onCheckCustomServer={handleCheckCustomServer}
              appInferenceUrl={appInferenceUrl}
              onSetAppInferenceUrl={handleSetAppInferenceUrl}
            />
          </div>
        )}

        {/* Memory Store Tab */}
        {(activeTab === 'memoryText' || activeTab === 'memoryImages') && (
          <div className="px-4">
            <MemoryStoreTab kind={activeTab === 'memoryImages' ? 'image' : 'text'} />
          </div>
        )}

        {/* Recordings Tab */}
        {activeTab === 'recordings' && (
          <div className="px-4">
            <RecordingsViewer />
          </div>
        )}

        {/* Settings Tab */}
        {activeTab === 'settings' && (
          <div className="px-4">
            <SettingsTab isDarkMode={isDarkMode} onToggleDarkMode={toggleDarkMode} onOpenAccount={() => setIsAccountModalOpen(true)} />
          </div>
        )}

        {/* Subscription (opened from upgrade prompts) lives inside Settings */}
        {activeTab === 'obServer' && (
          <div className="px-4">
            <SettingsTab
              initialView="subscription"
              isDarkMode={isDarkMode}
              onToggleDarkMode={toggleDarkMode}
              onOpenAccount={() => setIsAccountModalOpen(true)}
            />
          </div>
        )}

        {/* Fallback for unknown tabs */}
        {!['myAgents', 'observerChat', 'community', 'models', 'recordings', 'memoryText', 'memoryImages', 'settings', 'obServer'].includes(activeTab) && (
          <div className="text-center p-8">
            <p className="text-gray-500">This feature is coming soon!</p>
          </div>
        )}
        </main>
      </div>

      <SimpleCreatorModal
        isOpen={isSimpleCreatorOpen}
        onClose={() => setIsSimpleCreatorOpen(false)}
        onNext={handleSimpleCreatorNext}
        isAuthenticated={isAuthenticated}
        hostingContext={hostingContext}
        userEmail={user?.email}
        isProUser={isProUser}
      />


      {isEditModalOpen && (
        <EditAgentModal
          isOpen={isEditModalOpen}
          onClose={() => {
            setIsEditModalOpen(false);
            setStagedAgentConfig(null);
          }}
          createMode={isCreateMode}
          agent={stagedAgentConfig ? stagedAgentConfig.agent : (selectedAgent ? agents.find(a => a.id === selectedAgent) : undefined)}
          code={stagedAgentConfig ? stagedAgentConfig.code : (selectedAgent ? agentCodes[selectedAgent] : undefined)}
          onSave={handleSaveAgent}
          onImportComplete={fetchAgents}
          setError={setError}
          getToken={getToken}
          isProUser={isProUser}
        />
      )}

      {isMemoryManagerOpen && memoryAgentId && (
        <MemoryManager
          agentId={memoryAgentId}
          agentName={agents.find(a => a.id === memoryAgentId)?.name || memoryAgentId}
          isOpen={isMemoryManagerOpen}
          onClose={() => {
            setIsMemoryManagerOpen(false);
            setMemoryAgentId(null);
          }}
        />
      )}

      {showGlobalLogs && (
        <GlobalLogsViewer
          isOpen={showGlobalLogs}
          onClose={() => setShowGlobalLogs(false)}
        />
      )}

      {activityModalOpen && activityModalAgentId && (
        <AgentActivityModal
          isOpen={activityModalOpen}
          onClose={() => {
            setActivityModalOpen(false);
            setActivityModalAgentId(null);
          }}
          agentId={activityModalAgentId}
          agentName={agents.find(a => a.id === activityModalAgentId)?.name || activityModalAgentId}
          getToken={getToken}
          isAuthenticated={isAuthenticated}
        />
      )}

      {showStartupDialog && (
        <StartupDialogs
          onDismiss={handleDismissStartupDialog}
          onServerConnected={() => { setCustomServers(getCustomServers()); fetchModels(); }}
          onLogin={login}
          onToggleObServer={() => setIsUsingObServer(true)}
          isAuthenticated={isAuthenticated}
          hostingContext={hostingContext}
          hasPendingImport={hasPendingImport}
        />
      )}

      <WelcomeModal
        isOpen={showLocalModeWarning}
        onClose={() => setShowLocalModeWarning(false)}
        onViewAllTiers={() => setActiveTab('obServer')}
        mode="local"
        onContinueLocal={() => setIsLocalOnboardingActive(true)}
      />

      <LocalOnboardingTutorial
        isActive={isLocalOnboardingActive}
        onDismiss={() => setIsLocalOnboardingActive(false)}
      />

      <FeedbackDialog
        isOpen={isFeedbackOpen}
        onClose={() => setIsFeedbackOpen(false)}
        getToken={getToken}
        isAuthenticated={isAuthenticated}
      />

      {whitelistModalInfo && (
        <WhitelistModal
          phoneNumbers={whitelistModalInfo.phoneNumbers}
          onClose={() => setWhitelistModalInfo(null)}
          onStartAgent={whitelistModalInfo.onStartAgent}
          channel={whitelistModalInfo.channel}
          onStartAnyway={
            whitelistModalInfo.agentId
              ? async () => {
                  const agentId = whitelistModalInfo.agentId!;
                  setWhitelistModalInfo(null);
                  setStartingAgents(prev => {
                    const updated = new Set(prev);
                    updated.add(agentId);
                    return updated;
                  });
                  try {
                    await startAgentLoop(agentId, getToken, true);
                  } catch (err) {
                    const errorMessage = err instanceof Error ? err.message : 'Unknown error';
                    setError(errorMessage);
                    Logger.error('APP', `Failed to start agent: ${errorMessage}`, err);
                  } finally {
                    setStartingAgents(prev => {
                      const updated = new Set(prev);
                      updated.delete(agentId);
                      return updated;
                    });
                  }
                }
              : undefined
          }
          getToken={getToken}
        />
      )}

      <LiveStream />


     </MCPProvider>
    </div>
  );
}

export function App() {
  const isAuthDisabled = import.meta.env.VITE_DISABLE_AUTH === 'true';

  Logger.info('AUTH', `is Auth disabled?: ${isAuthDisabled}`);

  if (isAuthDisabled) {
    Logger.info('isAuthDisabled', "Auth0 is disabled for local development.");
    return (
      <BrowserRouter>
        <Routes>
          <Route path="/*" element={<AppContent />} />
        </Routes>
      </BrowserRouter>
    );
  }

  return (
    <Auth0Provider
      domain="auth.observer-ai.com"
      clientId="R5iv3RVkWjGZrexFSJ6HqlhSaaGLyFpm"
      authorizationParams={{
        redirect_uri: (window.location.origin.startsWith('tauri://') && getPlatform() === 'linux') ? 'http://localhost:3838' : window.location.origin,
        audience: 'https://api.observer-ai.com',
        scope: 'openid profile email offline_access'
      }}
      cacheLocation="localstorage"
      useRefreshTokens={true}
      useRefreshTokensFallback={true}
      onRedirectCallback={(appState) => {
        window.history.replaceState(
          {},
          document.title,
          appState?.returnTo || window.location.pathname
        );
      }}
    >
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/upgrade-success" element={<UpgradeSuccessPage />} />
            <Route path="/join" element={<JoinOrgPage />} />
            <Route path="/team" element={<TeamPage />} />
            <Route path="/marketplace/:agentId" element={<AgentShareLandingPage />} />
            <Route path="/*" element={<AppContent />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </Auth0Provider>
  );
}

export default App;
