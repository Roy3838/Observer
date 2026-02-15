import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Terminal, MessageSquare, ChevronUp, X } from 'lucide-react';
import { Auth0Provider } from '@auth0/auth0-react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AuthProvider, useAuth } from '@contexts/AuthContext';
import { useIOSKeyboard } from '@hooks/useIOSKeyboard';
import { isMobile, confirm } from '@utils/platform';
import {
  listAgents,
  getAgentCode,
  deleteAgent,
  saveAgent,
  CompleteAgent,
} from '@utils/agent_database';
import { startAgentLoop, stopAgentLoop } from '@utils/main_loop';
import { Logger, type WhitelistChannel } from '@utils/logging';
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
import JupyterServerModal from '@components/JupyterServerModal';
import { generateAgentFromSimpleConfig } from '@utils/agentTemplateManager';
import SimpleCreatorModal from '@components/EditAgent/SimpleCreatorModal';
import ConversationalGeneratorModal from '@components/AICreator/ConversationalGeneratorModal';
import RecordingsViewer from '@components/RecordingsViewer';
import SettingsTab from '@components/SettingsTab';
import MemoryStoreTab from '@components/MemoryStoreTab';
import { UpgradeSuccessPage } from '../pages/UpgradeSuccessPage';
import { ObServerTab } from '@components/ObServerTab';
import { UpgradeModal } from '@components/UpgradeModal';
import { WelcomeModal } from '@components/WelcomeModal';
import AgentActivityModal from '@components/AgentCard/AgentActivityModal';
import TerminalModal from '@components/TerminalModal';
import FeedbackDialog from '@components/FeedbackDialog';
import { startCommandSSE, updateCommandSSEToken } from '@utils/commandSSE';
import { fetchModels } from '@utils/inferenceServer';
import WhitelistModal from '@components/WhitelistModal';
import InteractiveTutorial from '@components/InteractiveTutorial';

// Main app content - uses the unified auth hook
function AppContent() {
  const { isAuthenticated, isLoading, user, login, logout, getAccessToken } = useAuth();

  // Handle iOS keyboard - updates CSS variables when keyboard shows/hides
  useIOSKeyboard();

  const [agents, setAgents] = useState<CompleteAgent[]>([]);
  const [agentCodes, setAgentCodes] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<'unchecked' | 'online' | 'offline'>('unchecked');
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
  const [activeTab, setActiveTab] = useState('myAgents');
  const [isUsingObServer, setIsUsingObServer] = useState(false);
  const [isJupyterModalOpen, setIsJupyterModalOpen] = useState(false);
  const [isSimpleCreatorOpen, setIsSimpleCreatorOpen] = useState(false);
  const [stagedAgentConfig, setStagedAgentConfig] = useState<{ agent: CompleteAgent, code: string } | null>(null);
  const [isConversationalModalOpen, setIsConversationalModalOpen] = useState(false);
  const [aiEditMessage, setAiEditMessage] = useState<string | undefined>();
  const [hasCompletedStartupCheck, setHasCompletedStartupCheck] = useState(false);

  // Quota error state
  const [agentsWithQuotaError, setAgentsWithQuotaError] = useState<Set<string>>(new Set());
  const [isUpgradeModalOpen, setIsUpgradeModalOpen] = useState(false);
  const [isHalfwayWarning, setIsHalfwayWarning] = useState(false);
  const [currentQuotaType, setCurrentQuotaType] = useState<string>('monitor');

  // Activity modal state
  const [activityModalOpen, setActivityModalOpen] = useState(false);
  const [activityModalAgentId, setActivityModalAgentId] = useState<string | null>(null);

  // Terminal modal state
  const [noModels, setNoModels] = useState(false);
  const [isTerminalModalOpen, setIsTerminalModalOpen] = useState(false);

  // Quota info state
  const [quotaInfo, setQuotaInfo] = useState<{
    used: number;
    remaining: number;
    limit: number;
    tier: string;
  } | null>(null);

  // Mobile UI state
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isMobileFooterOpen, setIsMobileFooterOpen] = useState(false);

  // Feedback dialog state
  const [isFeedbackOpen, setIsFeedbackOpen] = useState(false);

  // Whitelist modal state
  const [whitelistModalInfo, setWhitelistModalInfo] = useState<{
    phoneNumbers: Array<{ number: string; isWhitelisted: boolean }>;
    agentId?: string;
    onStartAgent?: () => void;
    channel?: WhitelistChannel;
  } | null>(null);

  // Welcome modal state
  const [isWelcomeModalOpen, setIsWelcomeModalOpen] = useState(false);
  const [welcomeModalIntent, setWelcomeModalIntent] = useState<'local' | 'login' | null>(null);

  // Tutorial modal state
  const [tutorialModalInfo, setTutorialModalInfo] = useState<{
    agentName: string;
    agentId: string;
    hasPhoneTools: boolean;
  } | null>(null);

  const hasAutoStartedTutorialRef = useRef(false);

  // Dark mode state
  const [isDarkMode, setIsDarkMode] = useState(() => {
    const saved = localStorage.getItem('observer-dark-mode');
    return saved === 'true';
  });

  const isProUser = quotaInfo?.tier === 'pro' || quotaInfo?.tier === 'max';

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

  const hostingContext = useMemo(() => {
    const { protocol, hostname } = window.location;

    if (protocol === 'https:' && hostname !== 'localhost' && hostname !== '127.0.0.1') {
      return 'official-web';
    }

    return 'self-hosted';
  }, []);

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

  const handleAIEditClick = (agentId: string) => {
    setAiEditMessage(`Help me edit this agent @${agentId}`);
    setIsConversationalModalOpen(true);
    Logger.info('APP', `Opening AI Edit modal for agent ${agentId}`);
  };

  const handleTutorialComplete = () => {
    Logger.info('TUTORIAL', 'Tutorial completed');
    setTutorialModalInfo(null);
  };

  const handleTutorialDismiss = () => {
    Logger.info('TUTORIAL', 'Tutorial dismissed');
    setTutorialModalInfo(null);
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
      await fetchAgents();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      Logger.error('APP', `Failed to delete agent: ${errorMessage}`, err);
    }
  };

  const handleDismissStartupDialog = () => {
    setShowStartupDialog(false);
    setHasCompletedStartupCheck(true);
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

      if (isNew && user && 'sub' in user && user.sub) {
        const dismissed = localStorage.getItem(`observer_tutorial_dismissed_${user.sub}`);
        if (!dismissed) {
          const hasPhoneTools = code.includes('call(') ||
                               code.includes('sendSms(') ||
                               code.includes('sendWhatsapp(');
          setTutorialModalInfo({
            agentName: agent.name,
            agentId: agent.id,
            hasPhoneTools
          });
        }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      Logger.error('APP', `Failed to save agent: ${errorMessage}`, err);
    }
  };

  // Auto-start tutorial when first agent is created
  useEffect(() => {
    if (agents.length === 1 && user && 'sub' in user && user.sub && !hasAutoStartedTutorialRef.current) {
      const dismissed = localStorage.getItem(`observer_tutorial_dismissed_${user.sub}`);

      if (!dismissed && !tutorialModalInfo) {
        const firstAgent = agents[0];

        setTutorialModalInfo({
          agentName: firstAgent.name,
          agentId: firstAgent.id,
          hasPhoneTools: false
        });

        hasAutoStartedTutorialRef.current = true;

        Logger.info('TUTORIAL', `Auto-starting tutorial for first agent: ${firstAgent.name}`);
      }
    }
  }, [agents, user, tutorialModalInfo]);

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

  // Start command SSE for hotkey support (desktop only)
  useEffect(() => {
    if (hostingContext === 'self-hosted' && !isMobileDevice) {
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
        setShowStartupDialog(true);
      }
    }
  }, [isLoading, isAuthenticated]);

  // Welcome modal - check for login intent after auth completes
  useEffect(() => {
    if (!isLoading && isAuthenticated && user && 'sub' in user) {
      // Check if user just logged in (sessionStorage intent)
      const loginIntent = sessionStorage.getItem('observer_login_intent');
      if (loginIntent) {
        sessionStorage.removeItem('observer_login_intent'); // Clear immediately
        Logger.info('WELCOME', 'User just logged in, showing welcome modal with login intent');
        setWelcomeModalIntent('login');
        setIsWelcomeModalOpen(true);
        return;
      }

      // Otherwise, check if they've completed onboarding
      const hasCompletedOnboarding = localStorage.getItem(`observer_onboarding_complete_${user.sub}`);
      if (!hasCompletedOnboarding) {
        const checkTierAndShowModal = async () => {
          try {
            const token = await getAccessToken();
            if (!token) return;
            const response = await fetch('https://api.observer-ai.com/quota', {
              headers: { 'Authorization': `Bearer ${token}` }
            });
            if (response.ok) {
              const data = await response.json();
              if (data.tier === 'free' || !data.tier) {
                Logger.info('WELCOME', 'User is on free tier, showing welcome modal');
                setWelcomeModalIntent('login'); // They're already authenticated
                setIsWelcomeModalOpen(true);
              } else {
                Logger.info('WELCOME', `User is on ${data.tier} tier, skipping welcome modal`);
              }
            } else {
              Logger.info('WELCOME', 'Could not determine tier, showing welcome modal');
              setWelcomeModalIntent('login');
              setIsWelcomeModalOpen(true);
            }
          } catch (err) {
            Logger.error('WELCOME', 'Error checking tier:', err);
            setWelcomeModalIntent('login');
            setIsWelcomeModalOpen(true);
          }
        };
        checkTierAndShowModal();
      }
    }
  }, [isLoading, isAuthenticated, user, getAccessToken]);

  useEffect(() => {
    if (serverStatus === 'offline' && !hasCompletedStartupCheck && !isLoading && !isAuthenticated) {
      setShowStartupDialog(true);
    }
  }, [serverStatus, hasCompletedStartupCheck, isLoading, isAuthenticated]);

  // Reload agents when switching to My Agents tab
  useEffect(() => {
    if (activeTab === 'myAgents') {
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

  return (
    <div className="app-container bg-gray-50">
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

      <WelcomeModal
        isOpen={isWelcomeModalOpen}
        onClose={() => setIsWelcomeModalOpen(false)}
        onViewAllTiers={() => setActiveTab('obServer')}
        intent={welcomeModalIntent}
      />

      <AppHeader
        serverStatus={serverStatus}
        setServerStatus={setServerStatus}
        setError={setError}
        isUsingObServer={isUsingObServer}
        setIsUsingObServer={setIsUsingObServer}
        hostingContext={hostingContext}
        authState={{
          isLoading,
          isAuthenticated,
          user,
          loginWithRedirect: login,
          logout
        }}
        getToken={getToken}
        onUpgradeClick={() => {
          setIsHalfwayWarning(true);
          setIsUpgradeModalOpen(true);
        }}
        onShowTerminalModal={() => setNoModels(true)}
        quotaInfo={quotaInfo}
        setQuotaInfo={setQuotaInfo}
        onToggleMobileMenu={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        isDarkMode={isDarkMode}
        onToggleDarkMode={toggleDarkMode}
      />

      <PersistentSidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        isMobileMenuOpen={isMobileMenuOpen}
        onCloseMobileMenu={() => setIsMobileMenuOpen(false)}
      />

      <JupyterServerModal
        isOpen={isJupyterModalOpen}
        onClose={() => setIsJupyterModalOpen(false)}
      />

      <main className="w-full pt-4 px-0 md:px-4 pl-2 md:pl-20 wide:px-4 max-w-7xl mx-auto">
        {error && <ErrorDisplay message={error} />}

        {/* My Agents Tab */}
        <div className={activeTab !== 'myAgents' ? 'hidden' : ''}>
          {sortedAgents.length > 0 ? (
            <div className="px-4">
              <AgentImportHandler
                onAddAgent={handleAddAgentClick}
                agentCount={agents.length}
                activeAgentCount={runningAgents.size}
                isRefreshing={isRefreshing}
                onRefresh={fetchAgents}
                onGenerateAgent={() => setIsConversationalModalOpen(true)}
              />

              <div className="flex flex-wrap gap-6 items-start overflow-x-hidden">
                {sortedAgents.map(agent => {
                  const isAgentLive = runningAgents.has(agent.id) || startingAgents.has(agent.id);
                  return (
                    <div
                      key={agent.id}
                      className={`flex-shrink-0 transition-all duration-700 ease-in-out ${
                        isAgentLive ? 'w-full' : 'w-full xl:w-[calc(50%-12px)]'
                      }`}
                    >
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
                        onAIEdit={handleAIEditClick}
                        hostingContext={hostingContext}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <GetStarted
              onExploreCommunity={() => setActiveTab('community')}
              onCreateNewAgent={handleAddAgentClick}
              getToken={getToken}
              isAuthenticated={isAuthenticated}
              isUsingObServer={isUsingObServer}
              isPro={isProUser}
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
            />
          )}
        </div>

        {/* Community Tab */}
        <div className={`px-4 ${activeTab !== 'community' ? 'hidden' : ''}`}>
          <CommunityTab />
        </div>

        {/* Models Tab */}
        {activeTab === 'models' && (
          <div className="px-4">
            <AvailableModels isProUser={isProUser} />
          </div>
        )}

        {/* Memory Store Tab */}
        <div className={`px-4 ${activeTab !== 'memoryStore' ? 'hidden' : ''}`}>
          <MemoryStoreTab />
        </div>

        {/* Recordings Tab */}
        <div className={`px-4 ${activeTab !== 'recordings' ? 'hidden' : ''}`}>
          <RecordingsViewer />
        </div>

        {/* Settings Tab */}
        <div className={`px-4 ${activeTab !== 'settings' ? 'hidden' : ''}`}>
          <SettingsTab />
        </div>

        {/* ObServer Tab */}
        <div className={`px-4 ${activeTab !== 'obServer' ? 'hidden' : ''}`}>
          <ObServerTab />
        </div>

        {/* Fallback for unknown tabs */}
        {!['myAgents', 'community', 'models', 'recordings', 'memoryStore', 'settings', 'obServer'].includes(activeTab) && (
          <div className="text-center p-8">
            <p className="text-gray-500">This feature is coming soon!</p>
          </div>
        )}
      </main>

      <SimpleCreatorModal
        isOpen={isSimpleCreatorOpen}
        onClose={() => setIsSimpleCreatorOpen(false)}
        onNext={handleSimpleCreatorNext}
        isAuthenticated={isAuthenticated}
        hostingContext={hostingContext}
      />

      <ConversationalGeneratorModal
        isOpen={isConversationalModalOpen}
        onClose={() => {
          setIsConversationalModalOpen(false);
          setAiEditMessage(undefined);
        }}
        getToken={getToken}
        isAuthenticated={isAuthenticated}
        isUsingObServer={isUsingObServer}
        isPro={isProUser}
        onSignIn={login}
        onSwitchToObServer={() => setIsUsingObServer(true)}
        onUpgrade={() => {
          setActiveTab('obServer');
          setIsUsingObServer(true);
        }}
        onRefresh={fetchAgents}
        initialMessage={aiEditMessage}
        onUpgradeClick={() => {
          setIsHalfwayWarning(false);
          setIsUpgradeModalOpen(true);
        }}
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

      {/* Mobile FAB / Floating bubble */}
      {!isMobileFooterOpen ? (
        <button
          onClick={() => setIsMobileFooterOpen(true)}
          className="fixed bottom-4 right-4 z-50 md:hidden w-10 h-10 bg-white/95 backdrop-blur-sm text-gray-700 rounded-full shadow-lg border border-gray-200 flex items-center justify-center"
          style={{ marginBottom: 'env(safe-area-inset-bottom)' }}
        >
          <ChevronUp size={18} />
        </button>
      ) : (
        <>
          <div
            className="fixed inset-0 z-50 md:hidden"
            onClick={() => setIsMobileFooterOpen(false)}
          />
          <div
            className="fixed bottom-4 right-4 z-50 md:hidden flex items-center space-x-2 bg-white/95 backdrop-blur-sm rounded-full shadow-lg border border-gray-200 px-3 py-2"
            style={{ marginBottom: 'env(safe-area-inset-bottom)' }}
          >
            <button
              className="flex items-center justify-center w-8 h-8 bg-gray-100 rounded-full hover:bg-gray-200 transition"
              onClick={() => {
                setShowGlobalLogs(!showGlobalLogs);
                setIsMobileFooterOpen(false);
              }}
              title="Logs"
            >
              <Terminal className="h-4 w-4" />
            </button>
            <div className="w-px h-5 bg-gray-200" />
            <button
              onClick={() => {
                setIsFeedbackOpen(true);
                setIsMobileFooterOpen(false);
              }}
              className="p-1.5 text-gray-700 hover:text-blue-600 transition"
              title="Send feedback"
            >
              <MessageSquare className="h-4 w-4" />
            </button>
            <div className="w-px h-5 bg-gray-200" />
            <div className="flex items-center space-x-2">
              <a href="https://discord.gg/wnBb7ZQDUC" target="_blank" rel="noopener noreferrer" className="text-indigo-500 hover:text-indigo-600 transition" title="Discord">
                <svg className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor"><path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.127 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03z" /></svg>
              </a>
              <a href="https://x.com/AppObserverAI" target="_blank" rel="noopener noreferrer" className="text-gray-800 hover:text-gray-900 transition" title="X">
                <svg className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.244H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" /></svg>
              </a>
              <a href="https://buymeacoffee.com/roy3838" target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:text-gray-900 transition" title="Support">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" className="h-4 w-4"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" /></svg>
              </a>
              <a href="https://github.com/Roy3838/Observer" target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:text-gray-900 transition" title="GitHub">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" className="h-4 w-4"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" /></svg>
              </a>
            </div>
            <div className="w-px h-5 bg-gray-200" />
            <button
              onClick={() => setIsMobileFooterOpen(false)}
              className="p-1 text-gray-500 hover:text-gray-700 transition"
            >
              <X size={16} />
            </button>
          </div>
        </>
      )}

      {/* Desktop floating bubble */}
      <div className="fixed bottom-4 right-4 z-40 hidden md:flex items-center space-x-3 bg-white/95 backdrop-blur-sm rounded-full shadow-lg border border-gray-200 px-4 py-2.5">
        <button
          className="flex items-center justify-center w-8 h-8 bg-gray-100 rounded-full hover:bg-gray-200 transition"
          onClick={() => setShowGlobalLogs(!showGlobalLogs)}
          title="Logs"
        >
          <Terminal className="h-4 w-4" />
        </button>
        <div className="w-px h-5 bg-gray-200" />
        <button
          onClick={() => setIsFeedbackOpen(true)}
          className="flex items-center space-x-1.5 px-2 py-1 text-xs font-medium text-gray-700 hover:text-blue-600 transition"
          title="Send feedback"
        >
          <MessageSquare className="h-3.5 w-3.5" />
          <span>Feedback</span>
        </button>
        <div className="w-px h-5 bg-gray-200" />
        <span className="text-xs text-gray-500">Support the Project!</span>
        <div className="flex items-center space-x-2">
          <a href="https://discord.gg/wnBb7ZQDUC" target="_blank" rel="noopener noreferrer" className="text-indigo-500 hover:text-indigo-600 transition" title="Join our Discord community">
            <svg className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor"><path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.127 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03z" /></svg>
          </a>
          <a href="https://x.com/AppObserverAI" target="_blank" rel="noopener noreferrer" className="text-gray-800 hover:text-gray-900 transition" title="Follow us on X (Twitter)">
            <svg className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.244H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" /></svg>
          </a>
          <a href="https://buymeacoffee.com/roy3838" target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:text-gray-900 transition" title="Support the project">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" className="h-4 w-4"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" /></svg>
          </a>
          <a href="https://github.com/Roy3838/Observer" target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:text-gray-900 transition" title="GitHub Repository">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" className="h-4 w-4"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" /></svg>
          </a>
        </div>
      </div>

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
          onSkip={() => {
            setWelcomeModalIntent('local');
            setIsWelcomeModalOpen(true);
          }}
          onLogin={login}
          onToggleObServer={() => setIsUsingObServer(true)}
          isAuthenticated={isAuthenticated}
          hostingContext={hostingContext}
        />
      )}

      <TerminalModal
        isOpen={!showStartupDialog && (noModels || isTerminalModalOpen)}
        onClose={() => {
          setNoModels(false);
          setIsTerminalModalOpen(false);
        }}
        onPullComplete={async () => {
          await fetchModels();
        }}
        noModels={noModels}
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

      {tutorialModalInfo && (
        <InteractiveTutorial
          isActive={true}
          onComplete={handleTutorialComplete}
          onDismiss={handleTutorialDismiss}
          agentId={tutorialModalInfo.agentId}
          hasPhoneTools={tutorialModalInfo.hasPhoneTools}
        />
      )}
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
        redirect_uri: window.location.origin,
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
            <Route path="/*" element={<AppContent />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </Auth0Provider>
  );
}

export default App;
