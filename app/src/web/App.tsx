import { useState, useEffect, useCallback } from 'react';
import { Terminal, Server } from 'lucide-react';
import { Auth0Provider, useAuth0 } from '@auth0/auth0-react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { 
  listAgents, 
  getAgentCode,
  deleteAgent,
  saveAgent,
  CompleteAgent,
} from '@utils/agent_database';
import { startAgentLoop, stopAgentLoop, AGENT_STATUS_CHANGED_EVENT } from '@utils/main_loop';
import { Logger } from '@utils/logging';
import { MEMORY_UPDATE_EVENT } from '@components/MemoryManager';

// Imported Components
import AppHeader from '@components/AppHeader';
import AgentCard from '@components/AgentCard';
import EditAgentModal from '@components/EditAgent/EditAgentModal';
import StartupDialogs from '@components/StartupDialogs';
import GlobalLogsViewer from '@components/GlobalLogsViewer';
import ScheduleAgentModal from '@components/ScheduleAgentModal';
import MemoryManager from '@components/MemoryManager';
import ErrorDisplay from '@components/ErrorDisplay';
import AgentImportHandler from '@components/AgentImportHandler';
import PersistentSidebar from '@components/PersistentSidebar';
import AvailableModels from '@components/AvailableModels';
import CommunityTab from '@components/CommunityTab';
import GetStarted from '@components/GetStarted';
import JupyterServerModal from '@components/JupyterServerModal';
import { generateAgentFromSimpleConfig } from '@utils/agentTemplateManager';
import { themeManager } from '@utils/theme';
import SimpleCreatorModal from '@components/EditAgent/SimpleCreatorModal';
import ConversationalGeneratorModal from '@components/ConversationalGeneratorModal';
import RecordingsViewer from '@components/RecordingsViewer';
import SettingsTab from '@components/SettingsTab';
import { UpgradeSuccessPage } from '../pages/UpgradeSuccessPage';
import { ObServerTab } from '@components/ObServerTab';
import { UpgradeModal } from '@components/UpgradeModal';


function AppContent() {
  // Check our environment variable to see if Auth0 should be disabled
  const isAuthDisabled = import.meta.env.VITE_DISABLE_AUTH === 'true';
  Logger.debug('isAuthDisabled', `is it? lets see ${isAuthDisabled}`);
  
  // Add a loading state to help debug white screen issues
  const [isAppLoading, setIsAppLoading] = useState(true);

  // Create the auth object based on whether auth is disabled
  let isAuthenticated: boolean;
  let user: any;
  let loginWithRedirect: () => Promise<void>;
  let logout: (options?: any) => void;
  let isLoading: boolean;
  let getAccessTokenSilently: (options?: any) => Promise<string>;

  if (isAuthDisabled) {
    // Mock auth values for local development
    isAuthenticated = true;
    user = { name: 'Local Dev User', email: 'dev@localhost' };
    loginWithRedirect = () => Promise.resolve();
    logout = (_options?: any) => {};
    isLoading = false;
    getAccessTokenSilently = async (_options?: any) => 'mock_token';
  } else {
    // Use real Auth0 hook
    const auth0Data = useAuth0();
    isAuthenticated = auth0Data.isAuthenticated;
    user = auth0Data.user;
    loginWithRedirect = auth0Data.loginWithRedirect;
    logout = auth0Data.logout;
    isLoading = auth0Data.isLoading;
    getAccessTokenSilently = auth0Data.getAccessTokenSilently;
  } 

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
  const [showStartupDialog, setShowStartupDialog] = useState(() => {
    // Check if user has completed startup before
    const hasCompletedStartup = localStorage.getItem('observer-startup-completed');
    return !hasCompletedStartup;
  });
  const [showGlobalLogs, setShowGlobalLogs] = useState(false);
  const [isScheduleModalOpen, setIsScheduleModalOpen] = useState(false);
  const [schedulingAgentId, setSchedulingAgentId] = useState<string | null>(null);
  const [isMemoryManagerOpen, setIsMemoryManagerOpen] = useState(false);
  const [memoryAgentId, setMemoryAgentId] = useState<string | null>(null);
  const [flashingMemories, setFlashingMemories] = useState<Set<string>>(new Set());
  const [activeTab, setActiveTab] = useState('myAgents');
  const [isUsingObServer, setIsUsingObServer] = useState(() => {
    // Check for saved server choice, default to cloud
    const savedChoice = localStorage.getItem('observer-server-choice');
    return savedChoice === 'local' ? false : true;
  });
  const [isJupyterModalOpen, setIsJupyterModalOpen] = useState(false);
  const [isSimpleCreatorOpen, setIsSimpleCreatorOpen] = useState(false);
  const [stagedAgentConfig, setStagedAgentConfig] = useState<{ agent: CompleteAgent, code: string } | null>(null);
  const [isConversationalModalOpen, setIsConversationalModalOpen] = useState(false);

  // --- NEW STATE FOR QUOTA ERRORS AND MODAL ---
  const [agentsWithQuotaError, setAgentsWithQuotaError] = useState<Set<string>>(new Set());
  const [isUpgradeModalOpen, setIsUpgradeModalOpen] = useState(false);


  const fetchAgents = useCallback(async () => {
    try {
      setIsRefreshing(true);
      Logger.debug('APP', 'Fetching agents from database');
      const agentsData = await listAgents();
      setAgents(agentsData);
      Logger.debug('APP', `Found ${agentsData.length} agents`);

      // Fetch codes
      const codeResults = await Promise.all(
        agentsData.map(async (a) => ({ id: a.id, code: await getAgentCode(a.id) }))
      );
      const newCodes: Record<string, string> = {};
      codeResults.forEach((r) => {
        if (r.code) newCodes[r.id] = r.code;
      });
      setAgentCodes(newCodes);

      setError(null);
      
      // Refresh the ConversationalGenerator models after fetching agents
      window.dispatchEvent(new Event('refreshConversationalModels'));
      
    } catch (err) {
      setError('Failed to fetch agents from database');
      Logger.error('APP', `Error fetching agents:`, err);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  const getToken = useCallback(async () => {

    // If Auth0 is still loading its state, we can't get a token yet.
    if (isLoading) {
      Logger.warn('AUTH', 'getToken called while auth state is loading. Aborting.');
      return undefined;
    }

    // If loading is finished AND the user is not authenticated, abort.
    if (!isAuthenticated) {
      Logger.warn('AUTH', 'getToken called, but user is not authenticated.');
      try{ 
        const token = await getAccessTokenSilently({
          authorizationParams: {
            audience: 'https://api.observer-ai.com',
          },
        });
        Logger.error('AUTH', `trying getToken anyway and got token ${token}`);
        return token;
      }
      catch (error){
        Logger.warn('AUTH', `errored out in second try`);
      }
      return undefined;
    }

    try {
      const token = await getAccessTokenSilently({
        authorizationParams: {
          audience: 'https://api.observer-ai.com',
        },
      });
      return token;
    } catch (error) {
      Logger.error('AUTH', 'Failed to retrieve access token silently.', error);
      throw error;
    }
    // ✨ Add isLoading to the dependency array
  }, [isAuthenticated, isLoading, getAccessTokenSilently]);

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

    window.addEventListener(
      AGENT_STATUS_CHANGED_EVENT,
      handleAgentStatusChange as EventListener
    );
    return () => {
      window.removeEventListener(
        AGENT_STATUS_CHANGED_EVENT,
        handleAgentStatusChange as EventListener
      );
    };
  }, []);

  // --- USEEFFECT FOR QUOTA EVENT LISTENER ---
  useEffect(() => {
    const handleQuotaExceeded = (event: CustomEvent<{ agentId: string }>) => {
      const { agentId } = event.detail;
      setAgentsWithQuotaError(prevSet => {
        const newSet = new Set(prevSet);
        newSet.add(agentId);
        return newSet;
      });
    };

    window.addEventListener('quotaExceeded', handleQuotaExceeded as EventListener);

    return () => {
      window.removeEventListener('quotaExceeded', handleQuotaExceeded as EventListener);
    };
  }, []);

  // --- USEEFFECT FOR REFRESH AGENTS LIST EVENT LISTENER ---
  useEffect(() => {
    const handleRefreshAgentsList = () => {
      fetchAgents();
    };

    window.addEventListener('refreshAgentsList', handleRefreshAgentsList);

    return () => {
      window.removeEventListener('refreshAgentsList', handleRefreshAgentsList);
    };
  }, [fetchAgents]);

  const handleEditClick = async (agentId: string) => {
    setSelectedAgent(agentId);
    setIsCreateMode(false);
    setIsEditModalOpen(true);
    Logger.info('APP', `Opening editor for agent ${agentId}`);
  };

  const handleAddAgentClick = () => {
    setSelectedAgent(null);
    setIsCreateMode(true); // Keep this true to signal intent
    setStagedAgentConfig(null); // Clear any old staged config
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

  const handleAgentGenerated = (agent: CompleteAgent, code: string) => {
      Logger.info('APP', `Staging agent generated from conversation: "${agent.name}"`);
      setStagedAgentConfig({ agent, code });
      setIsCreateMode(true); // Signal that this is a new agent
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

  const handleDeleteClick = async (agentId: string) => {
    const agent = agents.find(a => a.id === agentId);
    if (!agent) return;
    
    if (window.confirm(`Are you sure you want to delete agent "${agent.name}"?`)) {
      try {
        setError(null);
        Logger.info('APP', `Deleting agent "${agent.name}" (${agentId})`);
        
        if (runningAgents.has(agentId)) {
          Logger.info(agentId, `Stopping agent before deletion`);
          stopAgentLoop(agentId);
        }
        
        await deleteAgent(agentId);
        Logger.info('APP', `Agent "${agent.name}" deleted successfully`);
        await fetchAgents();
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        setError(errorMessage);
        Logger.error('APP', `Failed to delete agent: ${errorMessage}`, err);
      }
    }
  };

  const handleDismissStartupDialog = () => {
    setShowStartupDialog(false);
  };

  const toggleAgent = async (id: string, isCurrentlyRunning: boolean): Promise<void> => {
    // If using Ob-Server and not logged in, trigger login instead of starting agent.
    if (isUsingObServer && !isAuthenticated) {
      Logger.info('AUTH', 'User attempted to use a protected feature while logged out. Redirecting to login.');
      loginWithRedirect();
      return; // Stop the function here.
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
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : 'Unknown error';
    setError(errorMessage);
    Logger.error('APP', `Failed to save agent: ${errorMessage}`, err);
  }
  };

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
    console.log('Observer App: Starting initialization...');
    Logger.info('APP', 'Application starting');
    
    // Initialize theme manager
    themeManager.getTheme();
    
    fetchAgents();
    
    const handleWindowError = (event: ErrorEvent) => {
      Logger.error('APP', `Uncaught error: ${event.message}`, {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        error: event.error
      });
    };

    window.addEventListener('error', handleWindowError);
    
    // Set loading to false after initial setup
    const timer = setTimeout(() => setIsAppLoading(false), 100);
    
    return () => {
      window.removeEventListener('error', handleWindowError);
      clearTimeout(timer);
    };
  }, []); // Remove dependencies to prevent repeated calls
  
  // Separate effect for auth logging that doesn't trigger fetchAgents
  useEffect(() => {
    if (isAuthenticated) {
      Logger.info('AUTH', `User authenticated: ${user?.name || user?.email || 'Unknown user'}`);
    } else if (!isLoading) {
      Logger.info('AUTH', 'User not authenticated');
    }
  }, [isAuthenticated, isLoading, user]);
  
  useEffect(() => {
    if (!isLoading) {
      Logger.info('AUTH', `Auth loading complete, authenticated: ${isAuthenticated}`);
    }
  }, [isLoading, isAuthenticated]);



  // Show loading screen if app is still initializing
  if (isAppLoading || (isLoading && !isAuthDisabled)) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading Observer...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-dark-bg">
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

        <UpgradeModal isOpen={isUpgradeModalOpen} onClose={() => setIsUpgradeModalOpen(false)} />

        {showStartupDialog && (
          <StartupDialogs 
            onDismiss={handleDismissStartupDialog}
            setUseObServer={setIsUsingObServer}
            onLogin={loginWithRedirect}
            isAuthenticated={isAuthenticated}
            />
        )}

        <AppHeader 
          serverStatus={serverStatus}
          setServerStatus={setServerStatus}
          setError={setError}
          isUsingObServer={isUsingObServer}
          setIsUsingObServer={setIsUsingObServer}
          authState={{
            isLoading,
            isAuthenticated,
            user,
            loginWithRedirect,
            logout
          }}
          getToken={getToken}
          onLogoClick={() => setActiveTab('myAgents')}
        />

        <PersistentSidebar 
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />

        <JupyterServerModal
          isOpen={isJupyterModalOpen}
          onClose={() => setIsJupyterModalOpen(false)}
        />

        <main className="ml-16 px-4 pt-24 pb-16">
          <div className="max-w-6xl mx-auto">

          {error && <ErrorDisplay message={error} />}

          {activeTab === 'myAgents' ? (
          <>
            {agents.length > 0 && (
              <AgentImportHandler 
                isRefreshing={isRefreshing}
                onRefresh={fetchAgents}
                onGenerateUsingAI={() => setIsConversationalModalOpen(true)}
                onBuildCustom={handleAddAgentClick}
              />
            )}
            <div className="flex flex-wrap gap-6">
              {agents.length > 0 ? agents.map(agent => (
                <div key={agent.id} className="w-full md:w-[calc(50%-12px)] lg:w-[calc(33.333%-16px)] flex-shrink-0">
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
                    onShowJupyterModal={() => setIsJupyterModalOpen(true)}
                    getToken={getToken}
                    isAuthenticated={isAuthenticated}
                    hasQuotaError={agentsWithQuotaError.has(agent.id)}
                    onUpgradeClick={() => setIsUpgradeModalOpen(true)}
                  />
                </div>
              )) : 
                <GetStarted 
                  onExploreCommunity={() => setActiveTab('community')}
                  onCreateNewAgent={handleAddAgentClick}
                  onAgentGenerated={handleAgentGenerated}
                  getToken={getToken}
                  isAuthenticated={isAuthenticated}
                />
              }
            </div>
          </>
          ) : activeTab === 'community' ? (
            <CommunityTab />
          ) : activeTab === 'models' ? (
            <AvailableModels />
          ) : activeTab === 'recordings' ? ( 
            <RecordingsViewer />            
          ) : activeTab === 'settings' ? ( 
            <SettingsTab />
          ) : activeTab === 'obServer' ? (
            <ObServerTab />
          ) : (
            <div className="text-center p-8">
              <p className="text-gray-500">This feature is coming soon!</p>
            </div>
          )}
          </div>
        </main>

        <SimpleCreatorModal 
          isOpen={isSimpleCreatorOpen}
          onClose={() => setIsSimpleCreatorOpen(false)}
          onNext={handleSimpleCreatorNext}
          isAuthenticated={isAuthenticated}
        />
        <ConversationalGeneratorModal
        isOpen={isConversationalModalOpen}
        onClose={() => setIsConversationalModalOpen(false)}
        onAgentGenerated={handleAgentGenerated}
        getToken={getToken}
        isAuthenticated={isAuthenticated}
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
        />
      )}
      
      {isScheduleModalOpen && schedulingAgentId && (
        <ScheduleAgentModal
          agentId={schedulingAgentId}
          isOpen={isScheduleModalOpen}
          onClose={() => {
            setIsScheduleModalOpen(false);
            setSchedulingAgentId(null);
          }}
          onUpdate={fetchAgents}
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

      <footer className="fixed bottom-0 left-0 right-0 bg-white dark:bg-dark-surface border-t dark:border-dark-border z-30">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex justify-between items-center">
            <div className="flex space-x-3">
              <button 
                className="flex items-center space-x-2 px-4 py-2 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"
                onClick={() => setShowGlobalLogs(!showGlobalLogs)}
              >
                <Terminal className="h-5 w-5 text-gray-600 dark:text-gray-300" />
              </button>
              
              <button 
                className="flex items-center space-x-2 px-4 py-2 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-md hover:bg-blue-100 dark:hover:bg-blue-900/40"
                onClick={() => setIsJupyterModalOpen(true)}
              >
                <Server className="h-5 w-5" />
              </button>
            </div>
            
            <div className="flex items-center space-x-4">
              <span className="text-xs text-gray-500 dark:text-gray-400">Support the Project!</span>
              <div className="flex items-center space-x-2">
                <a 
                  href="https://discord.gg/wnBb7ZQDUC"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-indigo-500 hover:text-indigo-600 dark:text-indigo-400 dark:hover:text-indigo-300"
                  title="Join our Discord community"
                >
                  <svg className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.127 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03z" />
                  </svg>
                </a>
                
                <a 
                  href="https://x.com/AppObserverAI"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-800 hover:text-gray-900 dark:text-gray-200 dark:hover:text-white"
                  title="Follow us on X (Twitter)"
                >
                  <svg className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                  </svg>
                </a>
                
                <a 
                  href="https://buymeacoffee.com/roy3838"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-600 hover:text-gray-900 dark:text-gray-300 dark:hover:text-white"
                  title="Support the project"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" className="h-4 w-4">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                  </svg>
                </a>
                
                <a 
                  href="https://github.com/Roy3838/Observer"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-600 hover:text-gray-900 dark:text-gray-300 dark:hover:text-white"
                  title="GitHub Repository"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" className="h-4 w-4">
                    <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" />
                  </svg>
                </a>
              </div>
            </div>
          </div>
        </div>
      </footer>
      
      {showGlobalLogs && (
        <GlobalLogsViewer 
          isOpen={showGlobalLogs}
          onClose={() => setShowGlobalLogs(false)}
        />
      )}
    </div>
  );
}

export function App() {
  const isAuthDisabled = import.meta.env.VITE_DISABLE_AUTH === 'true';

  if (isAuthDisabled) {
    Logger.info('isAuthDisabled',"Auth0 is disabled for local development.");
    // Even in dev mode, we need the router for consistency
    return (
      <BrowserRouter>
        <Routes>
          <Route path="/*" element={<AppContent />} />
        </Routes>
      </BrowserRouter>
    );
  }
  
  // This is the main production logic
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
      {/* The Router now lives inside the Auth0Provider */}
      {/* This ensures all pages can use the useAuth0() hook */}
      <BrowserRouter>
        <Routes>
          {/* Route 1: The special page for after payment */}
          <Route 
            path="/upgrade-success" 
            element={<UpgradeSuccessPage />} 
          />

          {/* Route 2: The catch-all for your main application */}
          {/* The "/*" means "match any other URL" */}
          <Route 
            path="/*" 
            element={<AppContent />} 
          />
        </Routes>
      </BrowserRouter>
    </Auth0Provider>
  );
}

export default App;
