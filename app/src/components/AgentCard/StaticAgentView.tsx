import React, { useRef, useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import {
    Brain, Clock, Eye, ChevronDown, AlertTriangle, Server, Wrench, ChevronRight, Zap, Settings, Cloud, Download, Cpu, CheckCircle, FileDown, StopCircle
} from 'lucide-react';
import { CompleteAgent } from '@utils/agent_database';
import { BROWSER_LOCAL_SENTINEL, LLAMA_CPP_LOCAL_SENTINEL } from '@utils/inferenceServer';
import { ModelManager, LocalModelState } from '@utils/ModelManager';
import { detectAgentCapabilities } from './agentCapabilities';
import SensorModal from './SensorModal';
import ToolsModal from './ToolsModal';
import ChangeDetectionSettings from '@components/ChangeDetectionSettings';




// --- HELPER FUNCTIONS ---

const formatBytes = (bytes: number, decimals = 1) => {
    if (!+bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`;
};

// --- GENERIC HELPER COMPONENTS ---

const InfoTag: React.FC<{ icon: React.ElementType; label: string; warning?: string; isBlocking?: boolean }> = ({ icon: Icon, label, warning, isBlocking }) => (
    <div className="relative group">
        <div className="inline-flex items-center gap-1.5 bg-gray-100 text-gray-700 px-2 py-1 rounded-md text-xs font-medium cursor-default">
            <Icon className="w-3.5 h-3.5" />
            <span>{label}</span>
            {warning && <AlertTriangle className={`w-3.5 h-3.5 ml-1 ${isBlocking ? 'text-red-500' : 'text-orange-500'}`} />}
        </div>
        {warning && (
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-max max-w-xs p-2 bg-gray-800 text-white text-xs rounded-md shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                {warning}
            </div>
        )}
    </div>
);

// Provider display names mapping
const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
    'gemini': 'Google AI Studio',
    'gemini-pro': 'Google AI Studio',
    'google': 'Google AI Studio',
    'fireworks': 'Fireworks.ai',
    'openrouter': 'OpenRouter',
    'openai': 'OpenAI',
    'anthropic': 'Anthropic',
    'together': 'Together AI',
    'groq': 'Groq',
};

// Provider privacy policy URLs
const PROVIDER_PRIVACY_URLS: Record<string, string> = {
    'Google AI Studio': 'https://ai.google.dev/gemini-api/terms',
    'Fireworks.ai': 'https://fireworks.ai/privacy-policy',
    'OpenRouter': 'https://openrouter.ai/privacy',
    'OpenAI': 'https://openai.com/policies/privacy-policy',
    'Anthropic': 'https://www.anthropic.com/privacy',
    'Together AI': 'https://www.together.ai/privacy',
    'Groq': 'https://groq.com/privacy-policy/',
    'Observer Cloud': 'https://observer-ai.com/#/Privacy',
};

// Helper to get provider display name from ownedBy or server
const getProviderName = (ownedBy?: string, server?: string): string | null => {
    if (ownedBy) {
        const key = ownedBy.toLowerCase();
        return PROVIDER_DISPLAY_NAMES[key] || ownedBy;
    }
    // Only show provider name for Observer-managed servers
    if (server?.includes('api.observer-ai.com')) {
        return 'Observer Cloud';
    }
    return null;
};

// Helper to check if model is Observer-managed (for privacy indicator)
const isCloudModel = (server?: string): boolean => {
    if (!server) return false;
    return server.includes('api.observer-ai.com');
};

// Helper to format bytes
//const formatBytes = (bytes: number, decimals = 2) => {
//    if (!+bytes) return '0 Bytes';
//    const k = 1024;
//    const dm = decimals < 0 ? 0 : decimals;
//    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
//    const i = Math.floor(Math.log(bytes) / Math.log(k));
//    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
//};

// Model Location Indicator - Shows cloud or local status with popover (uses portal)
const ModelLocationIndicator: React.FC<{
    isCloud: boolean;
    providerName?: string;
    sensors: { key: string; label: string; icon: React.ElementType }[];
    isLoading?: boolean;
    isUnloaded?: boolean;
    localModelId?: string;
    server?: string;
}> = ({ isCloud, providerName, sensors, isLoading = false, isUnloaded = false, localModelId, server }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [popoverPosition, setPopoverPosition] = useState<{ top: number; left: number } | null>(null);
    const buttonRef = useRef<HTMLButtonElement>(null);
    const popoverRef = useRef<HTMLDivElement>(null);

    // Get unified local model state from ModelManager
    const isLocalModel = server === LLAMA_CPP_LOCAL_SENTINEL || server === BROWSER_LOCAL_SENTINEL;
    const [localState, setLocalState] = useState<LocalModelState | null>(
        isLocalModel && server ? ModelManager.getInstance().getLocalModelState(server) : null
    );
    const [isNativeDownloading, setIsNativeDownloading] = useState(
        () => ModelManager.getInstance().getLocalModelState(LLAMA_CPP_LOCAL_SENTINEL)?.status === 'downloading'
    );

    const updatePopoverPosition = () => {
        if (buttonRef.current) {
            const rect = buttonRef.current.getBoundingClientRect();
            setPopoverPosition({
                top: rect.bottom + 8,
                left: rect.left + rect.width / 2 - 128, // 128 = half of w-64 (256px)
            });
        }
    };

    // Subscribe to ModelManager for state updates
    useEffect(() => {
        if (!isOpen || !isLocalModel || !server) return;
        const unsubscribe = ModelManager.getInstance().onModelsChange(() => {
            setLocalState(ModelManager.getInstance().getLocalModelState(server));
            setIsNativeDownloading(
                ModelManager.getInstance().getLocalModelState(LLAMA_CPP_LOCAL_SENTINEL)?.status === 'downloading'
            );
        });
        return unsubscribe;
    }, [isOpen, isLocalModel, server]);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (popoverRef.current && !popoverRef.current.contains(event.target as Node) &&
                buttonRef.current && !buttonRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        if (isOpen) {
            document.addEventListener('mousedown', handleClickOutside);
            return () => document.removeEventListener('mousedown', handleClickOutside);
        }
    }, [isOpen]);

    useEffect(() => {
        if (isOpen) {
            updatePopoverPosition();
            window.addEventListener('scroll', updatePopoverPosition, true);
            window.addEventListener('resize', updatePopoverPosition);
            return () => {
                window.removeEventListener('scroll', updatePopoverPosition, true);
                window.removeEventListener('resize', updatePopoverPosition);
            };
        }
    }, [isOpen]);

    const handleLoadModel = () => {
        if (!localModelId || !server) return;
        ModelManager.getInstance().loadLocalModel(localModelId, server);
    };

    const handleCancelLoad = () => {
        if (!server) return;
        ModelManager.getInstance().unloadLocalModel(server);
    };

    const handleToggle = () => {
        if (!isOpen) {
            updatePopoverPosition();
        }
        setIsOpen(!isOpen);
    };

    // Determine icon and colors
    let iconColor = 'text-green-500 hover:text-green-600 hover:bg-green-50';
    let IconComponent = Server;
    let title = 'Local model';

    if (isCloud) {
        iconColor = 'text-blue-500 hover:text-blue-600 hover:bg-blue-50';
        IconComponent = Cloud;
        title = 'Cloud model';
    } else if (isUnloaded) {
        iconColor = 'text-red-500 hover:text-red-600 hover:bg-red-50';
        IconComponent = Download;
        title = 'Model not loaded - click to load';
    } else if (isLoading) {
        iconColor = 'text-amber-500 hover:text-amber-600 hover:bg-amber-50';
        title = 'Loading local model...';
    }

    return (
        <div className="relative inline-block">
            <button
                ref={buttonRef}
                onClick={handleToggle}
                className={`p-1 rounded transition-colors ${iconColor}`}
                title={title}
            >
                <div className={isLoading ? 'animate-pulse' : isUnloaded ? 'animate-bounce' : ''}>
                    <IconComponent className="w-4 h-4" />
                </div>
            </button>
            {isOpen && popoverPosition && createPortal(
                <div
                    ref={popoverRef}
                    className="fixed w-64 p-3 bg-white border border-gray-200 rounded-lg shadow-lg z-[100]"
                    style={{ top: popoverPosition.top, left: popoverPosition.left }}
                >
                    {isCloud ? (
                        <>
                            <div className="text-xs font-semibold text-gray-800 mb-2">
                                → {providerName && PROVIDER_PRIVACY_URLS[providerName] ? (
                                    <a
                                        href={PROVIDER_PRIVACY_URLS[providerName]}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-blue-600 hover:text-blue-800 hover:underline"
                                        onClick={(e) => e.stopPropagation()}
                                    >
                                        {providerName} ↗
                                    </a>
                                ) : providerName}
                            </div>
                            {sensors.length > 0 && (
                                <div className="space-y-1 mb-2">
                                    <div className="text-xs text-gray-500 font-medium">Data sent to provider:</div>
                                    {sensors.map(sensor => (
                                        <div key={sensor.key} className="flex items-center gap-2 text-xs text-gray-600">
                                            <sensor.icon className="w-3.5 h-3.5 text-gray-400" />
                                            <span>{sensor.label}</span>
                                        </div>
                                    ))}
                                </div>
                            )}
                            <div className="mt-2 pt-2 border-t border-gray-100">
                                <p className="text-xs text-gray-400">
                                    Use local models for 100% privacy
                                </p>
                            </div>
                        </>
                    ) : isUnloaded ? (
                        // Unloaded state - show load button
                        <div className="space-y-3">
                            <div className="flex items-center gap-2">
                                <Download className="w-4 h-4 text-red-500" />
                                <p className="text-xs text-gray-700">
                                    <span className="font-semibold text-red-600">Model not loaded</span>
                                </p>
                            </div>
                            <button
                                onClick={handleLoadModel}
                                disabled={isNativeDownloading}
                                title={isNativeDownloading ? 'A model is currently downloading — please wait' : undefined}
                                className="w-full flex items-center justify-center gap-1.5 px-3 py-2 text-sm bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-colors"
                            >
                                <Cpu size={14} />
                                <span>{isNativeDownloading ? 'Downloading...' : 'Load Model'}</span>
                            </button>
                            <p className="text-xs text-gray-400 text-center">
                                All data stays on your device
                            </p>
                        </div>
                    ) : isLoading ? (
                        // Loading state - show progress
                        <div className="space-y-3">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <Cpu className="w-4 h-4 text-amber-500 animate-pulse" />
                                    <span className="text-xs font-semibold text-amber-600">Loading model...</span>
                                </div>
                                <button
                                    onClick={handleCancelLoad}
                                    className="p-1 text-gray-400 hover:text-red-600 rounded transition-colors"
                                    title="Cancel"
                                >
                                    <StopCircle size={14} />
                                </button>
                            </div>

                            {/* Progress bars (transformers.js only) */}
                            {localState && localState.progress.length > 0 && (
                                <div className="space-y-2 max-h-32 overflow-y-auto">
                                    {localState.progress.map((item) => (
                                        <div key={item.file}>
                                            <div className="flex justify-between items-center text-xs mb-0.5">
                                                <span className="text-gray-600 flex items-center gap-1 truncate max-w-[60%]">
                                                    {item.done
                                                        ? <CheckCircle className="h-3 w-3 text-green-500 flex-shrink-0" />
                                                        : <FileDown className="h-3 w-3 text-gray-400 flex-shrink-0" />
                                                    }
                                                    <span className="truncate">{item.file.split('/').pop()}</span>
                                                </span>
                                                <span className="font-medium text-gray-500 flex-shrink-0 text-xs">
                                                    {item.done
                                                        ? '✓'
                                                        : item.total > 0
                                                            ? `${formatBytes(item.loaded)}/${formatBytes(item.total)}`
                                                            : `${Math.round(item.progress)}%`
                                                    }
                                                </span>
                                            </div>
                                            <div className="w-full bg-gray-200 rounded-full h-1">
                                                <div
                                                    className={`h-1 rounded-full transition-all duration-300 ${item.done ? 'bg-green-500' : 'bg-blue-600'}`}
                                                    style={{ width: `${item.progress}%` }}
                                                />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* llama.cpp status (no detailed progress) */}
                            {localState?.engineInfo?.type === 'llama.cpp' && localState?.status === 'loading' && (
                                <div className="text-xs text-gray-500 text-center">
                                    Initializing model...
                                </div>
                            )}
                            {localState?.engineInfo?.type === 'llama.cpp' && localState?.status === 'unloading' && (
                                <div className="text-xs text-amber-600 text-center">
                                    Unloading model...
                                </div>
                            )}

                            {/* Show engine settings (transformers.js) */}
                            {localState?.engineInfo?.type === 'transformers.js' && localState?.engineInfo?.device && (
                                <div className="text-xs text-gray-500 text-center flex items-center justify-center gap-1.5 flex-wrap">
                                    <span>{localState.engineInfo.device} · {localState.engineInfo.dtype}</span>
                                    {localState.engineInfo.enableThinking ? (
                                        <span className="text-[10px] font-semibold text-purple-700 bg-purple-100 px-1.5 py-0.5 rounded">Thinking on</span>
                                    ) : (
                                        <span className="text-[10px] font-medium text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">Thinking off</span>
                                    )}
                                </div>
                            )}

                            <p className="text-xs text-gray-400 text-center">
                                All data stays on your device
                            </p>
                        </div>
                    ) : (
                        // Loaded state
                        <div className="space-y-3">
                            <div className="flex items-center gap-2">
                                <Server className="w-4 h-4 text-green-500" />
                                <p className="text-xs text-gray-700">
                                    <span className="font-semibold text-green-600">
                                        {localState?.engineInfo?.type === 'llama.cpp' ? 'Local On-Device' : 'Local In-Browser'}
                                    </span> all data stays on your device!
                                </p>
                            </div>
                            {/* Show engine info */}
                            {localState?.engineInfo?.type === 'llama.cpp' ? (
                                <div className="bg-gray-50 rounded-md p-2">
                                    <div className="flex justify-between text-xs">
                                        <span className="text-gray-500">Engine:</span>
                                        <span className="font-medium text-gray-700">llama.cpp</span>
                                    </div>
                                </div>
                            ) : localState?.engineInfo && (
                                <div className="bg-gray-50 rounded-md p-2 space-y-1">
                                    <div className="flex justify-between text-xs">
                                        <span className="text-gray-500">Device:</span>
                                        <span className="font-medium text-gray-700">{localState.engineInfo.device === 'webgpu' ? 'WebGPU (GPU)' : 'WASM (CPU)'}</span>
                                    </div>
                                    {localState.engineInfo.dtype && (
                                        <div className="flex justify-between text-xs">
                                            <span className="text-gray-500">Precision:</span>
                                            <span className="font-medium text-gray-700">{localState.engineInfo.dtype}</span>
                                        </div>
                                    )}
                                    <div className="flex justify-between text-xs">
                                        <span className="text-gray-500">Thinking:</span>
                                        {localState.engineInfo.enableThinking ? (
                                            <span className="text-[10px] font-semibold text-purple-700 bg-purple-100 px-1.5 py-0.5 rounded">on</span>
                                        ) : (
                                            <span className="text-[10px] font-medium text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded">off</span>
                                        )}
                                    </div>
                                </div>
                            )}
                            <button
                                onClick={handleCancelLoad}
                                className="w-full flex items-center justify-center gap-1.5 px-3 py-1.5 text-sm bg-gray-100 text-gray-600 rounded-lg hover:bg-red-100 hover:text-red-600 font-medium transition-colors"
                            >
                                <StopCircle size={14} />
                                <span>Unload Model</span>
                            </button>
                        </div>
                    )}
                </div>,
                document.body
            )}
        </div>
    );
};


const ModelDropdown: React.FC<{ currentModel: string; onModelChange: (modelName: string) => void; isProUser?: boolean; }> = ({ currentModel, onModelChange, isProUser = false }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [availableModels, setAvailableModels] = useState<{ name: string; multimodal?: boolean; pro?: boolean; server: string; ownedBy?: string; }[]>([]);
    const [dropdownPosition, setDropdownPosition] = useState<{ top: number; left: number } | null>(null);
    const buttonRef = useRef<HTMLButtonElement>(null);
    const dropdownRef = useRef<HTMLDivElement>(null);

    const fetchModels = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const response = ModelManager.getInstance().listModels();
            if (response.error) throw new Error(response.error);
            setAvailableModels(response.models);
        } catch (e) {
            setError(`Failed to fetch: ${e instanceof Error ? e.message : String(e)}`);
        } finally {
            setIsLoading(false);
        }
    };

    const updateDropdownPosition = () => {
        if (buttonRef.current) {
            const rect = buttonRef.current.getBoundingClientRect();
            setDropdownPosition({
                top: rect.bottom + 8,
                left: rect.right - 192, // 192px = w-48 (12rem)
            });
        }
    };

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node) &&
                buttonRef.current && !buttonRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    useEffect(() => {
        if (isOpen) {
            updateDropdownPosition();
            window.addEventListener('scroll', updateDropdownPosition, true);
            window.addEventListener('resize', updateDropdownPosition);
            return () => {
                window.removeEventListener('scroll', updateDropdownPosition, true);
                window.removeEventListener('resize', updateDropdownPosition);
            };
        }
    }, [isOpen]);

    const handleToggle = () => {
        if (!isOpen) {
            fetchModels();
            updateDropdownPosition();
        }
        setIsOpen(!isOpen);
    };

    return (
        <div className="relative inline-block text-left">
            <button ref={buttonRef} type="button" onClick={handleToggle} className="inline-flex justify-center w-full rounded-md border border-gray-300 shadow-sm px-3 py-2 md:px-2.5 md:py-1.5 bg-white text-xs font-medium text-gray-700 hover:bg-gray-50 min-h-[44px] md:min-h-0">
                <span className="truncate max-w-[120px]">{currentModel || 'Select Model'}</span>
                <ChevronDown className="-mr-1 ml-1.5 h-4 w-4" />
            </button>
            {isOpen && dropdownPosition && createPortal(
                <div
                    ref={dropdownRef}
                    className="fixed w-48 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 z-[100]"
                    style={{ top: dropdownPosition.top, left: dropdownPosition.left }}
                >
                    <div className="py-1 max-h-72 overflow-y-auto">
                        {isLoading && <div className="px-3 py-1.5 text-xs text-gray-500">Loading...</div>}
                        {error && <div className="px-3 py-1.5 text-xs text-red-600">{error}</div>}
                        {!isLoading && !error && availableModels.map((model) => (
                            <button
                                key={model.name}
                                onClick={() => {
                                    if (model.pro && !isProUser) return;
                                    onModelChange(model.name);
                                    setIsOpen(false);
                                }}
                                disabled={model.pro && !isProUser}
                                className={`${model.name === currentModel ? 'bg-gray-100' : ''} block w-full text-left px-3 py-1.5 text-xs hover:bg-gray-100 ${model.pro && !isProUser ? 'opacity-50 select-none cursor-not-allowed' : ''}`}>
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center">
                                    <span className="truncate">{model.name}</span>
                                    {model.pro && !isProUser && (
                                      <span className="ml-2 text-xs font-bold text-purple-600 bg-purple-100 px-1.5 py-0.5 rounded-full">
                                        PRO
                                      </span>
                                    )}
                                  </div>
                                  <div className="flex items-center space-x-1">
                                    {model.multimodal && <Eye className="h-4 w-4 text-purple-600" />}
                                    {!model.server.includes('api.observer-ai.com') && <Server className="h-4 w-4 text-gray-600" />}
                                  </div>
                                </div>
                            </button>
                        ))}
                    </div>
                </div>,
                document.body
            )}
        </div>
    );
};


// --- MAIN COMPONENT ---

interface StaticAgentViewProps {
    agent: CompleteAgent;
    code?: string;
    currentModel: string;
    onModelChange: (modelName: string) => void;
    onToggleSignificantChange: (enabled: boolean) => void;
    onSystemPromptChange?: (newPrompt: string) => void;
    onCodeChange?: (newCode: string) => void;
    startWarning: string | null;
    isProUser?: boolean;
    hostingContext?: 'official-web' | 'self-hosted' | 'tauri';
    getToken?: () => Promise<string | undefined>;
}


const StaticAgentView: React.FC<StaticAgentViewProps> = ({
    agent,
    code,
    currentModel,
    onModelChange,
    onToggleSignificantChange,
    onSystemPromptChange,
    onCodeChange,
    startWarning,
    isProUser = false,
    hostingContext,
    getToken,
}) => {
    const [detectedSensors, setDetectedSensors] = useState<any[]>([]);
    const [detectedTools, setDetectedTools] = useState<any[]>([]);
    const [isSensorModalOpen, setIsSensorModalOpen] = useState(false);
    const [isToolsModalOpen, setIsToolsModalOpen] = useState(false);
    const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);
    const [currentModelInfo, setCurrentModelInfo] = useState<{ server?: string; ownedBy?: string; status?: 'loaded' | 'loading' | 'unloaded' | 'unloading' | 'error'; localModelId?: string } | null>(null);

    // Look up current model info for location indicator
    useEffect(() => {
        let cancelled = false;

        const lookupModel = () => {
            if (cancelled) return;
            const models = ModelManager.getInstance().listModels().models;
            const modelInfo = models.find(m => m.name === currentModel);
            setCurrentModelInfo(modelInfo
                ? { server: modelInfo.server, ownedBy: modelInfo.ownedBy, status: modelInfo.status, localModelId: modelInfo.localModelId }
                : null
            );
        };

        // Try immediately
        lookupModel();

        // Subscribe to ModelManager for all model state changes
        // ModelManager handles forwarding from underlying managers (Gemma + Native)
        const unsubscribe = ModelManager.getInstance().onModelsChange(() => {
            lookupModel();
        });

        return () => {
            cancelled = true;
            unsubscribe();
        };
    }, [currentModel]);

    // Derive model location info
    const cloudProviderName = currentModelInfo
        ? getProviderName(currentModelInfo.ownedBy, currentModelInfo.server)
        : null;
    const isCloud = isCloudModel(currentModelInfo?.server);
    const isModelLoading = currentModelInfo?.status === 'loading';
    const isModelUnloaded = currentModelInfo?.status === 'unloaded';
    const showModelIndicator = currentModelInfo?.server;

    useEffect(() => {
        const loadCapabilities = async () => {
            try {
                const capabilities = await detectAgentCapabilities(agent.system_prompt || '', code || '', hostingContext);
                setDetectedSensors(capabilities.sensors);
                setDetectedTools(capabilities.tools);
            } catch (error) {
                console.error('Failed to load agent capabilities:', error);
                setDetectedSensors([]);
                setDetectedTools([]);
            }
        };

        loadCapabilities();
    }, [agent.system_prompt, code, hostingContext]);

    // ESC key to close settings modal
    useEffect(() => {
        const handleEscape = (e: KeyboardEvent) => {
            if (e.key === 'Escape' && isSettingsModalOpen) {
                setIsSettingsModalOpen(false);
            }
        };
        window.addEventListener('keydown', handleEscape);
        return () => window.removeEventListener('keydown', handleEscape);
    }, [isSettingsModalOpen]);

    const handleOpenToolsModal = () => {
        setIsToolsModalOpen(true);
    };

    return (
        <div className="animate-fade-in">
            {/* 3 Column Layout with Arrows - Responsive: vertical on mobile, horizontal on desktop */}
            <div className="flex flex-col md:flex-row items-center md:items-start gap-1 md:gap-4">
                {/* Column 1: Sensors */}
                <div className="flex flex-col flex-1 w-full md:w-auto">
                    <button
                        onClick={() => setIsSensorModalOpen(true)}
                        className="flex md:flex-col items-start md:items-center w-full text-left p-3 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors cursor-pointer group"
                        title="View system prompt"
                    >
                        <div className="flex justify-start mb-0 md:mb-4 w-6 md:w-auto flex-shrink-0 transition-colors">
                            <Eye className="w-5 h-5 text-gray-500 group-hover:text-indigo-600" />
                        </div>
                        <div className="flex flex-wrap gap-2 md:flex-col md:space-y-2 items-start md:items-center min-h-[44px] md:min-h-0 flex-1 ml-3 md:ml-0">
                            {detectedSensors.length > 0 ? (
                                detectedSensors.map(sensor => (
                                    <InfoTag key={sensor.key} icon={sensor.icon} label={sensor.label} />
                                ))
                            ) : (
                                <div className="text-sm text-gray-400 italic">No sensors</div>
                            )}
                        </div>
                    </button>
                </div>

                {/* Arrow 1 - Responsive: down on mobile, right on desktop */}
                <div className="flex items-center justify-start md:justify-center py-2 md:py-0 pl-1 md:pl-0">
                    <ChevronRight className="w-4 h-4 text-gray-400 rotate-90 md:rotate-0" />
                </div>

                {/* Column 2: Model */}
                <div className="flex flex-col flex-1 w-full md:w-auto">
                    {/* Desktop: vertical layout */}
                    <div className="hidden md:flex flex-col items-center">
                        <div className="flex justify-center mb-4">
                            <Brain className="w-5 h-5 text-gray-500" />
                        </div>
                        <div className="flex flex-col items-center space-y-3">
                            <div className="flex items-center gap-1">
                                <ModelDropdown currentModel={currentModel} onModelChange={onModelChange} isProUser={isProUser} />
                                {showModelIndicator && (
                                    <ModelLocationIndicator
                                        isCloud={isCloud}
                                        providerName={cloudProviderName || undefined}
                                        sensors={detectedSensors}
                                        isLoading={isModelLoading}
                                        isUnloaded={isModelUnloaded}
                                        localModelId={currentModelInfo?.localModelId}
                                        server={currentModelInfo?.server}
                                    />
                                )}
                            </div>
                            <div className="flex items-center gap-3">
                                <div className="flex items-center gap-1" data-tutorial-loop-timer={agent.id}>
                                    <Clock className="w-4 h-4 text-gray-500" />
                                    <span className="text-sm text-gray-600">{agent.loop_interval_seconds}s</span>
                                </div>
                                <div className="flex items-center gap-1.5">
                                    <div className="relative group">
                                        <button
                                            onClick={() => onToggleSignificantChange(!(agent.only_on_significant_change ?? false))}
                                            className={`flex items-center gap-2 px-3 py-1.5 rounded-md transition-colors ${
                                                (agent.only_on_significant_change ?? false)
                                                    ? 'bg-indigo-100 text-indigo-700 hover:bg-indigo-200'
                                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                            }`}
                                        >
                                            <Zap className="w-4 h-4" />
                                            <span className="text-xs font-medium">
                                                {(agent.only_on_significant_change ?? false) ? 'On' : 'Off'}
                                            </span>
                                        </button>
                                        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-max max-w-xs p-2 bg-gray-800 text-white text-xs rounded-md shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                                            Only run model when there's significant change in inputs
                                        </div>
                                    </div>
                                    <div className="relative group">
                                        <button
                                            onClick={() => setIsSettingsModalOpen(true)}
                                            className="p-0.5 rounded-md bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-indigo-600 transition-colors"
                                            title="Change detection settings"
                                        >
                                            <Settings className="w-3 h-3" />
                                        </button>
                                        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-max max-w-xs p-2 bg-gray-800 text-white text-xs rounded-md shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                                            Configure change detection sensitivity
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Mobile: 3-column grid */}
                    <div className="grid md:hidden grid-cols-[auto_1fr_auto] gap-3 items-center w-full">
                        {/* Column 1: Brain icon */}
                        <div className="flex items-center justify-center">
                            <Brain className="w-5 h-5 text-gray-500" />
                        </div>

                        {/* Column 2: Model dropdown + model location indicator */}
                        <div className="flex items-center justify-center gap-1">
                            <ModelDropdown currentModel={currentModel} onModelChange={onModelChange} isProUser={isProUser} />
                            {showModelIndicator && (
                                <ModelLocationIndicator
                                    isCloud={isCloud}
                                    providerName={cloudProviderName || undefined}
                                    sensors={detectedSensors}
                                    isLoading={isModelLoading}
                                    isUnloaded={isModelUnloaded}
                                    localModelId={currentModelInfo?.localModelId}
                                    server={currentModelInfo?.server}
                                />
                            )}
                        </div>

                        {/* Column 3: Timer, Flash button, and Settings button stacked */}
                        <div className="flex flex-col gap-1.5 items-center">
                            {/* Timer */}
                            <div className="flex items-center gap-1" data-tutorial-loop-timer={agent.id}>
                                <Clock className="w-4 h-4 text-gray-500" />
                                <span className="text-sm text-gray-600">{agent.loop_interval_seconds}s</span>
                            </div>

                            {/* Flash button and Settings button side by side */}
                            <div className="flex items-center gap-1.5">
                                <div className="relative group">
                                    <button
                                        onClick={() => onToggleSignificantChange(!(agent.only_on_significant_change ?? false))}
                                        className={`flex items-center gap-2 px-3 py-1.5 rounded-md transition-colors ${
                                            (agent.only_on_significant_change ?? false)
                                                ? 'bg-indigo-100 text-indigo-700 hover:bg-indigo-200'
                                                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                        }`}
                                    >
                                        <Zap className="w-4 h-4" />
                                        <span className="text-xs font-medium">
                                            {(agent.only_on_significant_change ?? false) ? 'On' : 'Off'}
                                        </span>
                                    </button>
                                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-max max-w-xs p-2 bg-gray-800 text-white text-xs rounded-md shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                                        Only run model when there's significant change in inputs
                                    </div>
                                </div>
                                <div className="relative group">
                                    <button
                                        onClick={() => setIsSettingsModalOpen(true)}
                                        className="p-0.5 rounded-md bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-indigo-600 transition-colors"
                                        title="Change detection settings"
                                    >
                                        <Settings className="w-3 h-3" />
                                    </button>
                                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-max max-w-xs p-2 bg-gray-800 text-white text-xs rounded-md shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                                        Configure change detection sensitivity
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Arrow 2 - Responsive: down on mobile, right on desktop */}
                <div className="flex items-center justify-start md:justify-center py-2 md:py-0 pl-1 md:pl-0">
                    <ChevronRight className="w-4 h-4 text-gray-400 rotate-90 md:rotate-0" />
                </div>

                {/* Column 3: Tools */}
                <div className="flex flex-col flex-1 w-full md:w-auto">
                    <button
                        onClick={handleOpenToolsModal}
                        data-tutorial-tools-button={agent.id}
                        className="flex md:flex-col items-start md:items-center w-full text-left p-3 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors cursor-pointer group"
                        title="View agent code"
                    >
                        <div className="flex justify-start mb-0 md:mb-4 w-6 md:w-auto flex-shrink-0 transition-colors">
                            <Wrench className="w-5 h-5 text-gray-500 group-hover:text-indigo-600" />
                        </div>
                        <div className="flex flex-wrap gap-2 md:flex-col md:space-y-2 items-start md:items-center min-h-[44px] md:min-h-0 flex-1 ml-3 md:ml-0">
                            {detectedTools.length > 0 ? (
                                detectedTools.map(tool => (
                                    <InfoTag
                                        key={tool.key}
                                        icon={tool.icon}
                                        label={tool.label}
                                        warning={tool.warning}
                                        isBlocking={tool.isBlocking}
                                    />
                                ))
                            ) : (
                                <div className="text-sm text-gray-400 italic">No tools</div>
                            )}
                        </div>
                    </button>
                </div>
            </div>

            {/* Other Warnings */}
            {startWarning && (
                <div className="mt-6 p-3 bg-yellow-50 border border-yellow-200 text-yellow-800 rounded-md text-sm flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5 flex-shrink-0" />
                    <span>{startWarning}</span>
                </div>
            )}

            {/* Sensor Modal */}
            <SensorModal
                isOpen={isSensorModalOpen}
                onClose={() => setIsSensorModalOpen(false)}
                systemPrompt={agent.system_prompt || ''}
                agentName={agent.name || 'Unnamed Agent'}
                agentId={agent.id}
                onSystemPromptChange={onSystemPromptChange}
            />

            {/* Tools Modal */}
            <ToolsModal
                isOpen={isToolsModalOpen}
                onClose={() => setIsToolsModalOpen(false)}
                code={code || ''}
                agentName={agent.name || 'Unnamed Agent'}
                agentId={agent.id}
                getToken={getToken}
                onCodeChange={onCodeChange}
            />

            {/* Change Detection Settings Modal */}
            {isSettingsModalOpen && (
                <div
                    className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[70]"
                    onClick={() => setIsSettingsModalOpen(false)}
                >
                    <div
                        className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] overflow-y-auto m-4"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="sticky top-0 bg-white border-b px-6 py-4 flex items-center justify-between">
                            <h2 className="text-xl font-bold text-gray-900">Change Detection Settings</h2>
                            <button
                                onClick={() => setIsSettingsModalOpen(false)}
                                className="text-gray-400 hover:text-gray-600 transition-colors"
                            >
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>
                        <div className="p-6">
                            <ChangeDetectionSettings
                                compact={true}
                                focusedThreshold="dhash"
                            />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default StaticAgentView;
