import React, { useRef, useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import {
    Brain, Clock, Eye, ChevronDown, AlertTriangle, Server, Wrench, ChevronRight, Zap, Settings, Cloud
} from 'lucide-react';
import { CompleteAgent } from '@utils/agent_database';
import { listModels, fetchModels } from '@utils/inferenceServer';
import { getInferenceAddresses } from '@utils/inferenceServer';
import { detectAgentCapabilities } from './agentCapabilities';
import SensorModal from './SensorModal';
import ToolsModal from './ToolsModal';
import ChangeDetectionSettings from '@components/ChangeDetectionSettings';




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
    // Check if it's a cloud server (not localhost)
    if (server && !server.includes('localhost') && !server.includes('127.0.0.1') && !server.includes('browser_local')) {
        if (server.includes('api.observer-ai.com')) return 'Observer Cloud';
        return 'Cloud Provider';
    }
    return null;
};

// Helper to check if model is cloud-based
const isCloudModel = (server?: string): boolean => {
    if (!server) return false;
    return !server.includes('localhost') && !server.includes('127.0.0.1') && !server.includes('browser_local');
};

// Model Location Indicator - Shows cloud or local status with popover
const ModelLocationIndicator: React.FC<{
    isCloud: boolean;
    providerName?: string;
    sensors: { key: string; label: string; icon: React.ElementType }[];
}> = ({ isCloud, providerName, sensors }) => {
    const [isOpen, setIsOpen] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        if (isOpen) {
            document.addEventListener('mousedown', handleClickOutside);
            return () => document.removeEventListener('mousedown', handleClickOutside);
        }
    }, [isOpen]);

    return (
        <div ref={containerRef} className="relative">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={`p-1 rounded transition-colors ${
                    isCloud
                        ? 'text-blue-500 hover:text-blue-600 hover:bg-blue-50'
                        : 'text-gray-400 hover:text-gray-500 hover:bg-gray-100'
                }`}
                title={isCloud ? "Cloud model" : "Local model"}
            >
                {isCloud ? <Cloud className="w-4 h-4" /> : <Server className="w-4 h-4" />}
            </button>
            {isOpen && (
                <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-56 p-3 bg-white border border-gray-200 rounded-lg shadow-lg z-50">
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
                    ) : (
                        <div className="flex items-center gap-2">
                            <Server className="w-4 h-4 text-green-500" />
                            <p className="text-xs text-gray-700">
                                <span className="font-semibold text-green-600">Local model</span> — all data stays on your device!
                            </p>
                        </div>
                    )}
                </div>
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
            const addresses = getInferenceAddresses();
            if (addresses.length === 0) throw new Error("No inference servers configured.");
            const response = listModels();
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
                                    {(model.server.includes('localhost') || model.server.includes('http://')) || model.server.includes('browser_local') && <Server className="h-4 w-4 text-gray-600" />}
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
    const [currentModelInfo, setCurrentModelInfo] = useState<{ server?: string; ownedBy?: string } | null>(null);

    // Look up current model info for location indicator
    useEffect(() => {
        let cancelled = false;

        const lookupModel = () => {
            if (cancelled) return;
            const models = listModels().models;
            const modelInfo = models.find(m => m.name === currentModel);
            setCurrentModelInfo(modelInfo
                ? { server: modelInfo.server, ownedBy: modelInfo.ownedBy }
                : null
            );
        };

        // Try immediately
        const models = listModels().models;
        if (models.length > 0) {
            lookupModel();
        } else if (currentModel) {
            // Models not loaded yet, fetch them (only if needed)
            fetchModels().then(lookupModel);
        }

        return () => { cancelled = true; };
    }, [currentModel]);

    // Derive model location info
    const cloudProviderName = currentModelInfo
        ? getProviderName(currentModelInfo.ownedBy, currentModelInfo.server)
        : null;
    const isCloud = isCloudModel(currentModelInfo?.server);
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
                                            onClick={() => onToggleSignificantChange(!(agent.only_on_significant_change ?? true))}
                                            className={`flex items-center gap-2 px-3 py-1.5 rounded-md transition-colors ${
                                                (agent.only_on_significant_change ?? true)
                                                    ? 'bg-indigo-100 text-indigo-700 hover:bg-indigo-200'
                                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                            }`}
                                        >
                                            <Zap className="w-4 h-4" />
                                            <span className="text-xs font-medium">
                                                {(agent.only_on_significant_change ?? true) ? 'On' : 'Off'}
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
                                        onClick={() => onToggleSignificantChange(!(agent.only_on_significant_change ?? true))}
                                        className={`flex items-center gap-2 px-3 py-1.5 rounded-md transition-colors ${
                                            (agent.only_on_significant_change ?? true)
                                                ? 'bg-indigo-100 text-indigo-700 hover:bg-indigo-200'
                                                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                        }`}
                                    >
                                        <Zap className="w-4 h-4" />
                                        <span className="text-xs font-medium">
                                            {(agent.only_on_significant_change ?? true) ? 'On' : 'Off'}
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
