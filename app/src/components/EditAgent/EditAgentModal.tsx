import React, {
  Suspense, useState, useEffect, useRef,
} from 'react';
import Modal from '@components/EditAgent/Modal';
import SensorInputText from '@components/EditAgent/SensorInputText';
import { CompleteAgent, importAgentsFromFiles } from '@utils/agent_database';
import {
  Download,
  Server,
  ChevronDown,
  Eye,
  ScanText,
  X,
  Zap,
  Monitor,
  Brain,
  Save,
  Activity,
  Edit3,
  Tag,
  FileUp,
  Clipboard,
  Mic,
  Camera,
  AlertTriangle,
  Volume2,
  Blend,
  Images,
  Info,
} from 'lucide-react';
import { Logger } from '@utils/logging';
import { useEditAgentModalLogic } from './useEditAgentModalLogic';

import LazyCodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { python } from '@codemirror/lang-python';
import { vscodeDark } from '@uiw/codemirror-theme-vscode';

import type { TokenProvider } from '@utils/main_loop';

/* ───────────────────────── props ───────────────────────── */
interface EditAgentModalProps {
  isOpen: boolean;
  onClose: () => void;
  createMode: boolean;
  agent?: CompleteAgent;
  code?: string;
  onSave: (agent: CompleteAgent, code: string) => void;
  onImportComplete?: () => Promise<void>;
  setError?: (message: string | null) => void;
  getToken: TokenProvider;
  isProUser?: boolean;
}

// --- HELPER COMPONENTS (Unchanged) ---
interface MobileTabNavProps {
  activeTab: 'config' | 'prompt' | 'code';
  setActiveTab: (tab: 'config' | 'prompt' | 'code') => void;
}
const MobileTabNav: React.FC<MobileTabNavProps> = ({ activeTab, setActiveTab }) => {
  const tabs = [
    { id: 'config', label: 'Config', icon: Brain },
    { id: 'prompt', label: 'Prompt', icon: ScanText },
    { id: 'code', label: 'Code', icon: Zap },
  ] as const;

  return (
    <div className="md:hidden flex-shrink-0 flex border-b border-gray-200 bg-gray-50">
      {tabs.map(tab => (
        <button
          key={tab.id}
          onClick={() => setActiveTab(tab.id)}
          className={`flex-1 flex justify-center items-center p-3 text-sm font-medium transition-colors ${
            activeTab === tab.id
              ? 'border-b-2 border-blue-600 text-blue-600 bg-white'
              : 'text-gray-500 hover:bg-gray-100'
          }`}
        >
          <tab.icon size={16} className="mr-2" />
          {tab.label}
        </button>
      ))}
    </div>
  );
};


// --- NEW HELPER COMPONENT: Sensor Button ---
const SensorButton = ({ icon: Icon, label, colorClass, onClick }: { icon: React.ElementType, label: string, colorClass?: string, onClick: () => void }) => (
  <button onClick={onClick} className={`flex-grow md:flex-grow-0 flex items-center justify-center space-x-2 px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors ${colorClass || 'text-gray-700'}`}>
    <Icon className="h-5 w-5" />
    <span className="text-sm font-medium">{label}</span>
  </button>
);

// --- NEW HELPER COMPONENT: Sensor Dropdown Button ---
const SensorDropdownButton = ({
  icon: Icon,
  label,
  colorClass,
  agents,
  onSelect
}: {
  icon: React.ElementType;
  label: string;
  colorClass?: string;
  agents: { id: string; name: string }[];
  onSelect: (agentId: string) => void;
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen]);

  if (agents.length === 0) return null;

  return (
    <div ref={dropdownRef} className="relative flex-grow md:flex-grow-0">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full flex items-center justify-center space-x-2 px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors ${colorClass || 'text-gray-700'}`}
      >
        <Icon className="h-5 w-5" />
        <span className="text-sm font-medium">{label}</span>
        <ChevronDown className={`h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute z-30 mt-1 w-full min-w-[200px] max-h-60 bg-white border border-gray-300 rounded-md shadow-lg overflow-y-auto">
          {agents.map((agent) => (
            <button
              key={agent.id}
              onClick={() => {
                onSelect(agent.id);
                setIsOpen(false);
              }}
              className="w-full text-left px-3 py-2 text-sm hover:bg-gray-100 transition-colors"
            >
              {agent.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};


/* ───────────────────────── REFACTORED CONTENT COMPONENTS ───────────────────────── */

// --- Config Content Component ---
interface ConfigContentProps {
  name: string;
  setName: (name: string) => void;
  agentId: string;
  setAgentId: (id: string) => void;
  createMode: boolean;
  currentModel: string;
  setCurrentModel: (model: string) => void;
  isModelDropdownOpen: boolean;
  setIsModelDropdownOpen: (isOpen: boolean) => void;
  loadingModels: boolean;
  modelsError: string | null;
  availableModels: { name: string; multimodal?: boolean; pro?: boolean; server: string; }[];
  loopInterval: number;
  setLoopInterval: (interval: number) => void;
  onlyOnSignificantChange: boolean;
  setOnlyOnSignificantChange: (value: boolean) => void;
  description: string;
  setDescription: (desc: string) => void;
  isProUser?: boolean;
}
const ConfigContent: React.FC<ConfigContentProps> = ({
  name, setName, agentId, setAgentId, createMode, currentModel, setCurrentModel,
  isModelDropdownOpen, setIsModelDropdownOpen, loadingModels, modelsError,
  availableModels, loopInterval, setLoopInterval, onlyOnSignificantChange, setOnlyOnSignificantChange,
  description, setDescription, isProUser = false,
}) => {
  return (
    <div className="p-4 bg-white rounded-lg border border-gray-200 shadow-sm">
      {/* Mobile-only header */}
      <h3 className="text-lg font-semibold text-blue-700 md:hidden mb-4">
        Agent Configuration
      </h3>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
        {/* Row 1: Name, ID */}
        <div>
          <label className="block text-gray-600 mb-1 flex items-center"><Edit3 size={14} className="mr-1.5 text-gray-500" />Name <span className="text-red-500">*</span></label>
          <input value={name} onChange={(e) => setName(e.target.value)} className="w-full p-2 bg-gray-100 border-gray-300 rounded-md" placeholder="Agent name" />
        </div>
        <div>
          <label className="block text-gray-600 mb-1 flex items-center"><Tag size={14} className="mr-1.5 text-gray-500" />ID {createMode && <span className="text-red-500">*</span>}</label>
          <input value={agentId} onChange={(e) => { if (createMode) { setAgentId(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, '')); } }} readOnly={!createMode} className={`w-full p-2 bg-gray-100 border-gray-300 rounded-md ${createMode ? 'focus:ring-blue-500' : 'opacity-70 cursor-not-allowed bg-gray-200'}`} placeholder="my_agent_id" />
        </div>

        {/* Row 2: Model + Params (left, aligned with Name) | Loop + Only on Change (right, aligned with ID) */}
        {/* Left column: Model + Params */}
        <div>
          <label className="block text-gray-600 mb-1 flex items-center"><Brain size={14} className="mr-1.5 text-gray-500" />Model <span className="text-red-500">*</span></label>
            <div className="relative">
              <button onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)} disabled={loadingModels} className="w-full p-2 bg-gray-100 border-gray-300 rounded-md flex justify-between items-center text-left">
                <span className="truncate">{currentModel || (loadingModels ? 'Loading…' : 'Select model')}</span>
                <ChevronDown className={`h-4 w-4 text-gray-400 transition-transform ${isModelDropdownOpen ? 'rotate-180' : ''}`} />
              </button>
              {isModelDropdownOpen && (
                <div className="absolute z-20 mt-1 w-full max-h-40 bg-white border border-gray-300 rounded-md shadow-lg overflow-y-auto">
                  {loadingModels && <div className="px-3 py-2 text-sm text-gray-500">Loading…</div>}
                  {modelsError && <div className="px-3 py-2 text-sm text-red-600">{modelsError}</div>}
                  {!loadingModels && !modelsError && availableModels.length === 0 && <div className="px-3 py-2 text-sm text-gray-500">No models. Ensure Ollama is running.</div>}
                  {!loadingModels && !modelsError && availableModels.map((m) => (
                      <button
                        key={m.name}
                        onClick={() => {
                          if (m.pro && !isProUser) return;
                          setCurrentModel(m.name);
                          setIsModelDropdownOpen(false);
                        }}
                        disabled={m.pro && !isProUser}
                        className={`w-full text-left px-3 py-2 text-xs flex justify-between items-center ${
                          currentModel === m.name
                            ? 'bg-blue-500 text-white'
                            : 'hover:bg-gray-100'
                        } ${m.pro && !isProUser ? 'opacity-50 select-none cursor-not-allowed' : ''}`}
                      >
                        <div className="flex items-center truncate pr-2">
                          <span className="truncate">{m.name}</span>
                          {m.pro && !isProUser && (
                            <span className="ml-2 text-xs font-bold text-purple-600 bg-purple-100 px-1.5 py-0.5 rounded-full">
                              PRO
                            </span>
                          )}
                        </div>
                        <div className="flex items-center space-x-1">
                          {m.multimodal && (
                            <span title="Supports Vision" className={`inline-flex items-center text-xs font-medium px-2 py-0.5 rounded ${currentModel === m.name ? 'bg-blue-400 text-white' : 'text-purple-600 bg-purple-100'}`}>
                              <Eye className="h-3.5 w-3.5 mr-1" />
                              Vision
                            </span>
                          )}
                          {(m.server.includes('localhost') || m.server.includes('http://')) && (
                            <span title="Running Locally" className={`inline-flex items-center text-xs font-medium px-2 py-0.5 rounded ${currentModel === m.name ? 'bg-blue-400 text-white' : 'text-gray-600 bg-gray-100'}`}>
                              <Server className="h-3.5 w-3.5 mr-1" />
                              Local
                            </span>
                          )}
                        </div>
                      </button>
                  ))}
                  {!loadingModels && (modelsError || availableModels.length === 0) && (
                    <div className="p-2 border-t">
                      <input value={currentModel} onChange={(e) => setCurrentModel(e.target.value)} className="w-full p-1.5 bg-gray-100 border-gray-300 rounded-md text-xs" placeholder="Custom model name" onClick={(e) => e.stopPropagation()} />
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        {/* Right column: Loop + Only on Change */}
        <div className="flex gap-4">
          <div className="flex-1">
            <label className="block text-gray-600 mb-1 flex items-center"><Activity size={14} className="mr-1.5 text-gray-500" />Loop (s)</label>
            <input type="number" min="1" step="1" value={loopInterval} onChange={(e) => setLoopInterval(Math.max(1, parseFloat(e.target.value) || 30))} className="w-full p-2 bg-gray-100 border-gray-300 rounded-md" />
          </div>
          <div className="flex-1">
            <label className="block text-gray-600 mb-1 flex items-center"><Zap size={14} className="mr-1.5 text-gray-500" />Only on Change</label>
            <label className="relative inline-flex items-center cursor-pointer mt-1">
              <input type="checkbox" checked={onlyOnSignificantChange} onChange={(e) => setOnlyOnSignificantChange(e.target.checked)} className="sr-only peer" />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
              <span className="ml-3 text-sm font-medium text-gray-700">{onlyOnSignificantChange ? 'On' : 'Off'}</span>
            </label>
          </div>
        </div>
          <div className="col-span-1 sm:col-span-2">
            <label className="block text-gray-600 mb-1 flex items-center"><Edit3 size={14} className="mr-1.5 text-gray-500" />Description</label>
            <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={2} className="w-full p-2 bg-gray-100 border-gray-300 rounded-md" placeholder="Optional description" />
          </div>
        </div>
    </div>
  );
};

// --- Prompt Content Component (UPDATED) ---
interface PromptContentProps {
  systemPromptRef: React.RefObject<HTMLTextAreaElement>;
  systemPrompt: string;
  setSystemPrompt: (prompt: string) => void;
  insertSystemPromptText: (text: string) => void;
  availableAgentsForBlocks: { id: string; name: string }[];
  visionValidationError: string | null;
}
const PromptContent: React.FC<PromptContentProps> = ({
  systemPromptRef, systemPrompt, setSystemPrompt, insertSystemPromptText,
  availableAgentsForBlocks, visionValidationError
}) => (
  <div className="p-4 bg-white rounded-lg border border-gray-200 shadow-sm flex-grow flex flex-col h-full">
    <div className="flex justify-between items-center mb-3">
      <div>
        <h3 className="text-lg font-semibold text-blue-700">System Prompt</h3>
        <p className="text-xs text-gray-500">Define instructions for the agent</p>
      </div>
    </div>
    <SensorInputText
      value={systemPrompt}
      onChange={setSystemPrompt}
      textareaRef={systemPromptRef}
      className="flex-grow bg-gray-50"
      placeholder="Enter system prompt…"
      rows={10}
    />

    {/* --- NEW: Static Sensor Button Grid --- */}
    <div className="mt-4">
        <label className="block text-xs text-gray-500 mb-2 font-medium">INSERT SENSOR:</label>
        <div className="flex flex-wrap gap-2">
            <SensorButton icon={ScanText} label="Screen Text" onClick={() => insertSystemPromptText('$SCREEN_OCR')} colorClass="text-blue-600" />
            <SensorButton icon={Monitor} label="Screen Image" onClick={() => insertSystemPromptText('$SCREEN')} colorClass="text-purple-600" />
            <SensorButton icon={Camera} label="Camera" onClick={() => insertSystemPromptText('$CAMERA')} colorClass="text-purple-600" />
            <SensorButton icon={ScanText} label="Camera Text" onClick={() => insertSystemPromptText('$CAMERA_OCR')} colorClass="text-blue-600" />
            <SensorButton icon={Clipboard} label="Clipboard" onClick={() => insertSystemPromptText('$CLIPBOARD')} colorClass="text-sky-600" />
            <SensorButton icon={Mic} label="Microphone" onClick={() => insertSystemPromptText('$MICROPHONE')} colorClass="text-amber-600" />
            <SensorButton icon={Volume2} label="Screen Audio" onClick={() => insertSystemPromptText('$SCREEN_AUDIO')} colorClass="text-amber-600" />
            <SensorButton icon={Blend} label="All Audio" onClick={() => insertSystemPromptText('$ALL_AUDIO')} colorClass="text-orange-600" />

            <SensorDropdownButton
              icon={Save}
              label="Memory"
              colorClass="text-emerald-600"
              agents={availableAgentsForBlocks}
              onSelect={(agentId) => insertSystemPromptText(`$MEMORY@${agentId}`)}
            />

            <SensorDropdownButton
              icon={Images}
              label="Image Memory"
              colorClass="text-purple-600"
              agents={availableAgentsForBlocks}
              onSelect={(agentId) => insertSystemPromptText(`$IMEMORY@${agentId}`)}
            />

            <SensorDropdownButton
              icon={ScanText}
              label="Image Memory Text"
              colorClass="text-blue-600"
              agents={availableAgentsForBlocks}
              onSelect={(agentId) => insertSystemPromptText(`$IMEMORY_OCR@${agentId}`)}
            />
        </div>
    </div>

    {visionValidationError && (
        <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded-md flex items-center text-sm">
            <AlertTriangle className="h-5 w-5 text-yellow-500 mr-2 flex-shrink-0" />
            <p className="text-yellow-700">{visionValidationError}</p>
        </div>
    )}
  </div>
);



// --- Draggable Reference Popup Component ---
const DraggableReferencePopup: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const [position, setPosition] = useState({ x: 100, y: 100 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const popupRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('.popup-header')) {
      setIsDragging(true);
      setDragOffset({
        x: e.clientX - position.x,
        y: e.clientY - position.y
      });
    }
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        setPosition({
          x: e.clientX - dragOffset.x,
          y: e.clientY - dragOffset.y
        });
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, dragOffset]);

  return (
    <div
      ref={popupRef}
      className="fixed bg-white rounded-lg shadow-2xl border-2 border-blue-400 z-50 max-w-md"
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
        cursor: isDragging ? 'grabbing' : 'default'
      }}
      onMouseDown={handleMouseDown}
    >
      <div className="popup-header bg-gradient-to-r from-blue-600 to-blue-700 text-white px-4 py-2 rounded-t-lg flex justify-between items-center cursor-grab active:cursor-grabbing">
        <h4 className="font-semibold text-sm">Tools Reference</h4>
        <button
          onClick={onClose}
          className="p-1 rounded-full hover:bg-white/20 transition-colors"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="p-4 text-xs text-gray-600 max-h-96 overflow-y-auto">
        <p className="mb-3 font-medium text-gray-700">
          This code does what you want with the <code className="bg-blue-100 px-1 rounded">response</code> variable. You can parse it, log it, send it, whatever you want.
        </p>

        <div className="space-y-3">
          <div>
            <p className="font-semibold text-gray-800 mb-1.5">Notification Tools:</p>
            <ul className="space-y-0.5 pl-3">
              <li>• <code>sendEmail(email, message, images?)</code></li>
              <li>• <code>sendDiscord(discord_webhook, message, images?)</code></li>
              <li>• <code>sendTelegram(chat_id, message, images?)</code></li>
              <li>• <code>sendWhatsapp(phone_number, message)</code></li>
              <li>• <code>sendPushover(user_token, message, images?, title?)</code></li>
              <li>• <code>sendSms(phone_number, message, images?)</code></li>
              <li>• <code>notify(title, options)</code></li>
            </ul>
          </div>

          <div>
            <p className="font-semibold text-gray-800 mb-1.5">Agent Tools:</p>
            <ul className="space-y-0.5 pl-3">
              <li>• <code>getMemory(agentId)*</code> - Retrieve memory</li>
              <li>• <code>setMemory(agentId, content)*</code> - Replace memory</li>
              <li>• <code>appendMemory(agentId, content)*</code> - Add to memory</li>
              <li>• <code>getImageMemory(agentId)*</code> - Retrieve images</li>
              <li>• <code>setImageMemory(agentId, images)</code> - Set images</li>
              <li>• <code>appendImageMemory(agentId, images)</code> - Add images</li>
              <li>• <code>startAgent(agentId)*</code> - Start an agent</li>
              <li>• <code>stopAgent(agentId)*</code> - Stop an agent</li>
              <li>• <code>time()</code> - Get current time</li>
              <li>• <code>sleep(ms)</code> - Wait milliseconds</li>
            </ul>
            <p className="text-xs text-gray-500 mt-1.5 italic pl-3">
              * If you omit agentId, defaults to current agent
            </p>
          </div>

          <div>
            <p className="font-semibold text-gray-800 mb-1.5">Video Recording Tools:</p>
            <ul className="space-y-0.5 pl-3">
              <li>• <code>startClip()</code> - Start recording</li>
              <li>• <code>stopClip()</code> - Stop recording</li>
              <li>• <code>markClip(label)</code> - Add label to recording</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

// --- Code Editor Content Component ---
interface CodeEditorContentProps {
  isPythonMode: boolean;
  setIsPythonMode: (isPython: boolean) => void;
  editorIsLoaded: boolean;
  agentCode: string;
  setAgentCode: (code: string) => void;
}
const CodeEditorContent: React.FC<CodeEditorContentProps> = ({
  isPythonMode, setIsPythonMode,
  editorIsLoaded, agentCode, setAgentCode
}) => {
  const [showReference, setShowReference] = useState(false);

  return (
    <div className="p-4 bg-white rounded-lg border border-gray-200 shadow-sm flex-grow flex flex-col h-full">
      <div className="flex justify-between items-start mb-2">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold text-blue-700 mt-0.5">Agent Code</h3>
            <button
              onClick={() => setShowReference(!showReference)}
              className="p-1 rounded-full hover:bg-gray-100 text-blue-600 transition-colors"
              title="Show tools reference"
            >
              <Info className="h-5 w-5" />
            </button>
          </div>
          <div className="flex border border-gray-300 rounded-md overflow-hidden text-sm">
            <button onClick={() => setIsPythonMode(false)} className={`px-3 py-1 ${!isPythonMode ? 'bg-yellow-500 text-white' : 'bg-gray-50 text-gray-600'}`}>JS</button>
            <button onClick={() => setIsPythonMode(true)} className={`px-3 py-1 ${isPythonMode ? 'bg-blue-500 text-white' : 'bg-gray-50 text-gray-600'}`}>Py</button>
          </div>
      </div>

      {showReference && <DraggableReferencePopup onClose={() => setShowReference(false)} />}

      <div className="flex-grow border border-gray-300 rounded-md overflow-hidden relative">
           <Suspense fallback={<div className="absolute inset-0 flex items-center justify-center bg-gray-100 text-gray-500 text-sm">Loading editor…</div>}>
              {editorIsLoaded && (
                  <LazyCodeMirror value={agentCode} height="100%" className="h-full" theme={vscodeDark} extensions={[isPythonMode ? python() : javascript()]} onChange={(v) => setAgentCode(v)} basicSetup={{ lineNumbers: true, foldGutter: true, highlightActiveLine: true, autocompletion: true, bracketMatching: true, closeBrackets: true }} />
              )}
           </Suspense>
      </div>
    </div>
  );
};


/* ───────────────────────── MAIN COMPONENT ───────────────────────── */
const EditAgentModal: React.FC<EditAgentModalProps> = ({
  isOpen,
  onClose,
  createMode,
  agent,
  code: existingCode,
  onSave,
  onImportComplete,
  setError,
  getToken,
  isProUser = false,
}) => {
  const logic = useEditAgentModalLogic({
    isOpen, onClose, createMode, agent, code: existingCode, onSave, getToken
  });

  const [editorIsLoaded, setEditorIsLoaded] = useState(false);
  const [importStatus, setImportStatus] = useState<{ inProgress: boolean; results: {filename: string; success: boolean; agent?: CompleteAgent; error?: string;}[] }>({
    inProgress: false,
    results: []
  });
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [activeTab, setActiveTab] = useState<'config' | 'prompt' | 'code'>('config');

  useEffect(() => {
    if (isOpen && !editorIsLoaded) {
      import('@uiw/react-codemirror').then(() => setEditorIsLoaded(true));
    }
    if (!isOpen && editorIsLoaded) {
      setEditorIsLoaded(false);
    }
  }, [isOpen, editorIsLoaded]);

  const handleImportClick = () => {
    setImportStatus({ inProgress: false, results: [] });
    if (fileInputRef.current) fileInputRef.current.click();
    Logger.info('APP', 'Opening file selector for agent import');
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    Logger.info('APP', `Selected ${files.length} file(s) for import`);
    setImportStatus({ inProgress: true, results: [] });

    try {
      if (setError) setError(null);
      const results = await importAgentsFromFiles(Array.from(files));
      setImportStatus({ inProgress: false, results });

      const successCount = results.filter(r => r.success).length;
      if (successCount > 0 && onImportComplete) await onImportComplete();

      const failedImports = results.filter(r => !r.success);
      if (failedImports.length > 0 && setError) {
        setError(`Failed to import ${failedImports.length} agent(s): ${failedImports.map(r => `${r.filename}: ${r.error}`).join('; ')}`);
      }
    } catch (err) {
      const error = err as Error;
      if (setError) setError(`Import failed: ${error.message || 'Unknown error'}`);
      setImportStatus({ inProgress: false, results: [] });
    }

    if (fileInputRef.current) fileInputRef.current.value = '';
  };


  if (!isOpen) return null;

  /* ───────────────────────── RENDER LOGIC ───────────────────────── */

  return (
    <>
      <input type="file" ref={fileInputRef} onChange={handleFileSelect} accept=".yaml" multiple className="hidden" />
      <Modal
        open={isOpen}
        onClose={onClose}
        className="w-full max-w-7xl h-full md:max-h-[95vh] flex flex-col overflow-hidden"
      >
        {/* --- HEADER --- */}
        <div className="flex-shrink-0 flex justify-between items-center p-4 border-b border-gray-200 bg-gradient-to-r from-blue-600 to-blue-700 text-white">
          <h2 className="text-xl font-semibold truncate pr-4">
            {createMode ? 'Create New Agent' : <>Edit Agent: <span className="font-normal">{logic.name || 'Unnamed'}</span></>}
          </h2>
          <div className="flex items-center space-x-3">
            {!createMode && logic.agentId && (
              <button onClick={logic.handleExport} className="flex items-center space-x-1.5 px-3 py-1.5 bg-blue-500 hover:bg-blue-400 text-white rounded-md text-xs" title="Export">
                <Download className="h-4 w-4" />
                <span>Export</span>
              </button>
            )}
            {createMode && (
              <button onClick={handleImportClick} className="flex items-center space-x-1.5 px-3 py-1.5 bg-blue-500 hover:bg-blue-400 text-white rounded-md text-xs" disabled={importStatus.inProgress}>
                <FileUp className="h-4 w-4" />
                <span>{importStatus.inProgress ? 'Importing...' : 'Import Agents'}</span>
              </button>
            )}
            <button onClick={onClose} className="p-1.5 rounded-full hover:bg-blue-700 hover:bg-opacity-50 text-indigo-100 hover:text-white">
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* --- IMPORT RESULTS --- */}
        {importStatus.results.length > 0 && (
          <div className="flex-shrink-0 p-4 bg-blue-50">
            <h3 className="font-medium mb-2">Import Results:</h3>
            <ul className="list-disc pl-5">
              {importStatus.results.map((result, index) => (
                <li key={index} className={result.success ? 'text-green-600' : 'text-red-600'}>
                  {result.filename}: {result.success ? 'Success' : `Failed - ${result.error}`}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* --- MOBILE TAB NAV --- */}
        <MobileTabNav activeTab={activeTab} setActiveTab={setActiveTab} />

        <div className="flex-grow min-h-0 bg-gray-50 overflow-y-auto">
          <>
              {/* --- DESKTOP LAYOUT --- */}
              <div className="hidden md:flex flex-row flex-grow h-full">
                {/* Left Half: System Prompt (Full Height) */}
                <div className="w-1/2 p-5 flex flex-col overflow-y-auto border-r border-gray-200">
                  <PromptContent
                    systemPromptRef={logic.systemPromptRef}
                    systemPrompt={logic.systemPrompt} setSystemPrompt={logic.setSystemPrompt}
                    insertSystemPromptText={logic.insertSystemPromptText}
                    availableAgentsForBlocks={logic.availableAgentsForBlocks}
                    visionValidationError={logic.visionValidationError}
                  />
                </div>

                {/* Right Half: Config (Top) + Code (Bottom) */}
                <div className="w-1/2 p-5 flex flex-col overflow-y-auto space-y-5">
                  <div className="flex-shrink-0">
                    <ConfigContent
                      name={logic.name} setName={logic.setName}
                      agentId={logic.agentId} setAgentId={logic.setAgentId}
                      createMode={createMode}
                      currentModel={logic.currentModel} setCurrentModel={logic.setCurrentModel}
                      isModelDropdownOpen={logic.isModelDropdownOpen} setIsModelDropdownOpen={logic.setIsModelDropdownOpen}
                      loadingModels={logic.loadingModels} modelsError={logic.modelsError}
                      availableModels={logic.availableModels}
                      loopInterval={logic.loopInterval} setLoopInterval={logic.setLoopInterval}
                      onlyOnSignificantChange={logic.onlyOnSignificantChange} setOnlyOnSignificantChange={logic.setOnlyOnSignificantChange}
                      description={logic.description} setDescription={logic.setDescription}
                      isProUser={isProUser}
                    />
                  </div>
                  <div className="flex-grow min-h-0">
                    <CodeEditorContent
                      isPythonMode={logic.isPythonMode} setIsPythonMode={logic.setIsPythonMode}
                      editorIsLoaded={editorIsLoaded}
                      agentCode={logic.agentCode} setAgentCode={logic.setAgentCode}
                    />
                  </div>
                </div>
              </div>

              {/* --- MOBILE LAYOUT (Tab Content) --- */}
              <div className="md:hidden p-4 space-y-4">
                {activeTab === 'config' &&
                  <ConfigContent
                    name={logic.name} setName={logic.setName}
                    agentId={logic.agentId} setAgentId={logic.setAgentId}
                    createMode={createMode}
                    currentModel={logic.currentModel} setCurrentModel={logic.setCurrentModel}
                    isModelDropdownOpen={logic.isModelDropdownOpen} setIsModelDropdownOpen={logic.setIsModelDropdownOpen}
                    loadingModels={logic.loadingModels} modelsError={logic.modelsError}
                    availableModels={logic.availableModels}
                    loopInterval={logic.loopInterval} setLoopInterval={logic.setLoopInterval}
                    onlyOnSignificantChange={logic.onlyOnSignificantChange} setOnlyOnSignificantChange={logic.setOnlyOnSignificantChange}
                    description={logic.description} setDescription={logic.setDescription}
                    isProUser={isProUser}
                  />
                }

                {activeTab === 'prompt' && <div className="h-full flex flex-col">
                  <PromptContent
                    systemPromptRef={logic.systemPromptRef}
                    systemPrompt={logic.systemPrompt} setSystemPrompt={logic.setSystemPrompt}
                    insertSystemPromptText={logic.insertSystemPromptText}
                    availableAgentsForBlocks={logic.availableAgentsForBlocks}
                    visionValidationError={logic.visionValidationError}
                  />
                </div>}

                {activeTab === 'code' && (
                    <CodeEditorContent
                      isPythonMode={logic.isPythonMode} setIsPythonMode={logic.setIsPythonMode}
                      editorIsLoaded={editorIsLoaded}
                      agentCode={logic.agentCode} setAgentCode={logic.setAgentCode}
                    />
                )}
              </div>
            </>
        </div>

        {/* --- FOOTER --- */}
        <div className="flex-shrink-0 flex justify-end space-x-3 p-4 border-t border-gray-200 bg-gray-50">
          <button onClick={onClose} className="px-4 py-2 bg-white border border-gray-300 rounded-md text-sm text-gray-700 hover:bg-gray-50">Cancel</button>
          <button onClick={logic.handleSave} disabled={(createMode && !logic.agentId) || !logic.name || !logic.currentModel} className="px-5 py-2 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700 disabled:opacity-60">{createMode ? 'Create Agent' : 'Save Changes'}</button>
        </div>
      </Modal>

    </>
  );
};

export default EditAgentModal;
