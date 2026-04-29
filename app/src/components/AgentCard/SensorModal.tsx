import React, { useState, useEffect, useRef } from 'react';
import Modal from '@components/EditAgent/Modal';
import SensorInputText from '@components/EditAgent/SensorInputText';
import SensorPreviewPanel from './SensorPreviewPanel';
import { StreamManager, StreamState } from '@utils/streamManager';
import { listAgents } from '@utils/agent_database';
import { Eye, X, Monitor, ScanText, Camera, Clipboard, Mic, Volume2, Blend, Save, Images, ChevronDown, ChevronRight } from 'lucide-react';

interface SensorModalProps {
  isOpen: boolean;
  onClose: () => void;
  systemPrompt: string;
  agentName: string;
  agentId: string;
  onSystemPromptChange?: (newPrompt: string) => void;
}

// Sensor Button Helper Component
const SensorButton = ({ icon: Icon, label, colorClass, onClick }: { icon: React.ElementType, label: string, colorClass?: string, onClick: () => void }) => (
  <button onClick={onClick} className={`flex-grow md:flex-grow-0 flex items-center justify-center space-x-2 px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors ${colorClass || 'text-gray-700'}`}>
    <Icon className="h-5 w-5" />
    <span className="text-sm font-medium">{label}</span>
  </button>
);

// Sensor Dropdown Button Component
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
        <div className="absolute z-30 bottom-full mb-1 w-full min-w-[200px] max-h-60 bg-white border border-gray-300 rounded-md shadow-lg overflow-y-auto overscroll-contain" style={{ WebkitOverflowScrolling: 'touch' }}>
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

const SensorModal: React.FC<SensorModalProps> = ({ isOpen, onClose, systemPrompt, agentName, agentId, onSystemPromptChange }) => {
  const [streams, setStreams] = useState<StreamState>(() => StreamManager.getCurrentState());
  const [editedPrompt, setEditedPrompt] = useState(systemPrompt);
  const [availableAgents, setAvailableAgents] = useState<{ id: string; name: string }[]>([]);
  const [isPromptExpanded, setIsPromptExpanded] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Listen to stream state changes
  useEffect(() => {
    if (!isOpen) return;
    const handleStreamUpdate = (newState: StreamState) => setStreams(newState);
    StreamManager.addListener(handleStreamUpdate);
    return () => StreamManager.removeListener(handleStreamUpdate);
  }, [isOpen]);

  // Load available agents for memory sensor dropdowns
  useEffect(() => {
    if (!isOpen) return;
    const loadAgents = async () => {
      try {
        const agents = await listAgents();
        setAvailableAgents(agents.map(a => ({ id: a.id, name: a.name })));
      } catch (error) {
        console.error('Failed to load agents:', error);
      }
    };
    loadAgents();
  }, [isOpen]);

  // Reset edited prompt when modal opens or systemPrompt changes
  useEffect(() => {
    setEditedPrompt(systemPrompt);
  }, [systemPrompt, isOpen]);

  // Insert sensor variable at cursor position
  const insertSystemPromptText = (text: string) => {
    if (!textareaRef.current) return;
    const { selectionStart, selectionEnd, value } = textareaRef.current;
    const newPrompt = `${value.substring(0, selectionStart)} ${text} ${value.substring(selectionEnd)}`;
    setEditedPrompt(newPrompt);
    setTimeout(() => {
      textareaRef.current?.focus();
      const newPos = selectionStart + text.length + 2;
      textareaRef.current?.setSelectionRange(newPos, newPos);
    }, 0);
  };

  // Handle modal close with save
  const handleClose = () => {
    if (onSystemPromptChange && editedPrompt !== systemPrompt) {
      onSystemPromptChange(editedPrompt);
    }
    onClose();
  };

  if (!isOpen) return null;

  return (
    <Modal
      open={isOpen}
      onClose={handleClose}
      className="w-full max-w-6xl max-h-[90vh] flex flex-col"
    >
      {/* Header */}
      <div className="flex-shrink-0 flex justify-between items-center p-4 border-b border-gray-200 bg-gradient-to-r from-blue-600 to-blue-700 text-white">
        <div className="flex items-center space-x-3">
          <Eye className="h-6 w-6" />
          <div>
            <h2 className="text-xl font-semibold">System Prompt & Sensors</h2>
            <p className="text-sm text-blue-100">{agentName}</p>
          </div>
        </div>
        <button
          onClick={handleClose}
          className="p-1.5 rounded-full hover:bg-blue-700 hover:bg-opacity-50 text-blue-100 hover:text-white"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      {/* Content - Responsive Layout */}
      <div className="flex-grow p-4 md:p-6 overflow-y-auto bg-gray-50">
        <div className="grid md:grid-cols-2 gap-4 md:gap-6">
          {/* Mobile: Live Sensor Preview First */}
          <div className="md:order-2">
            <div className="mb-3">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Live Sensor Preview
              </label>
            </div>
            <SensorPreviewPanel
              agentId={agentId}
              streams={streams}
              systemPrompt={editedPrompt}
            />
          </div>

          {/* Mobile: Collapsible System Prompt / Desktop: Normal Left Column */}
          <div className="md:order-1">
            {/* Mobile: Expandable Header with Preview */}
            <button
              onClick={() => setIsPromptExpanded(!isPromptExpanded)}
              className="md:hidden w-full flex flex-col p-3 mb-3 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors text-left"
            >
              <div className="flex items-center justify-between w-full mb-2">
                <div className="flex items-center space-x-2">
                  <Eye className="h-4 w-4 text-gray-600" />
                  <span className="text-sm font-medium text-gray-700">System Prompt & Sensors</span>
                </div>
                <ChevronRight className={`h-5 w-5 text-gray-500 transition-transform flex-shrink-0 ${isPromptExpanded ? 'rotate-90' : ''}`} />
              </div>
              {!isPromptExpanded && (
                <div className="text-sm text-gray-600 italic line-clamp-2">
                  {editedPrompt || "No system prompt defined"}
                </div>
              )}
            </button>

            {/* Desktop: Always visible label */}
            <div className="hidden md:block mb-3">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                System Prompt with Sensors
              </label>
            </div>

            {/* Content (expanded on mobile, always visible on desktop) */}
            <div className={`${isPromptExpanded ? 'block' : 'hidden'} md:block`}>
              <SensorInputText
                value={editedPrompt}
                onChange={setEditedPrompt}
                textareaRef={textareaRef}
                className="h-64 md:h-96 bg-white"
                placeholder="No system prompt defined"
              />

              {/* Sensor Insertion Buttons */}
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
                    agents={availableAgents}
                    onSelect={(agentId) => insertSystemPromptText(`$MEMORY@${agentId}`)}
                  />

                  <SensorDropdownButton
                    icon={Images}
                    label="Image Memory"
                    colorClass="text-purple-600"
                    agents={availableAgents}
                    onSelect={(agentId) => insertSystemPromptText(`$IMEMORY@${agentId}`)}
                  />

                  <SensorDropdownButton
                    icon={ScanText}
                    label="Image Memory Text"
                    colorClass="text-blue-600"
                    agents={availableAgents}
                    onSelect={(agentId) => insertSystemPromptText(`$IMEMORY_OCR@${agentId}`)}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 flex justify-end p-4 border-t border-gray-200 bg-gray-50">
        <button
          onClick={handleClose}
          className="px-5 py-2 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700"
        >
          Close
        </button>
      </div>
    </Modal>
  );
};

export default SensorModal;
