// components/AgentCard/AgentCardFooter.tsx
import React from 'react';
import { Edit, Trash2, Terminal, Sparkles, Minus } from 'lucide-react';

interface AgentCardFooterProps {
    agentId: string;
    isPythonAgent: boolean;
    isJupyterConnected: boolean;
    isMemoryFlashing: boolean;
    onEdit: (agentId: string) => void;
    onDelete: (agentId: string) => void;
    onMemory: (agentId: string) => void;
    onActivity: (agentId: string) => void;
    onShowJupyterModal: () => void;
    onAIEdit?: (agentId: string) => void;
    onMinimize: () => void;
}

const AgentCardFooter: React.FC<AgentCardFooterProps> = ({
    agentId, isPythonAgent, isJupyterConnected, isMemoryFlashing,
    onEdit, onDelete, onMemory, onActivity, onShowJupyterModal, onAIEdit, onMinimize
}) => {
    return (
        <div className="border-t border-gray-200 bg-gray-50 px-4 py-2 flex justify-between items-center">
            {/* Left Side: Delete + Minimize */}
            <div className="flex items-center gap-2">
                <button onClick={() => onDelete(agentId)} className="flex items-center justify-center p-2 text-red-600 hover:bg-red-100 rounded-md" title="Delete"><Trash2 className="w-4 h-4" /></button>
                <button onClick={onMinimize} data-tutorial-minimize-button={agentId} className="flex items-center justify-center p-2 text-gray-400 hover:bg-gray-200 hover:text-gray-600 rounded-md" title="Minimize to tray"><Minus className="w-4 h-4" /></button>
            </div>

            {/* Center: Memory and Activity */}
            <div className="flex items-center gap-2">
                 {isPythonAgent ? (
                    <button onClick={onShowJupyterModal} className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md ${isJupyterConnected ? 'text-green-700 bg-green-50' : 'text-red-700 bg-red-50 hover:bg-red-100'}`}>
                        <Terminal className="w-4 h-4" /> Jupyter
                    </button>
                ) : (
                    <button onClick={() => onMemory(agentId)} className={`flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-200 rounded-md ${isMemoryFlashing ? 'animate-pulse' : ''}`}>
                        Memory
                    </button>
                )}
                <button onClick={() => onActivity(agentId)} className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-200 rounded-md">
                    Activity
                </button>
            </div>

            {/* Right Side: Edit and AI Edit */}
            <div className="flex items-center gap-2">
                <button onClick={() => onEdit(agentId)} className="flex items-center justify-center p-2 text-gray-600 hover:bg-gray-200 rounded-md" title="Edit"><Edit className="w-4 h-4" /></button>
                {onAIEdit && (
                    <button onClick={() => onAIEdit(agentId)} className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-purple-600 hover:bg-purple-100 rounded-md"><Sparkles className="w-4 h-4" /> AI Edit</button>
                )}
            </div>
        </div>
    );
};

export default AgentCardFooter;
