// src/utils/agentCapabilities.ts

import React from 'react';
import * as customIcons from './icons';

// --- Type Definitions ---
export interface DetectedSensor {
    key: string;
    label: string;
    icon: React.ElementType;
}

export interface DetectedTool {
    key: string;
    label: string;
    icon: React.ElementType;
    warning?: string;
    isBlocking?: boolean;
}

export interface AgentCapabilities {
    sensors: DetectedSensor[];
    tools: DetectedTool[];
}

// --- Configuration Objects ---

// Dynamic icon loaders
const loadLucideIcon = async (iconName: string) => {
    const icons = await import('lucide-react');
    return (icons as any)[iconName];
};

const loadCustomIcon = async (iconName: string) => {
    return (customIcons as any)[iconName];
};

export const SENSOR_CONFIG = {
    SCREEN_OCR: { label: 'Screen OCR', iconName: 'ScanText' },
    SCREEN_64: { label: 'Screen', iconName: 'Monitor' },
    CAMERA: { label: 'Camera', iconName: 'Camera' },
    MEMORY: { label: 'Memory', iconName: 'Save' },
    IMEMORY: { label: 'Memory', iconName: 'Images' },
    CLIPBOARD: { label: 'Clipboard', iconName: 'Clipboard' },
    MICROPHONE: { label: 'Microphone', iconName: 'Mic' },
    SCREEN_AUDIO: { label: 'Screen Audio', iconName: 'Volume2' },
    ALL_AUDIO: { label: 'All Audio', iconName: 'Blend' },
} as const;

interface ToolConfigEntry {
    label: string;
    iconName: string;
    iconType: 'lucide' | 'custom';
    regex: RegExp;
    warning?: string;
}

export const TOOL_CONFIG: Record<string, ToolConfigEntry> = {
    notify: { label: 'Browser Notification', iconName: 'Bell', iconType: 'lucide', regex: /\bnotify\s*\(/g, warning: 'Browser notifications are unreliable, preferably use system_notify ' },
    getMemory: { label: 'Get Memory', iconName: 'Save', iconType: 'lucide', regex: /getMemory\s*\(/g },
    setMemory: { label: 'Set Memory', iconName: 'SquarePen', iconType: 'lucide', regex: /setMemory\s*\(/g },
    appendMemory: { label: 'Append Memory', iconName: 'SquarePen', iconType: 'lucide', regex: /appendMemory\s*\(/g },
    startAgent: { label: 'Start Agent', iconName: 'PlayCircle', iconType: 'lucide', regex: /startAgent\s*\(/g },
    stopAgent: { label: 'Stop Agent', iconName: 'StopCircle', iconType: 'lucide', regex: /stopAgent\s*\(/g },
    time: { label: 'Get Time', iconName: 'Hourglass', iconType: 'lucide', regex: /time\s*\(/g },
    sendEmail: { label: 'Send Email', iconName: 'Mail', iconType: 'lucide', regex: /sendEmail\s*\(/g },
    sendPushover: { label: 'Pushover', iconName: 'Bell', iconType: 'lucide', regex: /sendPushover\s*\(/g },
    sendDiscordBot: { label: 'Discord Bot', iconName: 'DiscordIcon', iconType: 'custom', regex: /sendDiscord\s*\(/g },
    sendWhatsapp: { label: 'WhatsApp', iconName: 'WhatsAppIcon', iconType: 'custom', regex: /sendWhatsapp\s*\(/g, warning: 'To receive messages, you must first message: +1 (555) 783-4727.' },
    sendSms: { label: 'SMS', iconName: 'MessageSquarePlus', iconType: 'lucide', regex: /sendSms\s*\(/g, warning: 'Delivery to US/Canada is unreliable. Use email for now.' },
    sendTelegram: { label: 'Telegram', iconName: 'MessageCircle', iconType: 'lucide', regex: /sendTelegram\s*\(/g },
    startClip: { label: 'Start Clip', iconName: 'Video', iconType: 'lucide', regex: /startClip\s*\(/g },
    stopClip: { label: 'Stop Clip', iconName: 'VideoOff', iconType: 'lucide', regex: /stopClip\s*\(/g },
    markClip: { label: 'Mark Clip', iconName: 'Tag', iconType: 'lucide', regex: /markClip\s*\(/g },
    ask: { label: 'Ask Dialog', iconName: 'MessageSquareQuote', iconType: 'lucide', regex: /ask\s*\(/g },
    message: { label: 'Message Dialog', iconName: 'MessageSquare', iconType: 'lucide', regex: /message\s*\(/g },
    system_notify: { label: 'Sys Notify', iconName: 'Bell', iconType: 'lucide', regex: /system_notify\s*\(/g },
    overlay: { label: 'Overlay', iconName: 'Monitor', iconType: 'lucide', regex: /overlay\s*\(/g },
    click: { label: 'Mouse Click', iconName: 'MousePointer', iconType: 'lucide', regex: /\bclick\s*\(/g, warning: 'Position mouse before agent runs' },
    call: { label: 'Phone Call', iconName: 'Phone', iconType: 'lucide', regex: /\bcall\s*\(/g },
    celebrate: { label: 'Celebrate', iconName: 'PartyPopper', iconType: 'lucide', regex: /\bcelebrate\s*\(/g },
};

// --- Icon Loading Helpers ---

/**
 * Loads an icon component dynamically
 */
export async function loadIcon(iconName: string, iconType: 'lucide' | 'custom'): Promise<React.ElementType> {
    if (iconType === 'custom') {
        return loadCustomIcon(iconName);
    } else {
        return loadLucideIcon(iconName);
    }
}

/**
 * Loads sensor icon by sensor key
 */
export async function loadSensorIcon(sensorKey: string): Promise<React.ElementType> {
    const config = (SENSOR_CONFIG as any)[sensorKey];
    if (!config) throw new Error(`Unknown sensor: ${sensorKey}`);
    return loadLucideIcon(config.iconName);
}

/**
 * Loads tool icon by tool key
 */
export async function loadToolIcon(toolKey: string): Promise<React.ElementType> {
    const config = TOOL_CONFIG[toolKey];
    if (!config) throw new Error(`Unknown tool: ${toolKey}`);
    return loadIcon(config.iconName, config.iconType);
}

// --- Detection Functions ---

/**
 * Detects which sensors an agent uses based on its system prompt
 */
export async function detectAgentSensors(systemPrompt: string): Promise<DetectedSensor[]> {
    const foundSensors: DetectedSensor[] = [];

    for (const [key, config] of Object.entries(SENSOR_CONFIG)) {
        if (systemPrompt.includes(`$${key}`)) {
            const icon = await loadSensorIcon(key);
            foundSensors.push({
                key,
                label: config.label,
                icon
            });
        }
    }

    return foundSensors;
}

/**
 * Detects which tools an agent uses based on its code
 */
export async function detectAgentTools(code: string, hostingContext?: 'official-web' | 'self-hosted' | 'tauri'): Promise<DetectedTool[]> {
    const foundTools: DetectedTool[] = [];

    // Tools that don't work in official web environment
    const webIncompatibleTools = ['overlay', 'message', 'ask', 'system_notify', 'click'];

    for (const [key, tool] of Object.entries(TOOL_CONFIG)) {
        if (code.match(tool.regex)) {
            const icon = await loadToolIcon(key);
            let warning = tool.warning;

            // Add web incompatibility warning if in official web context
            let isBlocking = false;
            if (hostingContext === 'official-web' && webIncompatibleTools.includes(key)) {
                warning = 'These tools are only available when using the Observer App';
                isBlocking = true;
            }

            foundTools.push({
                key,
                label: tool.label,
                icon,
                warning,
                isBlocking
            });
        }
    }

    return foundTools;
}

/**
 * Detects all capabilities of an agent (sensors + tools)
 */
export async function detectAgentCapabilities(systemPrompt: string, code: string, hostingContext?: 'official-web' | 'self-hosted' | 'tauri'): Promise<AgentCapabilities> {
    const [sensors, tools] = await Promise.all([
        detectAgentSensors(systemPrompt),
        detectAgentTools(code, hostingContext)
    ]);

    return {
        sensors,
        tools
    };
}

/**
 * Checks if agent has a specific sensor capability
 */
export function agentHasSensor(systemPrompt: string, sensorKey: string): boolean {
    return systemPrompt.includes(`$${sensorKey}`);
}

/**
 * Checks if agent has any screen-related sensors (SCREEN_OCR or SCREEN_64)
 */
export function agentHasScreenSensor(systemPrompt: string): boolean {
    return agentHasSensor(systemPrompt, 'SCREEN_OCR') || agentHasSensor(systemPrompt, 'SCREEN_64');
}

/**
 * Checks if agent has camera sensor
 */
export function agentHasCameraSensor(systemPrompt: string): boolean {
    return agentHasSensor(systemPrompt, 'CAMERA');
}

/**
 * Checks if agent has any audio-related sensors
 */
export function agentHasAudioSensor(systemPrompt: string): boolean {
    return agentHasSensor(systemPrompt, 'MICROPHONE') ||
           agentHasSensor(systemPrompt, 'SCREEN_AUDIO') ||
           agentHasSensor(systemPrompt, 'ALL_AUDIO');
}
