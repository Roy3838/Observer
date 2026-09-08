// src/utils/agentTemplateManager.ts

import { CompleteAgent } from './agent_database';

export type SimpleTool = 'notification' | 'memory' | 'sms' | 'email' | 'whatsapp' | 'call' | 'start_clip' | 'mark_clip' | 'pushover' | 'discord' | 'telegram' | 'ask' | 'system_notify' | 'message' | 'overlay' | 'click' | 'celebrate' | 'sound';

export interface ToolData {
  smsPhoneNumber?: string;
  emailAddress?: string;
  whatsappPhoneNumber?: string;
  phoneNumber?: string;
  pushoverUserKey?: string;
  discordWebhookUrl?: string;
  telegramChatId?: string;
}

interface SensorContext {
  hasScreen: boolean;
  hasCamera: boolean;
}

function buildImageArg(ctx: SensorContext): string | null {
  const vars: string[] = [];
  if (ctx.hasScreen) vars.push('screen');
  if (ctx.hasCamera) vars.push('camera');
  if (vars.length === 0) return null;
  return vars.length === 1 ? vars[0] : `[${vars.join(', ')}]`;
}

const TOOL_CODE_SNIPPETS: Record<SimpleTool, (data: ToolData, ctx: SensorContext) => string> = {
  notification: () => `
// --- NOTIFICATION TOOL ---
// Sends the model's entire response as a desktop notification.
notify("Observer AI Agent", response);
`,
  memory: () => `
// --- MEMORY TOOL ---
// Appends the model's response to this agent's memory, with a timestamp.
const timestamp = time();
appendMemory(agentId, \`\\n[\${timestamp}] \${response}\`);
`,
  sms: (data: ToolData, ctx: SensorContext) => {
    const phoneNumber = data.smsPhoneNumber ? JSON.stringify(data.smsPhoneNumber) : '""';
    const imgArg = buildImageArg(ctx);
    const imgPart = imgArg ? `, ${imgArg}` : '';
    return `
// --- SMS TOOL ---
// Sends the model's response as an SMS to the specified number.
sendSms(${phoneNumber}, response${imgPart});
`;
  },
  whatsapp: (data: ToolData, ctx: SensorContext) => {
    const phoneNumber = data.whatsappPhoneNumber ? JSON.stringify(data.whatsappPhoneNumber) : '""';
    const imgArg = buildImageArg(ctx);
    const imgPart = imgArg ? `, ${imgArg}` : '';
    return `
// --- WHATSAPP TOOL ---
// Sends a pre-approved WhatsApp notification. The content is static for now.
// IMPORTANT: The 'response' variable is currently ignored for anti-spam reasons.
sendWhatsapp(${phoneNumber}, response${imgPart});
`;
  },
  email: (data: ToolData, ctx: SensorContext) => {
    const emailAddr = data.emailAddress ? JSON.stringify(data.emailAddress) : '""';
    const imgArg = buildImageArg(ctx);
    const imgPart = imgArg ? `, ${imgArg}` : '';
    return `
// --- EMAIL TOOL ---
// Sends the model's response as an email to the specified address.
sendEmail(${emailAddr}, response${imgPart});
`;
  },
  pushover: (data: ToolData, ctx: SensorContext) => {
    const userKey = data.pushoverUserKey ? JSON.stringify(data.pushoverUserKey) : '""';
    const imgArg = buildImageArg(ctx);
    const imgPart = imgArg ? `, ${imgArg}` : '';
    return `
// --- PUSHOVER TOOL ---
// Sends the model's response as a Pushover notification.
sendPushover(${userKey}, response${imgPart});
`;
  },
  discord: (data: ToolData, ctx: SensorContext) => {
    const webhookUrl = data.discordWebhookUrl ? JSON.stringify(data.discordWebhookUrl) : '""';
    const imgArg = buildImageArg(ctx);
    const imgPart = imgArg ? `, ${imgArg}` : '';
    return `
// --- DISCORD TOOL ---
// Sends the model's response to a Discord channel via a webhook.
sendDiscord(${webhookUrl}, response${imgPart});
`;
  },
  telegram: (data: ToolData, ctx: SensorContext) => {
    const chatId = data.telegramChatId ? JSON.stringify(data.telegramChatId) : '""';
    const imgArg = buildImageArg(ctx);
    const imgPart = imgArg ? `, ${imgArg}` : '';
    return `
// --- TELEGRAM TOOL ---
// Sends the model's response to a Telegram chat.
sendTelegram(${chatId}, response${imgPart});
`;
  },
  start_clip: () => `
// --- START RECORDING TOOL ---
// Starts a video recording. The recording will stop when the agent is stopped
// or if you manually add a stopClip() call to the code.
startClip();
`,
  mark_clip: () => `
// --- LABEL RECORDING TOOL ---
// Adds the model's response as a label to an active recording.
// Prompt your model to output just the desired label text.
markClip(response);
`,
  ask: () => `
// --- ASK USER TOOL ---
// Shows a dialog asking for user confirmation.
ask(response);
`,
  system_notify: () => `
// --- SYSTEM NOTIFICATION TOOL ---
// Shows a native system notification.
system_notify(response, "Observer AI");
`,
  message: () => `
// --- MESSAGE TOOL ---
// Shows a dialog message to the user.
message(response);
`,
  overlay: () => `
// --- OVERLAY TOOL ---
// Displays a message in the translucent overlay window.
overlay(response);
`,
  click: () => `
// --- MOUSE CLICK TOOL ---
// Triggers a mouse click at the current cursor position.
// IMPORTANT: Position the mouse before the agent runs.
click();
`,

  celebrate: () => `
// --- CELEBRATE TOOL ---
// Triggers a celebration animation in the Observer UI.
celebrate();
`,

  sound: () => `
// --- SOUND TOOL ---
// Plays a notification sound. Works on web and in the app.
sound();
`,
  call: (data: ToolData) => {
    const phoneNumber = data.phoneNumber ? JSON.stringify(data.phoneNumber) : '""';
    return `
// --- PHONE CALL TOOL ---
// Makes an automated phone call with the model's response.
call(${phoneNumber}, response);
`;
  },
};

interface SimpleConfig {
  agentData: Partial<CompleteAgent>;
  selectedTools: Map<SimpleTool, ToolData>;
  condition: {
    enabled: boolean;
    keyword: string;
  };
}

export function generateAgentFromSimpleConfig(
  config: SimpleConfig
): { agent: CompleteAgent; code: string } {
  
  const comments = [
    '// This code was auto-generated by the Simple Agent Creator.',
    '// You can edit it to add more complex logic.',
  ].join('\n');

  const systemPrompt = config.agentData.system_prompt || '';
  const ctx: SensorContext = {
    hasScreen: /\$SCREEN(?![A-Z_])/.test(systemPrompt),
    hasCamera: /\$CAMERA(?![A-Z_])/.test(systemPrompt),
  };

  let toolCode = Array.from(config.selectedTools.entries())
    .map(([tool, data]) => TOOL_CODE_SNIPPETS[tool](data, ctx).trim())
    .join('\n\n');

  if (config.condition.enabled && config.condition.keyword) {
    const safeKeyword = JSON.stringify(config.condition.keyword.toLowerCase()).slice(1, -1);
    const indentedToolCode = `  ${toolCode.replace(/\n/g, '\n  ')}`;
    toolCode = `if (response.toLowerCase().includes('${safeKeyword}')) {\n${indentedToolCode}\n}`;
  }

  const finalCode = `${comments}\n\n${toolCode}`;

  const agent: CompleteAgent = {
    id: config.agentData.id || `agent_${Date.now()}`,
    name: config.agentData.name || 'My New Agent',
    description: config.agentData.description || 'An agent created with the Simple Creator.',
    model_name: config.agentData.model_name || '',
    system_prompt: config.agentData.system_prompt || '',
    loop_interval_seconds: config.agentData.loop_interval_seconds || 60,
  };

  return {
    agent,
    code: finalCode,
  };
}
