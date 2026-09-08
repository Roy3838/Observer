// src/utils/handlers/utils.ts

import { Logger } from '../logging';
import { getAgentMemory as fetchAgentMemory, updateAgentMemory as saveAgentMemory, getAgentImageMemory as fetchAgentImageMemory, updateAgentImageMemory as saveAgentImageMemory, appendAgentImageMemory as addAgentImageMemory } from '../agent_database';
import { recordingManager } from '../recordingManager';
import { pauseAgentLoop } from '../main_loop';
import { platformFetch } from '../platform';

/**
 * Utility functions for handlers
 */

/**
 * Helper function to handle 429 quota exceeded responses.
 * Extracts quota type from error response and logs appropriately.
 * @param response The fetch response object to check
 * @param defaultQuotaType The default quota type to use if not specified in response
 * @throws Error if quota is exceeded (status 429)
 */
async function handleQuotaExceeded(response: Response, defaultQuotaType: string): Promise<void> {
  if (response.status !== 429) {
    return;
  }

  try {
    const errorBody = await response.json();
    const quotaType = errorBody?.detail?.quota_type || defaultQuotaType;
    const message = errorBody?.detail?.message || `${quotaType} quota exceeded`;

    Logger.error('quota', message, {
      logType: 'quota-exceeded',
      quotaType: quotaType,
      content: {
        quotaType,
        message,
        timestamp: new Date().toISOString()
      }
    });

    throw new Error(message);
  } catch (parseError) {
    if (parseError instanceof Error && parseError.message.includes('quota')) {
      throw parseError;
    }

    const defaultMessage = `${defaultQuotaType} quota exceeded`;
    Logger.error('quota', defaultMessage, {
      logType: 'quota-exceeded',
      quotaType: defaultQuotaType,
      content: {
        quotaType: defaultQuotaType,
        message: defaultMessage,
        timestamp: new Date().toISOString()
      }
    });
    throw new Error(defaultMessage);
  }
}

/**
 * Get the current time in a readable format
 */
export function time(): string {
  return new Date().toLocaleTimeString([], {
    hour: 'numeric', 
    minute: '2-digit',
    hour12: true
  }).toLowerCase();
}

/**
 * Get agent's memory value
 */
export async function getMemory(agentId: string): Promise<string> {
  return await fetchAgentMemory(agentId);
}

/**
 * Set agent's memory value
 */
export async function setMemory(agentId: string, memory: any): Promise<void> {
  await saveAgentMemory(agentId, memory);
}

/**
 * Append to agent's memory value
 */
export async function appendMemory(agentId: string, content: string, separator: string = '\n'): Promise<void> {
  const currentMemory = await fetchAgentMemory(agentId);
  const newMemory = currentMemory ? `${currentMemory}${separator}${content}` : content;
  await saveAgentMemory(agentId, newMemory);
}

/**
 * Get agent's image memory
 */
export async function getImageMemory(agentId: string): Promise<string[]> {
  return await fetchAgentImageMemory(agentId);
}

/**
 * Set agent's image memory (replaces entire array)
 */
export async function setImageMemory(agentId: string, images: string[]): Promise<void> {
  await saveAgentImageMemory(agentId, images);
}

/**
 * Append images to agent's image memory
 */
export async function appendImageMemory(agentId: string, images: string[]): Promise<void> {
  await addAgentImageMemory(agentId, images);
}

/**
 * Sends a notification
 */
export function notify(title: string, message: string): void {
  if (!("Notification" in window)) {
    console.error('Browser does not support notifications');
    return;
  }
  
  if (Notification.permission === "granted") {
    new Notification(title, { body: message });
  } else if (Notification.permission !== "denied") {
    Notification.requestPermission().then(permission => {
      if (permission === "granted") {
        new Notification(title, { body: message });
      }
    });
  }
}

// Bundled sounds. Adding a file to ./sounds/ is all it takes to add a new name:
// './sounds/airplane.mp3' becomes sound('airplane'). Vite emits each file as a
// separate hashed asset, so they cost nothing until one is actually played.
// Cast because tsconfig sets an explicit "types" array, which leaves out vite/client.
const SOUND_FILES = (import.meta as any).glob('./sounds/*.{mp3,ogg,wav}', {
  eager: true,
  query: '?url',
  import: 'default',
}) as Record<string, string>;

const SOUND_LIBRARY: Record<string, string> = Object.fromEntries(
  Object.entries(SOUND_FILES).map(([path, url]) => [
    path.split('/').pop()!.replace(/\.\w+$/, ''),
    url,
  ])
);

// Agents run on a loop, so a condition that stays true would otherwise fire every
// iteration. This only stops a burst; the real fix is stopAgent() after the sound.
const SOUND_COOLDOWN_MS = 3000;
let lastSoundPlayedAt = 0;

/**
 * Plays a notification sound in the browser. Works on both web and app.
 * @param name A bundled sound name (see ./sounds/), or a http(s)/data/blob/absolute URL.
 * @param volume Playback volume from 0 to 1. Defaults to 0.5.
 */
export async function sound(name: string = 'ping', volume: number = 0.5): Promise<void> {
  const now = Date.now();
  if (now - lastSoundPlayedAt < SOUND_COOLDOWN_MS) {
    Logger.debug('utils', `sound("${name}") skipped: within ${SOUND_COOLDOWN_MS}ms cooldown`);
    return;
  }
  lastSoundPlayedAt = now;

  const isUrl = /^(https?:|data:|blob:|\/)/.test(name);
  const src = SOUND_LIBRARY[name] ?? (isUrl ? name : null);
  if (!src) {
    throw new Error(
      `Unknown sound "${name}". Available: ${Object.keys(SOUND_LIBRARY).join(', ')} (or pass a URL).`
    );
  }

  const audio = new Audio(src);
  audio.volume = Math.min(Math.max(volume, 0), 1);

  try {
    await audio.play();
  } catch (error) {
    // Most likely the browser's autoplay policy: the page has had no user
    // interaction yet, so audible playback is blocked until it does.
    throw new Error(
      `Could not play sound "${name}": ${error instanceof Error ? error.message : String(error)}. ` +
      `If the browser blocked it, click anywhere on the Observer page and try again.`
    );
  }
}

/**
 * Sends an SMS message by calling the backend API.
 * This is the core utility function.
 * @param message The SMS message content to send.
 * @param number The phone number to send to.
 * @param authToken The authentication token for the Observer AI API.
 * @param images Optional array of base64-encoded images (without data:image prefix).
 * @param videos Optional array of base64-encoded videos (without data:video prefix).
 */
export async function sendSms(message: string, number: string, authToken: string, images?: string[], videos?: string[]): Promise<void> {
  const API_HOST = "https://api.observer-ai.com";

  if (!authToken) {
    throw new Error("Authentication error: Auth token is missing.");
  }

  try {
    const requestBody: { to_number: string; message: string; images?: string[]; videos?: string[] } = {
      to_number: number,
      message: message,
    };

    if (images && images.length > 0) {
      requestBody.images = images;
    }

    if (videos && videos.length > 0) {
      requestBody.videos = videos;
    }

    const response = await fetch(`${API_HOST}/tools/send-sms`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`,
      },
      body: JSON.stringify(requestBody),
    });

    await handleQuotaExceeded(response, 'sms');

    if (!response.ok) {
      // Check if this is a whitelist error (403 + user is authenticated)
      if (response.status === 403 && authToken) {
        Logger.error('whitelist', `Phone number ${number} not whitelisted for SMS`, {
          logType: 'whitelist-required',
          content: {
            phoneNumber: number,
            toolName: 'SMS',
            channel: 'sms',
            timestamp: new Date().toISOString()
          }
        });
        throw new Error(`Phone number ${number} is not whitelisted. Please verify your number first.`);
      }

      try {
        const errorData = await response.json();
        const errorMessage = typeof errorData.detail === 'string'
          ? errorData.detail
          : `Failed to send SMS: ${response.status} ${response.statusText}`;
        throw new Error(errorMessage);
      } catch (parseError) {
        // If JSON parsing fails, use the HTTP status
        throw new Error(`Failed to send SMS: ${response.status} ${response.statusText}`);
      }
    }
  } catch (error) {
    throw error;
  }
}

/**
 * Sends a WhatsApp notification using a pre-approved template.
 * @param message The WhatsApp message content to send.
 * @param number The phone number to send to.
 * @param authToken The authentication token for the Observer AI API.
 * @param images Optional array of base64-encoded images (without data:image prefix).
 * @param videos Optional array of base64-encoded videos (without data:video prefix).
 */
export async function sendWhatsapp(message: string, number: string, authToken: string, images?: string[], videos?: string[]): Promise<void> {
  const API_HOST = "https://api.observer-ai.com";

  if (!authToken) {
    throw new Error("Authentication error: Auth token is missing.");
  }

  try {
    const requestBody: { to_number: string; message: string; images?: string[]; videos?: string[] } = {
      to_number: number,
      message: message,
    };

    if (images && images.length > 0) {
      requestBody.images = images;
    }

    if (videos && videos.length > 0) {
      requestBody.videos = videos;
    }

    const response = await fetch(`${API_HOST}/tools/send-whatsapp`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`,
      },
      body: JSON.stringify(requestBody),
    });

    await handleQuotaExceeded(response, 'whatsapp');

    if (!response.ok) {
      // Check if this is a whitelist error (403 + user is authenticated)
      if (response.status === 403 && authToken) {
        Logger.error('whitelist', `Phone number ${number} not whitelisted for WhatsApp`, {
          logType: 'whitelist-required',
          content: {
            phoneNumber: number,
            toolName: 'WhatsApp',
            channel: 'whatsapp',
            timestamp: new Date().toISOString()
          }
        });
        throw new Error(`Phone number ${number} is not whitelisted. Please verify your number first.`);
      }

      try {
        const errorData = await response.json();
        const errorMessage = typeof errorData.detail === 'string'
          ? errorData.detail
          : `Failed to send WhatsApp message: ${response.status} ${response.statusText}`;
        throw new Error(errorMessage);
      } catch (parseError) {
        // If JSON parsing fails, use the HTTP status
        throw new Error(`Failed to send WhatsApp message: ${response.status} ${response.statusText}`);
      }
    }
  } catch (error) {
    throw error;
  }
}

/**
 * Sends an email by calling the backend API.
 * @param message The email message content.
 * @param emailAddress The recipient email address.
 * @param authToken The authentication token for the Observer AI API.
 * @param images Optional array of base64-encoded images (without data:image prefix).
 * @param videos Optional array of base64-encoded videos (without data:video prefix).
 */
export async function sendEmail(message: string, emailAddress: string, authToken: string, images?: string[], videos?: string[]): Promise<void> {
  const API_HOST = "https://api.observer-ai.com";

  if (!authToken) {
    throw new Error("Authentication error: Auth token is missing.");
  }

  try {
    const requestBody: { to_email: string; message: string; images?: string[]; videos?: string[] } = {
      to_email: emailAddress,
      message: message,
    };

    if (images && images.length > 0) {
      requestBody.images = images;
    }

    if (videos && videos.length > 0) {
      requestBody.videos = videos;
    }

    const response = await fetch(`${API_HOST}/tools/send-email`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`,
      },
      body: JSON.stringify(requestBody),
    });

    await handleQuotaExceeded(response, 'email');

    if (!response.ok) {
      try {
        const errorData = await response.json();
        const errorMessage = typeof errorData.detail === 'string'
          ? errorData.detail
          : `Failed to send email: ${response.status} ${response.statusText}`;
        throw new Error(errorMessage);
      } catch (parseError) {
        // If JSON parsing fails, use the HTTP status
        throw new Error(`Failed to send email: ${response.status} ${response.statusText}`);
      }
    }
  } catch (error) {
    throw error;
  }
}

/**
 * Starts a new global clip session.
 */
export async function startClip(): Promise<void> {
  await recordingManager.startClip();
}

/**
 * Stops the currently active global clip session.
 */
export async function stopClip(): Promise<void> {
  await recordingManager.stopClip();
}

/**
 * Marks a specific point in time with a label.
 */
export function markClip(label: string): void {
  if (!label || typeof label !== 'string') {
    console.warn('A valid string label must be provided.');
    return;
  }
  recordingManager.addMarker(label);
}

/**
 * Sends a Pushover notification by calling the backend API.
 * @param message The main content of the notification.
 * @param userKey The user's individual Pushover Key.
 * @param authToken The authentication token for the Observer AI API.
 * @param images Optional array of base64-encoded images (without data:image prefix).
 * @param title An optional title for the notification.
 * @param videos Optional array of base64-encoded videos (without data:video prefix).
 */
export async function sendPushover(message: string, userKey: string, authToken: string, images?: string[], title?: string, videos?: string[]): Promise<void> {
  const API_HOST = "https://api.observer-ai.com";

  if (!authToken) {
    throw new Error("Authentication error: Auth token is missing.");
  }

  if (!userKey) {
    throw new Error("Pushover user key is missing.");
  }

  try {
    const requestBody: { user_key: string; message: string; title?: string; images?: string[]; videos?: string[] } = {
      user_key: userKey, // Note: snake_case to match the Pydantic model on the backend
      message: message,
    };

    if (title) {
      requestBody.title = title;
    }

    if (images && images.length > 0) {
      requestBody.images = images;
    }

    if (videos && videos.length > 0) {
      requestBody.videos = videos;
    }

    const response = await fetch(`${API_HOST}/tools/send-pushover`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`,
      },
      body: JSON.stringify(requestBody),
    });

    await handleQuotaExceeded(response, 'pushover');

    if (!response.ok) {
      try {
        const errorData = await response.json();
        const errorMessage = typeof errorData.detail === 'string'
          ? errorData.detail
          : `Failed to send Pushover notification: ${response.status} ${response.statusText}`;
        throw new Error(errorMessage);
      } catch (parseError) {
        // If JSON parsing fails, use the HTTP status
        throw new Error(`Failed to send Pushover notification: ${response.status} ${response.statusText}`);
      }
    }
  } catch (error) {
    // Rethrow the error so the agent's execution log can catch it
    throw error;
  }
}

/**
 * Sends a Discord notification directly to a user-provided webhook.
 * @param message The main content of the notification.
 * @param webhookUrl The user's unique Discord Webhook URL.
 * @param images Optional array of base64-encoded images (without data:image prefix).
 * @param videos Optional array of base64-encoded videos (without data:video prefix).
 */
export async function sendDiscord(message: string, webhookUrl: string, images?: string[], videos?: string[]): Promise<void> {
  if (!webhookUrl) {
    throw new Error("Discord webhook URL is missing.");
  }

  // Guard against images being a string instead of array (strings are iterable!)
  if (images && typeof images === 'string') {
    Logger.warn('utils', 'sendDiscord received a string for images parameter, wrapping in array');
    images = [images];
  }

  // Guard against videos being a string instead of array
  if (videos && typeof videos === 'string') {
    Logger.warn('utils', 'sendDiscord received a string for videos parameter, wrapping in array');
    videos = [videos];
  }

  const DISCORD_MESSAGE_LIMIT = 1900;

  let messageToSend = message;

  // Check if the message is too long
  if (message.length > DISCORD_MESSAGE_LIMIT) {
      // Log a warning in the Observer AI logs so the developer knows this happened
      Logger.warn('utils', `Discord message was too long (${message.length} chars) and has been automatically truncated.`);

      // Truncate the message and add a clear indicator that it was shortened
      messageToSend = message.substring(0, DISCORD_MESSAGE_LIMIT) + "... (msg trunc)";
  }

  try {
    // Construct Discord embed payload
    const embed = {
      title: "Agent Notification",
      color: 3447003,  // Blue color (#3498db)
      description: messageToSend,
    };

    const payload = {
      username: "Observer AI",
      avatar_url: "https://raw.githubusercontent.com/Roy3838/Observer/dev/app/public/logo.png",
      embeds: [embed]
    };

    // Filter out null/undefined values
    const validImages = images?.filter(img => img != null) || [];
    const validVideos = videos?.filter(vid => vid != null) || [];
    const hasImages = validImages.length > 0;
    const hasVideos = validVideos.length > 0;

    // Send images/videos as file attachments if provided
    if (hasImages || hasVideos) {
      // For files, we need to use multipart/form-data
      const formData = new FormData();

      // Add the payload as JSON string
      formData.append('payload_json', JSON.stringify(payload));

      let fileIndex = 0;

      // Add each image as a file
      if (hasImages) {
        for (let i = 0; i < validImages.length; i++) {
          try {
            // Decode base64 image
            const base64Data = validImages[i].includes(',') ? validImages[i].split(',')[1] : validImages[i];
            const binaryString = atob(base64Data);
            const bytes = new Uint8Array(binaryString.length);
            for (let j = 0; j < binaryString.length; j++) {
              bytes[j] = binaryString.charCodeAt(j);
            }
            const blob = new Blob([bytes], { type: 'image/png' });
            formData.append(`file${fileIndex}`, blob, `image_${i+1}.png`);
            fileIndex++;
          } catch (error) {
            Logger.warn('utils', `Failed to process image ${i+1} for Discord: ${error}`);
          }
        }
      }

      // Add each video as a file
      if (hasVideos) {
        for (let i = 0; i < validVideos.length; i++) {
          try {
            // Decode base64 video
            const base64Data = validVideos[i].includes(',') ? validVideos[i].split(',')[1] : validVideos[i];
            const binaryString = atob(base64Data);
            const bytes = new Uint8Array(binaryString.length);
            for (let j = 0; j < binaryString.length; j++) {
              bytes[j] = binaryString.charCodeAt(j);
            }
            const blob = new Blob([bytes], { type: 'video/webm' });
            formData.append(`file${fileIndex}`, blob, `video_${i+1}.webm`);
            fileIndex++;
          } catch (error) {
            Logger.warn('utils', `Failed to process video ${i+1} for Discord: ${error}`);
          }
        }
      }

      const response = await fetch(webhookUrl, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text().catch(() => response.statusText);
        throw new Error(`Failed to send Discord notification: ${response.status} ${errorText}`);
      }
    } else {
      // Send as JSON if no images or videos
      const response = await fetch(webhookUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorText = await response.text().catch(() => response.statusText);
        throw new Error(`Failed to send Discord notification: ${response.status} ${errorText}`);
      }
    }
  } catch (error) {
    // Rethrow the error so the agent's execution log can see it
    throw error;
  }
}

/**
 * Sends a Telegram message by calling the backend API.
 * @param message The message content to send.
 * @param chatId The Telegram chat ID to send the message to.
 * @param authToken The authentication token for the Observer AI API.
 * @param images Optional array of base64-encoded images (without data:image prefix).
 * @param videos Optional array of base64-encoded videos (without data:video prefix).
 */
export async function sendTelegram(message: string, chatId: string, authToken: string, images?: string[], videos?: string[]): Promise<void> {
  const API_HOST = "https://api.observer-ai.com";

  if (!authToken) {
    throw new Error("Authentication error: Auth token is missing.");
  }

  try {
    const requestBody: { chat_id: string; message: string; images?: string[]; videos?: string[] } = {
      chat_id: chatId,
      message: message,
    };

    if (images && images.length > 0) {
      requestBody.images = images;
    }

    if (videos && videos.length > 0) {
      requestBody.videos = videos;
    }

    const response = await fetch(`${API_HOST}/tools/send-telegram`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`,
      },
      body: JSON.stringify(requestBody),
    });

    await handleQuotaExceeded(response, 'telegram');

    if (!response.ok) {
      try {
        const errorData = await response.json();
        const errorMessage = typeof errorData.detail === 'string'
          ? errorData.detail
          : `Failed to send Telegram message: ${response.status} ${response.statusText}`;
        throw new Error(errorMessage);
      } catch (parseError) {
        // If JSON parsing fails, use the HTTP status
        throw new Error(`Failed to send Telegram message: ${response.status} ${response.statusText}`);
      }
    }
  } catch (error) {
    throw error;
  }
}

/**
 * Sends a notification directly to a user's self-hosted Gotify server.
 * This function does NOT use the Observer AI backend API.
 * @param message The main content of the notification.
 * @param serverUrl The base URL of the user's Gotify server (e.g., "https://gotify.example.com").
 * @param appToken The Gotify application token for authentication.
 * @param title An optional title for the notification.
 * @param priority An optional priority for the notification (numeric).
 */
export async function sendGotify(message: string, serverUrl: string, appToken: string, title?: string, priority?: number): Promise<void> {
  // 1. Input Validation
  if (!serverUrl || !appToken) {
    throw new Error("Gotify server URL and application token are required.");
  }

  // Ensure the URL is clean and doesn't have a trailing slash for consistency
  const cleanServerUrl = serverUrl.endsWith('/') ? serverUrl.slice(0, -1) : serverUrl;
  
  // Construct the full API endpoint URL
  const endpoint = `${cleanServerUrl}/message?token=${appToken}`;

  // 2. Construct the Payload
  const payload: { message: string; title?: string; priority?: number } = {
    message: message,
  };

  if (title) {
    payload.title = title;
  }
  if (priority) {
    payload.priority = priority;
  }

  // 3. Perform the fetch request directly to the user's Gotify server
  // Using platformFetch since Gotify could be self-hosted on localhost
  const response = await platformFetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  // 4. Handle Response
  if (!response.ok) {
    // Try to get a more specific error from Gotify's response body
    const errorData = await response.json().catch(() => null); // Gracefully handle non-JSON responses
    const errorMessage = errorData?.error_description || `Request failed with status: ${response.status}`;
    throw new Error(`Failed to send Gotify notification: ${errorMessage}`);
  }
}

/**
 * Shows a native "ask" dialog and waits for the user's boolean response.
 * @param appUrl The base URL of the local Tauri server (e.g., "http://127.0.0.1:3838").
 * @param title The title of the dialog window.
 * @param question The main text/question in the dialog.
 * @returns A promise that resolves to `true` if the user clicks "Yes", and `false` otherwise.
 */
export async function ask(appUrl: string, title: string, question: string): Promise<boolean> {
  const response = await platformFetch(`${appUrl}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, question }),
  });

  if (!response.ok) {
    throw new Error(`Server responded with status: ${response.status}`);
  }

  const data = await response.json();
  return data.answer;
}

/**
 * Shows a native message dialog that the user must acknowledge.
 * @param appUrl The base URL of the local Tauri server.
 * @param title The title of the dialog window.
 * @param message The message to display.
 */
export async function message(appUrl: string, title: string, message: string): Promise<void> {
  const response = await platformFetch(`${appUrl}/message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, message }),
  });

  if (!response.ok) {
    throw new Error(`Server responded with status: ${response.status}`);
  }
}

/**
 * Sends a non-blocking native system notification.
 * This is different from the browser-based `notify` function.
 * @param appUrl The base URL of the local Tauri server.
 * @param title The title of the notification.
 * @param body The main content of the notification.
 */
export async function system_notify(appUrl: string, title: string, body: string): Promise<void> {
  const response = await platformFetch(`${appUrl}/notification`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, body }),
  });

  if (!response.ok) {
    throw new Error(`Server responded with status: ${response.status}`);
  }
}

/**
 * Displays a message in the overlay window.
 * The overlay appears on top of all other windows and shows agent messages.
 * @param appUrl The base URL of the local Tauri server.
 * @param message The message to display in the overlay (supports basic markdown).
 */
export async function overlay(appUrl: string, message: string): Promise<void> {
  const response = await platformFetch(`${appUrl}/overlay`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    throw new Error(`Server responded with status: ${response.status}`);
  }
}

/**
 * Sleep for a specified number of milliseconds
 * Optionally pauses the agent loop if agentId is provided
 */
export async function sleep(ms: number = 2000, agentId?: string): Promise<void> {
  if (agentId) {
    pauseAgentLoop(agentId, ms);
  }
  await new Promise(r => setTimeout(r, ms));
}

/**
 * Triggers a mouse click at the current cursor position.
 * User must position the mouse before calling this function.
 * @param appUrl The base URL of the local Tauri server.
 * @param button The mouse button to click ('left' or 'right'). Defaults to 'left'.
 */
export async function click(appUrl: string, button: 'left' | 'right' = 'left'): Promise<void> {
  const response = await platformFetch(`${appUrl}/click`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ button }),
  });

  if (!response.ok) {
    throw new Error(`Server responded with status: ${response.status}`);
  }
}

/**
 * Makes a phone call by calling the backend API.
 * @param message The message content to speak during the call.
 * @param number The phone number to call.
 * @param authToken The authentication token for the Observer AI API.
 */
export async function call(message: string, number: string, authToken: string): Promise<void> {
  const API_HOST = "https://api.observer-ai.com";

  if (!authToken) {
    throw new Error("Authentication error: Auth token is missing.");
  }

  try {
    const requestBody: { to_number: string; message: string } = {
      to_number: number,
      message: message,
    };

    const response = await fetch(`${API_HOST}/tools/make-call`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authToken}`,
      },
      body: JSON.stringify(requestBody),
    });

    await handleQuotaExceeded(response, 'voice_call');

    if (!response.ok) {
      // Check if this is a whitelist error (403 + user is authenticated)
      if (response.status === 403 && authToken) {
        Logger.error('whitelist', `Phone number ${number} not whitelisted for calls`, {
          logType: 'whitelist-required',
          content: {
            phoneNumber: number,
            toolName: 'Call',
            channel: 'voice',
            timestamp: new Date().toISOString()
          }
        });
        throw new Error(`Phone number ${number} is not whitelisted. Please verify your number first.`);
      }

      try {
        const errorData = await response.json();
        const errorMessage = typeof errorData.detail === 'string'
          ? errorData.detail
          : `Failed to make call: ${response.status} ${response.statusText}`;
        throw new Error(errorMessage);
      } catch (parseError) {
        // If JSON parsing fails, use the HTTP status
        throw new Error(`Failed to make call: ${response.status} ${response.statusText}`);
      }
    }
  } catch (error) {
    throw error;
  }
}
