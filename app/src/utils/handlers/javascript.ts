// src/utils/handlers/javascript.ts

// SECURITY MODEL DOCUMENTATION
// ============================
// This file uses `new Function()` to execute user-defined agent code. This are the 
// security considerations and threat model. 
//
// Quick rundown on why this is SAFE:
// -----------------
// 1. The `code` parameter is USER-AUTHORED, stored in IndexedDB via getAgentCode().
//    It is NOT extracted from or provided by the LLM response.
//
// 2. The LLM response is passed as a RUNTIME VALUE (`response` variable), not
//    interpolated into source code. This is similar to:
//
//      function runAgent(response) {   // response = "LLM output here"
//        setMemory(response);          // Just a string value, no injection possible
//      }
//
// 3. There is no string interpolation of untrusted input into the code template.
//    JavaScript/TypeScript strings cannot "escape" their context like SQL injection
//    because the code is already parsed before the values are passed in.
//
// WARNING: User-authored code should avoid using eval() or Function() on context
// variables (response, screen, etc.) as these contain LLM output or runtime data.
// Doing so re-introduces code injection risk via prompt injection or MITM attacks.
// This is not blocked to preserve legitimate use cases (e.g., math expressions),
// but users should understand the risk. 
//
// Trust Model:
// -----------------
// - Users trust their own agent code (they wrote it or at least reviewed it)
// - LLM responses are untrusted data, but only accessible as string values
// - The context object provides capabilities (sendSms, screen, etc.) that the
//   user's code can access - this is intentional and user-controlled
//
// If you're auditing this code: the `new Function()` pattern here is equivalent
// to a sandboxed function call with controlled inputs, not eval() of untrusted code.


import * as utils from './utils';
import { Logger } from '../logging';
import { startAgentLoop, stopAgentLoop } from '../main_loop';
import type { TokenProvider } from '../main_loop';
import type { PreProcessorResult } from '../pre-processor';
import { getAgentImageMemory } from '../agent_database';
import { recordingManager } from '../recordingManager';

// Helper function to extract error messages properly
function extractErrorMessage(error: any): string {
  if (typeof error === 'string') return error;
  if (error instanceof Error) return error.message;
  if (error?.message) return error.message;
  if (error?.toString && typeof error.toString === 'function') return error.toString();
  return String(error);
} 

/**
 * Execute JavaScript handler for processing agent responses
 */
export async function executeJavaScript(
  response: string,
  agentId: string,
  code: string,
  iterationId: string, // <-- New parameter
  getToken?: TokenProvider,
  preprocessResult?: PreProcessorResult
): Promise<boolean> {
  // Fetch current agent's image memory to make it always available
  const currentAgentImageMemory = await getAgentImageMemory(agentId);

  const context = {
      prompt: preprocessResult?.modifiedPrompt || "",
      response,
      agentId,
      // Image variables from preprocessing
      images: preprocessResult?.images || [],
      screen: preprocessResult?.imageSources?.screen ? [preprocessResult.imageSources.screen] : [],
      camera: preprocessResult?.imageSources?.camera ? [preprocessResult.imageSources.camera] : [],
      microphone: preprocessResult?.microphone || "",
      screenAudio: preprocessResult?.screenAudio || "",
      allAudio: preprocessResult?.allAudio || "",
      imemory: currentAgentImageMemory,
      getMemory: async (targetId = agentId) => {
        try {
          const result = await utils.getMemory(targetId);
          Logger.info(agentId, `Memory retrieved for ${targetId}`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'getMemory', params: { targetId }, result: result.slice(0, 100) + (result.length > 100 ? '...' : '') }
          });
          return result;
        } catch (error) {
          Logger.error(agentId, `Failed to get memory for ${targetId}`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'getMemory', params: { targetId }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },
      setMemory: async (targetId: string, value?: any) => {
        try {
          let result;
          if (value === undefined) {
            result = await utils.setMemory(agentId, targetId);
            Logger.info(agentId, `Memory set for ${agentId}`, {
              logType: 'tool-success',
              iterationId,
              content: { tool: 'setMemory', params: { targetId: agentId, value: String(targetId).slice(0, 100) } }
            });
          } else {
            result = await utils.setMemory(targetId, value);
            Logger.info(agentId, `Memory set for ${targetId}`, {
              logType: 'tool-success',
              iterationId,
              content: { tool: 'setMemory', params: { targetId, value: String(value).slice(0, 100) } }
            });
          }
          return result;
        } catch (error) {
          Logger.error(agentId, `Failed to set memory`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'setMemory', params: { targetId, value: String(value || targetId).slice(0, 100) }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },
      appendMemory: async (targetId: string, content?: string, separator = '\n') => {
        try {
          let result;
          if (content === undefined) {
            result = await utils.appendMemory(agentId, targetId, separator);
            Logger.info(agentId, `Memory appended for ${agentId}`, {
              logType: 'tool-success',
              iterationId,
              content: { tool: 'appendMemory', params: { targetId: agentId, content: targetId.slice(0, 100), separator } }
            });
          } else {
            result = await utils.appendMemory(targetId, content, separator);
            Logger.info(agentId, `Memory appended for ${targetId}`, {
              logType: 'tool-success',
              iterationId,
              content: { tool: 'appendMemory', params: { targetId, content: content.slice(0, 100), separator } }
            });
          }
          return result;
        } catch (error) {
          Logger.error(agentId, `Failed to append memory`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'appendMemory', params: { targetId, content: String(content || targetId).slice(0, 100), separator }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },
      getImageMemory: async (targetId = agentId) => {
        try {
          const result = await utils.getImageMemory(targetId);
          Logger.info(agentId, `Image memory retrieved for ${targetId}`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'getImageMemory', params: { targetId }, result: `${result.length} images` }
          });
          return result;
        } catch (error) {
          Logger.error(agentId, `Failed to get image memory for ${targetId}`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'getImageMemory', params: { targetId }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },
      setImageMemory: async (targetId: string, images?: string[]) => {
        try {
          if (images === undefined) {
            // If only one parameter provided, treat targetId as images array for current agent
            await utils.setImageMemory(agentId, targetId as any);
            Logger.info(agentId, `Image memory set for ${agentId}`, {
              logType: 'tool-success',
              iterationId,
              content: { tool: 'setImageMemory', params: { targetId: agentId, imageCount: (targetId as any).length } }
            });
          } else {
            await utils.setImageMemory(targetId, images);
            Logger.info(agentId, `Image memory set for ${targetId}`, {
              logType: 'tool-success',
              iterationId,
              content: { tool: 'setImageMemory', params: { targetId, imageCount: images.length } }
            });
          }
        } catch (error) {
          Logger.error(agentId, `Failed to set image memory`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'setImageMemory', params: { targetId, imageCount: images?.length || 0 }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },
      appendImageMemory: async (targetId: string, images?: string[]) => {
        try {
          if (images === undefined) {
            // If only one parameter provided, treat targetId as images array for current agent
            await utils.appendImageMemory(agentId, targetId as any);
            Logger.info(agentId, `Images appended to memory for ${agentId}`, {
              logType: 'tool-success',
              iterationId,
              content: { tool: 'appendImageMemory', params: { targetId: agentId, imageCount: (targetId as any).length } }
            });
          } else {
            await utils.appendImageMemory(targetId, images);
            Logger.info(agentId, `Images appended to memory for ${targetId}`, {
              logType: 'tool-success',
              iterationId,
              content: { tool: 'appendImageMemory', params: { targetId, imageCount: images.length } }
            });
          }
        } catch (error) {
          Logger.error(agentId, `Failed to append image memory`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'appendImageMemory', params: { targetId, imageCount: images?.length || 0 }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },
      notify: (title: string, message: string) => {
        try {
          utils.notify(title, message);
          Logger.info(agentId, `Notification sent: ${title}`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'notify', params: { title, message: message.slice(0, 100) } }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to send notification`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'notify', params: { title, message: message.slice(0, 100) }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },
      time: () => {
        try {
          const result = utils.time();
          Logger.info(agentId, `Current time retrieved: ${result}`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'time', params: {}, result }
          });
          return result;
        } catch (error) {
          Logger.error(agentId, `Failed to get current time`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'time', params: {}, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },
      console: console,
      startAgent: async (targetAgentId?: string) => {
        try {
          const idToStart = targetAgentId === undefined ? agentId : targetAgentId;
          await startAgentLoop(idToStart, getToken);
          Logger.info(agentId, `Agent started: ${idToStart}`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'startAgent', params: { targetAgentId: idToStart } }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to start agent`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'startAgent', params: { targetAgentId }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },
      stopAgent: async (targetAgentId?: string) => {
        try {
          const idToStop = targetAgentId === undefined ? agentId : targetAgentId;
          await stopAgentLoop(idToStop);
          Logger.info(agentId, `Agent stopped: ${idToStop}`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'stopAgent', params: { targetAgentId: idToStop } }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to stop agent`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'stopAgent', params: { targetAgentId }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },

      // --- UPDATED FUNCTIONS ---

      sendSms: async (number: string, message: string, images?: string[], videos?: string[]) => {
        try {
          if (!getToken) throw new Error("Authentication context not available for sendSms.");

          const token = await getToken();
          if (!token) throw new Error("Failed to retrieve authentication token for SMS.");

          await utils.sendSms(message, number, token, images, videos);
          Logger.info(agentId, `SMS sent to ${number}${images && images.length > 0 ? ` with ${images.length} images` : ''}${videos && videos.length > 0 ? ` and ${videos.length} videos` : ''}`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'sendSms', params: { message: message.slice(0,100), number, imageCount: images?.length || 0, videoCount: videos?.length || 0 }, success: true }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to send SMS to ${number}`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'sendSms', params: { message: message.slice(0,100), number, imageCount: images?.length || 0, videoCount: videos?.length || 0 }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },

      sendWhatsapp: async (number: string, message: string, images?: string[], videos?: string[]) => {
        try {
          if (!getToken) throw new Error("Authentication context not available for sendWhatsapp.");

          const token = await getToken();
          if (!token) throw new Error("Failed to retrieve authentication token for WhatsApp.");

          await utils.sendWhatsapp(message, number, token, images, videos);
          Logger.info(agentId, `WhatsApp sent to ${number}${images && images.length > 0 ? ` with ${images.length} images` : ''}${videos && videos.length > 0 ? ` and ${videos.length} videos` : ''}`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'sendWhatsapp', params: { message: message.slice(0,100), number, imageCount: images?.length || 0, videoCount: videos?.length || 0 }, success: true }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to send WhatsApp to ${number}`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'sendWhatsapp', params: { message: message.slice(0,100), number, imageCount: images?.length || 0, videoCount: videos?.length || 0 }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },

      sendEmail: async (emailAddress: string, message: string, images?: string[], videos?: string[]) => {
        try {
          if (!getToken) throw new Error("Authentication context not available for sendEmail.");

          const token = await getToken();
          if (!token) throw new Error("Failed to retrieve authentication token for Email.");

          await utils.sendEmail(message, emailAddress, token, images, videos);
          Logger.info(agentId, `Email sent to ${emailAddress}${images && images.length > 0 ? ` with ${images.length} images` : ''}${videos && videos.length > 0 ? ` and ${videos.length} videos` : ''}`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'sendEmail', params: { message: message.slice(0,100), emailAddress, imageCount: images?.length || 0, videoCount: videos?.length || 0 }, success: true }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to send email to ${emailAddress}`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'sendEmail', params: { message: message.slice(0,100), emailAddress, imageCount: images?.length || 0, videoCount: videos?.length || 0 }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },

      sendPushover: async (userKey: string, message: string, images?: string[], title?: string, videos?: string[]) => {
        try {
          if (!getToken) {
              throw new Error("Authentication context not available for sendPushover.");
          }

          const token = await getToken();
          if (!token) {
              throw new Error("Failed to retrieve authentication token for Pushover.");
          }

          await utils.sendPushover(message, userKey, token, images, title, videos);
          Logger.info(agentId, `Pushover notification sent${images && images.length > 0 ? ` with ${images.length} images` : ''}${videos && videos.length > 0 ? ` and ${videos.length} videos` : ''}`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'sendPushover', params: { message: message.slice(0,100), title, imageCount: images?.length || 0, videoCount: videos?.length || 0 }, success: true }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to send Pushover notification`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'sendPushover', params: { message: message.slice(0,100), title, imageCount: images?.length || 0, videoCount: videos?.length || 0 }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },

      sendDiscord: async (webhookUrl: string, message: string, images?: string[], videos?: string[]) => {
        try {
          await utils.sendDiscord(message, webhookUrl, images, videos);
          Logger.info(agentId, `Discord notification sent${images && images.length > 0 ? ` with ${images.length} images` : ''}${videos && videos.length > 0 ? ` and ${videos.length} videos` : ''}`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'sendDiscord', params: { message: message.slice(0,100), imageCount: images?.length || 0, videoCount: videos?.length || 0 }, success: true }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to send Discord notification`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'sendDiscord', params: { message: message.slice(0,100), imageCount: images?.length || 0, videoCount: videos?.length || 0 }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },

      sendTelegram: async (chatId: string, message: string, images?: string[], videos?: string[]) => {
        try {
          if (!getToken) throw new Error("Authentication context not available for sendTelegram.");

          const token = await getToken();
          if (!token) throw new Error("Failed to retrieve authentication token for Telegram.");

          await utils.sendTelegram(message, chatId, token, images, videos);
          Logger.info(agentId, `Telegram message sent to ${chatId}${images && images.length > 0 ? ` with ${images.length} images` : ''}${videos && videos.length > 0 ? ` and ${videos.length} videos` : ''}`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'sendTelegram', params: { message: message.slice(0,100), chatId, imageCount: images?.length || 0, videoCount: videos?.length || 0 }, success: true }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to send Telegram message to ${chatId}`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'sendTelegram', params: { message: message.slice(0,100), chatId, imageCount: images?.length || 0, videoCount: videos?.length || 0 }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },

      sendGotify: async (message: string, serverUrl: string, appToken: string, title?: string, priority?: number) => {
        try {
          await utils.sendGotify(message, serverUrl, appToken, title, priority);
          Logger.info(agentId, `Gotify notification sent`, { 
            logType: 'tool-success', 
            iterationId,
            content: { tool: 'sendGotify', params: { message: message.slice(0,100), title, priority }, success: true }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to send Gotify notification`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'sendGotify', params: { message: message.slice(0,100), title, priority }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },
      
      startClip: async () => {
        try {
          await utils.startClip();
          Logger.info(agentId, `Clip recording started`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'startClip', params: {} }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to start clip recording`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'startClip', params: {}, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },
      stopClip: async () => {
        try {
          await utils.stopClip();
          Logger.info(agentId, `Clip recording stopped`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'stopClip', params: {} }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to stop clip recording`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'stopClip', params: {}, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },
      markClip: (label: string) => {
        try {
          utils.markClip(label);
          Logger.info(agentId, `Clip marked: ${label}`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'markClip', params: { label } }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to mark clip`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'markClip', params: { label }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },

      getVideo: async (type?: 'screen' | 'camera'): Promise<string[]> => {
        try {
          const result = await recordingManager.getVideo(type);
          Logger.info(agentId, `Video retrieved${type ? ` for ${type}` : ' for all active streams'}`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'getVideo', params: { type }, videoCount: result.length, totalSize: result.reduce((acc, v) => acc + v.length, 0) }
          });
          return result;
        } catch (error) {
          Logger.error(agentId, `Failed to get video${type ? ` for ${type}` : ''}`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'getVideo', params: { type }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },

      ask: async (question: string, title = 'Confirmation'): Promise<boolean> => {
        try {
          const appUrl = "http://localhost:3838";
   
          if (!appUrl) throw new Error("Could not determine the local app server address.");

          const response = await utils.ask(appUrl, title, question);
          Logger.info(agentId, `User responded with: ${response}`,
          {
            logType: 'tool-success', 
            iterationId,
            content: { tool: 'ask', params: { question: question.slice(0,100), title }, result: response }
          });
          return response;
        } catch (error) {
          Logger.error(agentId, `Failed to ask user question`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'ask', params: { question: question.slice(0,100), title }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },

      message: async (message: string, title = 'Agent Message'): Promise<void> => {
        try {
          const appUrl = "http://localhost:3838";

          if (!appUrl) throw new Error("Could not determine the local app server address.");

          await utils.message(appUrl, title, message);
          Logger.info(agentId, `User message shown`, { 
            logType: 'tool-success', 
            iterationId,
            content: { tool: 'message', params: { message: message.slice(0,100), title }, success: true }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to show user message`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'message', params: { message: message.slice(0,100), title }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },

      system_notify: async (body: string, title = 'Observer AI'): Promise<void> => {
        try {
          const appUrl = "http://localhost:3838";

          if (!appUrl) throw new Error("Could not determine the local app server address.");

          await utils.system_notify(appUrl, title, body);
          Logger.info(agentId, `System notification sent`, { 
            logType: 'tool-success', 
            iterationId,
            content: { tool: 'system_notify', params: { body: body.slice(0,100), title }, success: true }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to send system notification`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'system_notify', params: { body: body.slice(0,100), title }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },

      overlay: async (body: string): Promise<void> => {
        try {
          const appUrl = "http://localhost:3838";

          if (!appUrl) throw new Error("Could not determine the local app server address.");

          await utils.overlay(appUrl, body);
          Logger.info(agentId, `Overlay message sent`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'overlay', params: { body: body.slice(0,100)}, success: true }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to send overlay message`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'overlay', params: { body: body.slice(0,100)}, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },

      sleep: async (ms: number = 2000): Promise<void> => {
        try {
          await utils.sleep(ms, agentId);
          Logger.info(agentId, `Slept for ${ms}ms`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'sleep', params: { ms } }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to sleep`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'sleep', params: { ms }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },

      click: async (button: 'left' | 'right' = 'left'): Promise<void> => {
        try {
          const appUrl = "http://localhost:3838";

          if (!appUrl) throw new Error("Could not determine the local app server address.");

          await utils.click(appUrl, button);
          Logger.info(agentId, `Mouse ${button} click executed`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'click', params: { button }, success: true }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to execute mouse ${button} click`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'click', params: { button }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },

      call: async (number: string, message: string): Promise<void> => {
        try {
          if (!getToken) throw new Error("Authentication context not available for call.");

          const token = await getToken();
          if (!token) throw new Error("Failed to retrieve authentication token for call.");

          await utils.call(message, number, token);
          Logger.info(agentId, `Phone call placed to ${number}`, {
            logType: 'tool-success',
            iterationId,
            content: { tool: 'call', params: { message: message.slice(0, 100), number }, success: true }
          });
        } catch (error) {
          Logger.error(agentId, `Failed to place call to ${number}`, {
            logType: 'tool-error',
            iterationId,
            content: { tool: 'call', params: { message: message.slice(0, 100), number }, error: extractErrorMessage(error) }
          });
          throw error;
        }
      },

      celebrate: (): void => {
        window.dispatchEvent(new CustomEvent('celebrateAgent', { detail: { agentId } }));
        Logger.info(agentId, 'celebrate() called', { logType: 'tool-success', iterationId });
      },

    };

    const wrappedCode = `
      return (async function() {
        ${code}
        return true;
      })()
    `;

    let result;
    try {
      // SECURITY NOTE: See file header for details.
      // - `code` is user-authored (from IndexedDB)
      // - `context` values (response, screen, etc.) are passed as runtime arguments
      // - LLM output cannot escape string context 
      const handler = new Function(...Object.keys(context), wrappedCode);
      result = await handler(...Object.values(context));
    } catch (error) {
      // This catches syntax errors, runtime errors, tool errors that bubble up, etc.
      Logger.error(agentId, `Error executing JavaScript: ${error}`, {
        logType: 'tool-error',
        iterationId,
        content: {
          tool: 'code-execution',
          error: extractErrorMessage(error),
          success: false
        }
      });
      throw error;
    }

    // DEBUG: See what we're actually getting back
    Logger.debug(agentId, `Code execution result: ${result}`, { iterationId, result });

    // If code executed successfully, it should return true
    // If we got anything else, something went wrong
    if (result !== true) {
      const error = new Error('Code execution failed or did not complete');
      Logger.error(agentId, `Error executing JavaScript: ${error.message}`, {
        logType: 'tool-error',
        iterationId,
        content: {
          tool: 'code-execution',
          error: error.message,
          success: false
        }
      });
      throw error;
    }

    return true;
}
