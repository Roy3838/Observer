import { Logger } from './logging';
import { executeJavaScript } from './handlers/javascript';
import { executePython } from './handlers/python';

/**
 * Process response using the JavaScript or Python handler
 * based on the presence of a '#python' marker
 */
export async function postProcess(agentId: string, response: string, code: string): Promise<boolean> {
  try {
    Logger.debug(agentId, 'Starting response post-processing');
    
    // Check if the code starts with #python
    if (code.trim().startsWith('#python')) {
      Logger.debug(agentId, 'Detected Python code, using Python handler');
      const result = await executePython(response, agentId, code);
      if (result) {
        Logger.debug(agentId, 'Response processed successfully');
      } else {
        Logger.debug(agentId, 'Python execution failed');
      }
    
      return result;
    } else {
      // Default to JavaScript handler (existing behavior)
      Logger.debug(agentId, 'Using JavaScript handler');
      
      const result = await executeJavaScript(response, agentId, code);
      
      if (result) {
        Logger.debug(agentId, 'Response processed quickly');
      } 
      
      return result;
    }
  } catch (error) {
    Logger.error(agentId, `Error in post-processing: ${error instanceof Error ? error.message : String(error)}`);
    return false;
  }
}
