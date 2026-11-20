// src/utils/agent_database.ts
// Database utilities for agent management with unified CompleteAgent type
//
import { dispatchMemoryUpdate } from '@components/MemoryManager';
import yaml from 'js-yaml';
import type { AgentClassification } from '@/mcp/ontology/schema';

export interface CompleteAgent {
  id: string;
  name: string;
  description: string;
  model_name: string;
  system_prompt: string;
  loop_interval_seconds: number;
  only_on_significant_change?: boolean;
  /** Set by the MCP: the agent's place in the ontology (mcp/ontology). UI-made agents have none. */
  classification?: AgentClassification;
}

// Database setup
const DB_NAME = 'observer-db';
const DB_VERSION = 4;
const AGENT_STORE = 'agents';
const CONFIG_STORE = 'configs';
const CODE_STORE = 'code';
const MEMORY_STORE = 'memories';
const IMAGE_MEMORY_STORE = 'image_memories';

// Open the database
export async function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    
    // Create the object stores when database is first created
    request.onupgradeneeded = () => {
      const db = request.result;

      // Store for agent metadata
      let agentStore: IDBObjectStore;
      if (!db.objectStoreNames.contains(AGENT_STORE)) {
        agentStore = db.createObjectStore(AGENT_STORE, { keyPath: 'id' });
      } else {
        agentStore = request.transaction!.objectStore(AGENT_STORE);
        if (agentStore.indexNames.contains('by_status')) {
          agentStore.deleteIndex('by_status');
        }
      }
      
      // Store for agent configurations
      if (!db.objectStoreNames.contains(CONFIG_STORE)) {
        db.createObjectStore(CONFIG_STORE, { keyPath: 'id' });
      }
      
      // Store for agent code
      if (!db.objectStoreNames.contains(CODE_STORE)) {
        db.createObjectStore(CODE_STORE, { keyPath: 'id' });
      }
      
      // Store for agent memories
      if (!db.objectStoreNames.contains(MEMORY_STORE)) {
        db.createObjectStore(MEMORY_STORE, { keyPath: 'id' });
      }
      
      // Store for agent image memories
      if (!db.objectStoreNames.contains(IMAGE_MEMORY_STORE)) {
        db.createObjectStore(IMAGE_MEMORY_STORE, { keyPath: 'id' });
      }
    };
  });
}

// Create or update an agent
export async function saveAgent(
  agent: CompleteAgent,
  code: string
): Promise<CompleteAgent> {
  // Validate agent ID (letters, numbers, underscores only)
  if (!agent.id.match(/^[a-zA-Z0-9_]+$/)) {
    throw new Error('Invalid agent ID. Use only letters, numbers, and underscores.');
  }
  
  const db = await openDB();
  
  // Start a transaction to save all related data
  const tx = db.transaction([AGENT_STORE, CONFIG_STORE, CODE_STORE], 'readwrite');
  
  // Save agent metadata - extract just the metadata fields
  const agentMetadata = {
    id: agent.id,
    name: agent.name,
    description: agent.description
  };
  
  const agentStore = tx.objectStore(AGENT_STORE);
  await new Promise<void>((resolve, reject) => {
    const request = agentStore.put(agentMetadata);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
  
  // Save agent config - extract just the config fields
  const agentConfig = {
    id: agent.id,
    model_name: agent.model_name,
    system_prompt: agent.system_prompt,
    loop_interval_seconds: agent.loop_interval_seconds,
    only_on_significant_change: agent.only_on_significant_change,
    classification: agent.classification
  };

  const configStore = tx.objectStore(CONFIG_STORE);
  await new Promise<void>((resolve, reject) => {
    const request = configStore.put(agentConfig);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
  
  // Save agent code
  const codeStore = tx.objectStore(CODE_STORE);
  await new Promise<void>((resolve, reject) => {
    const request = codeStore.put({
      id: agent.id,
      code
    });
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
  
  // Complete the transaction
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  
  // Initialize memory with empty string if it doesn't exist yet
  const memory = await getAgentMemory(agent.id);
  if (memory === null) {
    await updateAgentMemory(agent.id, '');
  }
  
  return agent;
}

// Get all agent IDs (lightweight for suggestions)
export async function getAllAgentIds(): Promise<string[]> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const tx = db.transaction(AGENT_STORE, 'readonly');
    const store = tx.objectStore(AGENT_STORE);
    const request = store.getAll();

    request.onsuccess = () => {
      const agents = request.result;
      const agentIds = agents.map(agent => agent.id);
      resolve(agentIds);
    };
    request.onerror = () => reject(request.error);
  });
}

// List all agents
export async function listAgents(): Promise<CompleteAgent[]> {
  const db = await openDB();
  
  // First get all agent metadata
  const agentsMetadata = await new Promise<any[]>((resolve, reject) => {
    const tx = db.transaction(AGENT_STORE, 'readonly');
    const store = tx.objectStore(AGENT_STORE);
    const request = store.getAll();
    
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  
  // Then get all configurations
  const configsPromises = agentsMetadata.map(agent => getAgentConfig(agent.id));
  const configs = await Promise.all(configsPromises);
  
  // Combine them into CompleteAgent objects
  return agentsMetadata.map((metadata, index) => {
    const config = configs[index] || {
      model_name: '',
      system_prompt: '',
      loop_interval_seconds: 10.0
    };
    
    return {
      ...metadata,
      ...config
    };
  });
}

// Get a single agent by ID
export async function getAgent(agentId: string): Promise<CompleteAgent | null> {
  const db = await openDB();
  
  // Get the agent metadata
  const agentMetadata = await new Promise<any>((resolve, reject) => {
    const tx = db.transaction(AGENT_STORE, 'readonly');
    const store = tx.objectStore(AGENT_STORE);
    const request = store.get(agentId);
    
    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => reject(request.error);
  });
  
  if (!agentMetadata) {
    return null;
  }
  
  // Get the agent config
  const config = await getAgentConfig(agentId);
  
  // Combine into a CompleteAgent
  return {
    ...agentMetadata,
    ...(config || {
      model_name: '',
      system_prompt: '',
      loop_interval_seconds: 1.0
    })
  };
}

// Get agent configuration (for internal use)
async function getAgentConfig(agentId: string): Promise<{
  model_name: string;
  system_prompt: string;
  loop_interval_seconds: number;
  only_on_significant_change?: boolean;
  classification?: AgentClassification;
} | null> {
  const db = await openDB();
  
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CONFIG_STORE, 'readonly');
    const store = tx.objectStore(CONFIG_STORE);
    const request = store.get(agentId);
    
    request.onsuccess = () => {
      const result = request.result;
      if (result) {
        // Remove id from the result to get just the config
        const { id, ...configWithoutId } = result;
        resolve(configWithoutId);
      } else {
        resolve(null);
      }
    };
    request.onerror = () => reject(request.error);
  });
}

// Get agent code
export async function getAgentCode(agentId: string): Promise<string | null> {
  const db = await openDB();
  
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CODE_STORE, 'readonly');
    const store = tx.objectStore(CODE_STORE);
    const request = store.get(agentId);
    
    request.onsuccess = () => {
      const result = request.result;
      resolve(result ? result.code : null);
    };
    request.onerror = () => reject(request.error);
  });
}

// Get agent memory
export async function getAgentMemory(agentId: string): Promise<string> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const tx = db.transaction(MEMORY_STORE, 'readonly');
    const store = tx.objectStore(MEMORY_STORE);
    const request = store.get(agentId);

    request.onsuccess = () => {
      const result = request.result;
      resolve(result ? result.memory : '');
    };
    request.onerror = () => reject(request.error);
  });
}

// Get all memory entries
export async function getAllMemories(): Promise<Array<{ id: string; memory: string }>> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const tx = db.transaction(MEMORY_STORE, 'readonly');
    const store = tx.objectStore(MEMORY_STORE);
    const request = store.getAll();

    request.onsuccess = () => resolve(request.result || []);
    request.onerror = () => reject(request.error);
  });
}

// Update agent memory
export async function updateAgentMemory(agentId: string, memory: string): Promise<void> {
  const db = await openDB();
  
  const tx = db.transaction(MEMORY_STORE, 'readwrite');
  const store = tx.objectStore(MEMORY_STORE);
  
  await new Promise<void>((resolve, reject) => {
    const request = store.put({
      id: agentId,
      memory
    });
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
  dispatchMemoryUpdate(agentId);
}

// Get agent image memory
export async function getAgentImageMemory(agentId: string): Promise<string[]> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const tx = db.transaction(IMAGE_MEMORY_STORE, 'readonly');
    const store = tx.objectStore(IMAGE_MEMORY_STORE);
    const request = store.get(agentId);

    request.onsuccess = () => {
      const result = request.result;
      resolve(result ? result.images : []);
    };
    request.onerror = () => reject(request.error);
  });
}

// Get all image memory entries
export async function getAllImageMemories(): Promise<Array<{ id: string; images: string[] }>> {
  const db = await openDB();

  return new Promise((resolve, reject) => {
    const tx = db.transaction(IMAGE_MEMORY_STORE, 'readonly');
    const store = tx.objectStore(IMAGE_MEMORY_STORE);
    const request = store.getAll();

    request.onsuccess = () => resolve(request.result || []);
    request.onerror = () => reject(request.error);
  });
}

// Update agent image memory (replaces entire array)
export async function updateAgentImageMemory(agentId: string, images: string[]): Promise<void> {
  const db = await openDB();
  
  const tx = db.transaction(IMAGE_MEMORY_STORE, 'readwrite');
  const store = tx.objectStore(IMAGE_MEMORY_STORE);
  
  await new Promise<void>((resolve, reject) => {
    const request = store.put({
      id: agentId,
      images
    });
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
  dispatchMemoryUpdate(agentId);
}

// Append images to agent's image memory
export async function appendAgentImageMemory(agentId: string, images: string[]): Promise<void> {
  const existingImages = await getAgentImageMemory(agentId);
  const updatedImages = [...existingImages, ...images];
  await updateAgentImageMemory(agentId, updatedImages);
}

// Clear agent image memory
export async function clearAgentImageMemory(agentId: string): Promise<void> {
  await updateAgentImageMemory(agentId, []);
}

// Delete a memory entry (text and images only, not agent)
export async function deleteMemory(memoryId: string): Promise<void> {
  const db = await openDB();

  const tx = db.transaction([MEMORY_STORE, IMAGE_MEMORY_STORE], 'readwrite');

  await Promise.all([
    new Promise<void>((resolve, reject) => {
      const request = tx.objectStore(MEMORY_STORE).delete(memoryId);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    }),
    new Promise<void>((resolve, reject) => {
      const request = tx.objectStore(IMAGE_MEMORY_STORE).delete(memoryId);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    })
  ]);

  // Complete the transaction
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// Delete an agent
export async function deleteAgent(agentId: string): Promise<void> {
  const db = await openDB();

  const tx = db.transaction([AGENT_STORE, CONFIG_STORE, CODE_STORE, MEMORY_STORE, IMAGE_MEMORY_STORE], 'readwrite');

  // Delete from all stores
  await Promise.all([
    new Promise<void>((resolve, reject) => {
      const request = tx.objectStore(AGENT_STORE).delete(agentId);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    }),
    new Promise<void>((resolve, reject) => {
      const request = tx.objectStore(CONFIG_STORE).delete(agentId);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    }),
    new Promise<void>((resolve, reject) => {
      const request = tx.objectStore(CODE_STORE).delete(agentId);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    }),
    new Promise<void>((resolve, reject) => {
      const request = tx.objectStore(MEMORY_STORE).delete(agentId);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    }),
    new Promise<void>((resolve, reject) => {
      const request = tx.objectStore(IMAGE_MEMORY_STORE).delete(agentId);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    })
  ]);

  // Complete the transaction
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export interface AgentExport {
  id: string;
  name: string;
  description: string;
  model_name: string;
  system_prompt: string;
  loop_interval_seconds: number;
  only_on_significant_change?: boolean;
  classification?: AgentClassification;
  code: string;
  memory: string;
}



/**
 * Import an agent from a file
 * @param file The YAML file containing the agent data
 * @returns A promise that resolves to the imported agent
 */
export async function importAgentFromFile(file: File): Promise<CompleteAgent> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = async (event) => {
      try {
        const content = event.target?.result as string;
        const agentData = yaml.load(content) as AgentExport;
        
        if (!agentData.id || !agentData.name || !agentData.code) {
          throw new Error('Invalid agent file format. Missing required fields.');
        }
        
        const agent: CompleteAgent = {
          id: agentData.id,
          name: agentData.name,
          description: agentData.description || '',
          model_name: agentData.model_name,
          system_prompt: agentData.system_prompt,
          loop_interval_seconds: agentData.loop_interval_seconds,
          only_on_significant_change: agentData.only_on_significant_change,
          classification: agentData.classification
        };
        
        await saveAgent(agent, agentData.code);
        await updateAgentMemory(agent.id, agentData.memory || '');
        
        resolve(agent);
      } catch (error) {
        reject(error);
      }
    };
    
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsText(file);
  });
}

/**
 * Import multiple agents from files
 * @param files Array of YAML files containing agent data
 * @returns A promise that resolves to an array of results with success/failure status
 */
export async function importAgentsFromFiles(files: File[]): Promise<Array<{
  filename: string;
  success: boolean;
  agent?: CompleteAgent;
  error?: string;
}>> {
  const results = [];
  
  for (const file of files) {
    try {
      const agent = await importAgentFromFile(file);
      results.push({
        filename: file.name,
        success: true,
        agent
      });
    } catch (error) {
      results.push({
        filename: file.name,
        success: false,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }
  
  return results;
}

/**
 * Export an agent to a YAML file
 * @param agentId The ID of the agent to export
 * @returns A promise that resolves to a Blob containing the agent data
 */
export async function exportAgentToFile(agentId: string): Promise<Blob> {
  const agent = await getAgent(agentId);
  if (!agent) throw new Error(`Agent ${agentId} not found`);
  
  const code = await getAgentCode(agentId);
  if (code === null) throw new Error(`Code for agent ${agentId} not found`);
  
  const memory = await getAgentMemory(agentId);
  
  const exportData: AgentExport = {
    id: agent.id,
    name: agent.name,
    description: agent.description,
    model_name: agent.model_name,
    system_prompt: agent.system_prompt,
    loop_interval_seconds: agent.loop_interval_seconds,
    only_on_significant_change: agent.only_on_significant_change,
    ...(agent.classification ? { classification: agent.classification } : {}), // js-yaml can't dump undefined
    code,
    memory
  };
  
  const yamlStr = yaml.dump(exportData);
  return new Blob([yamlStr], { type: 'application/x-yaml' });
}

export async function downloadAgent(agentId: string): Promise<void> {
  try {
    const agent = await getAgent(agentId);
    if (!agent) throw new Error(`Agent ${agentId} not found`);
    
    const code = await getAgentCode(agentId);
    if (code === null) throw new Error(`Code for agent ${agentId} not found`);
    
    const memory = await getAgentMemory(agentId);
    
    // Create YAML-formatted string directly
    const yamlContent = [
      `id: ${agent.id}`,
      `name: ${agent.name}`,
      `description: ${agent.description}`,
      `model_name: ${agent.model_name}`,
      `loop_interval_seconds: ${agent.loop_interval_seconds}`,
      `system_prompt: |`,
      `  ${agent.system_prompt.replace(/\n/g, '\n  ')}`,
      `code: |`,
      `  ${code.replace(/\n/g, '\n  ')}`,
      memory ? `memory: |` : 'memory: ""',
      memory ? `  ${memory.replace(/\n/g, '\n  ')}` : '',
    ].join('\n');

    const blob = new Blob([yamlContent], { type: 'application/x-yaml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `agent-${agentId}.yaml`;
    
    document.body.appendChild(a);
    a.click();
    
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 0);
  } catch (error) {
    console.error('Failed to download agent:', error);
    throw error;
  }
}
