// src/utils/inferenceConfigStore.ts
// Storage and retrieval of inference parameters (global defaults + per-agent overrides)

import { InferenceParams, DEFAULT_INFERENCE_PARAMS } from '../config/inference-params';

const STORAGE_PREFIX = 'observer-ai:inference';
const GLOBAL_DEFAULTS_KEY = `${STORAGE_PREFIX}:globalDefaults`;
const AGENT_OVERRIDES_PREFIX = `${STORAGE_PREFIX}:agent:`;

/**
 * Manages inference parameter configuration with global defaults and per-agent overrides.
 *
 * Architecture:
 * - Global defaults: Applied to all agents unless overridden
 * - Per-agent overrides: Specific settings for individual agents
 * - Effective params: Merge of defaults + overrides at runtime
 */
class InferenceConfigStore {
  // ==================== GLOBAL DEFAULTS ====================

  /**
   * Get the global inference defaults.
   * Returns hardcoded defaults merged with any user-configured defaults.
   */
  getGlobalDefaults(): InferenceParams {
    const stored = localStorage.getItem(GLOBAL_DEFAULTS_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        return { ...DEFAULT_INFERENCE_PARAMS, ...parsed };
      } catch (error) {
        console.warn('[InferenceConfigStore] Failed to parse global defaults:', error);
      }
    }
    return { ...DEFAULT_INFERENCE_PARAMS };
  }

  /**
   * Set global inference defaults.
   * Pass partial params to update only specific values.
   */
  setGlobalDefaults(params: Partial<InferenceParams>): void {
    const current = this.getGlobalDefaults();
    const merged = { ...current, ...params };
    // Remove undefined values to keep storage clean
    const cleaned = this.cleanParams(merged);
    localStorage.setItem(GLOBAL_DEFAULTS_KEY, JSON.stringify(cleaned));
  }

  /**
   * Reset global defaults to hardcoded values.
   */
  resetGlobalDefaults(): void {
    localStorage.removeItem(GLOBAL_DEFAULTS_KEY);
  }

  // ==================== PER-AGENT OVERRIDES ====================

  /**
   * Get inference overrides for a specific agent.
   * Returns only the overrides, not merged with defaults.
   */
  getAgentOverrides(agentId: string): Partial<InferenceParams> {
    const key = `${AGENT_OVERRIDES_PREFIX}${agentId}`;
    const stored = localStorage.getItem(key);
    if (stored) {
      try {
        return JSON.parse(stored);
      } catch (error) {
        console.warn(`[InferenceConfigStore] Failed to parse overrides for agent ${agentId}:`, error);
      }
    }
    return {};
  }

  /**
   * Set inference overrides for a specific agent.
   * Pass partial params to update only specific values.
   * Use replace=true to replace all overrides instead of merging.
   */
  setAgentOverrides(agentId: string, params: Partial<InferenceParams>, replace: boolean = false): void {
    const key = `${AGENT_OVERRIDES_PREFIX}${agentId}`;
    const current = replace ? {} : this.getAgentOverrides(agentId);
    const merged = { ...current, ...params };
    const cleaned = this.cleanParams(merged);

    if (Object.keys(cleaned).length === 0) {
      // No overrides, remove the key entirely
      localStorage.removeItem(key);
    } else {
      localStorage.setItem(key, JSON.stringify(cleaned));
    }
  }

  /**
   * Clear all overrides for a specific agent (revert to global defaults).
   */
  clearAgentOverrides(agentId: string): void {
    const key = `${AGENT_OVERRIDES_PREFIX}${agentId}`;
    localStorage.removeItem(key);
  }

  /**
   * Remove a specific override for an agent (revert that param to global default).
   */
  removeAgentOverride(agentId: string, paramKey: keyof InferenceParams): void {
    const overrides = this.getAgentOverrides(agentId);
    delete overrides[paramKey];
    this.setAgentOverrides(agentId, overrides);
  }

  /**
   * Check if an agent has any custom overrides.
   */
  hasAgentOverrides(agentId: string): boolean {
    return Object.keys(this.getAgentOverrides(agentId)).length > 0;
  }

  // ==================== EFFECTIVE PARAMS (MERGED) ====================

  /**
   * Get the effective inference params for an agent.
   * This merges global defaults with agent-specific overrides.
   * This is what should be passed to sendApi.
   */
  getEffectiveParams(agentId: string): InferenceParams {
    const globals = this.getGlobalDefaults();
    const overrides = this.getAgentOverrides(agentId);
    return { ...globals, ...overrides };
  }

  /**
   * Check if the effective params differ from global defaults.
   * Useful for UI to show "customized" indicator.
   */
  isCustomized(agentId: string): boolean {
    return this.hasAgentOverrides(agentId);
  }

  // ==================== UTILITIES ====================

  /**
   * Remove undefined values from params object.
   */
  private cleanParams(params: Partial<InferenceParams>): Partial<InferenceParams> {
    const cleaned: Partial<InferenceParams> = {};
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null) {
        (cleaned as any)[key] = value;
      }
    }
    return cleaned;
  }

  /**
   * Get all agent IDs that have custom overrides.
   * Useful for settings export/import.
   */
  getAgentIdsWithOverrides(): string[] {
    const ids: string[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith(AGENT_OVERRIDES_PREFIX)) {
        ids.push(key.slice(AGENT_OVERRIDES_PREFIX.length));
      }
    }
    return ids;
  }

  /**
   * Export all inference config (global + all agent overrides).
   * Useful for backup/sync.
   */
  exportAll(): { globalDefaults: InferenceParams; agentOverrides: Record<string, Partial<InferenceParams>> } {
    const agentOverrides: Record<string, Partial<InferenceParams>> = {};
    for (const agentId of this.getAgentIdsWithOverrides()) {
      agentOverrides[agentId] = this.getAgentOverrides(agentId);
    }
    return {
      globalDefaults: this.getGlobalDefaults(),
      agentOverrides,
    };
  }

  /**
   * Import inference config (global + agent overrides).
   * Useful for restore/sync.
   */
  importAll(data: { globalDefaults?: InferenceParams; agentOverrides?: Record<string, Partial<InferenceParams>> }): void {
    if (data.globalDefaults) {
      this.setGlobalDefaults(data.globalDefaults);
    }
    if (data.agentOverrides) {
      for (const [agentId, overrides] of Object.entries(data.agentOverrides)) {
        this.setAgentOverrides(agentId, overrides);
      }
    }
  }
}

// Export singleton instance
export const inferenceConfigStore = new InferenceConfigStore();
