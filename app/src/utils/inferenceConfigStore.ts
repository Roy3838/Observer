// src/utils/inferenceConfigStore.ts
// Thin wrapper — delegates per-model inference param storage to ModelManager.

import { InferenceParams } from '../config/inference-params';
import { ModelManager } from './ModelManager';

class InferenceConfigStore {
  getModelParams(modelName: string): Partial<InferenceParams> {
    return ModelManager.getInstance().getModelParams(modelName);
  }

  setModelParams(modelName: string, params: Partial<InferenceParams>): void {
    ModelManager.getInstance().setModelParams(modelName, params);
  }

  clearModelParams(modelName: string): void {
    ModelManager.getInstance().clearModelParams(modelName);
  }

  hasModelParams(modelName: string): boolean {
    return ModelManager.getInstance().hasModelParams(modelName);
  }
}

export const inferenceConfigStore = new InferenceConfigStore();
