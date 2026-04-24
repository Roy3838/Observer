import { useState, useEffect } from 'react';
import { Cpu, CheckCircle2, Download, ArrowRight, Sparkles, FileDown } from 'lucide-react';
import { NativeLlmManager } from '@utils/localLlm/NativeLlmManager';
import { NativeModelState } from '@utils/localLlm/types';
import { isTauri } from '@utils/platform';

// Preset for the recommended starter model
const GEMMA_E2B_PRESET = {
  label: 'Gemma 4 E2B',
  size: '~1.5 GB',
  gguf: 'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q3_K_S.gguf',
  mmproj: 'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf',
};

const formatBytes = (bytes: number, decimals = 2) => {
  if (!+bytes) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
};

interface LocalServerSetupDialogProps {
  onDismiss: () => void;
  onModelLoaded?: () => void;
}

type SetupStep = 'welcome' | 'downloading' | 'loading' | 'complete';

const LocalServerSetupDialog = ({
  onDismiss,
  onModelLoaded
}: LocalServerSetupDialogProps) => {
  const [step, setStep] = useState<SetupStep>('welcome');
  const [downloadStep, setDownloadStep] = useState<'gguf' | 'mmproj' | null>(null);
  const [nativeState, setNativeState] = useState<NativeModelState>(NativeLlmManager.getInstance().getState());
  const [error, setError] = useState<string | null>(null);

  // Only render for Tauri
  if (!isTauri()) {
    return null;
  }

  // Subscribe to native LLM state changes
  useEffect(() => {
    const unsubscribe = NativeLlmManager.getInstance().onStateChange(setNativeState);
    setNativeState(NativeLlmManager.getInstance().getState());
    return unsubscribe;
  }, []);

  // Auto-load model when download completes
  useEffect(() => {
    if (step === 'downloading' && downloadStep === 'mmproj' && nativeState.status !== 'downloading') {
      // mmproj download finished, now load the model
      handleLoadModel();
    }
  }, [step, downloadStep, nativeState.status]);

  // Check if model is loaded and complete the flow
  useEffect(() => {
    if (step === 'loading' && nativeState.status === 'loaded') {
      setStep('complete');
      // Notify parent after a brief delay for celebration animation
      setTimeout(() => {
        onModelLoaded?.();
      }, 2000);
    }
  }, [step, nativeState.status, onModelLoaded]);

  const handleStartDownload = async () => {
    setStep('downloading');
    setError(null);

    try {
      // Step 1: Download GGUF model
      setDownloadStep('gguf');
      await NativeLlmManager.getInstance().downloadModel(GEMMA_E2B_PRESET.gguf);

      // Step 2: Download mmproj vision projector
      setDownloadStep('mmproj');
      await NativeLlmManager.getInstance().downloadModel(GEMMA_E2B_PRESET.mmproj);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed');
      setStep('welcome');
      setDownloadStep(null);
    }
  };

  const handleLoadModel = async () => {
    setStep('loading');
    setError(null);

    try {
      // Find the downloaded model
      const models = await NativeLlmManager.getInstance().listModels();
      const gemmaModel = models.find(m => m.id.includes('gemma-4-E2B'));

      if (gemmaModel) {
        await NativeLlmManager.getInstance().loadModel(gemmaModel.filename, gemmaModel.mmprojFilename);
      } else {
        throw new Error('Model not found after download');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model');
      setStep('welcome');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-md w-full mx-4 overflow-hidden">
        {/* Welcome Step */}
        {step === 'welcome' && (
          <>
            <div className="text-center mb-6">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-green-100 to-emerald-100 mb-4">
                <Sparkles className="h-8 w-8 text-green-600" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Welcome to Observer!</h2>
              <p className="text-gray-600">
                Let's get you started with a local AI model that runs entirely on your device.
              </p>
            </div>

            <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl p-5 mb-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 rounded-lg bg-green-200 flex items-center justify-center">
                  <Cpu size={20} className="text-green-700" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">{GEMMA_E2B_PRESET.label}</h3>
                  <p className="text-sm text-gray-600">{GEMMA_E2B_PRESET.size} - Vision capable</p>
                </div>
              </div>
              <ul className="space-y-2 text-sm text-gray-700">
                <li className="flex items-center gap-2">
                  <CheckCircle2 size={16} className="text-green-600 flex-shrink-0" />
                  <span>Runs 100% on your device - no internet needed</span>
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 size={16} className="text-green-600 flex-shrink-0" />
                  <span>Understands images and text</span>
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 size={16} className="text-green-600 flex-shrink-0" />
                  <span>Fast and efficient - optimized for your hardware</span>
                </li>
              </ul>
            </div>

            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm mb-4">
                {error}
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={onDismiss}
                className="flex-1 py-3 px-4 border border-gray-300 text-gray-700 rounded-xl hover:bg-gray-50 transition-colors font-medium"
              >
                Skip for Now
              </button>
              <button
                onClick={handleStartDownload}
                className="flex-1 py-3 px-4 bg-green-600 text-white rounded-xl hover:bg-green-700 transition-colors flex items-center justify-center gap-2 font-medium shadow-sm"
              >
                <Download size={18} />
                Download Model
              </button>
            </div>
          </>
        )}

        {/* Downloading Step */}
        {step === 'downloading' && (
          <>
            <div className="text-center mb-6">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-blue-100 mb-4">
                <Download className="h-8 w-8 text-blue-600 animate-bounce" />
              </div>
              <h2 className="text-xl font-bold text-gray-900 mb-2">Downloading Model</h2>
              <p className="text-gray-600 text-sm">
                This may take a few minutes depending on your connection.
              </p>
            </div>

            <div className="space-y-4 mb-6">
              {/* GGUF Progress */}
              <div>
                <div className="flex justify-between items-center text-sm mb-2">
                  <span className="text-gray-700 flex items-center gap-2">
                    {downloadStep === 'mmproj' ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                    ) : (
                      <FileDown className="h-4 w-4 text-blue-500" />
                    )}
                    Model (.gguf)
                  </span>
                  <span className="font-medium text-gray-500">
                    {downloadStep === 'mmproj'
                      ? 'Complete'
                      : nativeState.totalBytes > 0
                        ? `${formatBytes(nativeState.downloadedBytes)} / ${formatBytes(nativeState.totalBytes)}`
                        : `${nativeState.downloadProgress}%`
                    }
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div
                    className={`h-2.5 rounded-full transition-all duration-300 ${
                      downloadStep === 'mmproj' ? 'bg-green-500' : 'bg-blue-600'
                    }`}
                    style={{ width: downloadStep === 'mmproj' ? '100%' : `${nativeState.downloadProgress}%` }}
                  />
                </div>
              </div>

              {/* mmproj Progress */}
              <div>
                <div className="flex justify-between items-center text-sm mb-2">
                  <span className="text-gray-700 flex items-center gap-2">
                    {downloadStep === 'mmproj' ? (
                      <FileDown className="h-4 w-4 text-purple-500" />
                    ) : (
                      <FileDown className="h-4 w-4 text-gray-300" />
                    )}
                    Vision capability (.gguf)
                  </span>
                  <span className="font-medium text-gray-500">
                    {downloadStep === 'mmproj'
                      ? nativeState.totalBytes > 0
                        ? `${formatBytes(nativeState.downloadedBytes)} / ${formatBytes(nativeState.totalBytes)}`
                        : `${nativeState.downloadProgress}%`
                      : 'Pending'
                    }
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div
                    className={`h-2.5 rounded-full transition-all duration-300 ${
                      downloadStep === 'mmproj' ? 'bg-purple-500' : 'bg-gray-200'
                    }`}
                    style={{ width: downloadStep === 'mmproj' ? `${nativeState.downloadProgress}%` : '0%' }}
                  />
                </div>
              </div>
            </div>

            <button
              onClick={async () => {
                await NativeLlmManager.getInstance().cancelDownload();
                setStep('welcome');
                setDownloadStep(null);
              }}
              className="w-full py-2.5 px-4 border border-gray-300 text-gray-700 rounded-xl hover:bg-gray-50 transition-colors font-medium text-sm"
            >
              Cancel
            </button>
          </>
        )}

        {/* Loading Step */}
        {step === 'loading' && (
          <>
            <div className="text-center mb-6">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-green-100 mb-4">
                <Cpu className="h-8 w-8 text-green-600 animate-pulse" />
              </div>
              <h2 className="text-xl font-bold text-gray-900 mb-2">Loading Model</h2>
              <p className="text-gray-600 text-sm">
                Initializing the AI model on your device...
              </p>
            </div>

            <div className="flex justify-center mb-6">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </>
        )}

        {/* Complete Step */}
        {step === 'complete' && (
          <>
            <div className="text-center mb-6">
              <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-green-100 mb-4 animate-pulse">
                <CheckCircle2 className="h-10 w-10 text-green-600" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">You're All Set!</h2>
              <p className="text-gray-600">
                Your local AI model is ready. You can now create agents that run entirely on your device.
              </p>
            </div>

            <div className="bg-green-50 border border-green-200 rounded-xl p-4 mb-6 text-center">
              <p className="text-green-800 font-medium">
                {GEMMA_E2B_PRESET.label} is now loaded
              </p>
              <p className="text-green-600 text-sm mt-1">
                No internet connection required for inference
              </p>
            </div>

            <button
              onClick={onDismiss}
              className="w-full py-3 px-4 bg-green-600 text-white rounded-xl hover:bg-green-700 transition-colors flex items-center justify-center gap-2 font-medium shadow-sm"
            >
              Get Started
              <ArrowRight size={18} />
            </button>
          </>
        )}
      </div>
    </div>
  );
};

export default LocalServerSetupDialog;
