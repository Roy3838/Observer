// src/components/AcceptToS.tsx

import React from 'react';
import { Shield, Monitor, Camera, Mic, Clipboard, Server } from 'lucide-react';

interface AcceptToSProps {
  isOpen: boolean;
  onAccept: () => void;
}

export const AcceptToS: React.FC<AcceptToSProps> = ({ isOpen, onAccept }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[10000] backdrop-blur-sm p-2 md:p-4">
      <div
        className="relative bg-white rounded-2xl shadow-xl border border-gray-200 w-full max-w-3xl max-h-[85vh] md:max-h-[90vh] overflow-y-auto transition-all duration-300"
        onClick={e => e.stopPropagation()}
      >
        <div className="p-6 md:p-8">
          {/* Header */}
          <div className="flex items-center gap-3 mb-6">
            <div className="flex items-center justify-center h-10 w-10 rounded-full bg-gray-100 flex-shrink-0">
              <Shield className="h-5 w-5 text-gray-700" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Privacy & Data Sharing</h2>
              <p className="text-sm text-gray-500 mt-0.5">Observer is open-source and designed to protect your privacy</p>
            </div>
          </div>

          <div className="space-y-5">
            {/* Data Disclosure Section */}
            <div>
              <p className="text-sm md:text-base text-gray-600 leading-relaxed mb-4">
                Using Cloud AI models, the data you choose will be sent to third-party AI providers for processing. View their privacy policies:{' '}
                <a href="https://ai.google.dev/gemini-api/terms" target="_blank" rel="noopener noreferrer" className="text-gray-900 hover:underline font-medium">Google AI Studio</a>
                {', '}
                <a href="https://openrouter.ai/privacy" target="_blank" rel="noopener noreferrer" className="text-gray-900 hover:underline font-medium">OpenRouter</a>
                {', '}
                <a href="https://fireworks.ai/privacy-policy" target="_blank" rel="noopener noreferrer" className="text-gray-900 hover:underline font-medium">Fireworks.ai</a>
              </p>
              <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
                <div className="grid grid-cols-2 gap-3">
                  <div className="flex items-center gap-2.5 text-sm text-gray-700">
                    <div className="w-8 h-8 rounded-full bg-white border border-gray-200 flex items-center justify-center flex-shrink-0">
                      <Monitor className="h-4 w-4 text-gray-500" />
                    </div>
                    <span>Screen captures & text</span>
                  </div>
                  <div className="flex items-center gap-2.5 text-sm text-gray-700">
                    <div className="w-8 h-8 rounded-full bg-white border border-gray-200 flex items-center justify-center flex-shrink-0">
                      <Camera className="h-4 w-4 text-gray-500" />
                    </div>
                    <span>Camera images</span>
                  </div>
                  <div className="flex items-center gap-2.5 text-sm text-gray-700">
                    <div className="w-8 h-8 rounded-full bg-white border border-gray-200 flex items-center justify-center flex-shrink-0">
                      <Mic className="h-4 w-4 text-gray-500" />
                    </div>
                    <span>Audio transcriptions</span>
                  </div>
                  <div className="flex items-center gap-2.5 text-sm text-gray-700">
                    <div className="w-8 h-8 rounded-full bg-white border border-gray-200 flex items-center justify-center flex-shrink-0">
                      <Clipboard className="h-4 w-4 text-gray-500" />
                    </div>
                    <span>Clipboard content</span>
                  </div>
                </div>
              </div>
              <p className="text-xs text-gray-400 mt-3 text-center">
                You control which sensors each agent uses when creating or configuring agents. To improve Observer, we use{' '}
              <a href="https://www.datadoghq.com/legal/privacy/" target="_blank" rel="noopener noreferrer" className="hover:underline">Datadog</a>
              {' '} to collect anonymous usage analytics and masked session recordings of the app interface.

              </p>
            </div>

            {/* Local Models Callout */}
            <div className="rounded-xl p-4 border border-gray-200">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center flex-shrink-0">
                  <Server className="h-5 w-5 text-gray-700" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-gray-900">Want 100% privacy?</h3>
                  <p className="text-sm text-gray-500">
                    Use <span className="font-medium text-gray-900">local models</span> — your data never leaves your device.
                  </p>
                </div>
              </div>
            </div>


            {/* Terms Link */}
            <p className="text-sm text-gray-500 text-center">
              By continuing, you agree to our{' '}
              <a href="https://observer-ai.com/#/Terms" target="_blank" rel="noopener noreferrer" className="text-gray-900 hover:underline font-medium">
                Terms
              </a>
              {' '}and{' '}
              <a href="https://observer-ai.com/#/Privacy" target="_blank" rel="noopener noreferrer" className="text-gray-900 hover:underline font-medium">
                Privacy Policy
              </a>.
            </p>

            {/* Accept Button */}
            <button
              onClick={onAccept}
              className="w-full px-6 py-3 bg-gray-900 text-white rounded-full hover:bg-black focus:outline-none focus:ring-2 focus:ring-gray-300 transition-colors font-medium text-base"
            >
              I Understand & Accept
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AcceptToS;
