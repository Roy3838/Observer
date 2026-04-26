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
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-5 rounded-t-xl -mx-6 -mt-6 md:-mx-8 md:-mt-8 mb-6">
            <div className="flex items-center gap-3">
              <Shield className="h-8 w-8 flex-shrink-0" />
              <div>
                <h2 className="text-2xl font-bold">Privacy & Data Sharing</h2>
                <p className="text-sm text-blue-100 mt-1">Observer is open-source and designed to protect your privacy</p>
              </div>
            </div>
          </div>

          <div className="space-y-5">
            {/* Data Disclosure Section */}
            <div>
              <p className="text-sm md:text-base text-gray-700 leading-relaxed mb-4">
                Using Cloud AI models, the data you choose will be sent to third-party AI providers for processing. View their privacy policies:{' '}
                <a href="https://ai.google.dev/gemini-api/terms" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline font-medium">Google AI Studio</a>
                {', '}
                <a href="https://openrouter.ai/privacy" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline font-medium">OpenRouter</a>
                {', '}
                <a href="https://fireworks.ai/privacy-policy" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline font-medium">Fireworks.ai</a>
              </p>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="grid grid-cols-2 gap-3">
                  <div className="flex items-center gap-2.5 text-sm text-gray-700">
                    <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">
                      <Monitor className="h-4 w-4 text-blue-600" />
                    </div>
                    <span>Screen captures & text</span>
                  </div>
                  <div className="flex items-center gap-2.5 text-sm text-gray-700">
                    <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center flex-shrink-0">
                      <Camera className="h-4 w-4 text-purple-600" />
                    </div>
                    <span>Camera images</span>
                  </div>
                  <div className="flex items-center gap-2.5 text-sm text-gray-700">
                    <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center flex-shrink-0">
                      <Mic className="h-4 w-4 text-indigo-600" />
                    </div>
                    <span>Audio transcriptions</span>
                  </div>
                  <div className="flex items-center gap-2.5 text-sm text-gray-700">
                    <div className="w-8 h-8 rounded-full bg-pink-100 flex items-center justify-center flex-shrink-0">
                      <Clipboard className="h-4 w-4 text-pink-600" />
                    </div>
                    <span>Clipboard content</span>
                  </div>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-3 text-center">
                You control which sensors each agent uses when creating or configuring agents.
              </p>
            </div>

            {/* Local Models Callout */}
            <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-4 border border-green-200">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center flex-shrink-0">
                  <Server className="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-gray-800">Want 100% Privacy?</h3>
                  <p className="text-sm text-gray-600">
                    Use <span className="font-medium text-green-700">local models</span>, your data never leaves your device.
                  </p>
                </div>
              </div>
            </div>

            {/* Terms Link */}
            <p className="text-sm text-gray-600 text-center">
              By continuing, you agree to our{' '}
              <a href="https://observer-ai.com/#/Terms" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline font-medium">
                Terms
              </a>
              {' '}and{' '}
              <a href="https://observer-ai.com/#/Privacy" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline font-medium">
                Privacy Policy
              </a>.
            </p>

            {/* Accept Button */}
            <button
              onClick={onAccept}
              className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 focus:outline-none focus:ring-4 focus:ring-blue-300 transition-all duration-200 font-semibold shadow-md hover:shadow-lg text-base"
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
