// src/components/InteractiveTutorial.tsx

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { X as CloseIcon, Sparkles, Play, Eye, Camera, CheckCircle2, Square, Zap, Heart, Star, Loader2 } from 'lucide-react';
import { useAuth } from '@contexts/AuthContext';
import { useApplePayments } from '@hooks/useApplePayments';
import { isIOS } from '@utils/platform';
import { CreditInfoButton } from './CreditVisualization';
import { Logger } from '@utils/logging';

export type TutorialStep = {
  id: string;
  targetSelector?: string;
  title: string;
  message: string;
  icon?: React.ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right' | 'center' | 'top-left';
  action?: 'click' | 'auto';
  waitForEvent?: string;
  noSpotlight?: boolean;
  noOverlay?: boolean;
};

interface InteractiveTutorialProps {
  isActive: boolean;
  onComplete: (agentId: string) => void;
  onDismiss: () => void;
  agentId: string;
  onImportAgent: () => Promise<void>;
  onViewAllTiers: () => void;
  tier?: string | null;
}

export const InteractiveTutorial: React.FC<InteractiveTutorialProps> = ({
  isActive,
  onComplete,
  onDismiss,
  agentId,
  onImportAgent,
  onViewAllTiers,
  tier,
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [targetRect, setTargetRect] = useState<DOMRect | null>(null);
  const [isImporting, setIsImporting] = useState(false);
  const [detected, setDetected] = useState(false);
  const detectedRef = useRef(false);

  // Upsell state
  const [upsellError, setUpsellError] = useState<string | null>(null);
  const [isButtonLoading, setIsButtonLoading] = useState(false);
  const { getAccessToken } = useAuth();
  const { isLoading: isAppleLoading, error: appleError, purchaseProduct } = useApplePayments();
  const isAppleDevice = isIOS();

  const steps: TutorialStep[] = [
    {
      id: 'welcome',
      title: 'Welcome to Observer!',
      message: '',
      noSpotlight: true,
      noOverlay: true,
    },
    {
      id: 'start-agent',
      targetSelector: `[data-tutorial-start-button="${agentId}"]`,
      title: 'Start Your Agent',
      message: 'Your agent is ready! Hit Start to watch it look for a person.',
      icon: <Play className="h-6 w-6 text-green-500" />,
      position: 'right',
      action: 'click',
    },
    {
      id: 'detecting',
      title: detected ? '🎉 Person Detected!' : 'Looking...',
      message: detected
        ? 'Amazing! Your first agent is working. Now let\'s stop it.'
        : 'Stand in front of your camera. Your agent is watching every 15 seconds.',
      icon: <Eye className="h-6 w-6 text-blue-500" />,
      position: 'top-left',
      noSpotlight: true,
      noOverlay: true,
      waitForEvent: 'celebrateAgent',
    },
    {
      id: 'stop-agent',
      targetSelector: `[data-tutorial-start-button="${agentId}"]`,
      title: 'Stop the Agent',
      message: 'Good job! Click Stop to pause your agent.',
      icon: <Square className="h-6 w-6 text-red-500" />,
      position: 'right',
      action: 'click',
    },
    {
      id: 'minimize-agent',
      targetSelector: `[data-tutorial-minimize-button="${agentId}"]`,
      title: 'Minimize to Tray',
      message: 'Click the minus button to minimize your agent to the bottom nav bar — it keeps running there!',
      icon: <Square className="h-6 w-6 text-gray-500" />,
      position: 'right',
      action: 'click',
    },
    {
      id: 'upsell',
      title: '',
      message: '',
      noSpotlight: true,
      noOverlay: true,
    },
  ];

  const currentStepData = steps[currentStep];

  const advanceStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(s => s + 1);
    } else {
      onComplete(agentId);
    }
  };

  const handleDismiss = () => onDismiss();

  const handleImport = async () => {
    setIsImporting(true);
    try {
      await onImportAgent();
      setCurrentStep(1);
    } finally {
      setIsImporting(false);
    }
  };

  // Upsell handlers (mirrored from WelcomeModal)
  const handleProCheckout = async () => {
    setIsButtonLoading(true);
    setUpsellError(null);
    try {
      const token = await getAccessToken();
      const response = await fetch('https://api.observer-ai.com/payments/create-checkout-session', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ return_base_url: window.location.origin }),
      });
      if (!response.ok) throw new Error('Failed to create checkout session');
      const { url } = await response.json();
      window.location.href = url;
    } catch (err) {
      setUpsellError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsButtonLoading(false);
    }
  };

  const handleApplePurchasePro = useCallback(async () => {
    setIsButtonLoading(true);
    setUpsellError(null);
    try {
      const result = await purchaseProduct('pro');
      if (result.success) {
        Logger.info('TUTORIAL', 'StoreKit purchase succeeded');
        window.location.href = '/upgrade-success';
      } else {
        setUpsellError(result.error || 'Purchase failed');
      }
    } catch (err) {
      setUpsellError(err instanceof Error ? err.message : 'Purchase failed');
    } finally {
      setIsButtonLoading(false);
    }
  }, [purchaseProduct]);

  // Listen for celebrateAgent to advance from detecting step
  useEffect(() => {
    if (!isActive || currentStepData?.id !== 'detecting') return;

    const handler = () => {
      if (detectedRef.current) return;
      detectedRef.current = true;
      setDetected(true);
      setTimeout(() => advanceStep(), 2000);
    };

    window.addEventListener('celebrateAgent', handler);
    return () => window.removeEventListener('celebrateAgent', handler);
  }, [isActive, currentStepData, agentId]);

  // Track target element position for spotlight steps
  useEffect(() => {
    if (!isActive || !currentStepData?.targetSelector || currentStepData.noSpotlight) {
      setTargetRect(null);
      return;
    }

    const update = () => {
      const el = document.querySelector(currentStepData.targetSelector!);
      if (el) setTargetRect(el.getBoundingClientRect());
    };

    update();
    let rafId: number;
    const loop = () => { update(); rafId = requestAnimationFrame(loop); };
    rafId = requestAnimationFrame(loop);
    window.addEventListener('scroll', update, true);
    window.addEventListener('resize', update);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener('scroll', update, true);
      window.removeEventListener('resize', update);
    };
  }, [isActive, currentStepData]);

  // Handle clicks on highlighted elements for click-to-advance steps
  useEffect(() => {
    if (!isActive || currentStepData?.action !== 'click' || !currentStepData.targetSelector) return;

    const handleClick = (e: Event) => {
      const target = e.target as HTMLElement;
      const highlighted = document.querySelector(currentStepData.targetSelector!);
      if (highlighted && (highlighted === target || highlighted.contains(target))) {
        advanceStep();
      }
    };

    document.addEventListener('click', handleClick, true);
    return () => document.removeEventListener('click', handleClick, true);
  }, [isActive, currentStep, currentStepData]);


  if (!isActive || !currentStepData) return null;

  // ── Welcome modal (step 0) ──────────────────────────────────────────────────
  if (currentStepData.id === 'welcome') {
    return createPortal(
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[110] backdrop-blur-sm">
        <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-md w-full mx-4 overflow-hidden">
          <div className="text-center mb-6">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-100 to-purple-100 mb-4">
              <Sparkles className="h-8 w-8 text-blue-600" />
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Welcome to Observer!</h2>
            <p className="text-gray-600">
              Let's run your first agent. It will watch your camera and celebrate when it sees you.
            </p>
          </div>

          <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-xl p-5 mb-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-lg bg-blue-200 flex items-center justify-center">
                <Camera size={20} className="text-blue-700" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Person Detector</h3>
                <p className="text-sm text-gray-600">Runs every 15 seconds</p>
              </div>
            </div>
            <ul className="space-y-2 text-sm text-gray-700">
              <li className="flex items-center gap-2">
                <CheckCircle2 size={16} className="text-blue-600 flex-shrink-0" />
                <span>Watches your camera feed in real time</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 size={16} className="text-blue-600 flex-shrink-0" />
                <span>Detects when a person is visible</span>
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 size={16} className="text-blue-600 flex-shrink-0" />
                <span>Celebrates with confetti when it finds you 🎉</span>
              </li>
            </ul>
          </div>

          <div className="flex gap-3">
            <button
              onClick={handleDismiss}
              className="flex-1 py-3 px-4 border border-gray-300 text-gray-700 rounded-xl hover:bg-gray-50 transition-colors font-medium"
            >
              Skip for Now
            </button>
            <button
              onClick={handleImport}
              disabled={isImporting}
              className="flex-1 py-3 px-4 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-60 transition-colors flex items-center justify-center gap-2 font-medium shadow-sm"
            >
              {isImporting ? (
                <>
                  <span className="h-4 w-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                  Importing...
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  Let's Go!
                </>
              )}
            </button>
          </div>
        </div>
      </div>,
      document.body
    );
  }

  // ── Upsell modal (final step) ──────────────────────────────────────────────
  if (currentStepData.id === 'upsell') {
    const isFreeUser = !tier || tier === 'free';
    return createPortal(
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-[10000] backdrop-blur-sm p-2 md:p-4">
        <div
          className="relative bg-white rounded-2xl shadow-xl border border-gray-200 w-full max-w-3xl max-h-[85vh] md:max-h-[90vh] overflow-y-auto"
          onClick={e => e.stopPropagation()}
        >
          <button onClick={() => onComplete(agentId)} className="absolute top-3 right-3 md:top-4 md:right-4 text-gray-400 hover:text-gray-700 z-10 transition-colors">
            <CloseIcon className="h-5 w-5 md:h-6 md:w-6" />
          </button>

          <div className="p-4 md:p-8">
            {/* Welcome Section */}
            <div className="text-center mb-4 pb-4 md:mb-6 md:pb-6 border-b-2 border-gray-200">
              <div className="flex justify-center items-center mb-2 md:mb-3">
                <img src="/eye-logo-black.svg" alt="Observer AI Logo" className="h-10 w-10 md:h-16 md:w-16 mr-2 md:mr-3" />
                <h1 className="text-2xl md:text-3xl font-bold text-gray-800 tracking-tight">🎉 You did it!</h1>
              </div>
              <p className="hidden md:block text-base text-gray-600 max-w-2xl mx-auto leading-relaxed">
                Your first agent detected a person. Now build something of your own.
              </p>
            </div>

            {/* Support Section */}
            <div className="text-center mb-4 md:mb-6">
              <p className="text-sm md:text-base text-gray-700 mb-1 flex items-center justify-center gap-2">
                <Heart className="h-4 w-4 md:h-5 md:w-5 text-pink-500" />
                <span className="font-semibold">Built by a solo developer</span>
              </p>
              <p className="text-xs md:text-sm text-gray-600 mb-4 md:mb-6">
                Try Observer Pro - give feedback and help development
              </p>

              {/* Observer Pro Card */}
              <div className="bg-gradient-to-br from-purple-50 to-blue-50 border-2 border-purple-300 rounded-xl p-4 md:p-6 max-w-md mx-auto mb-4 hover:shadow-xl transition-all duration-200 relative">
                {!isAppleDevice && (
                  <div className="absolute top-0 -translate-y-1/2 left-1/2 -translate-x-1/2">
                    <div className="bg-gradient-to-r from-purple-500 to-blue-500 text-white text-xs font-bold uppercase tracking-wider rounded-full px-4 py-1 whitespace-nowrap shadow-lg">
                      Free Trial
                    </div>
                  </div>
                )}

                <div className="flex items-center justify-center gap-2 md:gap-3 mb-3 md:mb-4 mt-2">
                  <Sparkles className="h-8 w-8 md:h-10 md:w-10 text-purple-500" />
                  <div className="text-left">
                    <h3 className="text-lg md:text-xl font-bold text-purple-900">Observer Pro</h3>
                    <p className="text-sm text-purple-700">
                      {isAppleDevice ? (
                        <span className="text-xl md:text-2xl font-bold text-purple-900">$22.99/month</span>
                      ) : (
                        <>
                          <span className="text-xl md:text-2xl font-bold text-purple-900">7 days free</span>
                          <span className="text-xs ml-1">then $20/month</span>
                        </>
                      )}
                    </p>
                  </div>
                </div>

                <div className="space-y-1.5 md:space-y-2 mb-3 md:mb-4 text-xs md:text-sm text-purple-900">
                  <div className="flex items-start">
                    <Sparkles className="h-3.5 w-3.5 md:h-4 md:w-4 text-purple-500 mr-2 flex-shrink-0 mt-0.5" />
                    <span><strong>Unlock AI Studio</strong> - multi-agent configs</span>
                  </div>
                  <div className="flex items-start">
                    <Zap className="h-3.5 w-3.5 md:h-4 md:w-4 text-purple-500 mr-2 flex-shrink-0 mt-0.5" />
                    <span>
                      <strong>8 hours/day</strong> cloud monitoring
                      <CreditInfoButton dailyCredits={480} tierName="Pro tier" className="ml-1 align-middle" />
                    </span>
                  </div>
                  <div className="flex items-start">
                    <Sparkles className="h-3.5 w-3.5 md:h-4 md:w-4 text-purple-500 mr-2 flex-shrink-0 mt-0.5" />
                    <span><strong>Premium AI models</strong> access</span>
                  </div>
                  <div className="flex items-start">
                    <Heart className="h-3.5 w-3.5 md:h-4 md:w-4 text-pink-500 mr-2 flex-shrink-0 mt-0.5" />
                    <span><strong>Support open source</strong> development</span>
                  </div>
                </div>

                {isFreeUser && (
                  <button
                    onClick={isAppleDevice ? handleApplePurchasePro : handleProCheckout}
                    disabled={isButtonLoading || isAppleLoading}
                    className="w-full px-4 py-2.5 md:px-6 md:py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:from-purple-700 hover:to-blue-700 focus:outline-none focus:ring-4 focus:ring-purple-300 transition-all duration-200 font-semibold text-sm md:text-base shadow-md hover:shadow-lg disabled:bg-slate-400 disabled:cursor-not-allowed flex items-center justify-center"
                  >
                    {isButtonLoading || isAppleLoading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : null}
                    Start Free Trial
                  </button>
                )}
                <p className="text-xs text-gray-500 mt-2 text-center">
                  <a href="https://observer-ai.com/#/Terms" target="_blank" rel="noopener noreferrer" className="hover:underline">Terms</a>
                  {' · '}
                  <a href="https://observer-ai.com/#/Privacy" target="_blank" rel="noopener noreferrer" className="hover:underline">Privacy</a>
                </p>
              </div>

              {/* Error Display */}
              {(upsellError || appleError) && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-2 md:p-3 mb-3 md:mb-4 text-center max-w-md mx-auto">
                  <p className="text-xs md:text-sm text-red-700">{upsellError || appleError}</p>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex flex-col md:flex-row items-center justify-center gap-2 md:gap-3">
                <button
                  onClick={() => window.open('https://github.com/Roy3838/Observer', '_blank')}
                  className="px-3 py-1.5 md:px-4 md:py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-900 transition-colors font-medium flex items-center gap-2 group shadow-md text-sm md:text-base"
                >
                  <Star className="h-4 w-4 text-yellow-400 fill-yellow-400 group-hover:scale-110 transition-transform" />
                  <span>Star on GitHub</span>
                  <span className="text-xs opacity-80">(1.2k)</span>
                </button>

                <div className="hidden md:block text-gray-300">|</div>

                <button
                  onClick={() => { onComplete(agentId); onViewAllTiers(); }}
                  className="text-xs md:text-sm text-purple-600 hover:text-purple-800 hover:underline transition-colors font-medium flex items-center gap-1"
                >
                  <Sparkles className="h-3.5 w-3.5 md:h-4 md:w-4" />
                  View all tiers →
                </button>

                <div className="hidden md:block text-gray-300">|</div>

                <button
                  onClick={() => onComplete(agentId)}
                  className="text-xs md:text-sm text-gray-600 hover:text-gray-800 hover:underline transition-colors font-medium"
                >
                  Continue with free tier →
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>,
      document.body
    );
  }

  // ── Spotlight / bubble steps (step 1+) ─────────────────────────────────────
  const getBubbleStyle = (): React.CSSProperties => {
    const padding = 20;
    const bubbleWidth = 300;
    const isMobile = window.innerWidth < 768;
    const position = currentStepData.position || 'right';

    if (position === 'top-left') {
      return {
        top: `calc(${padding}px + var(--sat, 0px))`,
        left: `calc(${padding}px + var(--sal, 0px))`,
        right: `calc(${padding}px + var(--sar, 0px))`,
      };
    }

    if (!targetRect || isMobile) {
      return {
        top: `calc(${padding}px + var(--sat, 0px))`,
        left: `calc(${padding}px + var(--sal, 0px))`,
        right: `calc(${padding}px + var(--sar, 0px))`,
      };
    }

    if (position === 'right') {
      if (targetRect.right + bubbleWidth + padding * 2 > window.innerWidth) {
        return { left: targetRect.left - bubbleWidth - padding, top: targetRect.top + targetRect.height / 2, transform: 'translateY(-50%)' };
      }
      return { left: targetRect.right + padding, top: targetRect.top + targetRect.height / 2, transform: 'translateY(-50%)' };
    }

    if (position === 'bottom') {
      return { left: targetRect.left + targetRect.width / 2, top: targetRect.bottom + padding, transform: 'translateX(-50%)' };
    }

    return { top: '50%', left: '50%', transform: 'translate(-50%, -50%)' };
  };

  const getArrowClass = () => {
    if (window.innerWidth < 768 || currentStepData.noSpotlight || currentStepData.position === 'top-left') return '';
    const position = currentStepData.position || 'right';
    if (position === 'right' && targetRect) {
      const bubbleWidth = 300;
      const padding = 20;
      return targetRect.right + bubbleWidth + padding * 2 > window.innerWidth ? 'arrow-right' : 'arrow-left';
    }
    if (position === 'bottom') return 'arrow-top';
    return '';
  };

  return (
    <>
      {/* Spotlight overlay */}
      {targetRect && !currentStepData.noSpotlight && (
        <>
          <div className="fixed left-0 right-0 bg-black/50 z-[100] pointer-events-auto" style={{ top: 0, height: targetRect.top - 4 }} />

          <div className="fixed left-0 right-0 bg-black/50 z-[100] pointer-events-auto" style={{ top: targetRect.bottom + 4, bottom: 0 }} />

          <div className="fixed left-0 bg-black/50 z-[100] pointer-events-auto" style={{ top: targetRect.top - 4, bottom: window.innerHeight - (targetRect.bottom + 4), width: targetRect.left - 4 }} />

          <div className="fixed right-0 bg-black/50 z-[100] pointer-events-auto" style={{ top: targetRect.top - 4, bottom: window.innerHeight - (targetRect.bottom + 4), left: targetRect.right + 4 }} />

          <div
            className="fixed pointer-events-none z-[101]"
            style={{
              left: targetRect.left - 4, top: targetRect.top - 4,
              width: targetRect.width + 8, height: targetRect.height + 8,
              borderRadius: '8px',
              animation: 'pulse 2s ease-in-out infinite',
            }}
          />
        </>
      )}

      {/* Speech bubble */}
      <div
        className={`fixed z-[101] pointer-events-auto bg-white rounded-xl shadow-2xl p-5 ${getArrowClass()}`}
        style={{
          ...getBubbleStyle(),
          ...(window.innerWidth >= 768 && { width: '300px' }),
          maxWidth: 'calc(100vw - 40px)',
        }}
      >
        <button onClick={handleDismiss} className="absolute top-3 right-3 text-gray-400 hover:text-gray-700 transition-colors">
          <CloseIcon className="h-4 w-4" />
        </button>

        <div className="mb-3">
          <div className="flex items-center gap-2 mb-1">
            {currentStepData.icon}
            <h3 className="text-base font-bold text-gray-900">{currentStepData.title}</h3>
          </div>
          <p className="text-sm text-gray-700 leading-relaxed">{currentStepData.message}</p>
        </div>

        {/* Progress dots (steps 1–4) */}
        <div className="flex items-center gap-1 mb-3">
          {[1, 2, 3, 4, 5].map(i => (
            <div
              key={i}
              className={`h-1.5 w-6 rounded-full transition-colors ${
                i === currentStep ? 'bg-blue-600' : i < currentStep ? 'bg-blue-300' : 'bg-gray-200'
              }`}
            />
          ))}
        </div>
      </div>

      <style>{`
        :root {
          --sat: env(safe-area-inset-top, 0px);
          --sal: env(safe-area-inset-left, 0px);
          --sar: env(safe-area-inset-right, 0px);
        }
        @keyframes pulse {
          0%, 100% { box-shadow: 0 0 0 4px rgba(59,130,246,0.5), 0 0 0 9999px rgba(0,0,0,0.6); }
          50%       { box-shadow: 0 0 0 8px rgba(59,130,246,0.3), 0 0 0 9999px rgba(0,0,0,0.6); }
        }
        .arrow-left::before {
          content: ''; position: absolute; left: -8px; top: 50%; transform: translateY(-50%);
          width: 0; height: 0; border-top: 8px solid transparent; border-bottom: 8px solid transparent; border-right: 8px solid white;
        }
        .arrow-right::before {
          content: ''; position: absolute; right: -8px; top: 50%; transform: translateY(-50%);
          width: 0; height: 0; border-top: 8px solid transparent; border-bottom: 8px solid transparent; border-left: 8px solid white;
        }
        .arrow-top::before {
          content: ''; position: absolute; top: -8px; left: 50%; transform: translateX(-50%);
          width: 0; height: 0; border-left: 8px solid transparent; border-right: 8px solid transparent; border-bottom: 8px solid white;
        }
      `}</style>
    </>
  );
};

export default InteractiveTutorial;
