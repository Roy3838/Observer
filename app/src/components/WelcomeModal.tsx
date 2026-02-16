// src/components/WelcomeModal.tsx

import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@contexts/AuthContext';
import { useApplePayments } from '@hooks/useApplePayments';
import { X as CloseIcon, X, Loader2, Sparkles, Zap, Heart, Star, Shield, ExternalLink, Check } from 'lucide-react';
import { openUrl } from '@tauri-apps/plugin-opener';
import { Logger } from '@utils/logging';
import { isIOS } from '../utils/platform';

interface WelcomeModalProps {
  isOpen: boolean;
  onClose: () => void;
  onViewAllTiers: () => void;
  intent: 'local' | 'login' | null;
}

export const WelcomeModal: React.FC<WelcomeModalProps> = ({ isOpen, onClose, onViewAllTiers, intent }) => {
  const [status, setStatus] = useState<'loading' | 'plus' | 'pro' | 'max' | 'free' | 'error'>('loading');
  const [error, setError] = useState<string | null>(null);
  const [isButtonLoading, setIsButtonLoading] = useState(false);
  const [dontShowAgain, setDontShowAgain] = useState(false);
  const [hasAcceptedPrivacy, setHasAcceptedPrivacy] = useState(false);

  const { getAccessToken, isAuthenticated, user, login } = useAuth();
  const {
    isLoading: isAppleLoading,
    error: appleError,
    purchaseProduct,
  } = useApplePayments();

  const isAppleDevice = isIOS();

  // Reset state when modal closes
  useEffect(() => {
    if (!isOpen) {
      setHasAcceptedPrivacy(false);
      setDontShowAgain(false);
      setError(null);
    }
  }, [isOpen]);

  // Fetch subscription status for authenticated users
  useEffect(() => {
    if (!isOpen || !isAuthenticated) {
      if (!isAuthenticated) setStatus('free');
      return;
    }

    const checkSubscriptionStatus = async () => {
      setStatus('loading');
      try {
        const token = await getAccessToken();
        const response = await fetch('https://api.observer-ai.com/quota', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (!response.ok) throw new Error(`API request failed: ${response.statusText}`);
        const data = await response.json();
        setStatus(data.tier || 'free');
      } catch (err) {
        Logger.error('WELCOME', 'Failed to check subscription status:', err);
        setError('Could not retrieve your subscription status.');
        setStatus('error');
      }
    };

    checkSubscriptionStatus();
  }, [isOpen, isAuthenticated, getAccessToken]);

  const handleApiAction = async (endpoint: 'create-checkout-session' | 'create-checkout-session-plus' | 'create-checkout-session-max' | 'create-customer-portal-session') => {
    setIsButtonLoading(true);
    setError(null);
    try {
      const token = await getAccessToken();
      const response = await fetch(`https://api.observer-ai.com/payments/${endpoint}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
      });
      if (!response.ok) throw new Error(`Failed to access ${endpoint}`);
      const { url } = await response.json();
      window.location.href = url;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Action failed: ${errorMessage}. Please try again later.`);
    } finally {
      setIsButtonLoading(false);
    }
  };

  const handleProCheckout = () => handleApiAction('create-checkout-session');

  // Apple In-App Purchase handler for Pro - just purchase and navigate
  // Verification is handled by UpgradeSuccessPage
  const handleApplePurchasePro = useCallback(async () => {
    setIsButtonLoading(true);
    setError(null);

    try {
      const purchaseResult = await purchaseProduct('pro');
      if (purchaseResult.success) {
        Logger.info('WELCOME', 'StoreKit purchase succeeded, navigating to /upgrade-success');
        window.location.href = '/upgrade-success';
      } else {
        setError(purchaseResult.error || 'Purchase failed');
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Purchase failed';
      setError(message);
    } finally {
      setIsButtonLoading(false);
    }
  }, [purchaseProduct]);

  const handleStarGithub = () => {
    window.open('https://github.com/Roy3838/Observer', '_blank');
  };

  const handleClose = () => {
    if (dontShowAgain && user?.sub) {
      localStorage.setItem(`observer_onboarding_complete_${user.sub}`, 'true');
      Logger.info('WELCOME', 'User set don\'t show again');
    }
    onClose();
  };

  const handleAcceptPrivacy = () => {
    setHasAcceptedPrivacy(true);
  };

  const handleSignIn = () => {
    login();
  };

  if (!isOpen) {
    return null;
  }

  // Determine which state to show based on intent prop
  const shouldShowLocalMode = intent === 'local' || (!isAuthenticated && intent === null);
  const shouldShowPrivacyConsent = (intent === 'login' || (isAuthenticated && intent === null)) && !hasAcceptedPrivacy;
  const shouldShowSubscription = (intent === 'login' || (isAuthenticated && intent === null)) && hasAcceptedPrivacy;

  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-[10000] backdrop-blur-sm p-2 md:p-4"
      onClick={shouldShowPrivacyConsent ? undefined : handleClose}
    >
      <div
        className="relative bg-white rounded-2xl shadow-xl border border-gray-200 w-full max-w-3xl max-h-[85vh] md:max-h-[90vh] overflow-y-auto transition-all duration-300"
        onClick={e => e.stopPropagation()}
      >
        {!shouldShowPrivacyConsent && (
          <button onClick={handleClose} className="absolute top-3 right-3 md:top-4 md:right-4 text-gray-400 hover:text-gray-700 z-10 transition-colors">
            <CloseIcon className="h-5 w-5 md:h-6 md:w-6" />
          </button>
        )}

        {/* STATE 1: Local Mode Warning (Not Signed In) */}
        {shouldShowLocalMode && (
          <div className="p-6 md:p-8">
            {/* Warning Header */}
            <div className="bg-amber-50 border-2 border-amber-400 rounded-xl p-4 md:p-5 mb-6">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-2xl">⚠️</span>
                <h2 className="text-xl md:text-2xl font-bold text-amber-900">Warning: 100% Offline Mode</h2>
              </div>
              <p className="text-sm md:text-base text-amber-800 leading-relaxed">
                It's really cool that you value your privacy! Observer is all in on that.
                <br />
                Everything will work if you host your own models.
              </p>
            </div>

            {/* API Features Warning */}
            <div className="mb-5">
              <p className="text-sm md:text-base text-gray-700 font-semibold mb-3">
                Just be mindful that API functions will not work:
              </p>
              <div className="space-y-1.5 text-sm md:text-base pl-2">
                <div className="flex items-center gap-2">
                  <X className="h-4 w-4 text-red-400 flex-shrink-0" />
                  <span className="text-gray-600">Telegram, SMS, Calling, Whatsapp, Email and Pushover</span>
                </div>
              </div>
            </div>

            {/* What Works */}
            <div className="space-y-2 text-sm md:text-base mb-6">
              <div className="flex items-center gap-2">
                <Check className="h-5 w-5 text-green-600 flex-shrink-0" />
                <span className="text-gray-800 font-medium">Discord will work!</span>
              </div>
              <div className="flex items-center gap-2">
                <Check className="h-5 w-5 text-green-600 flex-shrink-0" />
                <span className="text-gray-800 font-medium">All logging/recording features will work!</span>
              </div>
            </div>

            {/* Recommended */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
              <p className="text-sm md:text-base text-blue-900">
                <strong>Recommended:</strong> Sign in AND use local models, that way you have access to those features while keeping your sensitive data 100% on your device!
              </p>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col gap-3">
              <button
                onClick={handleSignIn}
                className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 focus:outline-none focus:ring-4 focus:ring-blue-300 transition-all duration-200 font-semibold shadow-md hover:shadow-lg"
              >
                Sign In
              </button>
              <button
                onClick={handleClose}
                className="w-full px-6 py-3 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 focus:outline-none focus:ring-4 focus:ring-gray-400 transition-all duration-200 font-semibold"
              >
                Continue Without Sign In
              </button>
            </div>

            {/* Don't show again checkbox */}
            <div className="flex items-center justify-center pt-4 mt-4 border-t border-gray-200">
              <label className="flex items-center cursor-pointer text-xs md:text-sm text-gray-600 hover:text-gray-800 transition-colors">
                <input
                  type="checkbox"
                  checked={dontShowAgain}
                  onChange={(e) => setDontShowAgain(e.target.checked)}
                  className="mr-2 h-3.5 w-3.5 md:h-4 md:w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                Don't show this again
              </label>
            </div>
          </div>
        )}

        {/* STATE 2 PAGE 1: Privacy & ToS Consent (Signed In, Not Accepted) */}
        {shouldShowPrivacyConsent && (
          <div className="p-6 md:p-8">
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-5 rounded-t-xl -mx-6 -mt-6 md:-mx-8 md:-mt-8 mb-6">
              <div className="flex items-center gap-3">
                <Shield className="h-8 w-8 flex-shrink-0" />
                <div>
                  <h2 className="text-2xl font-bold">Privacy & Data Sharing Notice</h2>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="space-y-5">
              {/* Simple Statement */}
              <p className="text-base text-gray-700 leading-relaxed">
                By using Observer, you agree to our{' '}
                <a
                  href="https://observer-ai.com/#/Terms"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline font-semibold inline-flex items-center gap-1"
                >
                  Terms of Service
                  <ExternalLink className="h-3.5 w-3.5" />
                </a>
                {' '}and{' '}
                <a
                  href="https://observer-ai.com/#/Privacy"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline font-semibold inline-flex items-center gap-1"
                >
                  Privacy Policy
                  <ExternalLink className="h-3.5 w-3.5" />
                </a>.
              </p>

              {/* Pro Tip - Hybrid Approach */}
              <div className="bg-green-50 border-2 border-green-300 rounded-xl p-5">
                <div className="flex items-start gap-3">
                  <Sparkles className="h-6 w-6 text-green-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="text-lg font-bold text-green-900 mb-2">💡 Pro Tip</h3>
                    <p className="text-sm md:text-base text-green-800 leading-relaxed">
                      Stay signed in and use local models for your agents! This makes your data <strong>100% private on your device</strong>.
                    </p>
                  </div>
                </div>
              </div>

              {/* Accept Button */}
              <button
                onClick={handleAcceptPrivacy}
                className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 focus:outline-none focus:ring-4 focus:ring-blue-300 transition-all duration-200 font-semibold shadow-md hover:shadow-lg text-base"
              >
                I Accept
              </button>
            </div>
          </div>
        )}

        {/* STATE 2 PAGE 2: Subscription Info (Signed In, Accepted) */}
        {shouldShowSubscription && status === 'loading' ? (
          <div className="flex justify-center items-center p-12 md:p-20">
            <Loader2 className="h-10 w-10 md:h-12 md:w-12 animate-spin text-gray-500" />
          </div>
        ) : shouldShowSubscription ? (
          <div className="p-4 md:p-8">
            {/* ============ WELCOME SECTION (Top 60%) ============ */}
            <div className="text-center mb-4 pb-4 md:mb-6 md:pb-6 border-b-2 border-gray-200">
              <div className="flex justify-center items-center mb-2 md:mb-3">
                <img src="/eye-logo-black.svg" alt="Observer AI Logo" className="h-10 w-10 md:h-16 md:w-16 mr-2 md:mr-3" />
                <h1 className="text-2xl md:text-3xl font-bold text-gray-800 tracking-tight">Welcome to Observer!</h1>
              </div>

              {/* Hidden on mobile for compactness */}
              <p className="hidden md:block text-base text-gray-600 max-w-2xl mx-auto leading-relaxed">
                Local micro-agents that watch, log, and react.
              </p>
            </div>

            {/* ============ SUPPORT SECTION (Bottom 40%) ============ */}
            <div className="text-center mb-4 md:mb-6">
              <p className="text-sm md:text-base text-gray-700 mb-1 flex items-center justify-center gap-2">
                <Heart className="h-4 w-4 md:h-5 md:w-5 text-pink-500" />
                <span className="font-semibold">Built by a solo developer</span>
              </p>
              <p className="text-xs md:text-sm text-gray-600 mb-4 md:mb-6">
                Try Observer Pro - give feedback and help development
              </p>

              {/* Observer Pro - Hero Free Trial Option */}
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
                        <span className="text-xl md:text-2xl font-bold text-purple-900">${'22.99'}/month</span>
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
                    <span><strong>8 hours/day</strong> cloud monitoring</span>
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

                <button
                  onClick={isAppleDevice ? handleApplePurchasePro : handleProCheckout}
                  disabled={isButtonLoading || isAppleLoading}
                  className="w-full px-4 py-2.5 md:px-6 md:py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:from-purple-700 hover:to-blue-700 focus:outline-none focus:ring-4 focus:ring-purple-300 transition-all duration-200 font-semibold text-sm md:text-base shadow-md hover:shadow-lg disabled:bg-slate-400 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {isButtonLoading || isAppleLoading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : null}
                  Start Free Trial
                </button>
                <p className="text-xs text-gray-500 mt-2 text-center">
                  {isAppleDevice ? (
                    <>
                      <button onClick={() => openUrl('https://observer-ai.com/#/Terms')} className="hover:underline">Terms</button>
                      {' · '}
                      <button onClick={() => openUrl('https://observer-ai.com/#/Privacy')} className="hover:underline">Privacy</button>
                    </>
                  ) : (
                    <>
                      <a href="https://observer-ai.com/#/Terms" target="_blank" rel="noopener noreferrer" className="hover:underline">Terms</a>
                      {' · '}
                      <a href="https://observer-ai.com/#/Privacy" target="_blank" rel="noopener noreferrer" className="hover:underline">Privacy</a>
                    </>
                  )}
                </p>
              </div>

              {/* Error Display */}
              {(error || appleError) && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-2 md:p-3 mb-3 md:mb-4 text-center max-w-md mx-auto">
                  <p className="text-xs md:text-sm text-red-700">{error || appleError}</p>
                </div>
              )}

              {/* Action Buttons - Stacked on mobile, inline on desktop */}
              <div className="flex flex-col md:flex-row items-center justify-center gap-2 md:gap-3">
                <button
                  onClick={handleStarGithub}
                  className="px-3 py-1.5 md:px-4 md:py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-900 transition-colors font-medium flex items-center gap-2 group shadow-md text-sm md:text-base"
                >
                  <Star className="h-4 w-4 text-yellow-400 fill-yellow-400 group-hover:scale-110 transition-transform" />
                  <span>Star on GitHub</span>
                  <span className="text-xs opacity-80">(1.2k)</span>
                </button>

                <div className="hidden md:block text-gray-300">|</div>

                <button
                  onClick={() => {
                    onViewAllTiers();
                    handleClose();
                  }}
                  className="text-xs md:text-sm text-purple-600 hover:text-purple-800 hover:underline transition-colors font-medium flex items-center gap-1"
                >
                  <Sparkles className="h-3.5 w-3.5 md:h-4 md:w-4" />
                  View all tiers →
                </button>

                <div className="hidden md:block text-gray-300">|</div>

                <button
                  onClick={handleClose}
                  className="text-xs md:text-sm text-gray-600 hover:text-gray-800 hover:underline transition-colors font-medium"
                >
                  Continue with free tier →
                </button>
              </div>
            </div>

            {/* Don't show again checkbox */}
            <div className="flex items-center justify-center pt-3 md:pt-4 border-t border-gray-200">
              <label className="flex items-center cursor-pointer text-xs md:text-sm text-gray-600 hover:text-gray-800 transition-colors">
                <input
                  type="checkbox"
                  checked={dontShowAgain}
                  onChange={(e) => setDontShowAgain(e.target.checked)}
                  className="mr-2 h-3.5 w-3.5 md:h-4 md:w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                Don't show this again
              </label>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
};
