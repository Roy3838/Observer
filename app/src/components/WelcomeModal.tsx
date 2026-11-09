// src/components/WelcomeModal.tsx

import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@contexts/AuthContext';
import { useApplePayments } from '@hooks/useApplePayments';
import { X as CloseIcon, Loader2, Sparkles, Zap, Heart, Star, Check, ArrowRight, ShieldCheck } from 'lucide-react';
import { Logger } from '@utils/logging';
import { Analytics } from '@utils/analytics';
import { isIOS, isWeb, openExternal } from '../utils/platform';
import { CreditInfoButton } from './CreditVisualization';

interface WelcomeModalProps {
  isOpen: boolean;
  onClose: () => void;
  onViewAllTiers: () => void;
  mode: 'local' | 'upsell';
  onContinueLocal?: () => void; // local mode only: called when user confirms they know what they're doing
  /** Upsell framing: 'onboarding' (right after ToS) or 'activation' (celebratory, after first agent starts). */
  variant?: 'onboarding' | 'activation';
}

export const WelcomeModal: React.FC<WelcomeModalProps> = ({ isOpen, onClose, mode, variant = 'onboarding' }) => {
  const upsellSource = variant === 'activation' ? 'activation' : 'welcome';
  const [error, setError] = useState<string | null>(null);
  const [isButtonLoading, setIsButtonLoading] = useState(false);

  const { getAccessToken, login } = useAuth();
  const {
    isLoading: isAppleLoading,
    error: appleError,
    purchaseProduct,
  } = useApplePayments();

  const isAppleDevice = isIOS();
  const isMobileWeb = isWeb();

  // Reset state when modal closes
  useEffect(() => {
    if (!isOpen) {
      setError(null);
    }
  }, [isOpen]);

  const handleProCheckout = async () => {
    setIsButtonLoading(true);
    setError(null);
    try {
      const token = await getAccessToken();
      const response = await fetch('https://api.observer-ai.com/payments/create-checkout-session', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ return_base_url: window.location.origin }),
      });
      if (!response.ok) throw new Error('Failed to create checkout session');
      const { url } = await response.json();
      window.location.href = url;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Action failed: ${errorMessage}. Please try again later.`);
    } finally {
      setIsButtonLoading(false);
    }
  };

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
    openExternal('https://github.com/Roy3838/Observer');
  };

  const handleClose = () => {
    onClose();
  };

  const handleSignIn = () => {
    Analytics.localModeSignIn();
    onClose();
    login();
  };

  // Escape closes the upsell sheet, same as the ask_user_info sheets.
  useEffect(() => {
    if (!isOpen || mode !== 'upsell') return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, mode, onClose]);

  if (!isOpen) {
    return null;
  }

  /* ============ MODE: Local Mode Warning (Not Signed In) ============ */
  if (mode === 'local') {
    return (
      <div className="fixed inset-0 bg-black/50 flex flex-col items-center justify-center z-[10000] backdrop-blur-sm p-2 md:p-4">
        <div
          className="relative bg-white rounded-2xl shadow-xl border border-gray-200 w-full max-w-3xl max-h-[85vh] md:max-h-[90vh] overflow-y-auto transition-all duration-300 flex flex-col"
          onClick={e => e.stopPropagation()}
        >
          <div className="p-4 md:p-8">
            {/* Header */}
            <div className="mb-3 md:mb-6">
              <h2 className="text-lg md:text-2xl font-semibold text-gray-900 mb-1">Trust me, the UX will suck.</h2>
              <p className="text-xs md:text-base text-gray-600 leading-relaxed">
                Observer <em>works</em> without an account. But a lot of things are limited:
              </p>
            </div>

            {/* What's limited */}
            <div className="space-y-2 mb-3 md:mb-6">
              <div className="bg-gray-50 border border-gray-200 rounded-xl p-2.5 md:p-3">
                <ul className="space-y-1.5">
                  <li className="flex items-center gap-2.5 text-xs md:text-sm font-medium text-gray-700">
                    <Sparkles className="h-4 w-4 text-gray-400 flex-shrink-0" />
                    Agent creator won't work, you'll have to create them manually!
                  </li>
                  <li className="flex items-center gap-2.5 text-xs md:text-sm font-medium text-gray-700">
                    <Zap className="h-4 w-4 text-gray-400 flex-shrink-0" />
                    Most notifications won't work
                  </li>
                </ul>
              </div>
            </div>

            {/* Mobile web + local model warning */}
            {isMobileWeb && (
              <div className="md:hidden bg-gray-50 border border-gray-200 rounded-xl p-2.5 mb-3 flex items-start gap-2">
                <span className="text-gray-400 text-base flex-shrink-0">⚠️</span>
                <p className="text-xs text-gray-600 leading-relaxed">
                  <strong className="text-gray-900">Local models crash mobile browsers.</strong> This is a known transformers.js limitation, download the app if you want to run local models on your phone.
                </p>
              </div>
            )}

            {/* Sign-in call to action */}
            <div className="mb-3 md:mb-6">
              <button
                onClick={handleSignIn}
                className="w-full px-6 py-3 bg-purple-600 text-white rounded-full hover:bg-purple-700 transition-colors font-medium"
              >
                Sign In
              </button>
              <p className="text-xs text-gray-500 text-center mt-2">
                I recommend this, you can sign in <strong>and</strong> use local models. Your data stays on-device either way :)
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* ============ MODE: Upsell — same sheet language as the ask_user_info modals ============ */
  const title = variant === 'activation' ? 'You did it! 🎉' : 'Welcome to Observer';
  const subtitle = variant === 'activation'
    ? 'You have a running agent. Help Observer grow — or continue free.'
    : 'The agent that monitors anything for you.';
  const busy = isButtonLoading || isAppleLoading;

  const primaryBtn = 'w-full inline-flex items-center justify-center gap-2 h-11 rounded-full bg-zinc-900 text-white dark:bg-white dark:text-zinc-900 text-sm font-medium hover:bg-zinc-800 dark:hover:bg-zinc-200 active:scale-[0.99] transition-all disabled:opacity-40 disabled:cursor-not-allowed';
  const textBtn = 'inline-flex items-center gap-1.5 text-xs font-medium text-zinc-500 hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200 transition-colors';

  const feature = (strong: React.ReactNode, rest: React.ReactNode, extra?: React.ReactNode) => (
    <li className="flex items-start gap-3">
      <Check className="h-4 w-4 mt-0.5 flex-shrink-0 text-zinc-900 dark:text-zinc-100" strokeWidth={2.5} />
      <span className="text-sm leading-5 text-zinc-500 dark:text-zinc-400">
        <span className="font-medium text-zinc-900 dark:text-zinc-100">{strong}</span> {rest}{extra}
      </span>
    </li>
  );

  const starButton = (primary: boolean) => (
    <button
      onClick={() => { Analytics.upsellGithub(upsellSource); handleStarGithub(); }}
      className={primary ? primaryBtn : textBtn}
    >
      <Star className={`${primary ? 'h-4 w-4' : 'h-3.5 w-3.5'} text-yellow-400 fill-yellow-400`} />
      Star on GitHub
      <span className="opacity-60">1.8k</span>
    </button>
  );

  const continueFree = (
    <button onClick={() => { Analytics.upsellContinueFree(upsellSource); handleClose(); }} className={textBtn}>
      Continue with free tier <ArrowRight className="h-3 w-3" />
    </button>
  );

  return (
    <div
      className="fixed inset-0 z-[10000] flex items-end sm:items-center justify-center bg-zinc-950/30 backdrop-blur-[2px] animate-sheet-fade"
      onMouseDown={e => { if (e.target === e.currentTarget) handleClose(); }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="relative w-full sm:max-w-md sm:mx-4 max-h-[92vh] overflow-y-auto rounded-t-[28px] sm:rounded-[28px] bg-white dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100 shadow-[0_24px_80px_-16px_rgba(0,0,0,0.35)] ring-1 ring-black/5 dark:ring-white/10 animate-sheet-in"
        onMouseDown={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-end px-5 pt-5">
          <button
            onClick={handleClose}
            aria-label="Close"
            className="flex items-center justify-center h-8 w-8 rounded-full text-zinc-400 hover:text-zinc-700 hover:bg-zinc-100 dark:hover:text-zinc-200 dark:hover:bg-zinc-800 transition-colors"
          >
            <CloseIcon className="h-4 w-4" />
          </button>
        </div>

        <div className="px-7 pb-7 -mt-2 flex flex-col items-center text-center">
          <div className="flex items-center justify-center h-12 w-12 rounded-2xl bg-zinc-100 dark:bg-zinc-800">
            <img src="/eye-logo-black.svg" alt="Observer" className="h-7 w-7 dark:invert" />
          </div>
          <h2 className="mt-4 text-[19px] font-semibold tracking-tight">{title}</h2>
          <p className="mt-1.5 text-sm leading-relaxed text-zinc-500 dark:text-zinc-400 max-w-xs">{subtitle}</p>

          <div className="mt-6 w-full">
            {variant === 'activation' ? (
              /* ============ ACTIVATION: Star + continue (no repeat Pro pitch) ============ */
              <div className="flex flex-col items-center gap-4">
                <p className="inline-flex items-center gap-1.5 text-xs text-zinc-500 dark:text-zinc-400">
                  <Heart className="h-3.5 w-3.5 text-pink-500 fill-pink-500" />
                  Built by a solo developer. Help more people find Observer.
                </p>
                {starButton(true)}
                {continueFree}
              </div>
            ) : (
              /* ============ ONBOARDING: Observer Pro free trial ============ */
              <div className="flex flex-col items-center gap-4">
                <div className="w-full rounded-2xl ring-1 ring-zinc-200 dark:ring-zinc-800 p-5 text-left">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h3 className="text-base font-semibold tracking-tight">Observer Pro</h3>
                      <p className="mt-0.5 text-sm text-zinc-500 dark:text-zinc-400">
                        {isAppleDevice ? (
                          <><span className="font-semibold text-zinc-900 dark:text-zinc-100">$22.99</span> / month</>
                        ) : (
                          <>Free for 7 days, then <span className="font-semibold text-zinc-900 dark:text-zinc-100">$20</span> / month</>
                        )}
                      </p>
                    </div>
                    {!isAppleDevice && (
                      <span className="flex-shrink-0 rounded-full bg-zinc-100 dark:bg-zinc-800 px-2.5 py-1 text-[11px] font-medium text-zinc-600 dark:text-zinc-300">
                        7-day trial
                      </span>
                    )}
                  </div>

                  <div className="my-4 h-px bg-zinc-100 dark:bg-zinc-800" />

                  <ul className="space-y-3">
                    {feature('Extended agent use', null)}
                    {feature('12 hr/day, 100 hr/mo', 'of cloud monitoring',
                      <CreditInfoButton dailyCredits={1440} monthlyCredits={12000} tierName="Pro tier" className="ml-1 align-middle" />)}
                    {feature('SMS, phone call & WhatsApp', 'alerts')}
                  </ul>
                </div>

                <button
                  onClick={() => {
                    Analytics.upsellFreeTrial(upsellSource);
                    if (isAppleDevice) handleApplePurchasePro(); else handleProCheckout();
                  }}
                  disabled={busy}
                  className={primaryBtn}
                >
                  {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <>Start free trial <ArrowRight className="h-4 w-4" /></>}
                </button>

                {(error || appleError) && (
                  <p className="-mt-1 text-xs text-red-500">{error || appleError}</p>
                )}

                {/* Refund guarantee. Not on iOS: App Store refunds go through Apple, not us. */}
                {!isAppleDevice && (
                  <div className="w-full flex items-start gap-3 rounded-2xl bg-zinc-50 dark:bg-zinc-800/50 px-4 py-3 text-left">
                    <ShieldCheck className="h-4 w-4 mt-0.5 flex-shrink-0 text-emerald-500" />
                    <p className="text-xs leading-5 text-zinc-500 dark:text-zinc-400">
                      <span className="font-medium text-zinc-900 dark:text-zinc-100">Didn't like it or forgot to cancel?</span>{' '}
                      Email me at{' '}
                      <a href="mailto:help@observer-ai.com?subject=Refund%20request" className="font-medium text-zinc-700 dark:text-zinc-200 underline underline-offset-2 hover:text-zinc-900 dark:hover:text-white">help@observer-ai.com</a>
                      . I refund everyone.
                    </p>
                  </div>
                )}

                <p className="-mt-1 inline-flex items-center gap-1.5 text-[11px] text-zinc-400">
                  <Heart className="h-3 w-3 text-pink-500 fill-pink-500" />
                  Built by a solo developer
                  <span>·</span>
                  <a href="https://observer-ai.com/#/Terms" target="_blank" rel="noopener noreferrer" className="hover:text-zinc-600 dark:hover:text-zinc-300">Terms</a>
                  <span>·</span>
                  <a href="https://observer-ai.com/#/Privacy" target="_blank" rel="noopener noreferrer" className="hover:text-zinc-600 dark:hover:text-zinc-300">Privacy</a>
                </p>

                <div className="w-full flex items-center justify-between pt-4 border-t border-zinc-100 dark:border-zinc-800">
                  {starButton(false)}
                  {continueFree}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
