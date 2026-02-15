// src/components/PricingTable.tsx

import React, { useState, useCallback } from 'react';
import {
  Loader2, Zap, ExternalLink, Heart, HeartCrack,
  Check, HardDrive, Server, Sparkles, RotateCcw
} from 'lucide-react';
import { isIOS } from '../utils/platform';
import { Logger } from '@utils/logging';
import type { UseApplePaymentsReturn } from '@hooks/useApplePayments';
import { openUrl } from '@tauri-apps/plugin-opener';

// Define the props this component will accept
interface PricingTableProps {
  headline: string;
  subheadline: string;
  status: 'loading' | 'plus' | 'pro' | 'max' | 'free' | 'error';
  isButtonLoading: boolean;
  isAuthenticated: boolean;
  error: string | null;
  onCheckout: () => void;
  onCheckoutPlus: () => void;
  onCheckoutMax: () => void;
  onManageSubscription: () => void;
  onLogin: () => void;
  isTriggeredByQuotaError?: boolean; // Optional prop for special styling
  isHalfwayWarning?: boolean; // New prop to indicate halfway warning vs full limit
  // Apple In-App Purchase - pass the whole hook result
  applePayments?: UseApplePaymentsReturn | null;
  onModalClose?: () => void; // Optional callback to close modal after purchase
}

export const PricingTable: React.FC<PricingTableProps> = ({
  headline,
  subheadline,
  status,
  isButtonLoading,
  isAuthenticated,
  error,
  onCheckout,
  onCheckoutPlus,
  onCheckoutMax,
  onManageSubscription,
  onLogin,
  isTriggeredByQuotaError = false,
  isHalfwayWarning = false,
  applePayments,
  onModalClose,
}) => {
  const isAppleDevice = isIOS();
  const [internalLoading, setInternalLoading] = useState(false);

  // Apple In-App Purchase handler - just purchase and navigate to /upgrade-success
  // Verification is handled by UpgradeSuccessPage
  const handleApplePurchase = useCallback(async (tier: 'plus' | 'pro' | 'max') => {
    if (!applePayments) return;

    Logger.info('PRICING_TABLE', `Starting Apple purchase for tier: ${tier}`);
    setInternalLoading(true);

    try {
      const purchaseResult = await applePayments.purchaseProduct(tier);
      Logger.info('PRICING_TABLE', 'Purchase result:', purchaseResult);

      if (purchaseResult.success) {
        Logger.info('PRICING_TABLE', 'StoreKit purchase succeeded, navigating to /upgrade-success');

        // Close modal if provided
        if (onModalClose) {
          onModalClose();
        }

        // Navigate to success page - it will handle verification and polling
        window.location.href = '/upgrade-success';
      }
      // If not successful, error is already set by the hook
    } catch (err) {
      Logger.error('PRICING_TABLE', 'Purchase failed:', err);
    } finally {
      setInternalLoading(false);
    }
  }, [applePayments, onModalClose]);

  const handleAppleRestore = useCallback(async () => {
    if (!applePayments) return;

    Logger.info('PRICING_TABLE', 'Starting Apple restore');
    setInternalLoading(true);

    try {
      const result = await applePayments.restorePurchases();
      Logger.info('PRICING_TABLE', 'Restore result:', result);

      if (result.success) {
        Logger.info('PRICING_TABLE', 'Restore successful! Navigating to /upgrade-success');

        // Close modal if provided
        if (onModalClose) {
          onModalClose();
        }

        // Navigate to success page
        window.location.href = '/upgrade-success';
      }
    } catch (err) {
      Logger.error('PRICING_TABLE', 'Restore failed:', err);
    } finally {
      setInternalLoading(false);
    }
  }, [applePayments, onModalClose]);

  // On Apple, redirect to Apple subscription management instead of Stripe portal
  const handleManageSubscription = isAppleDevice && applePayments
    ? () => {
        Logger.info('PRICING_TABLE', `handleManageSubscription: Apple device, opening App Store subscriptions. isAppleDevice=${isAppleDevice}, applePayments=${!!applePayments}`);
        openUrl('https://apps.apple.com/account/subscriptions')
          .then(() => Logger.info('PRICING_TABLE', 'openUrl succeeded'))
          .catch((err: unknown) => Logger.error('PRICING_TABLE', 'openUrl failed:', err));
      }
    : () => {
        Logger.info('PRICING_TABLE', `handleManageSubscription: falling back to onManageSubscription. isAppleDevice=${isAppleDevice}, applePayments=${!!applePayments}`);
        onManageSubscription();
      };

  // Determine which checkout handlers to use (Apple native or Stripe web)
  const handlePlusCheckout = isAppleDevice && applePayments
    ? () => handleApplePurchase('plus')
    : onCheckoutPlus;
  const handleProCheckout = isAppleDevice && applePayments
    ? () => handleApplePurchase('pro')
    : onCheckout;
  const handleMaxCheckout = isAppleDevice && applePayments
    ? () => handleApplePurchase('max')
    : onCheckoutMax;

  // Combine loading states
  const combinedLoading = isButtonLoading || internalLoading || (applePayments?.isLoading ?? false);
  const combinedError = error || (applePayments?.error ?? null);

  // Use smaller sizing for non-modal contexts (like ObServerTab)
  const containerClass = isTriggeredByQuotaError
    ? "w-full max-w-6xl mx-auto p-4 sm:p-6 md:p-8 bg-white rounded-lg"  // Modal - larger
    : "w-full max-w-6xl mx-auto p-4 sm:p-6 md:p-8 bg-white rounded-lg";  // Tab - same as modal

  return (
    <div className={containerClass}>
      <div className="flex items-start gap-4 mb-4">
        <div className="flex-shrink-0">
          <Zap className={isTriggeredByQuotaError ? "h-12 w-12 text-purple-500" : "h-8 w-8 text-purple-500"} />
        </div>
        <div className="flex-1">
          <h1 className={isTriggeredByQuotaError ? "text-2xl sm:text-3xl font-bold text-gray-800 tracking-tight mb-2" : "text-xl sm:text-2xl font-bold text-gray-800 tracking-tight mb-2"}>{headline}</h1>
          <p className={isTriggeredByQuotaError ? "text-base sm:text-lg text-gray-600 mb-3" : "text-sm sm:text-base text-gray-600 mb-2"}>{subheadline}</p>
          {isTriggeredByQuotaError && (
            <div className="flex items-center space-x-3 text-xs text-gray-500">
              <span>🚀 1k+ users</span>
              <span>•</span>
              <span>⭐ 1k+ GitHub stars</span>
              <span>•</span>
              <span>⚡ Processing 100k+ captures daily</span>
            </div>
          )}
        </div>
      </div>

      <div className={isTriggeredByQuotaError ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-0 items-start" : "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-0 items-start"}>
        {/* Column 1: Free Tier */}
        <div className={`relative border rounded-lg p-4 grid grid-rows-[auto_1fr_auto] h-full bg-white shadow-md ${status === 'free' && isTriggeredByQuotaError ? 'border-gray-400 border-2' : ''}`}>
           {status === 'free' && isTriggeredByQuotaError && (
             <div className="absolute top-0 -translate-y-1/2 left-1/2 -translate-x-1/2">
               <div className="bg-gray-500 text-white text-xs font-bold uppercase tracking-wider rounded-full px-6 py-1 whitespace-nowrap">Current</div>
             </div>
           )}
          <div className="text-center mb-4">
            <Server className="mx-auto h-12 w-12 text-gray-500 mb-3" />
            <h2 className="text-2xl font-bold text-gray-800">Quick Start</h2>
            <p className="text-gray-500 text-sm">Perfect for trying out Observer</p>
            <p className="text-3xl font-bold text-gray-900 mt-2">$0<span className="text-base font-normal">/month</span></p>
          </div>
          <ul className="space-y-2.5 mb-6 text-sm">
            <li className="flex items-start">
              <HardDrive className="h-5 w-5 text-blue-500 mr-2 flex-shrink-0 mt-0.5" />
              <span><strong>Local Monitoring 24/7 </strong></span>
            </li>
            <li className="flex items-start">
              <Check className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
              <span>
                <strong>Unlimited Discord Notifications</strong>
              </span>
            </li>
            <li className="flex items-start">
              <Check className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
              <span>
                <strong>Notifications: </strong> SMS, Phone Calling, Email, WhatsApp, Telegram, Pushover <strong>5/Day</strong>
              </span>
            </li>
            <li className="flex items-start">
              <Check className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
              <span>
                <strong>Cloud Monitoring:</strong> 30 min/day
                {isHalfwayWarning && <span className="block text-yellow-600 text-xs mt-1">(Halfway there!)</span>}
              </span>
            </li>
            <li className="flex items-start">
              <HeartCrack className="h-5 w-5 text-red-500 mr-2 flex-shrink-0 mt-0.5" />
              <span><strong>Support:</strong> Limited (solo dev!)</span>
            </li>
          </ul>
          <button
            disabled
            className="w-full inline-flex items-center justify-center px-8 py-3 border border-gray-300 text-base font-bold rounded-md text-gray-400 bg-gray-100 cursor-not-allowed"
          >
            Current Plan
          </button>
        </div>

        {/* Column 2: Plus Tier */}
        <div className={`relative border-2 border-blue-500 rounded-lg p-4 grid grid-rows-[auto_1fr_auto] h-full bg-blue-50 shadow-lg ${status === 'plus' && isTriggeredByQuotaError ? 'ring-2 ring-blue-600' : ''}`}>
          {status === 'plus' && isTriggeredByQuotaError && (
            <div className="absolute top-0 -translate-y-1/2 left-1/2 -translate-x-1/2">
              <div className="bg-blue-600 text-white text-xs font-bold uppercase tracking-wider rounded-full px-6 py-1 whitespace-nowrap">Current</div>
            </div>
          )}
          {status !== 'plus' && status === 'free' && (
            <div className="absolute top-0 -translate-y-1/2 left-1/2 -translate-x-1/2">
              <div className="bg-blue-500 text-white text-xs font-bold uppercase tracking-wider rounded-full px-4 py-1 whitespace-nowrap">Popular</div>
            </div>
          )}
          <div className="text-center mb-4">
            <Zap className="mx-auto h-12 w-12 text-blue-500 mb-3" />
            <h2 className="text-2xl font-bold text-blue-800">Observer Plus</h2>
            <p className="text-blue-700 text-sm">Unlimited alerts for local monitoring</p>
            <p className="text-3xl font-bold text-blue-900 mt-2">${isAppleDevice ? '5.99' : '5'}<span className="text-base font-normal">/month</span></p>
          </div>
          <ul className="space-y-2.5 mb-6 text-sm">
            <li className="flex items-start">
              <Check className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
              <span><strong>Unlimited notifications:</strong> SMS, Phone Calling, Email, WhatsApp, Telegram, Pushover
</span>
            </li>
            <li className="flex items-start">
              <Check className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
              <span><strong>Unlimited Alert Builder</strong></span>
            </li>
            <li className="flex items-start">
              <Heart className="h-5 w-5 text-pink-500 mr-2 flex-shrink-0 mt-0.5" />
              <span><strong>Better support</strong></span>
            </li>
          </ul>
          {status === 'plus' ? (
            <button
              onClick={handleManageSubscription}
              disabled={combinedLoading}
              className="w-full inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-blue-700 bg-blue-100 hover:bg-blue-200 disabled:bg-gray-300 transition-colors"
            >
              {combinedLoading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <ExternalLink className="mr-2 h-4 w-4" />}
              Manage Subscription
            </button>
          ) : (
            <button
              onClick={isAuthenticated ? handlePlusCheckout : onLogin}
              disabled={combinedLoading}
              className="w-full inline-flex items-center justify-center px-8 py-3 border border-transparent text-base font-bold rounded-md text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 transition-colors"
            >
              {combinedLoading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : null}
              Upgrade to Plus
            </button>
          )}
        </div>

        {/* Column 3: Pro Tier */}
        <div className={`relative border-2 border-purple-500 rounded-lg p-4 grid grid-rows-[auto_1fr_auto] h-full bg-purple-50 shadow-lg ${status === 'pro' && isTriggeredByQuotaError ? 'ring-2 ring-purple-600' : ''}`}>
          {status === 'pro' && isTriggeredByQuotaError && (
            <div className="absolute top-0 -translate-y-1/2 left-1/2 -translate-x-1/2">
              <div className="bg-purple-600 text-white text-xs font-bold uppercase tracking-wider rounded-full px-6 py-1 whitespace-nowrap">Current</div>
            </div>
          )}
          {status !== 'pro' && (
            <div className="absolute top-0 -translate-y-1/2 left-1/2 -translate-x-1/2">
              <div className="bg-purple-500 text-white text-xs font-bold uppercase tracking-wider rounded-full px-4 py-1 whitespace-nowrap">Best Value</div>
            </div>
          )}
          <div className="text-center mb-4">
            <Sparkles className="mx-auto h-12 w-12 text-purple-500 mb-3" />
            <h2 className="text-2xl font-bold text-purple-800">Observer Pro</h2>
            <p className="text-purple-700 text-sm">Coordinate agent teams, solve complex tasks</p>
            <p className="text-3xl font-bold text-purple-900 mt-2">${isAppleDevice ? '22.99' : '20'}<span className="text-base font-normal">/month</span></p>
          </div>
          <ul className="space-y-2.5 mb-6 text-sm">
            <li className="flex items-start">
              <Check className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
              <span><strong>Unlimited notifications</strong></span>
            </li>
            <li className="flex items-start">
              <Check className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
              <span><strong>Unlimited Alert Builder</strong></span>
            </li>
            <li className="flex items-start">
              <Check className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
              <span><strong>Cloud Monitoring:</strong> 8 hours/day!<br/></span>
            </li>
            <li className="flex items-start">
              <Check className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
              <span><strong>Premium Models</strong></span>
            </li>
            <li className="flex items-start">
              <Sparkles className="h-5 w-5 text-purple-500 mr-2 flex-shrink-0 mt-0.5" />
              <span><strong>Unlock AI Studio: Multi-Agent configurations</strong></span>
            </li>
            <li className="flex items-start">
              <Heart className="h-5 w-5 text-pink-500 mr-2 flex-shrink-0 mt-0.5" />
              <span><strong>Better support</strong></span>
            </li>
          </ul>
          {status === 'pro' ? (
            <button
              onClick={handleManageSubscription}
              disabled={combinedLoading}
              className="w-full inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-purple-700 bg-purple-100 hover:bg-purple-200 disabled:bg-gray-300 transition-colors"
            >
              {combinedLoading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <ExternalLink className="mr-2 h-4 w-4" />}
              Manage Subscription
            </button>
          ) : (
            <button
              onClick={isAuthenticated ? handleProCheckout : onLogin}
              disabled={combinedLoading}
              className="w-full inline-flex items-center justify-center px-8 py-3 border border-transparent text-base font-bold rounded-md text-white bg-purple-600 hover:bg-purple-700 disabled:bg-gray-300 transition-colors"
            >
              {combinedLoading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : null}
              Start Free Trial
            </button>
          )}
        </div>

        {/* Column 4: Max Tier */}
        <div className={`relative border-2 border-amber-500 rounded-lg p-4 grid grid-rows-[auto_1fr_auto] h-full bg-gradient-to-br from-amber-50 to-yellow-50 shadow-xl ${status === 'max' && isTriggeredByQuotaError ? 'ring-2 ring-amber-600' : ''}`}>
          {status === 'max' && isTriggeredByQuotaError && (
            <div className="absolute top-0 -translate-y-1/2 left-1/2 -translate-x-1/2">
              <div className="bg-amber-600 text-white text-xs font-bold uppercase tracking-wider rounded-full px-6 py-1 whitespace-nowrap">Current</div>
            </div>
          )}
          {status !== 'max' && (
            <div className="absolute top-0 -translate-y-1/2 left-1/2 -translate-x-1/2">
              <div className="bg-gradient-to-r from-amber-500 to-yellow-500 text-white text-xs font-bold uppercase tracking-wider rounded-full px-4 py-1 whitespace-nowrap shadow-lg">Premium</div>
            </div>
          )}
          <div className="text-center mb-4">
            <div className="flex items-center justify-center gap-2 mb-3">
              <Zap className="h-12 w-12 text-amber-500" />
            </div>
            <h2 className="text-2xl font-bold text-amber-900">Observer Max</h2>
            <p className="text-amber-800 text-sm font-semibold">Absolutely everything, unlimited</p>
            <p className="text-3xl font-bold text-amber-900 mt-2">${isAppleDevice ? '99.99' : '80'}<span className="text-base font-normal">/month</span></p>
          </div>
          <ul className="space-y-2.5 mb-6 text-sm">
            <li className="flex items-start">
              <Zap className="h-5 w-5 text-amber-500 mr-2 flex-shrink-0 mt-0.5" />
              <span><strong>24/7 Unlimited Cloud Monitoring</strong></span>
            </li>
            <li className="flex items-start">
              <Check className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
              <span><strong>Everything Unlimited</strong></span>
            </li>
            <li className="flex items-start">
              <Sparkles className="h-5 w-5 text-amber-500 mr-2 flex-shrink-0 mt-0.5" />
              <span><strong>Unlimited Premium Models</strong></span>
            </li>
            <li className="flex items-start">
              <Sparkles className="h-5 w-5 text-amber-500 mr-2 flex-shrink-0 mt-0.5" />
              <span><strong>Unlock AI Studio</strong></span>
            </li>
            <li className="flex items-start">
              <Heart className="h-5 w-5 text-pink-500 mr-2 flex-shrink-0 mt-0.5" />
              <span><strong>Priority Support</strong></span>
            </li>
          </ul>
          {status === 'max' ? (
            <button
              onClick={handleManageSubscription}
              disabled={combinedLoading}
              className="w-full inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-amber-700 bg-amber-100 hover:bg-amber-200 disabled:bg-gray-300 transition-colors"
            >
              {combinedLoading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <ExternalLink className="mr-2 h-4 w-4" />}
              Manage Subscription
            </button>
          ) : (
            <button
              onClick={isAuthenticated ? handleMaxCheckout : onLogin}
              disabled={combinedLoading}
              className="w-full inline-flex items-center justify-center px-8 py-3 border border-transparent text-base font-bold rounded-md text-white bg-gradient-to-r from-amber-500 to-yellow-500 hover:from-amber-600 hover:to-yellow-600 disabled:bg-gray-300 transition-all shadow-md"
            >
              {combinedLoading ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : null}
              Upgrade to Max
            </button>
          )}
        </div>
      </div>
      {combinedError && (
        <div className="text-center mt-8">
          <p className="text-sm text-red-600 font-semibold">{combinedError}</p>
        </div>
      )}
      {/* iOS Debug & Restore Purchases */}
      {isAppleDevice && applePayments && isAuthenticated && (
        <div className="text-center mt-4 flex justify-center gap-4">
          {applePayments.loadProducts && (
            <button
              onClick={applePayments.loadProducts}
              disabled={combinedLoading}
              className="inline-flex items-center text-sm text-blue-600 hover:text-blue-800 disabled:text-gray-300 font-medium"
            >
              <Sparkles className="h-4 w-4 mr-1" />
              Load Products
            </button>
          )}
          <button
            onClick={handleAppleRestore}
            disabled={combinedLoading}
            className="inline-flex items-center text-sm text-gray-500 hover:text-gray-700 disabled:text-gray-300"
          >
            <RotateCcw className="h-4 w-4 mr-1" />
            Restore Purchases
          </button>
        </div>
      )}
      {/* Terms of Service & Privacy Policy - Required for App Store */}
      <div className="text-center mt-6 pt-4 border-t border-gray-200">
        <p className="text-xs text-gray-500">
          By subscribing, you agree to our{' '}
          {isIOS() ? (
            <>
              <button
                onClick={() => openUrl('https://observer-ai.com/#/Terms')}
                className="text-blue-600 hover:text-blue-800 underline"
              >
                Terms of Service
              </button>
              {' '}and{' '}
              <button
                onClick={() => openUrl('https://observer-ai.com/#/Privacy')}
                className="text-blue-600 hover:text-blue-800 underline"
              >
                Privacy Policy
              </button>
            </>
          ) : (
            <>
              <a href="https://observer-ai.com/#/Terms" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800 underline">Terms of Service</a>
              {' '}and{' '}
              <a href="https://observer-ai.com/#/Privacy" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800 underline">Privacy Policy</a>
            </>
          )}
          .
        </p>
      </div>
    </div>
  );
};
