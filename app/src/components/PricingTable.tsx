// src/components/PricingTable.tsx

import React, { useState, useCallback } from 'react';
import {
  Loader2, Zap, ExternalLink,
  Check, X, Sparkles, RotateCcw, Building2, Users
} from 'lucide-react';
import { CreditInfoButton } from './CreditVisualization';
import { InfoTooltip } from './InfoTooltip';
import { isIOS } from '../utils/platform';
import { Logger } from '@utils/logging';
import type { UseApplePaymentsReturn } from '@hooks/useApplePayments';
import { openUrl } from '@tauri-apps/plugin-opener';

interface PricingTableProps {
  headline: string;
  subheadline: string;
  status: 'loading' | 'plus' | 'pro' | 'max' | 'free' | 'error' | 'enterprise';
  /** Tier the org seat grants ('pro' | 'max'), only meaningful when status is 'enterprise'. */
  orgTier?: string | null;
  isButtonLoading: boolean;
  isAuthenticated: boolean;
  error: string | null;
  onCheckout: () => void;
  onCheckoutPlus: () => void;
  onCheckoutMax: () => void;
  onManageSubscription: () => void;
  onLogin: () => void;
  isTriggeredByQuotaError?: boolean;
  isHalfwayWarning?: boolean;
  applePayments?: UseApplePaymentsReturn | null;
  onModalClose?: () => void;
}

interface FeatureRow {
  label: string;
  sparkle?: boolean;
  notLoggedIn: boolean | string;
  free: boolean | string;
  pro: boolean | string;
  max: boolean | string;
  creditInfo?: { free?: number; pro?: number; max?: number };
  monthlyCreditInfo?: { free?: number; pro?: number; max?: number };
  info?: { notLoggedIn?: string; free?: string; pro?: string; max?: string };
}

interface FeatureGroup {
  group: string;
  rows: FeatureRow[];
}

const featureGroups: FeatureGroup[] = [
  {
    group: 'Core — Free Forever',
    rows: [
      { label: 'Local Models',             notLoggedIn: true,  free: true,       pro: true,        max: true },
      { label: 'Logging & Recording',      notLoggedIn: true,  free: true,       pro: true,        max: true },
      { label: 'Discord Notifications',    notLoggedIn: true,  free: true,       pro: true,        max: true },
    ],
  },
  {
    group: 'It drives itself',
    rows: [
      { label: 'Micro-Agent Builder', sparkle: true, notLoggedIn: false, free: '3 agents / day', pro: 'Unlimited', max: 'Unlimited',
        info: {
          free: 'Building a micro-agent takes ~15 messages with Observer on average, and the free tier gives you 45/day, about 3 full builds. Plenty to design and iterate.',
          pro: "1,000 messages/day with Observer, roughly 67 micro-agents in a single day. If you genuinely need to spin up more than 67 a day, one subscription was never going to cover that 😅. Reach out and we'll figure it out.",
          max: "1,000 messages/day with Observer, roughly 67 micro-agents in a single day. If you genuinely need to spin up more than 67 a day, one subscription was never going to cover that 😅. Reach out and we'll figure it out.",
        } },
      { label: 'Cloud Monitoring',         notLoggedIn: false, free: '1 hr / day', pro: '8 hr/day · 100 hr/mo', max: '24 / 7', creditInfo: { free: 60, pro: 480, max: 2880 }, monthlyCreditInfo: { pro: 6000 } },
    ],
  },
  {
    group: 'Notifications',
    rows: [
      { label: 'Telegram, Email & Pushover', notLoggedIn: false, free: true,      pro: true,        max: true },
      { label: 'SMS, Phone & WhatsApp',      notLoggedIn: false, free: '5 / day', pro: 'Unlimited', max: 'Unlimited',
        info: {
          pro: "Practically unlimited 100/day as a guard against abuse. I use Observer every day and have never come close. Need more for a legit use case? Just email me, it's a solo project and I'm happy to help.",
          max: "Practically unlimited 100/day as a guard against abuse. I use Observer every day and have never come close. Need more for a legit use case? Just email me, it's a solo project and I'm happy to help.",
        } },
    ],
  },
  {
    group: 'Support',
    rows: [
      { label: 'Support', notLoggedIn: "We don't know you!", free: 'Limited', pro: 'Better', max: 'Priority' },
    ],
  },
];

const CheckMark = () => <Check className="h-5 w-5 text-green-500 mx-auto" />;
const CrossMark = () => <X className="h-5 w-5 text-gray-300 dark:text-gray-600 mx-auto" />;

const renderCell = (value: boolean | string, dailyCredits?: number, tierName?: string, info?: string, monthlyCredits?: number) => {
  const infoBtn = info ? <InfoTooltip body={info} className="align-middle" /> : null;
  if (value === true) {
    return (
      <span className="inline-flex items-center justify-center gap-1">
        <CheckMark />
        {infoBtn}
      </span>
    );
  }
  if (value === false) return <CrossMark />;
  return (
    <span className="inline-flex items-center justify-center gap-1 text-sm font-semibold text-gray-700 dark:text-gray-300">
      {value}
      {dailyCredits !== undefined && tierName && (
        <CreditInfoButton dailyCredits={dailyCredits} monthlyCredits={monthlyCredits} tierName={tierName} className="align-middle" />
      )}
      {infoBtn}
    </span>
  );
};

export const PricingTable: React.FC<PricingTableProps> = ({
  headline,
  subheadline,
  status,
  orgTier,
  isButtonLoading,
  isAuthenticated,
  error,
  onCheckout,
  onCheckoutMax,
  onManageSubscription,
  onLogin,
  isTriggeredByQuotaError = false,
  applePayments,
  onModalClose,
}) => {
  const isAppleDevice = isIOS();
  const [internalLoading, setInternalLoading] = useState(false);

  // plus is a legacy tier — treat it as pro
  const effectiveStatus = status === 'plus' ? 'pro' : status;

  const handleApplePurchase = useCallback(async (tier: 'pro' | 'max') => {
    if (!applePayments) return;
    Logger.info('PRICING_TABLE', `Starting Apple purchase for tier: ${tier}`);
    setInternalLoading(true);
    try {
      const result = await applePayments.purchaseProduct(tier);
      Logger.info('PRICING_TABLE', 'Purchase result:', result);
      if (result.success) {
        if (onModalClose) onModalClose();
        window.location.href = '/upgrade-success';
      }
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
        if (onModalClose) onModalClose();
        window.location.href = '/upgrade-success';
      }
    } catch (err) {
      Logger.error('PRICING_TABLE', 'Restore failed:', err);
    } finally {
      setInternalLoading(false);
    }
  }, [applePayments, onModalClose]);

  const handleManageSubscription = isAppleDevice && applePayments
    ? () => openUrl('https://apps.apple.com/account/subscriptions')
        .catch((err: unknown) => Logger.error('PRICING_TABLE', 'openUrl failed:', err))
    : onManageSubscription;

  const handleProCheckout  = isAppleDevice && applePayments ? () => handleApplePurchase('pro')  : onCheckout;
  const handleMaxCheckout  = isAppleDevice && applePayments ? () => handleApplePurchase('max')  : onCheckoutMax;

  const combinedLoading = isButtonLoading || internalLoading || (applePayments?.isLoading ?? false);
  const combinedError   = error || (applePayments?.error ?? null);

  // ── table column helpers ──────────────────────────────────────────────────
  const headerBase = "text-center align-bottom px-3 pt-2 pb-3 md:pt-3 md:pb-4 text-sm font-bold";
  const cellBase   = "text-center px-3 py-2 md:py-3";

  const getHeaderClass = (tier: 'free' | 'pro' | 'max') => {
    const current = effectiveStatus === tier;
    if (tier === 'pro') return `${headerBase} rounded-t-lg ${current ? 'bg-purple-600 text-white dark:bg-purple-700' : 'bg-purple-50 text-purple-900 dark:bg-purple-900/30 dark:text-purple-200'}`;
    if (tier === 'max') return `${headerBase} rounded-t-lg ${current ? 'bg-amber-500  text-white dark:bg-amber-600' : 'bg-amber-50  text-amber-900 dark:bg-amber-900/30 dark:text-amber-200'}`;
    return `${headerBase} bg-gray-50 text-gray-600 dark:bg-gray-900 dark:text-gray-400`;
  };

  const getCellClass = (tier: 'notLoggedIn' | 'free' | 'pro' | 'max') => {
    const current = tier !== 'notLoggedIn' && effectiveStatus === tier;
    if (tier === 'pro') return `${cellBase} ${current ? 'bg-purple-50/60 dark:bg-purple-900/20' : ''}`;
    if (tier === 'max') return `${cellBase} ${current ? 'bg-amber-50/60 dark:bg-amber-900/20'  : ''}`;
    return cellBase;
  };

  const dataColCount = isAuthenticated ? 4 : 3; // label + data columns

  // Enterprise seats are billed to the org, not the user — there is no personal
  // subscription to show or manage, so point them at their team page instead.
  if (status === 'enterprise') {
    const tierLabel = orgTier === 'max' ? 'Max' : 'Pro';
    return (
      <div className="w-full max-w-2xl mx-auto p-5 md:p-8 bg-white dark:bg-gray-800 rounded-none md:rounded-lg text-center">
        <Building2 className="mx-auto h-12 w-12 text-purple-500 mb-3" />
        <h1 className="text-xl sm:text-2xl font-bold text-gray-800 dark:text-gray-100 tracking-tight">
          You're on an Enterprise plan
        </h1>
        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
          Your seat is provided by your organization and includes all {tierLabel} features,
          drawing from a shared monthly hours pool quoted for your organization.
          Billing, seats, and the pool are managed by your team's owner.
        </p>
        <a
          href="/team"
          className="mt-5 inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg text-sm font-bold text-white bg-purple-600 hover:bg-purple-700 transition-colors shadow-md"
        >
          <Users className="h-4 w-4" />
          View your team
        </a>
        {combinedError && (
          <p className="mt-3 text-sm text-red-600 dark:text-red-400 font-semibold">{combinedError}</p>
        )}
      </div>
    );
  }

  return (
    <div className="w-full max-w-4xl mx-auto p-3 md:p-5 bg-white dark:bg-gray-800 rounded-none md:rounded-lg">

      {/* ── Header ── */}
      <div className="flex items-start gap-3 mb-3 md:mb-4">
        <Zap className={`flex-shrink-0 mt-0.5 text-purple-500 ${isTriggeredByQuotaError ? 'h-8 w-8' : 'h-6 w-6'}`} />
        <div>
          <h1 className={`font-bold text-gray-800 dark:text-gray-100 tracking-tight ${isTriggeredByQuotaError ? 'text-xl sm:text-2xl' : 'text-lg sm:text-xl'}`}>
            {headline}
          </h1>
          <p className="text-xs sm:text-sm text-gray-500 dark:text-gray-400 mt-0.5">{subheadline}</p>
          {isTriggeredByQuotaError && (
            <div className="flex items-center gap-3 mt-1.5 text-xs text-gray-400 dark:text-gray-500">
              <span>🚀 2k+ users</span>
              <span>•</span>
              <span>⭐ 1k+ GitHub stars</span>
              <span>•</span>
              <span>⚡ 100k+ captures/day</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Comparison Table with pricing + CTA embedded in headers ── */}
      <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm mb-3 md:mb-4">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              <th className="text-left px-4 py-3 text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500 bg-gray-50 dark:bg-gray-900 w-2/5">
                Features
              </th>
              {isAuthenticated ? (
                <>
                  {/* Free column */}
                  <th className={getHeaderClass('free')}>
                    <div aria-hidden className="mb-1 inline-block rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide invisible">Most Popular</div>
                    <div>Quick Start</div>
                    <div className="text-xs font-normal text-gray-400 dark:text-gray-500 mt-0.5">$0 / mo</div>
                    <div className="text-[11px] font-semibold text-gray-400 dark:text-gray-500 mt-1">You drive</div>
                    {effectiveStatus === 'free' ? (
                      <span className="mt-2 block text-xs bg-gray-200 text-gray-600 dark:bg-gray-700 dark:text-gray-300 px-2 py-0.5 rounded-full font-medium">
                        Current
                      </span>
                    ) : (
                      <button disabled className="mt-2 w-full py-1.5 rounded-lg border border-gray-200 dark:border-gray-600 text-sm font-medium text-gray-400 dark:text-gray-500 bg-gray-100 dark:bg-gray-700/50 cursor-not-allowed">
                        Free Forever
                      </button>
                    )}
                  </th>
                  {/* Pro column */}
                  <th className={getHeaderClass('pro')}>
                    <div className={`mb-1 inline-block rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide ${effectiveStatus === 'pro' ? 'invisible' : 'bg-purple-600 text-white dark:bg-purple-500'}`}>
                      Most Popular
                    </div>
                    <div>Pro</div>
                    <div className={`text-xs font-normal mt-0.5 ${effectiveStatus === 'pro' ? 'text-purple-200' : 'text-purple-500'}`}>
                      ${isAppleDevice ? '22.99' : '20'} / mo
                    </div>
                    <div className={`text-[11px] font-semibold mt-1 ${effectiveStatus === 'pro' ? 'text-purple-100' : 'text-purple-600 dark:text-purple-300'}`}>It drives itself</div>
                    {effectiveStatus === 'pro' ? (
                      <>
                        <span className="mt-2 block text-xs bg-purple-200 text-purple-700 dark:bg-purple-800 dark:text-purple-200 px-2 py-0.5 rounded-full font-medium">
                          Current
                        </span>
                        <button
                          onClick={handleManageSubscription}
                          disabled={combinedLoading}
                          className="mt-1.5 w-full py-1.5 rounded-lg border border-purple-300 dark:border-purple-700 text-sm font-medium text-purple-700 dark:text-purple-200 bg-purple-100 dark:bg-purple-900/40 hover:bg-purple-200 dark:hover:bg-purple-900/60 disabled:opacity-50 transition-colors flex items-center justify-center gap-1"
                        >
                          {combinedLoading ? <Loader2 className="h-3 w-3 animate-spin" /> : <ExternalLink className="h-3 w-3" />}
                          Manage
                        </button>
                      </>
                    ) : (
                      <button
                        onClick={handleProCheckout}
                        disabled={combinedLoading}
                        className="mt-2 w-full py-2 rounded-lg text-sm font-bold text-white bg-purple-600 hover:bg-purple-700 hover:scale-105 disabled:opacity-50 transition-all flex items-center justify-center gap-1 shadow-md ring-2 ring-purple-300"
                      >
                        {combinedLoading && <Loader2 className="h-3 w-3 animate-spin" />}
                        Free Trial
                      </button>
                    )}
                  </th>
                  {/* Max column */}
                  <th className={getHeaderClass('max')}>
                    <div aria-hidden className="mb-1 inline-block rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide invisible">Most Popular</div>
                    <div>Max</div>
                    <div className={`text-xs font-normal mt-0.5 ${effectiveStatus === 'max' ? 'text-amber-100' : 'text-amber-500'}`}>
                      ${isAppleDevice ? '99.99' : '80'} / mo
                    </div>
                    <div className={`text-[11px] font-semibold mt-1 ${effectiveStatus === 'max' ? 'text-amber-100' : 'text-amber-600 dark:text-amber-300'}`}>Always on</div>
                    {effectiveStatus === 'max' ? (
                      <>
                        <span className="mt-2 block text-xs bg-amber-200 text-amber-700 dark:bg-amber-800 dark:text-amber-200 px-2 py-0.5 rounded-full font-medium">
                          Current
                        </span>
                        <button
                          onClick={handleManageSubscription}
                          disabled={combinedLoading}
                          className="mt-1.5 w-full py-1.5 rounded-lg border border-amber-300 dark:border-amber-700 text-sm font-medium text-amber-700 dark:text-amber-200 bg-amber-100 dark:bg-amber-900/40 hover:bg-amber-200 dark:hover:bg-amber-900/60 disabled:opacity-50 transition-colors flex items-center justify-center gap-1"
                        >
                          {combinedLoading ? <Loader2 className="h-3 w-3 animate-spin" /> : <ExternalLink className="h-3 w-3" />}
                          Manage
                        </button>
                      </>
                    ) : (
                      <button
                        onClick={handleMaxCheckout}
                        disabled={combinedLoading}
                        className="mt-2 w-full py-2 rounded-lg text-sm font-bold text-white bg-gradient-to-r from-amber-500 to-yellow-500 hover:from-amber-600 hover:to-yellow-600 hover:scale-105 disabled:opacity-50 transition-all flex items-center justify-center gap-1 shadow-md ring-2 ring-amber-300"
                      >
                        {combinedLoading && <Loader2 className="h-3 w-3 animate-spin" />}
                        Upgrade
                      </button>
                    )}
                  </th>
                </>
              ) : (
                <>
                  <th className={`${headerBase} bg-gray-50 dark:bg-gray-900 text-gray-500 dark:text-gray-400`}>Not Logged In</th>
                  <th className={`${headerBase} bg-purple-50 dark:bg-purple-900/30 text-purple-900 dark:text-purple-200 rounded-t-lg`}>
                    <div>Quick Start</div>
                    <div className="text-xs font-normal text-purple-500 dark:text-purple-300 mt-0.5">Free</div>
                    <div className="text-[11px] font-semibold text-purple-400 dark:text-purple-300 mt-1">You drive</div>
                    <button
                      onClick={onLogin}
                      className="mt-2 w-full py-2 rounded-lg text-sm font-bold text-white bg-purple-600 hover:bg-purple-700 hover:scale-105 transition-all shadow-md ring-2 ring-purple-300"
                    >
                      Get Started
                    </button>
                  </th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {featureGroups.map((group) => (
              <React.Fragment key={group.group}>
                <tr className="bg-gray-50/80 dark:bg-gray-900/50">
                  <td colSpan={dataColCount} className="px-4 py-1 md:py-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-gray-400 dark:text-gray-500">
                      {group.group}
                    </span>
                  </td>
                </tr>
                {group.rows.filter(row => isAuthenticated || !row.sparkle).map((row, i) => (
                  <tr key={row.label} className={`border-t border-gray-100 dark:border-gray-700 ${row.sparkle ? 'bg-purple-200/60 dark:bg-purple-900/30' : i % 2 === 1 ? 'bg-gray-50/30 dark:bg-gray-900/30' : ''}`}>
                    <td className="px-4 py-3 text-sm font-medium">
                      {row.sparkle ? (
                        <span className="inline-flex items-center gap-1.5 text-gray-700 dark:text-gray-300 font-medium">
                          {row.label}
                          <Sparkles className="h-4 w-4 text-purple-500 flex-shrink-0" />
                        </span>
                      ) : (
                        <span className="text-gray-700 dark:text-gray-300">{row.label}</span>
                      )}
                    </td>
                    {isAuthenticated ? (
                      <>
                        <td className={getCellClass('free')}>{renderCell(row.free, row.creditInfo?.free, 'Free tier', row.info?.free, row.monthlyCreditInfo?.free)}</td>
                        <td className={getCellClass('pro')}>{renderCell(row.pro, row.creditInfo?.pro, 'Pro tier', row.info?.pro, row.monthlyCreditInfo?.pro)}</td>
                        <td className={getCellClass('max')}>{renderCell(row.max, row.creditInfo?.max, 'Max tier', row.info?.max, row.monthlyCreditInfo?.max)}</td>
                      </>
                    ) : (
                      <>
                        <td className={getCellClass('notLoggedIn')}>{renderCell(row.notLoggedIn)}</td>
                        <td className={getCellClass('free')}>{renderCell(row.free, row.creditInfo?.free, 'Free tier', row.info?.free, row.monthlyCreditInfo?.free)}</td>
                      </>
                    )}
                  </tr>
                ))}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>

      {/* ── Error ── */}
      {combinedError && (
        <div className="text-center mt-3">
          <p className="text-sm text-red-600 dark:text-red-400 font-semibold">{combinedError}</p>
        </div>
      )}

      {/* ── iOS: Restore & Load Products ── */}
      {isAppleDevice && applePayments && isAuthenticated && (
        <div className="text-center mt-3 flex justify-center gap-4">
          {applePayments.loadProducts && (
            <button
              onClick={applePayments.loadProducts}
              disabled={combinedLoading}
              className="inline-flex items-center text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 disabled:text-gray-300 dark:disabled:text-gray-600 font-medium"
            >
              <Sparkles className="h-4 w-4 mr-1" />
              Load Products
            </button>
          )}
          <button
            onClick={handleAppleRestore}
            disabled={combinedLoading}
            className="inline-flex items-center text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 disabled:text-gray-300 dark:disabled:text-gray-600"
          >
            <RotateCcw className="h-4 w-4 mr-1" />
            Restore Purchases
          </button>
        </div>
      )}

      {/* ── Terms (required for App Store) ── */}
      <div className="text-center mt-3 pt-3 border-t border-gray-100 dark:border-gray-700">
        <p className="text-xs text-gray-400 dark:text-gray-500">
          By subscribing, you agree to our{' '}
          {isIOS() ? (
            <>
              <button onClick={() => openUrl('https://observer-ai.com/#/Terms')} className="text-blue-500 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 underline">
                Terms of Service
              </button>
              {' '}and{' '}
              <button onClick={() => openUrl('https://observer-ai.com/#/Privacy')} className="text-blue-500 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 underline">
                Privacy Policy
              </button>
            </>
          ) : (
            <>
              <a href="https://observer-ai.com/#/Terms" target="_blank" rel="noopener noreferrer" className="text-blue-500 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 underline">Terms of Service</a>
              {' '}and{' '}
              <a href="https://observer-ai.com/#/Privacy" target="_blank" rel="noopener noreferrer" className="text-blue-500 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 underline">Privacy Policy</a>
            </>
          )}
          .
        </p>
      </div>
    </div>
  );
};
