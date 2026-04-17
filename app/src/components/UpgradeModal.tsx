// src/components/UpgradeModal.tsx

import React, { useState, useEffect } from 'react';
import { useAuth } from '@contexts/AuthContext';
import { useApplePayments } from '@hooks/useApplePayments';
import { X as CloseIcon, Loader2 } from 'lucide-react';
import { Logger } from '@utils/logging';
import { PricingTable } from './PricingTable';

interface UpgradeModalProps {
  isOpen: boolean;
  onClose: () => void;
  isHalfwayWarning?: boolean;
  quotaType?: string;
}

// Configuration for quota-specific messaging
const QUOTA_MESSAGES: Record<string, {
  headline: string;
  subheadline: string;
  recommendedTier: 'plus' | 'pro' | 'max';
  isUrgent: boolean;
}> = {
  monitor: {
    headline: "You've Reached Your Daily Monitoring Limit!",
    subheadline: "Upgrade to Observer Pro for 8 hours/day of cloud monitoring!",
    recommendedTier: 'pro',
    isUrgent: true
  },
  agent_creator: {
    headline: "AI Agent Creator Limit Reached",
    subheadline: "Unlock unlimited agent builds with Observer Pro!",
    recommendedTier: 'pro',
    isUrgent: true
  },
  email: {
    headline: "Email Notification Quota Reached",
    subheadline: "Get unlimited email alerts with Observer Plus for just $5/month!",
    recommendedTier: 'plus',
    isUrgent: false
  },
  sms: {
    headline: "SMS Notification Quota Reached",
    subheadline: "Upgrade to Plus for unlimited SMS alerts - just 1¢ per hour of monitoring!",
    recommendedTier: 'plus',
    isUrgent: false
  },
  whatsapp: {
    headline: "WhatsApp Notification Quota Reached",
    subheadline: "Never miss an alert! Get unlimited WhatsApp notifications with Plus.",
    recommendedTier: 'plus',
    isUrgent: false
  },
  telegram: {
    headline: "Telegram Notification Quota Reached",
    subheadline: "Unlock unlimited Telegram alerts with Observer Plus!",
    recommendedTier: 'plus',
    isUrgent: false
  },
  pushover: {
    headline: "Pushover Notification Quota Reached",
    subheadline: "Get unlimited push notifications with Observer Plus!",
    recommendedTier: 'plus',
    isUrgent: false
  },
  discord: {
    headline: "Discord Notification Quota Reached",
    subheadline: "Upgrade to Plus for unlimited Discord webhooks!",
    recommendedTier: 'plus',
    isUrgent: false
  },
  voice_call: {
    headline: "Voice Call Quota Reached",
    subheadline: "Get unlimited voice call alerts with Observer Plus!",
    recommendedTier: 'plus',
    isUrgent: false
  }
};

export const UpgradeModal: React.FC<UpgradeModalProps> = ({ isOpen, onClose, isHalfwayWarning = false, quotaType }) => {
  const [status, setStatus] = useState<'loading' | 'plus' | 'pro' | 'max' | 'free' | 'error'>('loading');
  const [error, setError] = useState<string | null>(null);
  const [isButtonLoading, setIsButtonLoading] = useState(false);

  const { getAccessToken, isAuthenticated, login } = useAuth();
  const applePayments = useApplePayments();

  // This logic is copied from ObServerTab - it's specific to this data-fetching context
  useEffect(() => {
    if (!isOpen || !isAuthenticated) {
      if (!isAuthenticated) setStatus('free');
      return;
    }
    
    const checkProStatus = async () => {
      setStatus('loading');
      try {
        const token = await getAccessToken();
        const response = await fetch('https://api.observer-ai.com/quota', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (!response.ok) throw new Error(`API request failed: ${response.statusText}`);
        const data = await response.json();
        // Backend returns tier: 'free' | 'plus' | 'pro' | 'max'
        setStatus(data.tier || (data.pro_status ? 'pro' : 'free'));
      } catch (err) {
        Logger.error('PAYMENTS', 'Failed to check pro status:', err);
        setError('Could not retrieve your subscription status.');
        setStatus('error');
      }
    };

    checkProStatus();
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
        body: JSON.stringify({ return_base_url: window.location.origin }),
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

  const handleCheckout = () => handleApiAction('create-checkout-session');
  const handlePlusCheckout = () => handleApiAction('create-checkout-session-plus');
  const handleMaxCheckout = () => handleApiAction('create-checkout-session-max');

  // Select quota-specific messaging
  const quotaConfig = quotaType
    ? QUOTA_MESSAGES[quotaType] || QUOTA_MESSAGES.monitor
    : null;

  // Only use halfway warning for monitor quota type
  const headline = (quotaType === 'monitor' && isHalfwayWarning)
    ? "You've Used Half of Your Daily Limit!"
    : (quotaConfig?.headline || "You've Reached Your Daily Limit!");

  const subheadline = (quotaType === 'monitor' && isHalfwayWarning)
    ? "Upgrade to Observer Pro for 8 hours/day of cloud monitoring!"
    : (quotaConfig?.subheadline || "Upgrade to Observer Pro!");

  if (!isOpen) {
    return null;
  }

  return (
    <div 
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-[110] backdrop-blur-sm p-4"
      onClick={onClose} // Close modal on overlay click
    >
      <div 
        className="relative bg-white rounded-xl shadow-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto transition-all duration-300"
        onClick={e => e.stopPropagation()} // Prevent clicks inside modal from closing it
      >
        <button onClick={onClose} className="absolute top-4 right-4 text-gray-400 hover:text-gray-700 z-10">
          <CloseIcon className="h-6 w-6" />
        </button>

        {status === 'loading' ? (
          <div className="flex justify-center items-center p-20">
            <Loader2 className="h-12 w-12 animate-spin text-gray-500" />
          </div>
        ) : (
          <PricingTable
            headline={headline}
            subheadline={subheadline}
            status={status}
            isButtonLoading={isButtonLoading}
            isAuthenticated={isAuthenticated}
            error={error}
            onCheckout={handleCheckout}
            onCheckoutPlus={handlePlusCheckout}
            onCheckoutMax={handleMaxCheckout}
            onManageSubscription={() => handleApiAction('create-customer-portal-session')}
            onLogin={login}
            isTriggeredByQuotaError={true}
            isHalfwayWarning={isHalfwayWarning}
            applePayments={applePayments}
            onModalClose={onClose}
          />
        )}
      </div>
    </div>
  );
};
