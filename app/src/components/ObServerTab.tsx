// src/components/ObServerTab.tsx

import React, { useState, useEffect } from 'react';
import { useAuth } from '@contexts/AuthContext';
import { useApplePayments } from '@hooks/useApplePayments';
import { Loader2, AlertTriangle } from 'lucide-react';
import { Logger } from '@utils/logging';
import { PricingTable } from './PricingTable';

export const ObServerTab: React.FC = () => {
  const [status, setStatus] = useState<'loading' | 'plus' | 'pro' | 'max' | 'free' | 'error'>('loading');
  const [error, setError] = useState<string | null>(null);
  const [isButtonLoading, setIsButtonLoading] = useState(false);

  const { getAccessToken, isAuthenticated, login } = useAuth();
  const applePayments = useApplePayments();

  // This logic stays here, as this component is responsible for fetching its own data.
  useEffect(() => {
    if (!isAuthenticated) {
      setStatus('free');
      return;
    }
    
    const checkProStatus = async () => {
      setStatus('loading');
      try {
        const token = await getAccessToken();
        const response = await fetch('https://api.observer-ai.com/quota', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (!response.ok) throw new Error(`API request failed with status: ${response.status}`);
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
  }, [isAuthenticated, getAccessToken]);

  // This logic also stays here. The PricingTable component calls these functions via props.
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

  // --- RENDER LOGIC ---

  // Handle the loading state
  if (status === 'loading') {
    return (
      <div className="flex justify-center items-center p-20">
        <Loader2 className="h-12 w-12 animate-spin text-gray-500" />
      </div>
    );
  }
  
  // Handle the error state
  if (status === 'error') {
      return (
        <div className="max-w-3xl mx-auto p-8 bg-white rounded-lg shadow-md text-center border-l-4 border-red-500">
            <AlertTriangle className="mx-auto h-16 w-16 text-red-500 mb-4" />
            <h1 className="text-3xl font-bold text-gray-800 mb-2">Something Went Wrong</h1>
            <p className="text-gray-600">{error || 'We could not load your subscription details. Please refresh the page.'}</p>
        </div>
      )
  }

  // The main render path is now incredibly simple.
  // It renders the reusable table with props tailored for the general "Ob-Server" tab.
  return (
    <div className="w-full pt-8">
      <PricingTable
        headline="Choose Your Way to Observe"
        subheadline=""
        status={status}
        isButtonLoading={isButtonLoading}
        isAuthenticated={isAuthenticated}
        error={error}
        onCheckout={handleCheckout}
        onCheckoutPlus={handlePlusCheckout}
        onCheckoutMax={handleMaxCheckout}
        onManageSubscription={() => handleApiAction('create-customer-portal-session')}
        onLogin={login}
        isTriggeredByQuotaError={false}
        applePayments={applePayments}
      />
    </div>
  );
};
