import { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@contexts/AuthContext';
import { useApplePayments } from '@hooks/useApplePayments';
import { Loader2, CheckCircle, RotateCcw } from 'lucide-react';
import { Logger } from '@utils/logging';

export function UpgradeSuccessPage() {
  const { refreshSession, logout } = useAuth();
  const navigate = useNavigate();
  const { isAppleDevice, getActiveSubscription, verifyTransaction, restorePurchases, isLoading: appleLoading } = useApplePayments();

  const [pageStatus, setPageStatus] = useState<'verifying' | 'polling' | 'success' | 'timeout' | 'sync-issue'>('verifying');
  const [hasSyncIssue, setHasSyncIssue] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string>('Processing your purchase...');
  const [currentTier, setCurrentTier] = useState<string | null>(null);
  const [attempts, setAttempts] = useState(0);

  // Prevent multiple runs of the effect
  const hasStartedRef = useRef(false);
  const isCancelledRef = useRef(false);

  useEffect(() => {
    // Only run once
    if (hasStartedRef.current) return;
    hasStartedRef.current = true;

    const processUpgrade = async () => {
      // Step 1: For iOS, verify the Apple transaction first
      if (isAppleDevice) {
        Logger.info('UPGRADE_SUCCESS', 'iOS device detected, checking for active subscription...');
        setStatusMessage('Checking your purchase with Apple...');

        const subscription = await getActiveSubscription();

        if (isCancelledRef.current) return;

        if (subscription?.hasActiveSubscription && subscription.jwsRepresentation) {
          Logger.info('UPGRADE_SUCCESS', 'Found active subscription, verifying with server...');
          setStatusMessage('Activating your subscription...');

          const verifyResult = await verifyTransaction(subscription.jwsRepresentation);

          if (isCancelledRef.current) return;

          if (!verifyResult.success) {
            Logger.error('UPGRADE_SUCCESS', 'Transaction verification failed', { error: verifyResult.error });
            // Mark that we had a sync issue - will show helpful message if polling also fails
            setHasSyncIssue(true);
            Logger.info('UPGRADE_SUCCESS', 'Continuing to poll despite verify failure...');
          } else {
            Logger.info('UPGRADE_SUCCESS', 'Transaction verified successfully', { tier: verifyResult.tier });
          }
        } else {
          Logger.info('UPGRADE_SUCCESS', 'No active Apple subscription found, proceeding to poll');
        }
      }

      // Step 2: Poll for quota update (works for both Stripe and Apple)
      setPageStatus('polling');
      setStatusMessage('Updating your account...');

      const maxAttempts = 15;
      let attemptCount = 0;

      while (attemptCount < maxAttempts && !isCancelledRef.current) {
        try {
          const token = await refreshSession();

          if (isCancelledRef.current) return;

          if (!token) {
            attemptCount++;
            setAttempts(attemptCount);
            if (attemptCount < maxAttempts) {
              await new Promise(resolve => setTimeout(resolve, 2000));
            }
            continue;
          }

          const response = await fetch('https://api.observer-ai.com/quota', {
            headers: { Authorization: `Bearer ${token}` }
          });

          if (isCancelledRef.current) return;

          if (response.ok) {
            const data = await response.json();
            if (data.tier && data.tier !== 'free') {
              setCurrentTier(data.tier);
              setPageStatus('success');
              setTimeout(() => navigate('/'), 1500);
              return;
            }
          }
        } catch (error) {
          Logger.error('UPGRADE_SUCCESS', 'Polling error on attempt', { attempt: attemptCount + 1, error });
        }

        attemptCount++;
        setAttempts(attemptCount);

        if (attemptCount < maxAttempts && !isCancelledRef.current) {
          await new Promise(resolve => setTimeout(resolve, 2000));
        }
      }

      if (!isCancelledRef.current) {
        setPageStatus('timeout');
      }
    };

    processUpgrade();

    return () => {
      isCancelledRef.current = true;
    };
  }, []); // Empty deps - run once on mount

  const handleRestorePurchases = async () => {
    setPageStatus('verifying');
    setStatusMessage('Restoring your purchase...');

    const result = await restorePurchases();
    if (result.success) {
      setCurrentTier(result.tier || null);
      setPageStatus('success');
      setTimeout(() => navigate('/'), 1500);
    } else {
      setPageStatus('timeout');
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-b from-gray-50 to-gray-100">
      <div className="p-10 bg-white rounded-2xl shadow-lg max-w-md w-full mx-4 text-center">

        {/* Icon */}
        <div className="flex justify-center mb-5">
          {(pageStatus === 'verifying' || pageStatus === 'polling') && (
            <div className="h-16 w-16 rounded-full bg-blue-50 flex items-center justify-center">
              <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
            </div>
          )}
          {pageStatus === 'success' && (
            <div className="h-16 w-16 rounded-full bg-green-50 flex items-center justify-center">
              <CheckCircle className="h-8 w-8 text-green-500" />
            </div>
          )}
          {pageStatus === 'timeout' && (
            <div className="h-16 w-16 rounded-full bg-yellow-50 flex items-center justify-center">
              <CheckCircle className="h-8 w-8 text-yellow-500" />
            </div>
          )}
        </div>

        {/* Heading */}
        <h1 className="text-2xl font-bold text-gray-900 mb-2">
          {pageStatus === 'success' ? 'Welcome!' : 'Payment received!'}
        </h1>

        {/* Subtext */}
        {(pageStatus === 'verifying' || pageStatus === 'polling') && (
          <>
            <p className="text-gray-500 text-sm mb-1">
              {statusMessage}
            </p>
            {pageStatus === 'polling' && (
              <p className="text-xs text-gray-400">
                Checking... ({attempts}/{15})
              </p>
            )}
          </>
        )}

        {pageStatus === 'success' && (
          <p className="text-green-600 font-medium text-sm">
            {currentTier
              ? `Welcome to Observer ${currentTier.charAt(0).toUpperCase() + currentTier.slice(1)}!`
              : 'Your account has been upgraded!'}
          </p>
        )}

        {pageStatus === 'timeout' && (
          <>
            {hasSyncIssue && isAppleDevice ? (
              // Sync issue - purchase succeeded but server update failed
              <>
                <p className="text-gray-600 text-sm mb-2">
                  Your purchase was successful!
                </p>
                <p className="text-gray-500 text-sm mb-6">
                  We had trouble syncing with our servers. Tap "Restore Purchases" to activate your subscription.
                </p>
              </>
            ) : (
              <p className="text-gray-500 text-sm mb-6">
                We're still updating your account on our servers — this usually
                only takes a moment.
              </p>
            )}

            <div className="flex flex-col gap-3">
              {/* For iOS users, show Restore Purchases prominently */}
              {isAppleDevice && (
                <button
                  onClick={handleRestorePurchases}
                  disabled={appleLoading}
                  className="w-full px-5 py-2.5 bg-blue-500 text-white text-sm font-medium rounded-xl hover:bg-blue-600 transition-colors disabled:bg-gray-300 flex items-center justify-center gap-2"
                >
                  {appleLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RotateCcw className="h-4 w-4" />
                  )}
                  Restore Purchases
                </button>
              )}

              <div className="flex gap-3 justify-center">
                <button
                  onClick={() => window.location.reload()}
                  className="px-5 py-2.5 bg-gray-100 text-gray-700 text-sm font-medium rounded-xl hover:bg-gray-200 transition-colors"
                >
                  Retry
                </button>
                <button
                  onClick={() => logout()}
                  className="px-5 py-2.5 bg-gray-100 text-gray-700 text-sm font-medium rounded-xl hover:bg-gray-200 transition-colors"
                >
                  Log Out
                </button>
              </div>
            </div>
          </>
        )}

      </div>
    </div>
  );
}
