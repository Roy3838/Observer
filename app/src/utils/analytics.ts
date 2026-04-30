import { datadogRum } from '@datadog/browser-rum';

// Thin wrapper so callers never import datadogRum directly.
// All funnel events go through here — easy to swap or extend later.
export function track(action: string, context?: Record<string, unknown>) {
  datadogRum.addAction(action, context);
}

// ── Startup dialog ────────────────────────────────────────────────────────────
export const Analytics = {
  startupShown: () => track('startup_dialog_shown'),
  startupSignIn: () => track('startup_sign_in_clicked'),
  startupSkip: () => track('startup_skip_clicked'),

  // ── Local mode warning ──────────────────────────────────────────────────────
  localModeShown: () => track('local_mode_warning_shown'),
  localModeSignIn: () => track('local_mode_sign_in_clicked'),
  localModeContinue: () => track('local_mode_continue_clicked'),

  // ── Tutorial ────────────────────────────────────────────────────────────────
  tutorialStarted: () => track('tutorial_started'),
  tutorialDismissed: (atStep: string) => track('tutorial_dismissed_early', { at_step: atStep }),
  tutorialCompleted: () => track('tutorial_completed'),

  // ── Upsell ──────────────────────────────────────────────────────────────────
  upsellShown: (source: 'tutorial_complete' | 'tutorial_dismissed') =>
    track('upsell_shown', { source }),
  upsellFreeTrial: (source: 'tutorial_complete' | 'tutorial_dismissed') =>
    track('upsell_free_trial_clicked', { source }),
  upsellGithub: (source: 'tutorial_complete' | 'tutorial_dismissed') =>
    track('upsell_github_clicked', { source }),
  upsellContinueFree: (source: 'tutorial_complete' | 'tutorial_dismissed') =>
    track('upsell_continue_free_clicked', { source }),
  upsellViewTiers: (source: 'tutorial_complete' | 'tutorial_dismissed') =>
    track('upsell_view_tiers_clicked', { source }),
};
