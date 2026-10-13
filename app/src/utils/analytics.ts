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

  // ── Recipe builder (IFTTT onboarding hero) ──────────────────────────────────
  recipeShown: () => track('recipe_builder_shown'),
  recipeBuilt: (trigger: string, action: string, modelMode?: string) => track('recipe_built', { trigger, action, modelMode }),
  tutorialStarted: (demo: string, action: string) => track('recipe_tutorial_started', { demo, action }),
  tutorialSkipped: () => track('recipe_tutorial_skipped'),

  // ── Attribution (asked right after ToS, before the welcome upsell) ──────────
  attributionShown: () => track('attribution_shown'),
  attributionSelected: (source: AttributionSource, detail?: string) => track('attribution_selected', { source, detail }),

  // ── Upsell (WelcomeModal — 'welcome' after ToS, 'activation' after first agent starts)
  upsellShown: (source: UpsellSource) => track('upsell_shown', { source }),
  upsellFreeTrial: (source: UpsellSource) => track('upsell_free_trial_clicked', { source }),
  upsellGithub: (source: UpsellSource) => track('upsell_github_clicked', { source }),
  upsellContinueFree: (source: UpsellSource) => track('upsell_continue_free_clicked', { source }),
  upsellViewTiers: (source: UpsellSource) => track('upsell_view_tiers_clicked', { source }),
};

type UpsellSource = 'welcome' | 'activation';

export type AttributionSource = 'reddit' | 'youtube' | 'tiktok' | 'instagram' | 'github' | 'google' | 'friend' | 'other';
