// src/types/quota.ts

export interface QuotaBlock {
  used: number;
  limit: number;
  resets_at: string;
}

export interface MonthlyQuotaBlock extends QuotaBlock {
  scope: 'user' | 'org';
  your_contribution?: number;
}

export interface QuotaInfo {
  tier: string;
  daily: QuotaBlock;
  monthly: MonthlyQuotaBlock;
  org_id: string | null;
  is_enterprise: boolean;
}

export async function fetchQuota(token: string): Promise<QuotaInfo | null> {
  const response = await fetch('https://api.observer-ai.com/quota', {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    if (response.status === 401) throw new Error('unauthorized');
    throw new Error(`quota fetch failed: ${response.status}`);
  }
  return response.json();
}

export const remaining = (block: QuotaBlock): number => Math.max(0, block.limit - block.used);
