import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@contexts/AuthContext';
import { Loader2, UserPlus, Trash2, ArrowLeft, AlertCircle, Users, CheckCircle2 } from 'lucide-react';
import { Logger } from '@utils/logging';
import { fetchQuota, type MonthlyQuotaBlock } from '@/types/quota';
import { SeatGrid } from '@components/SeatGrid';

const API = 'https://api.observer-ai.com';

// Mirrors DASHBOARD_SERVICES in api/orgs.py
const SERVICES = ['monitor', 'agent_creator', 'email', 'sms', 'whatsapp', 'telegram', 'voice_call', 'pushover'] as const;

const SERVICE_LABELS: Record<string, string> = {
  monitor: 'Monitor',
  agent_creator: 'Agent',
  email: 'Email',
  sms: 'SMS',
  whatsapp: 'WhatsApp',
  telegram: 'Telegram',
  voice_call: 'Calling',
  pushover: 'Pushover',
};

interface Member {
  email: string;
  status: 'invited' | 'active';
  joined_at: string | null;
  usage: Record<string, number>;
}

interface Org {
  org_id: string;
  name: string;
  tier: string;
  status: string;
  seats_purchased: number;
  seats_used: number;
  is_owner: boolean;
  owner_email: string;
  members: Member[];
}

const Card = ({ children, className = '' }: { children: React.ReactNode; className?: string }) => (
  <div className={`rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 ${className}`}>
    {children}
  </div>
);

export function TeamPage() {
  const { getAccessToken } = useAuth();
  const navigate = useNavigate();

  const [org, setOrg] = useState<Org | null>(null);
  const [monthly, setMonthly] = useState<MonthlyQuotaBlock | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviting, setInviting] = useState(false);
  const [notice, setNotice] = useState<{ kind: 'success' | 'error'; text: string } | null>(null);
  const [tab, setTab] = useState<'overview' | 'members'>('overview');

  const load = useCallback(async () => {
    try {
      const token = await getAccessToken();
      const res = await fetch(`${API}/orgs/me`, { headers: { Authorization: `Bearer ${token}` } });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setError(data.detail || 'Could not load your organization.');
        return;
      }
      setOrg(await res.json());
      setError(null);

      // The org's shared monthly hours pool lives on /quota (scope: 'org' for a member of an org).
      if (token) {
        try {
          const quota = await fetchQuota(token);
          if (quota?.monthly?.scope === 'org') setMonthly(quota.monthly);
        } catch (e) {
          Logger.error('TEAM', 'Failed to load monthly quota', { error: e });
        }
      }
    } catch (e) {
      Logger.error('TEAM', 'Failed to load org', { error: e });
      setError('Could not reach the server.');
    } finally {
      setLoading(false);
    }
  }, [getAccessToken]);

  useEffect(() => { load(); }, [load]);

  const invite = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inviteEmail.trim()) return;
    setInviting(true);
    setNotice(null);
    try {
      const token = await getAccessToken();
      const res = await fetch(`${API}/orgs/members`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: inviteEmail.trim() }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setNotice({ kind: 'error', text: data.detail || 'Could not send that invite.' });
        return;
      }
      setNotice({
        kind: 'success',
        text: data.status === 'active'
          ? `${data.email} already had an Observer account and is active now.`
          : `Invite sent to ${data.email}.`,
      });
      setInviteEmail('');
      await load();
    } finally {
      setInviting(false);
    }
  };

  const remove = async (email: string) => {
    if (!confirm(`Remove ${email} from the team? They will lose access immediately.`)) return;
    const token = await getAccessToken();
    const res = await fetch(`${API}/orgs/members`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ email }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      setNotice({ kind: 'error', text: data.detail || 'Could not remove that member.' });
      return;
    }
    await load();
  };

  if (loading) {
    return (
      <div className="h-full w-full flex flex-col items-center justify-center gap-3 bg-gray-50 dark:bg-gray-950">
        <Loader2 className="h-6 w-6 text-gray-400 animate-spin" />
      </div>
    );
  }

  if (error || !org) {
    return (
      <div className="h-full w-full flex flex-col items-center justify-center bg-gray-50 dark:bg-gray-950 px-4">
        <div className="p-10 bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 max-w-md w-full text-center">
          <AlertCircle className="h-8 w-8 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
          <p className="text-gray-600 dark:text-gray-300 text-sm mb-6">{error}</p>
          <button
            onClick={() => navigate('/')}
            className="px-5 py-2.5 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-200 text-sm font-medium rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
          >
            Go to Observer
          </button>
        </div>
      </div>
    );
  }

  const seatsLeft = org.seats_purchased - org.seats_used;
  const tierLabel = org.tier.charAt(0).toUpperCase() + org.tier.slice(1);
  const monitorToday = org.members.reduce((sum, m) => sum + (m.usage?.monitor ?? 0), 0);

  return (
    <div className="h-full w-full overflow-y-auto bg-gray-50 dark:bg-gray-950">
      <div className="max-w-4xl mx-auto px-6 py-10">
        <button
          onClick={() => navigate('/')}
          className="inline-flex items-center gap-1.5 text-sm text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 mb-8 transition-colors"
        >
          <ArrowLeft className="h-3.5 w-3.5" /> Back to Observer
        </button>

        {/* ── Header ── */}
        <div className="flex items-start justify-between flex-wrap gap-3 mb-1">
          <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100 tracking-tight">{org.name}</h1>
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-gray-500 dark:text-gray-400">Observer {tierLabel}</span>
            {org.status !== 'active' && (
              <span className="px-2 py-0.5 rounded-md text-xs font-medium bg-amber-50 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300">
                {org.status}
              </span>
            )}
          </div>
        </div>
        <p className="text-sm text-gray-400 dark:text-gray-500 mb-8">
          Managed by {org.owner_email}{org.is_owner && ' · you'}
        </p>

        {/* ── Tabs ── */}
        <div className="flex items-center gap-6 border-b border-gray-200 dark:border-gray-800 mb-8">
          {(['overview', 'members'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`pb-3 -mb-px text-sm font-medium border-b-2 transition-colors ${
                tab === t
                  ? 'border-gray-900 dark:border-gray-100 text-gray-900 dark:text-gray-100'
                  : 'border-transparent text-gray-400 dark:text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
              }`}
            >
              {t === 'overview' ? 'Overview' : `Members · ${org.members.length}`}
            </button>
          ))}
        </div>

        {tab === 'overview' ? (
          <div className="space-y-6">
            {/* ── Seats · Credits today ── */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Card className="px-5 py-4">
                <div className="flex items-baseline justify-between">
                  <span className="text-xs font-medium text-gray-400 dark:text-gray-500">Seats</span>
                  <span className="text-xs text-gray-400 dark:text-gray-500">
                    {seatsLeft > 0 ? `${seatsLeft} available` : 'All in use'}
                  </span>
                </div>
                <div className="mt-1 text-2xl font-semibold text-gray-900 dark:text-gray-100 tabular-nums">
                  {org.seats_used} <span className="text-gray-300 dark:text-gray-600 font-normal">/ {org.seats_purchased}</span>
                </div>
                <div className="mt-4">
                  <SeatGrid filled={org.seats_used} total={org.seats_purchased} />
                </div>
              </Card>

              <Card className="px-5 py-4">
                <span className="text-xs font-medium text-gray-400 dark:text-gray-500">Monitoring minutes today</span>
                <div className="mt-1 text-2xl font-semibold text-gray-900 dark:text-gray-100 tabular-nums">
                  {monitorToday.toLocaleString()}
                </div>
                <div className="mt-0.5 text-xs text-gray-400 dark:text-gray-500">Across the whole team</div>
              </Card>
            </div>

            {/* ── Monthly hours pool ── */}
            {monthly && (
              <Card className="px-5 py-4">
                <div className="flex items-baseline justify-between mb-3">
                  <span className="text-xs font-medium text-gray-400 dark:text-gray-500">Monitoring minutes this month</span>
                  <span className="text-sm font-semibold text-gray-900 dark:text-gray-100 tabular-nums">
                    {monthly.used.toLocaleString()}
                    <span className="text-gray-300 dark:text-gray-600 font-normal"> / {monthly.limit.toLocaleString()}</span>
                  </span>
                </div>
                <div className="h-1.5 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${
                      monthly.used >= monthly.limit ? 'bg-amber-500' : 'bg-gray-900 dark:bg-gray-100'
                    }`}
                    style={{ width: `${Math.min(100, (monthly.used / Math.max(1, monthly.limit)) * 100)}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-gray-400 dark:text-gray-500 mt-2">
                  <span>Resets {new Date(monthly.resets_at).toLocaleDateString()}</span>
                  {typeof monthly.your_contribution === 'number' && (
                    <span>Your contribution: {monthly.your_contribution.toLocaleString()}</span>
                  )}
                </div>
              </Card>
            )}

            {/* ── Invite (owners only) ── */}
            {org.is_owner && (
              <Card className="p-5">
                <h2 className="text-sm font-medium text-gray-900 dark:text-gray-100">Invite a teammate</h2>
                <p className="text-sm text-gray-400 dark:text-gray-500 mt-1 mb-4">
                  They'll get an email with a link to claim a seat.
                </p>
                <form onSubmit={invite} className="flex gap-2 flex-wrap">
                  <input
                    type="email"
                    value={inviteEmail}
                    onChange={(e) => setInviteEmail(e.target.value)}
                    placeholder="teammate@company.com"
                    disabled={seatsLeft <= 0}
                    className="flex-1 min-w-[220px] px-3.5 py-2 border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 rounded-lg text-sm focus:outline-none focus:ring-1 focus:ring-gray-400 dark:focus:ring-gray-500 focus:border-transparent disabled:bg-gray-50 dark:disabled:bg-gray-800 disabled:cursor-not-allowed transition-shadow"
                  />
                  <button
                    type="submit"
                    disabled={inviting || seatsLeft <= 0}
                    className="px-4 py-2 bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium rounded-lg hover:bg-gray-700 dark:hover:bg-white disabled:bg-gray-200 dark:disabled:bg-gray-700 disabled:text-gray-400 dark:disabled:text-gray-500 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
                  >
                    {inviting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <UserPlus className="h-3.5 w-3.5" />}
                    Send invite
                  </button>
                </form>
                {seatsLeft <= 0 && (
                  <p className="text-xs text-amber-700 dark:text-amber-400 mt-3">
                    All seats are in use. Contact Observer to add more.
                  </p>
                )}
                {notice && (
                  <div
                    className={`mt-4 flex items-start gap-2 rounded-lg px-3 py-2.5 text-sm ${
                      notice.kind === 'success'
                        ? 'bg-green-50 text-green-800 dark:bg-green-900/30 dark:text-green-200'
                        : 'bg-red-50 text-red-700 dark:bg-red-900/30 dark:text-red-200'
                    }`}
                  >
                    {notice.kind === 'success'
                      ? <CheckCircle2 className="h-4 w-4 mt-0.5 flex-shrink-0" />
                      : <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />}
                    <span>{notice.text}</span>
                  </div>
                )}
              </Card>
            )}
          </div>
        ) : (
          <Card className="overflow-hidden">
            <div className="px-5 py-3.5 border-b border-gray-100 dark:border-gray-800 flex items-center justify-between gap-3 flex-wrap">
              <h2 className="text-sm font-medium text-gray-900 dark:text-gray-100">Members</h2>
              <span className="text-xs text-gray-400 dark:text-gray-500">Usage resets daily at 00:00 UTC</span>
            </div>

            {org.members.length === 0 ? (
              <div className="px-6 py-14 text-center">
                <Users className="h-7 w-7 text-gray-200 dark:text-gray-700 mx-auto mb-3" />
                <p className="text-sm text-gray-400 dark:text-gray-500">
                  No one on the team yet.{org.is_owner && ' Invite your first teammate from Overview.'}
                </p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-gray-400 dark:text-gray-500 border-b border-gray-100 dark:border-gray-800">
                      <th className="px-5 py-2.5 font-medium">Member</th>
                      {SERVICES.map((s) => (
                        <th key={s} className="px-3 py-2.5 font-medium text-right whitespace-nowrap">{SERVICE_LABELS[s]}</th>
                      ))}
                      {org.is_owner && <th className="px-5 py-2.5" />}
                    </tr>
                  </thead>
                  <tbody>
                    {org.members.map((m) => {
                      const isOwner = m.email === org.owner_email;
                      return (
                        <tr
                          key={m.email}
                          className="border-b border-gray-50 dark:border-gray-800/60 last:border-0 hover:bg-gray-50/70 dark:hover:bg-gray-800/30 transition-colors"
                        >
                          <td className="px-5 py-2.5">
                            <div className="flex items-center gap-2.5">
                              <div className="min-w-0">
                                <div className="text-gray-900 dark:text-gray-100 truncate">{m.email}</div>
                                <div className="flex items-center gap-1.5 mt-0.5">
                                  <span
                                    className={`text-[11px] ${
                                      m.status === 'active'
                                        ? 'text-green-600 dark:text-green-400'
                                        : 'text-gray-400 dark:text-gray-500'
                                    }`}
                                  >
                                    {m.status === 'active' ? 'Active' : 'Pending'}
                                  </span>
                                  {isOwner && (
                                    <span className="text-[11px] text-gray-400 dark:text-gray-500">· Owner</span>
                                  )}
                                </div>
                              </div>
                            </div>
                          </td>
                          {SERVICES.map((s) => {
                            const n = m.usage?.[s] ?? 0;
                            return (
                              <td
                                key={s}
                                className={`px-3 py-2.5 text-right tabular-nums ${
                                  n > 0 ? 'text-gray-700 dark:text-gray-200' : 'text-gray-300 dark:text-gray-600'
                                }`}
                              >
                                {n > 0 ? n.toLocaleString() : '—'}
                              </td>
                            );
                          })}
                          {org.is_owner && (
                            <td className="px-5 py-2.5 text-right">
                              {!isOwner && (
                                <button
                                  onClick={() => remove(m.email)}
                                  className="p-1.5 rounded-md text-gray-300 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors"
                                  title={`Remove ${m.email}`}
                                >
                                  <Trash2 className="h-3.5 w-3.5" />
                                </button>
                              )}
                            </td>
                          )}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        )}
      </div>
    </div>
  );
}
