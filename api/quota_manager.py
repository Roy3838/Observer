import datetime
import logging
import os
import asyncio
import httpx
from typing import Dict

import r2_store
from redis_client import get_redis
from usage_log import log_usage

logger = logging.getLogger('quota_manager')

# --- Configuration ---
QUOTA_LIMITS = {
    "monitor": 30,
    "agent_creator": 45,   # 3 agent sessions × ~15 msgs
    "sms": 5,
    "whatsapp": 5,
    "email": 2880,
    "pushover": 5,
    "discord": 5,
    "telegram": 2880,
    "slack": 5,
    "teams": 5,
    "voice_call": 5,
}

# Plus user limits (unlimited alerts, limited chat)
PLUS_QUOTA_LIMITS = {
    "monitor": 60,
    "agent_creator": 1000,  # plus legacy tier
    "sms": 100,
    "whatsapp": 100,
    "email": 2880,
    "pushover": 2880,
    "discord": 2880,
    "telegram": 2880,
    "slack": 100,
    "teams": 100,
    "voice_call": 100,
}

# Pro user limits (anti-abuse measure)
PRO_QUOTA_LIMITS = {
    "monitor": 480,
    "agent_creator": 1000,
    "sms": 100,
    "whatsapp": 100,
    "email": 2880,
    "pushover": 2880,
    "discord": 2880,
    "telegram": 2880,
    "slack": 2880,
    "teams": 2880,
    "voice_call": 100,
}

# Max user limits (highest tier)
MAX_QUOTA_LIMITS = {
    "monitor": 2880, # 30s interval for 24h = 2/minx60x24=2880
    "agent_creator": 1000,
    "sms": 100,
    "whatsapp": 100,
    "email": 2880,
    "pushover": 2880,
    "discord": 2880,
    "telegram": 2880,
    "slack": 2880,
    "teams": 2880,
    "voice_call": 100,
}

# --- Monthly credit caps ----------------------------------------------------
#
# The daily tables above bound the *rate*; these bound the *budget*. Both are
# enforced, and that pairing is deliberate: the daily cap is what makes a
# monthly cap safe to ship. A runaway agent costs a user one day rather than
# their whole month, and a subscriber who signs up on the 28th cannot
# front-load a month's credits into the three days before the calendar rolls.
#
# Only "monitor" is metered monthly - that is what "credits" means on the
# pricing page. Messaging and agent_creator stay daily-only. NO_MONTHLY_LIMIT
# means the monthly key is never read or written, so those services cost
# nothing extra on the hot path.
NO_MONTHLY_LIMIT = -1

# Services with a monthly budget. Anything not listed is daily-only.
MONTHLY_METERED = ("monitor",)

FREE_MONTHLY_LIMITS = {"monitor": 600}       # 10 hours
PRO_MONTHLY_LIMITS  = {"monitor": 6_000}     # 100 hours

# Max is capped by its daily limit alone: 2880/day cannot reach any monthly
# number worth writing down, so a cap here would be config that never applies.
MAX_MONTHLY_LIMITS  = {"monitor": NO_MONTHLY_LIMIT}

# Plus is a closed legacy tier. It is left uncapped rather than quietly given
# terms nobody agreed to; its 60/day ceiling already bounds the damage.
PLUS_MONTHLY_LIMITS = {"monitor": NO_MONTHLY_LIMIT}

# Monthly counters carry their period in the key name, so the reset is a key
# rotation rather than a TTL expiry and a lost EXPIRE can never strand a user
# at their cap. The TTL is therefore garbage collection only, and just has to
# outlive the longest month by a comfortable margin.
MONTHLY_KEY_TTL = 35 * 86400

# How long a Redis-cached org pool size is trusted. Only a safety net: admin
# edits call invalidate_org_limit() and take effect on the next request.
ORG_LIMIT_CACHE_TTL = 3600


# Rate limiting configuration (requests per minute)
RATE_LIMIT_PER_MINUTE = 30

# Audio second limits per provider per tier
CHIRP_SECOND_LIMITS = {
    "free":   2_700,   # 45 min
    "plus":   2_700,   # 45 min
    "pro":   10_800,   # 3 hours
    "max":   10_800,   # 3 hours
}
GEMINI_SECOND_LIMITS = {
    "free":   2_700,   # 45 min
    "plus":   2_700,   # 45 min
    "pro":   54_000,   # 15 hours
    "max":   54_000,   # 15 hours
}

def _seconds_until_midnight() -> int:
    """
    Seconds until the next UTC midnight.

    Explicitly UTC so daily quota resets do not depend on the container's
    ambient timezone - they line up with the UTC day keys used by
    observability, and setting TZ on the container cannot silently shift
    every user's reset.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    midnight = datetime.datetime.combine(
        now.date() + datetime.timedelta(days=1),
        datetime.time.min,
        tzinfo=datetime.timezone.utc,
    )
    return int((midnight - now).total_seconds())

def _next_month_start() -> datetime.datetime:
    """First instant of the next UTC calendar month."""
    now = datetime.datetime.now(datetime.timezone.utc)
    year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
    return datetime.datetime(year, month, 1, tzinfo=datetime.timezone.utc)


def _next_midnight() -> datetime.datetime:
    now = datetime.datetime.now(datetime.timezone.utc)
    return datetime.datetime.combine(
        now.date() + datetime.timedelta(days=1),
        datetime.time.min,
        tzinfo=datetime.timezone.utc,
    )


def current_month() -> str:
    """The period stamp embedded in monthly keys. UTC calendar month.

    Calendar months rather than per-subscriber billing cycles: a billing-aligned
    period needs a boundary stored per user, which does not exist for individual
    subscribers today. The period is a string in the key, so if that changes the
    only thing that moves is this function.
    """
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")


def daily_key(user_id: str, service: str) -> str:
    return f"quota:{user_id}:{service}"


def monthly_key(user_id: str | None, service: str, org_id: str | None = None) -> str:
    """
    Enterprise seats count against one shared org key; everyone else against
    their own. Both prefixes deliberately sit outside "quota:" - get_all_usage_data()
    scans quota:* and splits on ":" expecting exactly three parts.
    """
    if org_id:
        return f"orgquota:{org_id}:{service}:{current_month()}"
    return f"mquota:{user_id}:{service}:{current_month()}"


def org_limit_key(org_id: str) -> str:
    return f"orglimit:{org_id}"


async def _send_abuse_alert_async(user_id: str, service: str):
    try:
        telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not telegram_bot_token:
            logger.warning("Cannot send abuse alert: TELEGRAM_BOT_TOKEN not configured")
            return

        admin_chat_id = os.getenv("ADMIN_TELEGRAM_CHAT_ID")
        message = f"⚠️ Rate limit exceeded!\n\nUser ID: {user_id}\nService: {service}\nTime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"

        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(url, json={"chat_id": admin_chat_id, "text": message})
        logger.info(f"Sent abuse alert for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to send abuse alert: {e}")

# Rate limit and quota are checked and consumed in a single Lua script so the
# two cannot interleave. The previous check_usage()/increment_usage() pair read
# the counter, decided, then incremented in a separate round trip: with four
# uvicorn workers two concurrent requests could both read 59 against a limit of
# 60, both pass, and both increment to 61. It also collapses six sequential
# round trips into one, which matters once Redis is not on localhost.
# KEYS: 1 ratelimit  2 daily quota  3 monthly quota (user or org)
# ARGV: 1 rate limit  2 daily limit  3 daily ttl  4 monthly limit  5 monthly ttl
#
# All three limits are checked before anything is incremented, so a call refused
# by the monthly budget does not burn a daily credit or rate-limit headroom.
# A monthly limit of -1 skips KEYS[3] entirely: messaging services and uncapped
# tiers never read or write a monthly counter.
_CONSUME_LUA = """
local rl = tonumber(redis.call('GET', KEYS[1]) or '0')
if rl >= tonumber(ARGV[1]) then return {-1, rl} end

local used = tonumber(redis.call('GET', KEYS[2]) or '0')
if used >= tonumber(ARGV[2]) then return {-2, used} end

local mlimit = tonumber(ARGV[4])
if mlimit >= 0 then
  local mused = tonumber(redis.call('GET', KEYS[3]) or '0')
  if mused >= mlimit then return {-3, mused} end
end

local nrl = redis.call('INCR', KEYS[1])
if nrl == 1 then redis.call('EXPIRE', KEYS[1], 60) end

local nq = redis.call('INCR', KEYS[2])
if nq == 1 then redis.call('EXPIRE', KEYS[2], tonumber(ARGV[3])) end

if mlimit >= 0 then
  local nm = redis.call('INCR', KEYS[3])
  if nm == 1 then redis.call('EXPIRE', KEYS[3], tonumber(ARGV[5])) end
end

return {0, nq}
"""

_consume_script = None


def limit_for(service: str, is_pro: bool = False, is_max: bool = False, is_plus: bool = False) -> int:
    """The daily limit that applies to this user's tier for this service."""
    if is_max:
        return MAX_QUOTA_LIMITS[service]
    if is_pro:
        return PRO_QUOTA_LIMITS[service]
    if is_plus:
        return PLUS_QUOTA_LIMITS[service]
    return QUOTA_LIMITS[service]


def monthly_limit_for(service: str, is_pro: bool = False, is_max: bool = False, is_plus: bool = False) -> int:
    """
    The monthly budget for this tier, or NO_MONTHLY_LIMIT. Enterprise seats do
    not use this - their budget is the org pool, see org_monthly_limit().
    """
    if is_max:
        table = MAX_MONTHLY_LIMITS
    elif is_pro:
        table = PRO_MONTHLY_LIMITS
    elif is_plus:
        table = PLUS_MONTHLY_LIMITS
    else:
        table = FREE_MONTHLY_LIMITS
    return table.get(service, NO_MONTHLY_LIMIT)


async def org_monthly_limit(org_id: str) -> int:
    """
    The org's negotiated monthly pool, R2 as source of truth and Redis as cache.

    Orgs with no monthly_credits set are uncapped, so this ships dark: existing
    enterprise customers keep exactly the behaviour they have until a number is
    negotiated onto their record.

    Fails open. R2 being unreachable should not cut off every enterprise seat at
    once, and the daily per-seat limit still applies underneath.
    """
    r = await get_redis()
    cached = await r.get(org_limit_key(org_id))
    if cached is not None:
        return int(cached)

    try:
        org, _ = await r2_store.get_json(r2_store.org_key(org_id))
    except Exception as e:
        logger.error(f"Could not read org {org_id} for its monthly pool, allowing: {e}")
        return NO_MONTHLY_LIMIT

    limit = int((org or {}).get("monthly_credits", NO_MONTHLY_LIMIT))
    await r.setex(org_limit_key(org_id), ORG_LIMIT_CACHE_TTL, limit)
    return limit


async def invalidate_org_limit(org_id: str) -> None:
    """Call after changing an org's monthly_credits so all workers pick it up."""
    r = await get_redis()
    await r.delete(org_limit_key(org_id))


async def try_consume(
    user_id: str, service: str,
    is_pro: bool = False, is_max: bool = False, is_plus: bool = False,
    org_id: str | None = None,
) -> tuple[bool, int, str | None]:
    """
    Atomically check the rate limit, daily quota and monthly budget, and consume
    one unit if all three allow it.

    Returns (allowed, count, reason). On success reason is None and count is
    the new daily total. On refusal reason is "rate_limit", "quota" or
    "monthly_quota" and count is the value that blocked it. Nothing is consumed
    when refused, so a rejected request does not eat rate-limit budget - same as
    the behaviour of the check/increment pair this replaces.

    Pass org_id for enterprise seats: their monthly credits come out of one
    shared org pool rather than a per-user budget, while the daily limit stays
    per seat. Prefer try_consume_for(), which fills this in from the JWT.
    """
    global _consume_script
    r = await get_redis()
    if _consume_script is None:
        # register_script sends EVALSHA and falls back to EVAL on NOSCRIPT, so
        # this stays one round trip and re-loads itself after a Redis restart.
        _consume_script = r.register_script(_CONSUME_LUA)

    limit = limit_for(service, is_pro=is_pro, is_max=is_max, is_plus=is_plus)

    if service not in MONTHLY_METERED:
        monthly = NO_MONTHLY_LIMIT
    elif org_id:
        monthly = await org_monthly_limit(org_id)
    else:
        monthly = monthly_limit_for(service, is_pro=is_pro, is_max=is_max, is_plus=is_plus)

    code, count = await _consume_script(
        keys=[
            f"ratelimit:{user_id}",
            daily_key(user_id, service),
            monthly_key(user_id, service, org_id),
        ],
        args=[
            RATE_LIMIT_PER_MINUTE, limit, _seconds_until_midnight(),
            monthly, MONTHLY_KEY_TTL,
        ],
    )
    code, count = int(code), int(count)

    if code == -1:
        asyncio.create_task(_send_abuse_alert_async(user_id, service))
        return False, count, "rate_limit"
    if code == -2:
        return False, count, "quota"
    if code == -3:
        return False, count, "monthly_quota"

    # Every service routes its consumption through here, so this one call
    # covers monitor, agent_creator and all eight messaging channels. It is
    # synchronous and non-blocking by design - see usage_log.
    log_usage(user_id, service)
    return True, count, None


async def try_consume_for(user, service: str) -> tuple[bool, int, str | None]:
    """
    try_consume() for an AuthenticatedUser. Every call site should use this:
    it is the only thing that guarantees an enterprise seat's org_id reaches
    the limiter, and forgetting it silently bills the user's own budget
    instead of the org pool.

    Deliberately duck-typed rather than importing AuthenticatedUser, to keep
    quota_manager free of a dependency on the auth layer.
    """
    return await try_consume(
        user.id, service,
        is_pro=user.is_pro, is_max=user.is_max, is_plus=user.is_plus,
        org_id=user.org_id,
    )


async def get_monthly_usage(
    user_id: str | None, service: str = "monitor", org_id: str | None = None
) -> int:
    """Credits spent this month - the org's total for an enterprise seat."""
    r = await get_redis()
    val = await r.get(monthly_key(user_id, service, org_id))
    return int(val) if val else 0


def daily_resets_at() -> str:
    return _next_midnight().isoformat().replace("+00:00", "Z")


def monthly_resets_at() -> str:
    return _next_month_start().isoformat().replace("+00:00", "Z")


async def get_usage_for_service(user_id: str, service: str) -> int:
    r = await get_redis()
    val = await r.get(f"quota:{user_id}:{service}")
    return int(val) if val else 0

async def get_all_usage_data() -> dict:
    r = await get_redis()
    usage_data: Dict[str, Dict[str, int]] = {}
    chirp_data: Dict[str, float] = {}
    gemini_data: Dict[str, float] = {}

    async for key in r.scan_iter("quota:*"):
        parts = key.split(":", 2)
        if len(parts) == 3:
            _, user_id, service = parts
            val = await r.get(key)
            if val:
                usage_data.setdefault(user_id, {})[service] = int(val)

    async for key in r.scan_iter("audio:*"):
        parts = key.split(":", 2)
        if len(parts) == 3:
            _, user_id, provider = parts
            val = await r.get(key)
            if val:
                if provider == "chirp3":
                    chirp_data[user_id] = float(val)
                else:
                    gemini_data[user_id] = float(val)

    from auth0_manager import get_email_by_id

    all_user_ids = set(usage_data) | set(chirp_data) | set(gemini_data)
    enriched_data = {}
    for user_id in all_user_ids:
        try:
            email = get_email_by_id(user_id)
            key = email if email else user_id
        except Exception as e:
            logger.error(f"Error fetching email for {user_id}: {e}")
            key = user_id

        entry = dict(usage_data.get(user_id, {}))
        chirp_secs = chirp_data.get(user_id, 0.0)
        gemini_secs = gemini_data.get(user_id, 0.0)
        if chirp_secs:
            entry["chirp_seconds"] = round(chirp_secs)
        if gemini_secs:
            entry["gemini_seconds"] = round(gemini_secs)
        enriched_data[key] = entry

    return enriched_data

async def check_provider_seconds_quota(
    user_id: str, audio_seconds: float, provider: str,
    is_pro: bool = False, is_max: bool = False, is_plus: bool = False,
) -> bool:
    tier = "max" if is_max else "pro" if is_pro else "plus" if is_plus else "free"
    limits = CHIRP_SECOND_LIMITS if provider == "chirp3" else GEMINI_SECOND_LIMITS
    limit = limits[tier]

    r = await get_redis()
    val = await r.get(f"audio:{user_id}:{provider}")
    current = float(val) if val else 0.0
    return current + audio_seconds > limit

async def increment_provider_seconds(user_id: str, audio_seconds: float, provider: str) -> float:
    r = await get_redis()
    key = f"audio:{user_id}:{provider}"
    new_total = await r.incrbyfloat(key, audio_seconds)
    # Only set TTL on first write
    if new_total == audio_seconds:
        await r.expire(key, _seconds_until_midnight())
    return new_total
