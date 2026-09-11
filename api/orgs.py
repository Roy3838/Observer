# orgs.py
"""
Enterprise organizations: seat-based billing, invites, and the team dashboard.

Design notes
------------
* `org_id` in Auth0 app_metadata is the third subscription sentinel, alongside
  `stripe_subscription_id` (Stripe) and `apple_original_transaction_id` (Apple).
  A user has at most one of the three. `is_pro`/`is_max` remain the effective
  entitlement flags, so every existing gate (quota, model access, the frontend's
  `isProUser`) keeps working with no changes.

* The org's own `stripe_customer_id` lives ONLY on the R2 org record, never in a
  member's metadata. Otherwise any employee could open the billing portal
  (payments._create_billing_portal) and cancel the company subscription.

* There are no roles. Any member can invite, bounded by `seats_purchased`.
  Removal is restricted to `owner_email` — the billing contact — because that is
  the one destructive action.
"""

import asyncio
import logging
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import stripe
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import ClickTracking, Mail, TrackingSettings

import r2_store
import usage_log
from admin_auth import get_admin_access
from auth import AuthUser
from auth0_manager import (
    CLEAR_FIELD,
    find_user_by_email,
    get_user_app_metadata,
    check_existing_subscription,
    update_user_subscription_status,
)
from quota_manager import (
    NO_MONTHLY_LIMIT,
    get_monthly_usage,
    invalidate_org_limit,
    monthly_resets_at,
    org_monthly_limit,
)
from redis_client import get_redis

logger = logging.getLogger('orgs')

orgs_router = APIRouter()

APP_BASE_URL = os.environ.get("APP_BASE_URL", "https://app.observer-ai.com")
INVITE_TTL_DAYS = 14

# Services surfaced on the team dashboard, in display order.
DASHBOARD_SERVICES = ["monitor", "agent_creator", "email", "sms", "whatsapp", "telegram", "voice_call", "pushover"]


# --- Models -----------------------------------------------------------------

class OrgCreateRequest(BaseModel):
    name: str
    admin_email: EmailStr
    tier: str = "pro"           # "pro" | "max"
    seats: int
    # Shared monthly monitor-credit pool for the whole org. -1 is uncapped,
    # which is what an org gets until a number is actually negotiated.
    monthly_credits: int = NO_MONTHLY_LIMIT
    price_id: Optional[str] = None
    days_until_due: int = 30
    dry_run: bool = False


class InviteRequest(BaseModel):
    email: EmailStr


class RemoveRequest(BaseModel):
    email: EmailStr


class ClaimRequest(BaseModel):
    token: str


class OrgCreditsRequest(BaseModel):
    monthly_credits: int


# --- Helpers ----------------------------------------------------------------

def sget(obj, key, default=None):
    """
    Read a key from a Stripe object.

    stripe.StripeObject implements __getitem__ but not .get() — calling .get()
    on one raises AttributeError rather than returning a default. This works for
    both StripeObject and plain dicts.
    """
    if obj is None:
        return default
    try:
        return obj[key]
    except (KeyError, TypeError, AttributeError, IndexError):
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(name: str) -> str:
    keep = "".join(c.lower() if c.isalnum() else "-" for c in name)
    return "-".join(p for p in keep.split("-") if p)[:40] or "org"


def _seats_used(org: dict) -> int:
    return sum(1 for m in org["members"] if m["status"] in ("invited", "active"))


def _find_member(org: dict, email: str) -> Optional[dict]:
    email = email.lower()
    return next((m for m in org["members"] if m["email"] == email), None)


def _tier_flags(tier: str) -> tuple[bool, bool]:
    """(is_pro, is_max) for an org tier."""
    return (tier != "max", tier == "max")


async def _require_org(user: AuthUser) -> tuple[str, dict]:
    """Resolve the caller's org from their JWT. The org_id never comes from the request."""
    org_id = (user.app_metadata or {}).get("org_id")
    if not org_id:
        raise HTTPException(status_code=403, detail="You are not a member of an organization.")

    org, _ = await r2_store.get_json(r2_store.org_key(org_id))
    if not org:
        logger.error(f"User {user.id} has org_id {org_id} but no org record exists")
        raise HTTPException(status_code=404, detail="Organization not found.")

    return org_id, org


def _send_invite_email(email: str, org_name: str, token: str) -> None:
    api_key = os.getenv("SENDGRID_API_KEY")
    from_email = os.getenv("SENDGRID_FROM_EMAIL")
    if not all([api_key, from_email]):
        logger.error("Cannot send invite: SENDGRID_API_KEY or SENDGRID_FROM_EMAIL missing")
        return

    join_url = f"{APP_BASE_URL}/join?token={token}"
    message = Mail(
        from_email=from_email,
        to_emails=email,
        subject=f"You've been added to {org_name} on Observer AI",
        plain_text_content=(
            f"You've been given a seat on {org_name}'s Observer AI plan.\n\n"
            f"Accept your seat here:\n{join_url}\n\n"
            f"This link is single-use and expires in {INVITE_TTL_DAYS} days.\n"
            f"It must be accepted with this email address ({email})."
        ),
    )
    # SendGrid click tracking rewrites the join link to a ct.sendgrid.net
    # redirect. On a link that grants account access that reads as phishing to
    # recipients and to corporate mail filters, so send the real URL.
    message.tracking_settings = TrackingSettings(
        click_tracking=ClickTracking(enable=False, enable_text=False)
    )
    try:
        SendGridAPIClient(api_key).send(message)
        logger.info(f"Sent org invite to {email}")
    except Exception:
        logger.exception(f"Failed to send org invite email to {email}")


async def grant_seat(email: str, org_id: str, org_name: str, tier: str) -> dict:
    """
    Give `email` a seat on `org_id`.

    If they already have an Observer account, entitle them immediately. If not,
    park a single-use invite in R2 that they claim after signing up.

    Used by both provisioning (for the first admin) and member invites, so the
    "does this Auth0 user exist yet" fork only exists in one place.
    """
    email = email.lower()
    is_pro, is_max = _tier_flags(tier)

    user_id = await find_user_by_email(email)
    if user_id:
        ok = await update_user_subscription_status(
            user_id=user_id,
            is_pro=is_pro,
            is_max=is_max,
            is_plus=False,
            org_id=org_id,
            org_tier=tier,
            stripe_customer_id=CLEAR_FIELD,
        )
        if not ok:
            raise HTTPException(status_code=502, detail="Could not update the user's account in Auth0.")
        logger.info(f"Granted {email} ({user_id}) an active seat on {org_id}")
        return {"status": "active", "auth0_user_id": user_id}

    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now(timezone.utc) + timedelta(days=INVITE_TTL_DAYS)).isoformat()
    await r2_store.put_json(
        r2_store.invite_key(token),
        {"org_id": org_id, "email": email, "tier": tier, "expires_at": expires_at, "created_at": _now_iso()},
    )
    await asyncio.to_thread(_send_invite_email, email, org_name, token)
    logger.info(f"Parked invite for {email} on {org_id}")
    return {"status": "invited", "auth0_user_id": None}


async def _reserve_and_grant(org_id: str, org: dict, email: str) -> dict:
    """
    Claim a seat for `email` on `org_id`, then entitle or invite them.

    The seat is reserved in R2 first. If the grant then fails we are left with a
    reserved seat and no invite — visible on the team page and revocable — which
    is the safe direction; the reverse would be a live invite for a seat nobody
    counted against the cap.

    The guards run again inside the mutator because update_json re-invokes it
    against freshly read data on an ETag conflict, so two admins inviting at the
    same moment cannot both pass a stale seat check.
    """
    def reserve(rec):
        if _seats_used(rec) >= rec["seats_purchased"]:
            raise HTTPException(status_code=409, detail="All seats are in use.")
        m = _find_member(rec, email)
        if m and m["status"] in ("invited", "active"):
            raise HTTPException(status_code=409, detail=f"{email} is already on this team.")
        if m:
            m.update({"status": "invited", "invited_at": _now_iso(), "joined_at": None, "auth0_user_id": None})
        else:
            rec["members"].append({
                "email": email, "auth0_user_id": None, "status": "invited",
                "invited_at": _now_iso(), "joined_at": None,
            })

    await r2_store.update_json(r2_store.org_key(org_id), reserve)

    result = await grant_seat(email, org_id, org["name"], org["tier"])

    if result["status"] == "active":
        def activate(rec):
            m = _find_member(rec, email)
            if m:
                m["status"] = "active"
                m["auth0_user_id"] = result["auth0_user_id"]
                m["joined_at"] = _now_iso()
        await r2_store.update_json(r2_store.org_key(org_id), activate)

    return result


async def _assert_no_personal_subscription(email: str) -> None:
    """A user paying for their own plan must cancel before taking a seat, or they'd be billed twice."""
    user_id = await find_user_by_email(email)
    if not user_id:
        return
    metadata = await get_user_app_metadata(user_id)
    has_sub, _, provider = check_existing_subscription(metadata)
    if has_sub and provider != "org":
        raise HTTPException(
            status_code=409,
            detail=f"{email} has an active personal {provider} subscription. "
                   f"They need to cancel it before joining an organization.",
        )


# --- Provisioning (Observer admin) ------------------------------------------

@orgs_router.post("/admin/orgs", dependencies=[Depends(get_admin_access)], summary="Provision an enterprise org")
async def create_org(body: OrgCreateRequest):
    """
    Called by hand after a deal closes. Creates the Stripe customer and an
    invoiced subscription, writes the org record to R2, and grants the first
    seat to the billing contact.

    Returns the hosted invoice URL and the join link so they can be pasted into
    the sales email — this does not depend on Stripe's own invoice email being
    configured or enabled.

    Pass dry_run=true to validate inputs without touching Stripe or R2.
    """
    if body.tier not in ("pro", "max"):
        raise HTTPException(status_code=400, detail="tier must be 'pro' or 'max'")
    if body.seats < 1:
        raise HTTPException(status_code=400, detail="seats must be at least 1")

    price_id = body.price_id or os.environ.get("STRIPE_ENTERPRISE_SEAT_PRICE_ID")
    if not price_id:
        raise HTTPException(status_code=400, detail="No price_id given and STRIPE_ENTERPRISE_SEAT_PRICE_ID is not set.")

    org_id = f"obs_org_{_slugify(body.name)}_{secrets.token_hex(3)}"
    admin_email = body.admin_email.lower()

    if body.dry_run:
        return {
            "dry_run": True, "org_id": org_id, "price_id": price_id,
            "seats": body.seats, "tier": body.tier, "owner_email": admin_email,
            "monthly_credits": body.monthly_credits,
        }

    await _assert_no_personal_subscription(admin_email)

    try:
        customer = await asyncio.to_thread(
            lambda: stripe.Customer.create(
                email=admin_email,
                name=body.name,
                metadata={"org_id": org_id},
            )
        )
        subscription = await asyncio.to_thread(
            lambda: stripe.Subscription.create(
                customer=customer.id,
                items=[{"price": price_id, "quantity": body.seats}],
                collection_method="send_invoice",
                days_until_due=body.days_until_due,
                metadata={"org_id": org_id},
                expand=["latest_invoice"],
            )
        )
    except Exception as e:
        logger.exception(f"Stripe provisioning failed for org {org_id}")
        raise HTTPException(status_code=502, detail=f"Stripe provisioning failed: {e}")

    # Everything past this point can still fail, and a half-provisioned org
    # leaves a live subscription billing a customer who has no seats. Unwind the
    # Stripe objects rather than leaving them orphaned.
    try:
        invoice = sget(subscription, "latest_invoice")
        if isinstance(invoice, str):
            invoice_id, invoice_url = invoice, None
        else:
            invoice_id = sget(invoice, "id")
            invoice_url = sget(invoice, "hosted_invoice_url")

        org = {
            "org_id": org_id,
            "name": body.name,
            "owner_email": admin_email,
            "tier": body.tier,
            "seats_purchased": body.seats,
            "monthly_credits": body.monthly_credits,
            "status": subscription.status,
            "stripe_customer_id": customer.id,
            "stripe_subscription_id": subscription.id,
            "features": {},
            "members": [{
                "email": admin_email,
                "auth0_user_id": None,
                "status": "invited",
                "invited_at": _now_iso(),
                "joined_at": None,
            }],
            "created_at": _now_iso(),
        }
        # if_none_match="*" so a colliding org_id fails loudly instead of overwriting.
        await r2_store.put_json(r2_store.org_key(org_id), org, if_none_match="*")

        result = await grant_seat(admin_email, org_id, body.name, body.tier)
        if result["status"] == "active":
            def activate(rec):
                m = _find_member(rec, admin_email)
                m["status"] = "active"
                m["auth0_user_id"] = result["auth0_user_id"]
                m["joined_at"] = _now_iso()
            await r2_store.update_json(r2_store.org_key(org_id), activate)

    except Exception as e:
        logger.exception(f"Provisioning failed after Stripe setup for org {org_id}; unwinding")
        try:
            await asyncio.to_thread(lambda: stripe.Subscription.cancel(subscription.id))
            await asyncio.to_thread(lambda: stripe.Customer.delete(customer.id))
            logger.info(f"Unwound Stripe customer {customer.id} for failed org {org_id}")
        except Exception:
            logger.exception(
                f"MANUAL CLEANUP NEEDED: could not unwind Stripe customer {customer.id} "
                f"/ subscription {subscription.id} for failed org {org_id}"
            )
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Provisioning failed after Stripe setup: {e}")

    logger.info(f"Provisioned org {org_id} ({body.seats} {body.tier} seats) for {admin_email}")
    return {
        "org_id": org_id,
        "owner_email": admin_email,
        "seats": body.seats,
        "tier": body.tier,
        "stripe_customer_id": customer.id,
        "stripe_subscription_id": subscription.id,
        "invoice_id": invoice_id,
        "hosted_invoice_url": invoice_url,
        "owner_status": result["status"],
        "team_url": f"{APP_BASE_URL}/team",
    }


@orgs_router.get("/admin/orgs/{org_id}", dependencies=[Depends(get_admin_access)], summary="Read an org record")
async def admin_get_org(org_id: str):
    """
    Support/debug view of an org. Unlike /orgs/me this needs no membership, so
    it works before anyone has claimed a seat.
    """
    org, _ = await r2_store.get_json(r2_store.org_key(org_id))
    if not org:
        raise HTTPException(status_code=404, detail=f"No org record for {org_id}")

    member_ids = [m["auth0_user_id"] for m in org["members"]
                  if m["status"] == "active" and m["auth0_user_id"]]
    usage = await _usage_for_members(member_ids)

    return {
        **org,
        "seats_used": _seats_used(org),
        "monthly_pool": await _pool_summary(org_id),
        "members": [{**m, "usage": usage.get(m.get("auth0_user_id"), {})} for m in org["members"]],
    }


@orgs_router.post("/admin/orgs/{org_id}/credits", dependencies=[Depends(get_admin_access)], summary="Set the org's monthly credit pool")
async def admin_set_org_credits(org_id: str, body: OrgCreditsRequest):
    """
    Set the monthly monitor-credit pool shared by every seat in the org. Seats
    keep their own per-tier *daily* limit; this is the monthly budget they all
    draw from. -1 means uncapped.

    R2 is the source of truth and quota_manager caches the number in Redis, so
    the cached copy is dropped here rather than left to age out - otherwise a
    renegotiated pool would not take effect on every worker for up to an hour.
    """
    if body.monthly_credits < NO_MONTHLY_LIMIT:
        raise HTTPException(
            status_code=400,
            detail="monthly_credits must be -1 (uncapped) or a non-negative number.",
        )

    org, _ = await r2_store.get_json(r2_store.org_key(org_id))
    if not org:
        raise HTTPException(status_code=404, detail=f"No org record for {org_id}")

    def apply(rec):
        rec["monthly_credits"] = body.monthly_credits

    updated = await r2_store.update_json(r2_store.org_key(org_id), apply)
    await invalidate_org_limit(org_id)

    logger.info(f"Org {org_id} monthly credit pool set to {body.monthly_credits}")
    return {"org_id": org_id, "monthly_credits": updated["monthly_credits"]}


@orgs_router.post("/admin/orgs/{org_id}/sync", dependencies=[Depends(get_admin_access)], summary="Force a Stripe resync")
async def admin_sync_org(org_id: str):
    """Reconcile an org against Stripe by hand, for when a webhook was missed."""
    return await sync_org_from_stripe(org_id)


@orgs_router.post("/admin/orgs/{org_id}/members", dependencies=[Depends(get_admin_access)], summary="Invite a member as admin")
async def admin_invite_member(org_id: str, body: InviteRequest):
    """
    Seat someone without being a member of the org yourself. Used from the admin
    dashboard for support ("add this person for me") and to seat the first few
    members before the owner has logged in.

    Enforces the same seat cap and personal-subscription guard as the org-facing
    endpoint — the only thing skipped is membership authorization.
    """
    org, _ = await r2_store.get_json(r2_store.org_key(org_id))
    if not org:
        raise HTTPException(status_code=404, detail=f"No org record for {org_id}")

    email = body.email.lower()
    existing = _find_member(org, email)
    if existing and existing["status"] in ("invited", "active"):
        raise HTTPException(status_code=409, detail=f"{email} is already on this team.")
    if _seats_used(org) >= org["seats_purchased"]:
        raise HTTPException(status_code=409, detail=f"All {org['seats_purchased']} seats are in use.")

    await _assert_no_personal_subscription(email)
    await _reserve_and_grant(org_id, org, email)
    return {"email": email, "org_id": org_id}


# --- Claiming an invite -----------------------------------------------------

@orgs_router.post("/orgs/claim", summary="Claim a seat invite")
async def claim_invite(current_user: AuthUser, body: ClaimRequest):
    """
    Called by /join?token=... after login. Writes the org into Auth0 metadata;
    the client must then force a token refresh before the new entitlement is
    visible anywhere.
    """
    key = r2_store.invite_key(body.token)
    invite, _ = await r2_store.get_json(key)
    if not invite:
        raise HTTPException(status_code=404, detail="This invite is invalid or has already been used.")

    if datetime.fromisoformat(invite["expires_at"]) < datetime.now(timezone.utc):
        await r2_store.delete(key)
        raise HTTPException(status_code=410, detail="This invite has expired. Ask your admin to send a new one.")

    # The invite is for one specific address — a forwarded link must not work.
    if not current_user.email or current_user.email.lower() != invite["email"]:
        raise HTTPException(
            status_code=403,
            detail=f"This invite was sent to {invite['email']}. Log in with that address to accept it.",
        )

    org_id = invite["org_id"]
    org, _ = await r2_store.get_json(r2_store.org_key(org_id))
    if not org:
        raise HTTPException(status_code=404, detail="That organization no longer exists.")
    if org["status"] not in ("active", "trialing", "past_due"):
        raise HTTPException(status_code=403, detail="That organization's subscription is not active.")

    is_pro, is_max = _tier_flags(org["tier"])
    ok = await update_user_subscription_status(
        user_id=current_user.id,
        is_pro=is_pro,
        is_max=is_max,
        is_plus=False,
        org_id=org_id,
        org_tier=org["tier"],
        stripe_customer_id=CLEAR_FIELD,
    )
    if not ok:
        raise HTTPException(status_code=502, detail="Could not activate your seat. Please try again.")

    def activate(rec):
        m = _find_member(rec, current_user.email)
        if m is None:
            # Seat was revoked between invite and claim.
            raise HTTPException(status_code=403, detail="Your seat is no longer available.")
        m["status"] = "active"
        m["auth0_user_id"] = current_user.id
        m["joined_at"] = _now_iso()

    await r2_store.update_json(r2_store.org_key(org_id), activate)
    await r2_store.delete(key)

    logger.info(f"User {current_user.id} ({current_user.email}) claimed a seat on {org_id}")
    return {"status": "active", "org_id": org_id, "org_name": org["name"], "tier": org["tier"]}


# --- Team page --------------------------------------------------------------

async def _usage_for_members(member_ids: list[str]) -> dict:
    """Today's Redis counters for a specific set of users. Direct GETs, no scan."""
    if not member_ids:
        return {}
    r = await get_redis()
    pipe = r.pipeline(transaction=False)
    for uid in member_ids:
        for svc in DASHBOARD_SERVICES:
            pipe.get(f"quota:{uid}:{svc}")
    values = await pipe.execute()

    out, i = {}, 0
    for uid in member_ids:
        row = {}
        for svc in DASHBOARD_SERVICES:
            row[svc] = int(values[i]) if values[i] else 0
            i += 1
        out[uid] = row
    return out


async def _pool_summary(org_id: str) -> dict:
    """
    The org's shared monthly credit pool and how much of it is spent.

    Read from the same counter the limiter increments, so the number on the
    dashboard is the number the next request will be checked against. The
    per-member figures alongside it are today's daily counters - different
    window, same property.
    """
    limit = await org_monthly_limit(org_id)
    used = await get_monthly_usage(None, "monitor", org_id)
    uncapped = limit == NO_MONTHLY_LIMIT
    return {
        "used": used,
        "limit": limit,
        "remaining": None if uncapped else max(0, limit - used),
        "unlimited": uncapped,
        "resets_at": monthly_resets_at(),
    }


@orgs_router.get("/orgs/me", summary="Org record, roster, credit pool and today's usage")
async def get_my_org(current_user: AuthUser):
    org_id, org = await _require_org(current_user)

    member_ids = [m["auth0_user_id"] for m in org["members"]
                  if m["status"] == "active" and m["auth0_user_id"]]
    usage = await _usage_for_members(member_ids)

    is_owner = bool(current_user.email) and current_user.email.lower() == org["owner_email"]

    return {
        "org_id": org_id,
        "name": org["name"],
        "tier": org["tier"],
        "status": org["status"],
        "seats_purchased": org["seats_purchased"],
        "seats_used": _seats_used(org),
        "monthly_pool": await _pool_summary(org_id),
        "is_owner": is_owner,
        "owner_email": org["owner_email"],
        "members": [
            {
                "email": m["email"],
                "status": m["status"],
                "joined_at": m.get("joined_at"),
                "usage": usage.get(m.get("auth0_user_id"), {}),
            }
            for m in org["members"] if m["status"] != "removed"
        ],
    }


@orgs_router.get("/orgs/usage/history", summary="Historical usage for the org")
async def get_org_usage_history(
    current_user: AuthUser,
    days: int = 30,
    bucket: str = "day",
):
    """
    Usage per member over time, from the Postgres event log.

    Deliberately a separate endpoint from /orgs/me rather than more fields on
    it. The two answer different questions from different stores: /orgs/me
    reports today's counters straight from Redis, which is what the limiter
    enforces and therefore the only number safe to show next to a limit, while
    this is a best-effort historical log that may be missing rows dropped under
    load or written to the other box during a failover. Keeping them apart
    means neither one's failure mode becomes the other's, and the frontend
    always knows which kind of number it is holding.

    Returns an empty series - not an error - when usage logging is disabled or
    the query fails, so a team page never breaks over analytics.
    """
    _org_id, org = await _require_org(current_user)

    if bucket not in ("day", "week", "month"):
        raise HTTPException(status_code=400, detail="bucket must be day, week or month")
    days = max(1, min(days, 365))

    members = [m for m in org["members"] if m["status"] == "active" and m["auth0_user_id"]]
    email_by_id = {m["auth0_user_id"]: m["email"] for m in members}

    rows = await usage_log.get_usage_history(
        list(email_by_id), days=days, bucket=bucket,
    )

    return {
        "days": days,
        "bucket": bucket,
        "available": usage_log.enabled(),
        "series": [
            {
                "period": r["period"],
                "email": email_by_id.get(r["user_id"], r["user_id"]),
                "service": r["service"],
                "count": r["count"],
            }
            for r in rows
        ],
    }


@orgs_router.post("/orgs/members", summary="Invite someone to the org")
async def invite_member(current_user: AuthUser, body: InviteRequest):
    org_id, org = await _require_org(current_user)
    email = body.email.lower()

    if org["status"] not in ("active", "trialing", "past_due"):
        raise HTTPException(status_code=403, detail="This organization's subscription is not active.")

    existing = _find_member(org, email)
    if existing and existing["status"] in ("invited", "active"):
        raise HTTPException(status_code=409, detail=f"{email} is already on this team.")

    if _seats_used(org) >= org["seats_purchased"]:
        raise HTTPException(
            status_code=409,
            detail=f"All {org['seats_purchased']} seats are in use. Contact Observer to add more.",
        )

    await _assert_no_personal_subscription(email)

    result = await _reserve_and_grant(org_id, org, email)
    logger.info(f"{current_user.email} invited {email} to {org_id} -> {result['status']}")
    return {"email": email, "status": result["status"]}


async def _revoke_seat(email: str, auth0_user_id: Optional[str]) -> None:
    """Strip org entitlement from a user. Their next /quota call reports free."""
    user_id = auth0_user_id or await find_user_by_email(email)
    if not user_id:
        return
    await update_user_subscription_status(
        user_id=user_id,
        is_pro=False,
        is_max=False,
        is_plus=False,
        org_id=CLEAR_FIELD,
        org_tier=CLEAR_FIELD,
    )


@orgs_router.delete("/orgs/members", summary="Remove someone from the org (owner only)")
async def remove_member(current_user: AuthUser, body: RemoveRequest):
    org_id, org = await _require_org(current_user)
    email = body.email.lower()

    # Removal is the one destructive action, so it is restricted to the billing
    # contact. This is why there is an owner_email field and not a role model.
    if not current_user.email or current_user.email.lower() != org["owner_email"]:
        raise HTTPException(status_code=403, detail="Only the organization owner can remove members.")
    if email == org["owner_email"]:
        raise HTTPException(status_code=400, detail="The owner cannot be removed. Contact Observer to change the billing contact.")

    member = _find_member(org, email)
    if not member or member["status"] == "removed":
        raise HTTPException(status_code=404, detail=f"{email} is not on this team.")

    await _revoke_seat(email, member.get("auth0_user_id"))

    def revoke(rec):
        m = _find_member(rec, email)
        if m:
            m["status"] = "removed"
            m["removed_at"] = _now_iso()
    await r2_store.update_json(r2_store.org_key(org_id), revoke)

    logger.info(f"{current_user.email} removed {email} from {org_id}")
    return {"email": email, "status": "removed"}


# --- Stripe sync ------------------------------------------------------------

async def sync_org_from_stripe(org_id: str) -> dict:
    """
    Reconcile an org record against Stripe. Called from the webhook when the
    event's customer carries an org_id in its metadata.

    Stripe is the source of truth for status and seat count. When the
    subscription stops being payable, every member's entitlement is revoked.
    """
    key = r2_store.org_key(org_id)
    org, _ = await r2_store.get_json(key)
    if not org:
        logger.error(f"sync_org_from_stripe: no org record for {org_id}")
        return {"status": "error", "detail": "Org not found"}

    try:
        subscription = await asyncio.to_thread(
            lambda: stripe.Subscription.retrieve(org["stripe_subscription_id"])
        )
    except Exception as e:
        logger.error(f"Could not retrieve subscription for org {org_id}: {e}")
        return {"status": "error", "detail": "Failed to retrieve subscription"}

    new_status = subscription.status
    try:
        new_seats = subscription["items"]["data"][0]["quantity"]
    except (KeyError, IndexError):
        new_seats = org["seats_purchased"]

    # active/trialing/past_due keep service running: past_due means an invoice is
    # late, and cutting off a net-30 enterprise the day AP is slow is not the
    # behaviour we want. Entitlements drop on unpaid/canceled.
    entitled = new_status in ("active", "trialing", "past_due")
    was_entitled = org["status"] in ("active", "trialing", "past_due")

    def apply(rec):
        rec["status"] = new_status
        rec["seats_purchased"] = new_seats
    await r2_store.update_json(key, apply)

    if was_entitled and not entitled:
        logger.warning(f"Org {org_id} subscription is {new_status}; revoking {len(org['members'])} seats")
        for m in org["members"]:
            if m["status"] == "active":
                await _revoke_seat(m["email"], m.get("auth0_user_id"))

    elif entitled and not was_entitled:
        logger.info(f"Org {org_id} subscription recovered to {new_status}; restoring seats")
        is_pro, is_max = _tier_flags(org["tier"])
        for m in org["members"]:
            if m["status"] == "active" and m.get("auth0_user_id"):
                await update_user_subscription_status(
                    user_id=m["auth0_user_id"], is_pro=is_pro, is_max=is_max,
                    is_plus=False, org_id=org_id, org_tier=org["tier"],
                )

    logger.info(f"Synced org {org_id}: status={new_status}, seats={new_seats}")
    return {"status": "success", "org_id": org_id, "subscription_status": new_status, "seats": new_seats}
