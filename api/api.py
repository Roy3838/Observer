from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import uvicorn
import argparse
import asyncio
import logging
import os
import re
import stripe
import hashlib
import httpx
import uuid
from pathlib import Path

# Logging must be configured before any other module logs at import time.
from logging_config import setup_logging, request_id_var
setup_logging()

from auth import AuthUser
from auth0_manager import delete_user
import r2_store
from redis_client import close_redis

# Import routers from our modules
import marketplace
from marketplace import marketplace_router
from compute import compute_router
from tools_router import tools_router
from messaging import messaging_router
from remote import remote_router, deadman_sweeper
from payments import payments_router, _list_personal_customers
from apple_payments import apple_payments_router
from transcriptions import transcriptions_router
from orgs import orgs_router
import api_handlers

logger = logging.getLogger('api-server')

# modelo = randomforest()

# @app.post("/resultados")
# def enviar_resultados(datos):
#     resultados = modelo(datos)
#     return resultados

MAX_BODY_SIZE = 20 * 1024 * 1024  # 20 MB

# Threads behind asyncio.to_thread, per worker process. Every synchronous SDK
# call runs there (R2/boto3, Stripe, Twilio, SendGrid, Pillow), and a thread
# waiting on the network holds its slot without using CPU, so the pool has to
# cover slow calls in flight, not cores. Python's default is min(32, CPUs + 4),
# 12 on an 8-core box, which one slow provider minute can fill.
IO_THREADS = 32

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once in each uvicorn worker, so each gets its own pool.
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=IO_THREADS, thread_name_prefix="io")
    )
    await api_handlers.startup_handlers()
    # One per worker; the sweep claims atomically, so each stopped session is announced once.
    sweeper = asyncio.create_task(deadman_sweeper())
    yield
    sweeper.cancel()
    await api_handlers.shutdown_handlers()
    await close_redis()

# Setup FastAPI app
app = FastAPI(lifespan=lifespan)

# Outbound message media, served from R2 rather than a StaticFiles mount over a
# local directory. The URL messaging.py hands to WhatsApp and Telegram points at
# a hostname that load balances across boxes, so the provider's fetch does not
# necessarily come back to the box that wrote the file. The path is unchanged,
# so nothing outside this process notices.
#
# Proxied rather than redirected to a presigned URL: R2 egress is free, the
# volume is one fetch per message, and this assumes nothing about whether a
# given provider follows redirects.
_TEMP_MEDIA_NAME = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.(jpg|mp4)$")


@app.get("/temp-images/{filename}")
async def serve_temp_media(filename: str):
    # StaticFiles sanitised the path for us; this route has to do it itself.
    # The names are generated UUIDs, so an exact-shape match is both sufficient
    # and tight enough to make traversal impossible.
    if not _TEMP_MEDIA_NAME.match(filename):
        raise HTTPException(status_code=404, detail="Not found")

    found = await r2_store.get_bytes(r2_store.temp_media_key(filename))
    if found is None:
        raise HTTPException(status_code=404, detail="Not found")

    body, content_type = found
    return Response(
        content=body,
        media_type=content_type,
        # Immutable: the name is a UUID, so a given URL's bytes never change.
        # Capped below the bucket lifecycle so nothing caches past expiry.
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )

# Stamp every request with an id so its log lines can be correlated. Four
# uvicorn workers share one stdout, so without this a single request's lines are
# indistinguishable from everyone else's.
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        request_id_var.reset(token)

# Reject oversized request bodies before they are read into memory
@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_SIZE:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body too large. Maximum size is 20MB."},
        )
    return await call_next(request)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers - without prefixes to maintain original URL structure
app.include_router(marketplace_router)
# Mount compute router last since it has a catch-all route
app.include_router(compute_router)
# Mount tools router
app.include_router(tools_router)
# Mount twilio router
app.include_router(messaging_router)
# Remote control mailbox (WhatsApp/Telegram <-> browser MCP)
app.include_router(remote_router)
# Payments router (Stripe)
app.include_router(
    payments_router,
    prefix="/payments",
    tags=["Payments"]
)
# Apple payments router
app.include_router(
    apple_payments_router,
    prefix="/payments",
    tags=["Payments"]
)
# Transcriptions router
app.include_router(transcriptions_router)
# Enterprise orgs router
app.include_router(orgs_router, tags=["Organizations"])

YOUTUBE_CHANNEL_ID = "UCgXTVhPSngONO6XhQiLhftg"
LIVE_CACHE_TTL = 120  # seconds

_live_cache: dict = {"live": False, "videoId": None, "last_checked": 0.0}

@app.get("/live")
async def live_status():
    import time
    now = time.monotonic()
    if now - _live_cache["last_checked"] < LIVE_CACHE_TTL:
        return {"live": _live_cache["live"], "videoId": _live_cache["videoId"]}

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        return {"live": False, "videoId": None}

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "id",
        "channelId": YOUTUBE_CHANNEL_ID,
        "eventType": "live",
        "type": "video",
        "key": api_key,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, timeout=10)

    if resp.status_code != 200:
        _live_cache["last_checked"] = now
        return {"live": False, "videoId": None}

    items = resp.json().get("items", [])
    result = {"live": bool(items), "videoId": items[0]["id"]["videoId"] if items else None}
    _live_cache.update({**result, "last_checked": now})
    return result


# Root path to check if service is running
@app.get("/")
async def root():
    return {"status": "API server is running"}


# Thread-side Stripe work for account deletion. The stripe SDK is synchronous,
# so delete_account runs each of these through asyncio.to_thread.

def _cancel_active_subscriptions(customer_id: str, user_id: str) -> None:
    subscriptions = stripe.Subscription.list(customer=customer_id, status='all', limit=100)
    for sub in subscriptions.data:
        if sub.status in ('active', 'trialing'):
            stripe.Subscription.cancel(sub.id)
            logger.info(f"Cancelled Stripe subscription {sub.id} for user {user_id}")


def _redact_stripe_customers(email: str, user_id: str) -> None:
    """
    Cancel, detach cards and redact every *personal* Stripe customer for `email`.

    An enterprise org's Stripe customer carries its billing contact's email, so
    a plain email lookup returns it too. It pays for every seat in the company:
    cancelling it here would cut off the whole org because one person deleted
    their own account. _list_personal_customers filters it out (and logs it).
    """
    email_hash = hashlib.sha256(email.encode()).hexdigest()
    ghost_email = f"{email_hash}@deleted.invalid"

    for cust in _list_personal_customers(email):
        # Cancel active subscriptions
        _cancel_active_subscriptions(cust.id, user_id)

        # Detach all payment methods
        payment_methods = stripe.PaymentMethod.list(customer=cust.id, type="card")
        for pm in payment_methods.data:
            stripe.PaymentMethod.detach(pm.id)

        # Replace email with hash, wipe all other PII
        stripe.Customer.modify(
            cust.id,
            email=ghost_email,
            name="",
            phone="",
            address={},
            metadata={"deleted": "true"}
        )
        logger.info(f"Redacted Stripe customer {cust.id} for user {user_id}")


@app.delete("/delete-account", summary="Permanently delete user account")
async def delete_account(current_user: AuthUser):
    """
    Permanently deletes the authenticated user's account.
    This will:
    1. Cancel any active Stripe subscription
    2. Delete all marketplace agents created by the user
    3. Delete the Auth0 user account

    This action is irreversible.
    """
    user_id = current_user.id
    logger.info(f"Account deletion requested for user: {user_id}")

    if current_user.org_id:
        # Their org's subscription is deliberately left alone (see
        # _redact_stripe_customers), but the org record in R2 still lists them
        # as an active member, and possibly as owner_email, the billing contact.
        logger.warning(
            f"Deleted user {user_id} held a seat on org {current_user.org_id}; "
            f"their seat and, if they were the owner, the billing contact need updating by hand"
        )

    # 1. Cancel Stripe subscription if exists
    stripe_customer_id = None
    if hasattr(current_user, 'app_metadata') and isinstance(current_user.app_metadata, dict):
        stripe_customer_id = current_user.app_metadata.get("stripe_customer_id")

    email = (getattr(current_user, 'email', None) or '').lower()
    if email:
        try:
            await asyncio.to_thread(_redact_stripe_customers, email, user_id)
        except Exception as e:
            logger.error(f"Failed to process Stripe data for user {user_id}: {e}")
            # Continue with deletion even if Stripe fails
    elif stripe_customer_id:
        # Fallback: no email available, cancel by customer ID only
        try:
            await asyncio.to_thread(_cancel_active_subscriptions, stripe_customer_id, user_id)
        except Exception as e:
            logger.error(f"Failed to cancel Stripe subscriptions for user {user_id}: {e}")

    # 2. Delete marketplace agents created by this user
    try:
        deleted_agents = await marketplace.delete_agents_by_author(user_id)
        logger.info(f"Deleted {deleted_agents} marketplace agents for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to delete marketplace agents for user {user_id}: {e}")
        # Continue with deletion even if marketplace cleanup fails

    # 3. Delete Auth0 user account
    deleted = await delete_user(user_id)
    if not deleted:
        logger.error(f"Failed to delete Auth0 account for user {user_id}")
        raise HTTPException(status_code=500, detail="Failed to delete account. Please try again or contact support.")

    logger.info(f"Successfully deleted account for user {user_id}")
    return {"success": True, "message": "Account deleted successfully"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Observer AI API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--proxy-target", default="https://compute.observer-ai.com", help="Target service URL for proxy")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    os.environ["AI_SERVICE_URL"] = args.proxy_target

    print("\n\033[1m OBSERVER AI API SERVER \033[0m ready")
    print(f"  ➜  \033[36mLocal:   \033[0mhttp://localhost:{args.port}/")
    print(f"\n  Marketplace routes: http://localhost:{args.port}/agents")
    print(f"  Compute quota: http://localhost:{args.port}/quota")
    print(f"  Proxy forwarding to: {args.proxy_target}")

    uvicorn.run("api:app", host="0.0.0.0", port=args.port, workers=4)
