# redis_client.py
"""
Shared Redis connection pool.

Both quota_manager and observability need Redis. Owning the client here keeps
them from building a pool each, and keeps observability from having to import
quota_manager just to get a handle (a strange dependency direction: analytics
should not depend on billing limits).

--- History ------------------------------------------------------------------

This module briefly targeted a managed Redis (Upstash) reached over the public
internet. That is overkill at current scale: a box-local Redis container is
simpler, ~100x faster per op, and has no per-command bill. So the pool below is
back to the minimal localhost form.

The managed-Redis configuration is preserved verbatim at the bottom of this
file. Restore it (and point REDIS_URL back at the rediss:// endpoint) if the
API ever runs as more than one *concurrent* box and needs shared state again.
Everything it added - timeouts, a connection cap, a blocking pool, health
checks, timeout retries - exists only to survive a Redis that is a network hop
away, and every one of those is a liability against a local unix/loopback
socket (in particular retry_on_timeout retried the non-idempotent INCR inside
quota_manager's consume script and could double-charge a credit).
"""

import os

import redis.asyncio as aioredis

_redis: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        url = os.getenv("REDIS_URL")
        if not url:
            # No localhost default on purpose: a missing REDIS_URL should fail
            # loudly, not silently split quota across the wrong database.
            raise RuntimeError(
                "REDIS_URL is not set. Expected redis://redis:6379/0 "
                "(the local redis service in docker-compose)."
            )
        _redis = aioredis.Redis.from_url(url, decode_responses=True)
    return _redis


async def close_redis() -> None:
    """Called from the FastAPI lifespan shutdown hook."""
    global _redis
    if _redis is None:
        return
    # aclose() is redis-py >= 5.0.1; close() on older releases.
    closer = getattr(_redis, "aclose", None) or _redis.close
    await closer()
    _redis = None


# ---------------------------------------------------------------------------
# Managed-Redis (Upstash, over the public internet) configuration.
#
# Kept for reference. To go back to a shared Redis:
#   1. set REDIS_URL to the native TCP endpoint
#      (rediss://default:<password>@<host>:6379), NOT the REST URL
#   2. replace get_redis()'s from_url() call with the pool built below
#   3. add `retry_on_timeout` back only if you also make the consume script
#      idempotent - otherwise a lost-reply timeout double-counts.
#
# MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "40"))
# POOL_TIMEOUT = float(os.getenv("REDIS_POOL_TIMEOUT", "5"))
#
# Redis is a managed service reached over the public internet, not a container
# on the same host. That changes three defaults:
#
#   - No localhost fallback. A missing REDIS_URL used to land on the local
#     container and quietly work; now it would either fail on every request
#     with a confusing connection error, or - worse, on a box that still has a
#     local Redis - succeed against the wrong database and split quota in two.
#   - Timeouts. Without them a dropped connection hangs a request forever
#     instead of failing it.
#   - A connection cap. redis-py defaults to 100 per pool and the API runs four
#     uvicorn workers, so the process group can open 400 - more than most
#     managed plans allow.
#   - A blocking pool. redis-py's default pool *raises* once max_connections
#     is reached, so a burst would 500 rather than queue. BlockingConnectionPool
#     waits POOL_TIMEOUT seconds for a free connection first, which is what a
#     web API wants: a little latency under burst, not an error.
#
#     pool = aioredis.BlockingConnectionPool.from_url(
#         url,
#         decode_responses=True,
#         max_connections=MAX_CONNECTIONS,
#         timeout=POOL_TIMEOUT,
#         socket_connect_timeout=5,
#         socket_timeout=5,
#         socket_keepalive=True,
#         # Managed Redis drops idle connections; without this the first
#         # command on a stale one fails instead of transparently reconnecting.
#         health_check_interval=30,
#         retry_on_timeout=True,
#     )
#     _redis = aioredis.Redis(connection_pool=pool)
