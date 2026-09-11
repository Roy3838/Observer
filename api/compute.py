# compute.py

from fastapi import APIRouter, Request, HTTPException, status, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
import logging
import json

# --- Local Imports ---
from auth import AuthUser
from admin_auth import get_admin_access
# Import the new, specific functions and the QUOTA_LIMITS dictionary
from quota_manager import (
    try_consume_for, limit_for, monthly_limit_for, org_monthly_limit,
    get_usage_for_service, get_monthly_usage,
    daily_resets_at, monthly_resets_at, NO_MONTHLY_LIMIT,
    QUOTA_LIMITS, PRO_QUOTA_LIMITS, MAX_QUOTA_LIMITS, PLUS_QUOTA_LIMITS,
)
import observability

# Logging is configured once in api.py via logging_config.setup_logging()
logger = logging.getLogger('compute_router')

# --- Observer AI Handler Integration ---
try:
    import api_handlers
    from api_handlers import HandlerError, ConfigError, BackendAPIError
    logger.info("Successfully imported api_handlers. Available handlers: %s", list(api_handlers.API_HANDLERS.keys()))
    HANDLERS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Could not import api_handlers: {e}. Backend routing will not work.", exc_info=True)
    api_handlers, HandlerError, ConfigError, BackendAPIError, HANDLERS_AVAILABLE = (None, Exception, Exception, Exception, False)
# --- End Integration ---

compute_router = APIRouter()

# --- Agent Creator Models Configuration ---
AGENT_CREATOR_MODELS = {
    "gemini-2.0-flash-lite-free",
    "gemini-2.5-flash-lite-free"
}

async def _log_streaming_response(stream_iterator, ctx: dict):
    """
    Wrapper that records the complete streaming response with timing metrics.
    Accumulates content from OpenAI SSE chunks and records when the stream ends.

    `ctx` carries the static fields for the request (user, model, handler, tier,
    service, prompt, ...) and is passed straight through to
    observability.record_request. The recording happens after the final chunk
    has been yielded, so it is off the user-visible latency path.
    """
    import time

    response_parts = []
    start_time = time.time()
    first_token_time = None
    total_chunks = 0

    try:
        async for chunk in stream_iterator:
            # Yield chunk immediately for streaming
            yield chunk

            # Parse chunk to extract content for logging
            if isinstance(chunk, (str, bytes)):
                chunk_str = chunk.decode() if isinstance(chunk, bytes) else chunk
                if chunk_str.startswith("data: ") and not chunk_str.startswith("data: [DONE]"):
                    try:
                        json_data = chunk_str[6:].strip()  # Remove "data: " prefix
                        if json_data:
                            chunk_json = json.loads(json_data)
                            choices = chunk_json.get("choices", [])
                            if choices and "delta" in choices[0]:
                                content = choices[0]["delta"].get("content")
                                if content:
                                    # Mark time to first token
                                    if first_token_time is None:
                                        first_token_time = time.time()
                                    response_parts.append(content)
                                    total_chunks += 1
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # Skip malformed chunks
                        continue

        # Calculate timing metrics
        end_time = time.time()
        total_duration = end_time - start_time

        time_to_first_token_ms = None
        if first_token_time is not None:
            time_to_first_token_ms = round((first_token_time - start_time) * 1000, 2)

        chunks_per_second = None
        if total_chunks > 0 and total_duration > 0:
            chunks_per_second = round(total_chunks / total_duration, 2)

        # Record the complete response when the stream finishes
        complete_response = ''.join(response_parts)
        await observability.record_request(
            **ctx,
            response_text=complete_response,
            status_code=200,
            ttft_ms=time_to_first_token_ms,
            chunks_per_second=chunks_per_second,
        )

    except Exception as e:
        # Record the failure if the stream breaks partway through
        await observability.record_request(
            **ctx,
            response_text=f"STREAM_ERROR: {str(e)}",
            status_code=500,
        )

# --- API Routes ---

@compute_router.get("/admin/metrics", tags=["Admin"], summary="Get the anonymised request digest for a day")
async def get_all_metrics(
    date: str | None = Query(
        None,
        description="UTC day to read, YYYY-MM-DD. Defaults to today. The nightly "
                    "digest routine runs at ~00:15 UTC and should ask for yesterday.",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
    limit: int | None = Query(None, gt=0, description="Cap the number of entries returned, newest first."),
    is_admin: bool = Depends(get_admin_access),
):
    """
    (Admin) Returns the anonymised request digest for one UTC day: prompts,
    responses, models, timings and error status. User ids are hashed at write
    time and never stored in the clear.

    Entries expire from Redis after OBS_DIGEST_TTL_HOURS (32h by default), so
    only today and yesterday are retrievable.

    Requires a valid X-Admin-Key header.
    """
    return await observability.get_digest(day=date, limit=limit)

@compute_router.get("/status", tags=["Status"], summary="Get model availability and uptime statistics")
async def get_status():
    """
    Public endpoint showing model availability and hourly uptime statistics.
    Returns success rates for each model over the last 24 hours.
    No authentication required.
    """
    known_models = list(api_handlers.MODEL_TO_HANDLER.keys()) if HANDLERS_AVAILABLE else None
    return await observability.get_hourly_status(known_models=known_models)

@compute_router.get("/quota", summary="Check remaining API credits for the authenticated user")
async def check_quota_endpoint(current_user: AuthUser):
    """
    Returns the daily MONITOR credit usage for the authenticated user.
    Requires a valid JWT. Pro and Max users will show their tier limits.
    """
    # Determine user tier and limits
    if current_user.is_max:
        tier = "max"
        limit = MAX_QUOTA_LIMITS["monitor"]
    elif current_user.is_plus:
        tier = "plus"
        limit = PLUS_QUOTA_LIMITS["monitor"]
    elif current_user.is_pro:
        tier = "pro"
        limit = PRO_QUOTA_LIMITS["monitor"]
    else:
        tier = "free"
        limit = QUOTA_LIMITS["monitor"]

    # Use the new specific function for the 'monitor' service
    used = await get_usage_for_service(current_user.id, "monitor")
    remaining = max(0, limit - used)

    # Enterprise seats carry the same entitlement flags as a personal subscription,
    # so tier/limit above are already correct. org_id tells the frontend the seat is
    # org-managed (no Stripe portal for this user — send them to /team instead).
    app_metadata = current_user.app_metadata or {}
    org_id = current_user.org_id

    # The monthly budget. For an enterprise seat this is the whole org's pool and
    # the count is the team's, not this user's - scope says which, and the
    # frontend has to label it accordingly ("your team has used", not "you have
    # used"). A limit of -1 is uncapped and the frontend should hide the bar.
    if org_id:
        monthly_limit = await org_monthly_limit(org_id)
        monthly_scope = "org"
    else:
        monthly_limit = monthly_limit_for(
            "monitor",
            is_pro=current_user.is_pro, is_max=current_user.is_max,
            is_plus=current_user.is_plus,
        )
        monthly_scope = "user"

    monthly_used = await get_monthly_usage(current_user.id, "monitor", org_id)
    uncapped = monthly_limit == NO_MONTHLY_LIMIT

    return JSONResponse(content={
        # used/remaining/limit stay at the top level, and stay daily, so clients
        # built against the old shape keep working untouched.
        "used": used,
        "remaining": remaining,
        "limit": limit,
        "resets_at": daily_resets_at(),
        "daily": {
            "used": used,
            "remaining": remaining,
            "limit": limit,
            "resets_at": daily_resets_at(),
        },
        "monthly": {
            "used": monthly_used,
            "limit": monthly_limit,
            "remaining": None if uncapped else max(0, monthly_limit - monthly_used),
            "unlimited": uncapped,
            "scope": monthly_scope,
            "resets_at": monthly_resets_at(),
        },
        "tier": tier,
        "org_id": org_id,
        "org_tier": app_metadata.get("org_tier"),
        "is_enterprise": bool(org_id),
    })


@compute_router.post("/v1/chat/completions", summary="Process chat completion requests")
async def handle_chat_completions_endpoint(request: Request, current_user: AuthUser):
    """
    Processes a chat completion request. Requires a valid JWT.
    Each call will consume one daily MONITOR credit or AGENT_CREATOR credit depending on model.
    """
    if not HANDLERS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend LLM handlers are not available.")

    # Parse Request Data first to determine model
    try:
        request_data = await request.json()
        model_name = request_data.get("model")
        if not model_name:
            raise HTTPException(status_code=400, detail="Request body must include a 'model' field.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON request body.")

    # --- NEW: Model-based Quota Routing ---
    # Determine which quota to use based on model type
    service_type = "agent_creator" if model_name in AGENT_CREATOR_MODELS else "monitor"

    # Check and consume quota for all users (each tier has limits as anti-abuse)
    allowed, usage_count, reason = await try_consume_for(current_user, service_type)
    user_type = "max" if current_user.is_max else ("plus" if current_user.is_plus else ("pro" if current_user.is_pro else "free"))

    if not allowed:
        limit_value = limit_for(
            service_type,
            is_pro=current_user.is_pro, is_max=current_user.is_max, is_plus=current_user.is_plus,
        )
        logger.warning(f"{service_type.capitalize()} limit exceeded for {user_type} user: {current_user.id} (reason: {reason}, daily limit: {limit_value})")
        # Three refusals with three different remedies: slow down, wait for
        # midnight, or wait for the 1st. An org seat that exhausts the pool
        # cannot fix it alone, so that message has to point at the owner.
        if reason == "rate_limit":
            message = "Rate limit exceeded. Please slow down your requests or try again later."
        elif reason == "monthly_quota" and current_user.org_id:
            message = (
                "Your team's monthly credits are used up. They reset on the 1st "
                "(UTC); your org owner can add more before then."
            )
        elif reason == "monthly_quota":
            message = "You have used all of this month's credits. They reset on the 1st (UTC)."
        else:
            message = "You have used all of today's credits. They reset at midnight UTC."

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "message": message,
                "quota_type": service_type,
                "reason": reason,
            }
        )

    logger.info(f"Processing {service_type} request for {user_type.upper()} user: {current_user.id} (Daily {service_type} request #{usage_count})")
    # --- END of Quota Logic ---

    # 5. Find the appropriate handler
    selected_handler = api_handlers.MODEL_TO_HANDLER.get(model_name)

    if not selected_handler:
        logger.warning(f"Request for unsupported model: {model_name}")
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' is not found or supported.")

    # 6. Check tier-based access control
    model_info = next((m for m in selected_handler.get_models() if m["name"] == model_name), None)
    if model_info:
        # Check if model requires max tier
        if model_info.get("max", False) and not current_user.is_max:
            logger.warning(f"Non-max user {current_user.id} attempted to access max model: {model_name}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Model '{model_name}' requires a Max subscription. Please upgrade to access this model."
            )
        # Check if model requires pro tier (or higher)
        elif model_info.get("pro", False) and not (current_user.is_pro or current_user.is_max):
            logger.warning(f"Free user {current_user.id} attempted to access pro model: {model_name}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Model '{model_name}' requires a Pro subscription. Please upgrade to access premium models."
            )

    # 7. Execute handler logic with centralized observability

    # Static fields for this request, shared by the success and error paths.
    # Agent Creator is conversational, so the digest wants the latest user turn
    # and the untruncated exchange; monitoring agents send standalone prompts.
    messages = request_data.get("messages", [])
    prompt_text, image_count = observability.extract_prompt(messages)
    is_agent_creator = model_name in AGENT_CREATOR_MODELS
    if is_agent_creator:
        prompt_text = observability.extract_latest_user_message(messages)

    obs_ctx = {
        "user_id": current_user.id,
        "model": model_name,
        "handler": selected_handler.name,
        "prompt_text": prompt_text,
        "image_count": image_count,
        "tier": user_type,
        "service": service_type,
        "truncate": not is_agent_creator,
    }

    try:
        response_payload = await selected_handler.handle_request(request_data)

        # Wrap StreamingResponse with logging (all requests are streaming)
        if hasattr(response_payload, '__class__') and response_payload.__class__.__name__ == 'StreamingResponse':
            return StreamingResponse(
                _log_streaming_response(response_payload.body_iterator, obs_ctx),
                media_type=response_payload.media_type,
                headers=response_payload.headers
            )

        # Fallback for non-streaming responses (shouldn't happen but defensive)
        return JSONResponse(content=response_payload)

    except (HandlerError, ConfigError, BackendAPIError) as e:
        status_code = getattr(e, 'status_code', 500)

        await observability.record_request(
            **obs_ctx,
            response_text=f"ERROR: {str(e)}",
            status_code=status_code,
        )

        logger.error(f"Handler error for model '{model_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status_code, detail=str(e))

    except Exception as e:
        await observability.record_request(
            **obs_ctx,
            response_text=f"INTERNAL_ERROR: {str(e)}",
            status_code=500,
        )

        logger.exception(f"Unexpected error processing request with handler {selected_handler.name}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


@compute_router.get("/v1/models", summary="List available models (OpenAI v1 compatible)")
async def list_models_v1_endpoint():
    """
    Provides an OpenAI-compatible /v1/models endpoint.

    This endpoint returns a list of available models in a standardized format,
    while also including custom 'parameter_size' and 'multimodal' fields
    that the Observer AI frontend uses for a richer UI.
    """
    if not HANDLERS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend handlers are not available.")

    # Exclude agent creator models and other hidden models from public listing
    EXCLUDED = AGENT_CREATOR_MODELS | {"gemini-2.5-pro"}
    
    # This list will hold the model data in the new format.
    model_data_list = []

    if api_handlers and api_handlers.API_HANDLERS:
        for handler in api_handlers.API_HANDLERS.values():
            try:
                for model_info in handler.get_models():
                    name = model_info.get("name", "")
                    if name in EXCLUDED:
                        continue
                    
                    # Create the new model entry in the OpenAI-compatible format
                    new_model_entry = {
                        "id": name, # The standard uses 'id' for the model name
                        "object": "model",
                        "created": 0, # Placeholder, as it's not strictly needed by the UI
                        "owned_by": handler.name,

                        # --- Custom fields needed by the Observer frontend ---
                        "parameter_size": model_info.get("parameters", "N/A"),
                        "multimodal": model_info.get("multimodal", False),
                        "pro": model_info.get("pro", False)
                    }
                    model_data_list.append(new_model_entry)

            except Exception as e:
                logger.error(f"Failed to get v1/models from handler {handler.name}: {e}")
    else:
        logger.warning("/v1/models called but no handlers are loaded.")

    # The final response must be a dictionary with 'object' and 'data' keys
    return JSONResponse(content={
        "object": "list",
        "data": model_data_list
    })


