# tools_router.py

import os
import asyncio
import hmac
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
import httpx

# Third-party imports
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment
import base64

# --- Local Imports ---
from auth import AuthUser
from admin_auth import get_admin_access
# Import the new, unified quota manager functions and constants
from quota_manager import try_consume_for, get_usage
from messaging import save_temp_image
import remote

# --- Setup (logging is configured once in api.py via logging_config.setup_logging()) ---
logger = logging.getLogger('tools_router')
tools_router = APIRouter()

SENDGRID_TIMEOUT = 15.0


# --- Pydantic Models ---

class EmailRequest(BaseModel):
    to_email: str = Field(..., description="The destination email address.", examples=["user@example.com"])
    message: str | None = Field(None, description="The email body content (optional if images provided).")
    images: list[str] | None = Field(None, description="Optional base64-encoded images (without data:image prefix)")
    videos: list[str] | None = Field(None, description="Optional base64-encoded videos (without data:video prefix)")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        super().__pydantic_init_subclass__(**kwargs)


class PushoverRequest(BaseModel):
    user_key: str = Field(..., description="The user's personal Pushover key.")
    message: str | None = Field(None, description="The notification content (optional if images provided).")
    title: str | None = Field("Alert from Observer AI", description="Optional title for the notification.")
    images: list[str] | None = Field(None, description="Optional base64-encoded images (without data:image prefix)")

class TelegramRequest(BaseModel):
    chat_id: str = Field(..., description="The Telegram chat ID (can be user ID or group ID starting with -), or a whitelist code linked to one.")
    message: str | None = Field(None, max_length=4096, description="The message content (optional if images provided).")
    images: list[str] | None = Field(None, description="Optional base64-encoded images (without data:image prefix)")
    videos: list[str] | None = Field(None, description="Optional base64-encoded videos (without data:video prefix)")


# --- API Endpoints ---

@tools_router.get("/admin/usage", tags=["Admin"], summary="Daily and monthly usage counters")
async def admin_usage(
    user_id: str | None = None,
    emails: bool = False,
    is_admin: bool = Depends(get_admin_access),
):
    """
    (Admin) Today's and this month's counters, straight from Redis. Pass
    user_id for one user (direct reads, no scan); omit it for everyone active
    today. emails=true labels users by email at the cost of one Auth0 lookup
    each. Requires a valid X-Admin-Key header.
    """
    return await get_usage(user_id=user_id, emails=emails)


@tools_router.post("/tools/send-email", tags=["Tools"])
async def send_email(
    request_data: EmailRequest,
    current_user: AuthUser
):
    """Sends an email, checking against the in-memory email quota."""
    # 1. Self-email check - recipient must be the authenticated user's own email
    if not current_user.email or request_data.to_email.lower() != current_user.email.lower():
        raise HTTPException(
            status_code=403,
            detail="Email recipient must be the same as your account email."
        )

    # 2. Quota Check (using the "email" service)
    allowed, _usage_count, _reason = await try_consume_for(current_user, "email")
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Daily email quota has been exceeded.",
                "quota_type": "email"
            }
        )
    logger.info(f"Processing email for user_id: {current_user.id} to {request_data.to_email}")

    # 4. Action: Send the Email
    sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
    from_email = os.getenv("SENDGRID_FROM_EMAIL")

    if not all([sendgrid_api_key, from_email]):
        logger.error("Server is missing SENDGRID_API_KEY or SENDGRID_FROM_EMAIL.")
        raise HTTPException(status_code=500, detail="Email service is not configured on the server.")

    # Determine if we have media for smart default
    has_media = bool(request_data.images) or bool(request_data.videos)

    message = Mail(
        from_email=from_email,
        to_emails=request_data.to_email,
        subject='An Alert from your Observer AI Agent',
        plain_text_content=request_data.message or ("Media from Observer AI" if has_media else "Alert from Observer AI")
    )

    # Add image attachments if provided
    if request_data.images:
        for i, image_b64 in enumerate(request_data.images):
            try:
                # Decode base64 image
                image_data = base64.b64decode(image_b64)

                # Create attachment
                attachment = Attachment(
                    file_content=base64.b64encode(image_data).decode(),
                    file_name=f"image_{i+1}.png",
                    file_type="image/png",
                    disposition="attachment"
                )
                message.add_attachment(attachment)
            except Exception as e:
                logger.warning(f"Failed to process image {i+1} for user {current_user.id}: {str(e)}")

    # Add video attachments if provided
    if request_data.videos:
        for i, video_b64 in enumerate(request_data.videos):
            try:
                # Decode base64 video
                video_data = base64.b64decode(video_b64)

                # Create attachment
                attachment = Attachment(
                    file_content=base64.b64encode(video_data).decode(),
                    file_name=f"video_{i+1}.mp4",
                    file_type="video/mp4",
                    disposition="attachment"
                )
                message.add_attachment(attachment)
            except Exception as e:
                logger.warning(f"Failed to process video {i+1} for user {current_user.id}: {str(e)}")

    try:
        sendgrid_client = SendGridAPIClient(sendgrid_api_key)
        # The SDK is synchronous and has no timeout by default; in a worker
        # thread a hung send would hold that thread forever.
        sendgrid_client.client.timeout = SENDGRID_TIMEOUT
        response = await asyncio.to_thread(sendgrid_client.send, message)
        logger.info(f"Email successfully sent to SendGrid for user {current_user.id}. Status: {response.status_code}")
        return {"success": True, "detail": "Email sent successfully."}
    except Exception as e:
        logger.exception(f"An unexpected error occurred while sending email via SendGrid for user {current_user.id}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


@tools_router.post("/tools/send-pushover", tags=["Tools"])
async def send_pushover(
    request_data: PushoverRequest,
    current_user: AuthUser
):
    """Sends a notification via Pushover, checking against a quota."""
    # 1. Get the secret application token from the server environment
    pushover_app_token = os.getenv("PUSHOVER_API_KEY")
    if not pushover_app_token:
        logger.error("Server is missing PUSHOVER_API_KEY environment variable.")
        raise HTTPException(status_code=500, detail="Notification service (Pushover) is not configured on the server.")

    allowed, _usage_count, _reason = await try_consume_for(current_user, "pushover")
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Daily Pushover notification quota has been exceeded.",
                "quota_type": "pushover"
            }
        )
    logger.info(f"Processing Pushover for user_id: {current_user.id}")

    # 4. Action: Send the notification to the Pushover API
    pushover_api_url = "https://api.pushover.net/1/messages.json"
    
    # Pushover expects standard form data, not a JSON body.
    # We will build the payload as a dictionary.
    # Determine if we have images for smart default
    has_images = bool(request_data.images)

    data = {
        "token": pushover_app_token,      # Your application token
        "user": request_data.user_key,    # The user's key from the request
        "title": request_data.title,      # The title (defaults to "Alert from Observer AI")
        "message": request_data.message or ("Image from Observer AI" if has_images else "Alert from Observer AI")
    }

    try:
        async with httpx.AsyncClient() as client:
            # Send images as file attachments if provided
            if request_data.images:
                files = {}
                for i, image_b64 in enumerate(request_data.images):
                    try:
                        # Decode base64 image
                        image_data = base64.b64decode(image_b64)
                        # Pushover supports "attachment" parameter for images
                        files["attachment"] = (f"image_{i+1}.png", image_data, "image/png")
                        # Pushover only supports one image, so we'll use the first one
                        break
                    except Exception as e:
                        logger.warning(f"Failed to process image {i+1} for user {current_user.id}: {str(e)}")
                
                response = await client.post(pushover_api_url, data=data, files=files)
            else:
                response = await client.post(pushover_api_url, data=data)
                
            response.raise_for_status() # Raises an exception for 4xx or 5xx status codes

        response_data = response.json()
        if response_data.get("status") != 1:
            # Pushover's API returns 200 OK but includes errors in the JSON body
            errors = ", ".join(response_data.get("errors", ["Unknown error"]))
            logger.error(f"Pushover API returned an error for user {current_user.id}: {errors}")
            raise HTTPException(status_code=400, detail=f"Pushover API error: {errors}")

        logger.info(f"Pushover notification sent successfully for user {current_user.id}. Request ID: {response_data.get('request')}")
        return {"success": True, "detail": "Pushover notification sent successfully."}
    
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling Pushover API for user {current_user.id}: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=502, detail="Failed to communicate with the Pushover service.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred while sending Pushover notification for user {current_user.id}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred.")



@tools_router.post("/tools/send-telegram", tags=["Tools"])
async def send_telegram(
    request_data: TelegramRequest,
    current_user: AuthUser
):
    """Sends a message via Telegram bot, checking against quota."""
    # 1. Get bot token from environment
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_bot_token:
        logger.error("Server is missing TELEGRAM_BOT_TOKEN environment variable.")
        raise HTTPException(status_code=500, detail="Telegram service is not configured on the server.")

    # 1.5. A whitelist code stands in for the chat it was linked to, for its owner only.
    chat_id = request_data.chat_id
    code = remote.normalize_code(chat_id)
    if code:
        chat_id = await remote.owned_address(code, "telegram", current_user.id)
        if not chat_id:
            raise HTTPException(
                status_code=403,
                detail="This code isn't linked to a Telegram chat on your account. Connect Telegram in Observer first.",
            )

    # 2. Quota Check
    allowed, _usage_count, _reason = await try_consume_for(current_user, "telegram")
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Daily Telegram notification quota has been exceeded.",
                "quota_type": "telegram"
            }
        )
    logger.info(f"Processing Telegram for user_id: {current_user.id} to chat_id: {chat_id}")

    # 4. Action: Send images first, then videos, then the message
    message_ids = []

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Send images first
            if request_data.images:
                for i, image_b64 in enumerate(request_data.images):
                    try:
                        # Decode base64 image
                        image_data = base64.b64decode(image_b64)

                        # Send photo via Telegram API
                        photo_url = f"https://api.telegram.org/bot{telegram_bot_token}/sendPhoto"
                        files = {"photo": ("image.png", image_data, "image/png")}
                        data = {"chat_id": chat_id}

                        photo_response = await client.post(photo_url, files=files, data=data)
                        photo_response.raise_for_status()

                        photo_data = photo_response.json()
                        if not photo_data.get("ok"):
                            error_description = photo_data.get("description", "Unknown error")
                            logger.error(f"Telegram photo API error for user {current_user.id}: {error_description}")
                            raise HTTPException(status_code=400, detail=f"Telegram photo API error: {error_description}")

                        message_ids.append(photo_data.get('result', {}).get('message_id'))
                        logger.info(f"Telegram image {i+1} sent successfully for user {current_user.id}")

                    except Exception as e:
                        logger.warning(f"Failed to process image {i+1} for user {current_user.id}: {str(e)}")

            # Send videos
            if request_data.videos:
                for i, video_b64 in enumerate(request_data.videos):
                    try:
                        # Decode base64 video
                        video_data = base64.b64decode(video_b64)

                        # Send video via Telegram API
                        video_url = f"https://api.telegram.org/bot{telegram_bot_token}/sendVideo"
                        files = {"video": ("video.mp4", video_data, "video/mp4")}
                        data = {"chat_id": chat_id}

                        video_response = await client.post(video_url, files=files, data=data)
                        video_response.raise_for_status()

                        video_data_response = video_response.json()
                        if not video_data_response.get("ok"):
                            error_description = video_data_response.get("description", "Unknown error")
                            logger.error(f"Telegram video API error for user {current_user.id}: {error_description}")
                            raise HTTPException(status_code=400, detail=f"Telegram video API error: {error_description}")

                        message_ids.append(video_data_response.get('result', {}).get('message_id'))
                        logger.info(f"Telegram video {i+1} sent successfully for user {current_user.id}")

                    except Exception as e:
                        logger.warning(f"Failed to process video {i+1} for user {current_user.id}: {str(e)}")

            # Determine if we sent media for smart default
            sent_media = bool(message_ids)  # message_ids populated from image/video sending above
            message_text = request_data.message or ("Media from Observer AI" if sent_media else "Alert from Observer AI")

            # Always send text message with default logic
            message_url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message_text,
                "parse_mode": "HTML"  # Allows basic HTML formatting
            }

            response = await client.post(message_url, json=payload)
            response.raise_for_status()

            response_data = response.json()
            if not response_data.get("ok"):
                error_description = response_data.get("description", "Unknown error")
                logger.error(f"Telegram API error for user {current_user.id}: {error_description}")
                raise HTTPException(status_code=400, detail=f"Telegram API error: {error_description}")

            message_ids.append(response_data.get('result', {}).get('message_id'))

        logger.info(f"Telegram message sent successfully for user {current_user.id}. Message IDs: {message_ids}")
        return {"success": True, "detail": "Telegram message sent successfully.", "message_ids": message_ids}

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling Telegram API for user {current_user.id}: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=502, detail="Failed to communicate with the Telegram service.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred while sending Telegram message for user {current_user.id}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


async def send_telegram_photo(chat_id: str, image_b64: str, caption: str | None = None) -> None:
    """Plain photo (base64, no data: prefix) send, for bot replies. No quota checks: callers own those."""
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_bot_token:
        raise HTTPException(status_code=500, detail="Telegram service is not configured on the server.")
    image_data = base64.b64decode(image_b64)
    files = {"photo": ("image.png", image_data, "image/png")}
    data = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption[:1024]  # Telegram's caption limit
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"https://api.telegram.org/bot{telegram_bot_token}/sendPhoto", files=files, data=data)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        description = payload.get("description", "Unknown error")
        logger.error(f"Telegram photo to {chat_id} failed: {description}")
        raise HTTPException(status_code=400, detail=f"Telegram photo API error: {description}")


async def send_telegram_text(chat_id: str, text: str, parse_mode: str | None = None) -> None:
    """Plain-text Telegram send, for bot replies. No quota checks: callers own those."""
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_bot_token:
        raise HTTPException(status_code=500, detail="Telegram service is not configured on the server.")
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage", json=payload)
    if not response.json().get("ok"):
        description = response.json().get("description", "Unknown error")
        logger.error(f"Telegram text to {chat_id} failed: {description}")
        raise HTTPException(status_code=400, detail=f"Telegram API error: {description}")


async def send_telegram_typing(chat_id: str) -> None:
    """Show "typing…" in the chat. Telegram clears it after ~5s, or when the bot's next message lands."""
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_bot_token:
        return
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            f"https://api.telegram.org/bot{telegram_bot_token}/sendChatAction",
            json={"chat_id": chat_id, "action": "typing"},
        )
    response.raise_for_status()


@tools_router.post("/tools/telegram-webhook", tags=["Tools"])
async def telegram_webhook(request: Request, update: dict):
    """
    Webhook endpoint for the Telegram bot. `/start <code>` (the deep link in the
    app's QR) links a private chat to a whitelist code for remote control; after
    that, the chat's messages go to the user's Observer session (remote.route_inbound).
    Anything else gets the chat ID reply, for the paste-a-chat-ID flow.

    Must be registered via setWebhook with a secret_token. The secret is required:
    this bot can drive a user's desktop session, so an unauthenticated webhook
    would let anyone inject messages as any chat.
    """
    webhook_secret = os.getenv("TELEGRAM_WEBHOOK_SECRET")
    if not webhook_secret:
        logger.error("TELEGRAM_WEBHOOK_SECRET not configured - rejecting Telegram webhook")
        raise HTTPException(status_code=403, detail="Forbidden")
    incoming = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not hmac.compare_digest(incoming, webhook_secret):
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        message = update.get("message")
        if not message:
            return {"ok": True}
        chat_id = str(message["chat"]["id"])
        chat_type = message["chat"]["type"]
        text = message.get("text") or ""

        update_id = update.get("update_id")
        if not await remote.first_delivery(f"tg:{update_id}" if update_id is not None else None):
            return {"ok": True}

        async def reply(body: str) -> None:
            try:
                await send_telegram_text(chat_id, body)
            except Exception as e:
                logger.error(f"Failed to reply to chat_id {chat_id}: {str(e)}")

        # Remote control is for private chats only; a bare /start is a chat ID request.
        if chat_type == "private" and text.strip() != "/start":
            body = text[len("/start"):].strip() if text.startswith("/start") else text
            if await remote.route_inbound("telegram", chat_id, body, reply):
                return {"ok": True}

        response_text = (
            f"Your chat ID is: <code>{chat_id}</code>\n\nChat type: {chat_type}\n\n"
            "Use this ID in your Observer AI agents, or connect with the QR code in Observer "
            "to chat with it from here!"
        )
        await send_telegram_text(chat_id, response_text, parse_mode="HTML")
        logger.info(f"Auto-responded to chat_id {chat_id} with their chat ID")
        return {"ok": True}
    except Exception as e:
        logger.exception("Error processing Telegram webhook")
        return {"ok": True}  # Always return OK to Telegram to avoid retries


