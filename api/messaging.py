# messaging.py

import os
import logging
import uuid
import secrets
from urllib.parse import quote
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field
from PIL import Image
import io
import base64
import phonenumbers

# Third-party imports
from better_profanity import profanity
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from twilio.twiml.voice_response import VoiceResponse
from twilio.request_validator import RequestValidator

# Local imports
from auth import AuthUser
import r2_store
from quota_manager import try_consume_for
from redis_client import get_redis

# Setup (logging is configured once in api.py via logging_config.setup_logging())
logger = logging.getLogger('twilio')
messaging_router = APIRouter()

# Temp images directory

WHITELIST_TTL = 86400  # 24 hours in seconds
VOICE_CALL_TTL = 3600  # 1 hour — how long a pending call message stays retrievable

DEFAULT_VOICE_MESSAGE = "Alert from Observer A I"

# Load better-profanity's bundled wordlist once at import. It is a maintained
# list of slurs/profanity and it normalises common character substitution
# (leetspeak, punctuation) before matching.
profanity.load_censor_words()


def assert_content_allowed(*texts: str | None) -> None:
    """
    Synchronous acceptable-use check on outbound message content, run before
    anything is handed to Twilio's client.messages.create / calls.create.

    Screens the caller-supplied text against a maintained slur/profanity
    wordlist. This is content-neutral enforcement that protects the Twilio
    account from Messaging Policy violations — the text is never stored or
    logged, it is only inspected in-memory on the send path.

    Raises HTTPException(400) if the content is rejected.
    """
    for text in texts:
        if text and profanity.contains_profanity(text):
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Message blocked: the content violates the acceptable use policy. Hate speech, slurs, and abusive language are not permitted.",
                    "error_type": "content_policy",
                },
            )


async def stash_voice_message(message: str) -> str:
    """
    Store the text a voice call should speak and return an opaque token for it.

    Lives in Redis rather than process memory so any uvicorn worker (or any
    server) can serve the callback. The token is minted *before* the Twilio call
    is created and travels in the callback URL, so the message is always in place
    by the time Twilio requests TwiML — and because Twilio signs the full URL
    including the query string, the token can't be tampered with.
    """
    r = await get_redis()
    token = secrets.token_urlsafe(16)
    await r.setex(f"voicecall:{token}", VOICE_CALL_TTL, message)
    return token


async def pop_voice_message(token: str) -> str | None:
    """Fetch and consume a stashed voice message. Returns None if unknown/expired."""
    if not token:
        return None
    r = await get_redis()
    message = await r.get(f"voicecall:{token}")
    if message is not None:
        await r.delete(f"voicecall:{token}")
    return message

def normalize_phone(phone: str) -> str:
    """
    Normalize phone number to E.164 format using phonenumbers library.
    Handles all country codes and formatting variations automatically.

    Examples:
    - "+52 811 500 0488" → "+528115000488"
    - "+1 (555) 123-4567" → "+15551234567"
    - Handles international formats properly

    Falls back to simple cleaning if parsing fails.
    """
    try:
        parsed = phonenumbers.parse(phone, None)
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except Exception as e:
        logger.warning(f"Failed to parse phone number '{phone}': {e}. Using fallback cleaning.")
        # Fallback: just remove non-digits except leading +
        return ''.join(c for c in phone if c.isdigit() or c == '+')

def is_subsequence(shorter: str, longer: str) -> bool:
    """Check if shorter is a subsequence of longer (all chars appear in same order)."""
    it = iter(longer)
    return all(c in it for c in shorter)

def phone_numbers_fuzzy_match(num1: str, num2: str) -> bool:
    """
    Fuzzy match two phone numbers using subsequence matching.

    Rules:
    1. Both must have at least 8 digits (prevents "911" matching long numbers)
    2. Length difference must be <= 1 digit (handles Mexico +521 vs +52, rejects wildly different numbers)
    3. Shorter number must be a subsequence of longer

    Examples:
    - "+5218115000488" vs "+528115000488" → Match! (13 vs 12 digits, diff=1, subsequence)
    - "+52 811 500 0488" vs "+528115000488" → Match! (same number, formatting removed)
    - "911" vs "+528115000488" → No match (911 has < 8 digits)
    - "+528115000488" vs "+529999999999" → No match (different subsequences)
    """
    # Extract digits only
    digits1 = ''.join(c for c in num1 if c.isdigit())
    digits2 = ''.join(c for c in num2 if c.isdigit())

    # Both must have at least 8 digits (security: prevents short number attacks)
    if len(digits1) < 8 or len(digits2) < 8:
        return False

    # Length difference must be <= 1 (handles single-digit variations like Mexico's +521)
    if abs(len(digits1) - len(digits2)) > 1:
        return False

    # Check if shorter is subsequence of longer
    if len(digits1) <= len(digits2):
        return is_subsequence(digits1, digits2)
    else:
        return is_subsequence(digits2, digits1)

async def resolve_to_phone(key_or_phone: str) -> str:
    """
    Resolve a key or phone number to an actual whitelisted phone number.

    First checks if the input is a key in the Redis key_to_phone mapping.
    If not, fuzzy matches against all whitelisted numbers (both phone and WhatsApp).
    If no match found, returns normalized E.164 format.
    """
    r = await get_redis()

    # Check if it's a key first
    mapped = await r.get(f"whitelist:key:{key_or_phone}")
    if mapped:
        return mapped

    # Fuzzy match against all whitelisted numbers
    phone_nums = [k.split(":", 2)[2] async for k in r.scan_iter("whitelist:phone:*")]
    wa_nums = [k.split(":", 2)[2] async for k in r.scan_iter("whitelist:whatsapp:*")]
    for whitelisted_num in phone_nums + wa_nums:
        if phone_numbers_fuzzy_match(key_or_phone, whitelisted_num):
            return whitelisted_num

    return normalize_phone(key_or_phone)

async def is_whitelisted(phone_number: str, channel: str = None) -> bool:
    """
    Check if phone number (or key) is whitelisted for messaging.
    Uses Redis so all workers share the same state.
    TTL-based expiry is handled by Redis automatically.
    """
    r = await get_redis()

    # Check if it's a key first
    mapped = await r.get(f"whitelist:key:{phone_number}")
    actual_phone = mapped if mapped else phone_number

    if channel == "whatsapp":
        nums = [k.split(":", 2)[2] async for k in r.scan_iter("whitelist:whatsapp:*")]
    else:
        phone_nums = [k.split(":", 2)[2] async for k in r.scan_iter("whitelist:phone:*")]
        wa_nums = [k.split(":", 2)[2] async for k in r.scan_iter("whitelist:whatsapp:*")]
        nums = phone_nums + wa_nums

    return any(phone_numbers_fuzzy_match(actual_phone, num) for num in nums)

async def add_to_whitelist(phone_number: str, key: str = None, channel: str = "phone") -> None:
    """
    Add phone number to appropriate whitelist with 24h expiry in Redis.
    Optionally map a user-chosen key (like "pizza" or "Hey, I'm Roy") to the phone number.
    """
    r = await get_redis()
    normalized_phone = normalize_phone(phone_number)

    if channel == "whatsapp":
        await r.setex(f"whitelist:whatsapp:{normalized_phone}", WHITELIST_TTL, "1")
        logger.info(f"Added {normalized_phone} (original: {phone_number}) to WhatsApp whitelist")
    else:
        await r.setex(f"whitelist:phone:{normalized_phone}", WHITELIST_TTL, "1")
        logger.info(f"Added {normalized_phone} (original: {phone_number}) to phone whitelist (SMS/Voice)")

    if key and key.strip():
        await r.setex(f"whitelist:key:{key}", WHITELIST_TTL, normalized_phone)
        logger.info(f"Mapped key '{key}' to phone number {normalized_phone}")

async def validate_twilio_request(request: Request) -> dict:
    """
    Validate that the webhook request actually came from Twilio and return form data.
    Prevents attackers from spoofing webhooks to whitelist arbitrary numbers.
    Returns the parsed form data as a dict.
    """
    # Get Twilio auth token
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    if not auth_token:
        logger.error("TWILIO_AUTH_TOKEN not configured - cannot validate webhooks!")
        raise HTTPException(status_code=500, detail="Server configuration error")

    # Create validator
    validator = RequestValidator(auth_token)

    # Get the signature from headers
    signature = request.headers.get('X-Twilio-Signature', '')
    if not signature:
        logger.warning(f"Webhook received without X-Twilio-Signature header from {request.client.host}")
        raise HTTPException(status_code=403, detail="Missing Twilio signature")

    # Get form data (FastAPI's FormData object)
    form_data = await request.form()

    # Reconstruct the public URL as Twilio signed it.
    # Behind a reverse proxy, request.url uses the internal scheme (http),
    # but Twilio signs the external HTTPS URL. Use X-Forwarded-Proto if present.
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
    url = str(request.url)
    if forwarded_proto == "https" and url.startswith("http://"):
        url = "https://" + url[len("http://"):]

    # Debug logging
    logger.info(f"Validating Twilio request: URL={url}, Signature={signature[:20]}..., Form keys={list(form_data.keys())}")

    # Validate the signature using Twilio's official method
    # Pass FormData object directly (not a parsed dict!)
    is_valid = validator.validate(url, form_data, signature)

    if not is_valid:
        logger.error(f"Invalid Twilio signature from {request.client.host} for URL {url}")
        logger.error(f"Rejecting potentially malicious webhook request")
        raise HTTPException(status_code=403, detail="Invalid Twilio signature - request rejected")

    logger.info(f"Twilio signature validated successfully for {url}")

    # Return FormData as a simple dict - much cleaner!
    return dict(form_data)

async def save_temp_image(image_b64: str) -> str:
    """Save base64 image to temp storage with compression and return public URL"""
    try:
        # Generate secure UUID filename
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.jpg"  # Use JPG for better compression

        # Decode base64 image
        image_data = base64.b64decode(image_b64)
        logger.info(f"Image input: base64 length={len(image_b64)}, decoded bytes={len(image_data)}")

        # Open image with PIL for compression
        with Image.open(io.BytesIO(image_data)) as img:
            logger.info(f"Image opened: size={img.size}, mode={img.mode}")
            # Convert to RGB if necessary (for JPG compatibility)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparent images
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if 'A' in img.mode else None)
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Compress image to ensure it's under 4MB (leaving 1MB buffer)
            quality = 95
            max_size = 4 * 1024 * 1024  # 4MB

            while quality > 10:
                # Save to memory buffer to check size
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=True)

                if buffer.tell() <= max_size:
                    # Size is acceptable
                    break

                # Reduce quality and try again
                quality -= 10
            else:
                # If still too large, resize the image
                img.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=80, optimize=True)

        data = buffer.getvalue()
        if not data:
            raise Exception("Image was not encoded properly")

        # To R2, not to local disk: the URL below points at a hostname that
        # load balances across boxes, so the provider's fetch will not
        # necessarily come back to the box that handled the send.
        await r2_store.put_bytes(
            r2_store.temp_media_key(filename), data, content_type="image/jpeg"
        )
        logger.info(f"Image compressed and uploaded: {len(data) / (1024 * 1024):.2f}MB")

        # Return public URL
        return f"https://api.observer-ai.com/temp-images/{filename}"

    except Exception as e:
        logger.error(f"Failed to save temp image: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save image")

async def save_temp_video(video_b64: str, max_size_mb: float = 50.0, transcode: bool = False) -> str:
    """Save base64 video to temp storage and return public URL.

    Args:
        video_b64: Base64-encoded video data
        max_size_mb: Maximum allowed file size in MB (default 50MB)
        transcode: If True, transcode to H.264/AAC for WhatsApp compatibility

    Returns:
        Public URL to the saved video
    """
    import subprocess
    import tempfile

    # ffmpeg works on real files, so transcoding needs a scratch directory.
    # It is per-request and torn down here, which is the difference that
    # matters: nothing outside this call ever has to find these files again.
    with tempfile.TemporaryDirectory(prefix="observer-video-") as scratch:
        return await _save_temp_video(video_b64, max_size_mb, transcode, Path(scratch))


async def _save_temp_video(
    video_b64: str, max_size_mb: float, transcode: bool, scratch: Path
) -> str:
    import subprocess

    try:
        # Generate secure UUID filename
        video_id = str(uuid.uuid4())
        input_filename = f"{video_id}_input.mp4"
        output_filename = f"{video_id}.mp4"
        input_filepath = scratch / input_filename
        output_filepath = scratch / output_filename

        # Decode base64 video
        video_data = base64.b64decode(video_b64)

        # Check file size
        file_size_mb = len(video_data) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise HTTPException(
                status_code=400,
                detail=f"Video too large ({file_size_mb:.1f}MB). Maximum allowed: {max_size_mb}MB"
            )

        # Save input video file
        with open(input_filepath, "wb") as f:
            f.write(video_data)
            f.flush()
            os.fsync(f.fileno())

        if transcode:
            # Transcode to H.264/AAC for WhatsApp compatibility
            # -c:v libx264: H.264 video codec (required by WhatsApp)
            # -c:a aac: AAC audio codec (required by WhatsApp)
            # -f lavfi -i anullsrc: Add silent audio if none exists (WhatsApp rejects silent videos)
            # -shortest: End when shortest input ends
            # -movflags +faststart: Enable streaming playback
            try:
                # First, check if video has audio
                probe_cmd = [
                    "ffprobe", "-v", "error", "-select_streams", "a",
                    "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                    str(input_filepath)
                ]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
                has_audio = bool(probe_result.stdout.strip())

                if has_audio:
                    # Video has audio, just transcode
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-i", str(input_filepath),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-c:a", "aac", "-b:a", "128k",
                        "-movflags", "+faststart",
                        str(output_filepath)
                    ]
                else:
                    # Video has no audio, add silent audio track (WhatsApp requirement)
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-i", str(input_filepath),
                        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-c:a", "aac", "-b:a", "128k",
                        "-shortest",
                        "-movflags", "+faststart",
                        str(output_filepath)
                    ]
                    logger.info("Adding silent audio track for WhatsApp compatibility")

                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=120)

                if result.returncode != 0:
                    logger.error(f"FFmpeg error: {result.stderr}")
                    # Fall back to original file if transcoding fails
                    os.rename(input_filepath, output_filepath)
                    logger.warning("FFmpeg transcoding failed, using original video")
                else:
                    # Remove input file after successful transcoding
                    input_filepath.unlink(missing_ok=True)
                    logger.info("Video transcoded to H.264/AAC for WhatsApp")

            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg timeout, using original video")
                os.rename(input_filepath, output_filepath)
            except FileNotFoundError:
                logger.warning("FFmpeg not installed, using original video")
                os.rename(input_filepath, output_filepath)
        else:
            # No transcoding, just rename input to output
            os.rename(input_filepath, output_filepath)

        # Verify file exists and is readable
        if not output_filepath.exists() or output_filepath.stat().st_size == 0:
            raise Exception("Video file was not saved properly")

        data = output_filepath.read_bytes()
        await r2_store.put_bytes(
            r2_store.temp_media_key(output_filename), data, content_type="video/mp4"
        )
        logger.info(f"Video uploaded: {len(data) / (1024 * 1024):.2f}MB")

        # Return public URL
        return f"https://api.observer-ai.com/temp-images/{output_filename}"

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save temp video: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save video")

# Pydantic Models
class SmsRequest(BaseModel):
    to_number: str = Field(..., description="The destination phone number in E.164 format.", examples=["+15551234567"])
    message: str | None = Field(None, max_length=1600, description="The text message content (optional if images provided).")
    images: list[str] | None = Field(None, description="Optional base64-encoded images (without data:image prefix)")
    videos: list[str] | None = Field(None, description="Optional base64-encoded videos (without data:video prefix).")

class WhatsAppRequest(BaseModel):
    to_number: str = Field(..., description="The destination phone number in E.164 format.", examples=["+15551234567"])
    message: str | None = Field(None, description="The message content (optional if images provided).")
    images: list[str] | None = Field(None, description="Optional base64-encoded images (without data:image prefix)")
    videos: list[str] | None = Field(None, description="Optional base64-encoded videos (without data:video prefix).")

class VoiceCallRequest(BaseModel):
    to_number: str = Field(..., description="The destination phone number in E.164 format.", examples=["+15551234567"])
    message: str | None = Field(None, description="The message to speak during the call (optional).", max_length=4096)

class IsWhitelistedRequest(BaseModel):
    phone_number: str = Field(..., description="The phone number to check in E.164 format.", examples=["+15551234567"])
    channel: str | None = Field(None, description="Optional channel to check ('whatsapp', 'sms', 'voice'). If 'whatsapp', checks only WhatsApp whitelist. Otherwise checks both.")

class TwilioConfig(BaseModel):
    account_sid: str
    auth_token: str
    from_number: str
    whatsapp_from_number: str

# Twilio Dependency
def get_twilio_config():
    """Dependency to load and validate Twilio credentials."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_PHONE_NUMBER")
    whatsapp_from_number = os.getenv("TWILIO_WHATSAPP_NUMBER")

    if not all([account_sid, auth_token, from_number, whatsapp_from_number]):
        logger.error("Server is missing required TWILIO environment variables (including WHATSAPP_NUMBER).")
        raise HTTPException(status_code=500, detail="Messaging service is not configured on the server.")
    return TwilioConfig(
        account_sid=account_sid,
        auth_token=auth_token,
        from_number=from_number,
        whatsapp_from_number=whatsapp_from_number
    )

# API Endpoints

@messaging_router.post("/tools/send-sms", tags=["Tools"])
async def send_sms(
    request_data: SmsRequest,
    current_user: AuthUser,
    config: TwilioConfig = Depends(get_twilio_config)
):
    """Sends an SMS to whitelisted numbers only."""
    # 1. Whitelist Check - unified anti-spam protection
    if not await is_whitelisted(request_data.to_number):
        raise HTTPException(
            status_code=403,
            detail=f"Number {request_data.to_number} not whitelisted! Send an SMS or WhatsApp message to whatsapp:+1 (555) 783-4727 or call us first to receive messages"
        )

    # 1.25. Acceptable-use content screen — reject slurs/hate terms before we
    # consume quota, upload media, or hand anything to Twilio.
    assert_content_allowed(request_data.message)

    # 1.5. Resolve key or phone number to normalized E.164 format
    resolved_phone = await resolve_to_phone(request_data.to_number)

    # 2. Quota Check (using the "sms" service)
    allowed, _usage_count, _reason = await try_consume_for(current_user, "sms")
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Daily SMS quota has been exceeded.",
                "quota_type": "sms"
            }
        )
    logger.info(f"Processing SMS for user_id: {current_user.id} to {resolved_phone} (original input: {request_data.to_number})")

    # 4. Action: Send the SMS/MMS
    try:
        client = Client(config.account_sid, config.auth_token)

        # Prepare message parameters
        message_params = {
            "to": resolved_phone,  # Use resolved phone number (handles keys and normalization)
            "from_": config.from_number,
        }

        # Process images first to determine default message
        media_urls = []
        if request_data.images:
            for i, image_b64 in enumerate(request_data.images):
                try:
                    url = await save_temp_image(image_b64)
                    media_urls.append(url)
                    logger.info(f"Image {i+1} saved and hosted for SMS/MMS")
                except Exception as e:
                    logger.warning(f"Failed to process image {i+1} for SMS/MMS: {str(e)}")

        # Process videos - send as text links (MMS video is unreliable across carriers)
        video_links = []
        if request_data.videos:
            for i, video_b64 in enumerate(request_data.videos):
                try:
                    url = await save_temp_video(video_b64, max_size_mb=50.0)
                    video_links.append(url)
                    logger.info(f"Video {i+1} saved and hosted for SMS (will send as link)")
                except Exception as e:
                    logger.warning(f"Failed to process video {i+1} for SMS: {str(e)}")

        # Build message body
        has_media = bool(media_urls) or bool(video_links)
        base_message = request_data.message or ("Media from Observer AI" if has_media else "Alert from Observer AI")

        # Append video links to message text (more reliable than MMS video)
        if video_links:
            video_text = "\n\n📹 Video" + ("s" if len(video_links) > 1 else "") + ":\n" + "\n".join(video_links)
            message_params["body"] = base_message + video_text
        else:
            message_params["body"] = base_message

        # Add media URLs if we have any
        if media_urls:
            message_params["media_url"] = media_urls

        message = client.messages.create(**message_params)
        logger.info(f"SMS/MMS sent successfully. SID: {message.sid}")
        return {"success": True, "message_sid": message.sid}
    except TwilioRestException as e:
        logger.error(f"Twilio API error: {e.msg}")
        raise HTTPException(status_code=400, detail=f"Failed to send SMS/MMS: {e.msg}")

@messaging_router.post("/webhooks/sms-incoming", tags=["Webhooks"], status_code=204)
async def sms_incoming_webhook(
    request: Request,
    form_data: dict = Depends(validate_twilio_request)
):
    """
    Webhook endpoint for incoming SMS messages.
    Automatically whitelists the sender's number for 24 hours across all phone channels.
    Validates Twilio signature to prevent spoofing attacks.
    """
    try:
        # Extract sender's phone number (form_data already validated and parsed)
        from_number = form_data.get("From", "")
        message_body = form_data.get("Body", "")

        if not from_number:
            logger.warning("Received SMS webhook without From number")
            return FastAPIResponse(status_code=204)

        # Add to phone whitelist (for SMS and voice)
        # Store message body as key for friendly lookup (e.g., "pizza", "Roy", etc.)
        if (len(message_body)>7 and message_body!="Hi! I'd like to whitelist my phone number for Observer"):
            await add_to_whitelist(from_number, key=message_body, channel="phone")
        else:
            await add_to_whitelist(from_number, channel="phone")
            message_body=None

        # Get Twilio config to respond
        try:
            config = get_twilio_config()
            client = Client(config.account_sid, config.auth_token)

            # Send unified confirmation response via SMS
            key_info = f" or use the key '{message_body}'" if message_body else ""
            response_message = f"🤖 This is the Observer Bot!\n\nYour number {from_number} is now whitelisted for 24 hours across SMS, WhatsApp, and voice calls!\n\nUse this number{key_info} in your Observer AI agents."

            client.messages.create(
                to=from_number,
                from_=config.from_number,
                body=response_message
            )

            logger.info(f"Auto-whitelisted and responded to SMS from {from_number}")

        except Exception as e:
            logger.error(f"Failed to send SMS confirmation to {from_number}: {str(e)}")
            # Still return success since whitelist was added

        return FastAPIResponse(status_code=204)

    except Exception as e:
        logger.error(f"Error processing SMS incoming webhook: {str(e)}")
        return FastAPIResponse(status_code=204)

@messaging_router.post("/tools/send-whatsapp", tags=["Tools"])
async def send_whatsapp(
    request_data: WhatsAppRequest,
    current_user: AuthUser,
    config: TwilioConfig = Depends(get_twilio_config)
):
    """Sends a WhatsApp message to whitelisted numbers only."""
    # 1. Whitelist Check - WhatsApp-specific (24h messaging window)
    if not await is_whitelisted(request_data.to_number, channel="whatsapp"):
        raise HTTPException(
            status_code=403,
            detail=f"Number {request_data.to_number} not whitelisted for WhatsApp! Send a WhatsApp message to +1 (555) 783-4727 first to receive WhatsApp messages"
        )

    # 1.25. Acceptable-use content screen — reject slurs/hate terms before we
    # consume quota, upload media, or hand anything to Twilio.
    assert_content_allowed(request_data.message)

    # 1.5. Resolve key or phone number to normalized E.164 format
    resolved_phone = await resolve_to_phone(request_data.to_number)

    # 2. Quota Check (using the "whatsapp" service)
    allowed, _usage_count, _reason = await try_consume_for(current_user, "whatsapp")
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Daily WhatsApp quota has been exceeded.",
                "quota_type": "whatsapp"
            }
        )
    logger.info(f"Processing WhatsApp for user_id: {current_user.id} to {resolved_phone} (original input: {request_data.to_number})")

    # 4. Action: Send the WhatsApp message
    client = Client(config.account_sid, config.auth_token)

    try:
        # Prepare media URLs if images provided
        media_urls = []
        if request_data.images:
            for i, image_b64 in enumerate(request_data.images):
                try:
                    url = await save_temp_image(image_b64)
                    media_urls.append(url)
                    logger.info(f"Image {i+1} saved and hosted for WhatsApp media message")
                except Exception as e:
                    logger.warning(f"Failed to process image {i+1} for WhatsApp: {str(e)}")

        # Process videos (transcode to H.264/AAC for WhatsApp compatibility)
        if request_data.videos:
            for i, video_b64 in enumerate(request_data.videos):
                try:
                    url = await save_temp_video(video_b64, max_size_mb=50.0, transcode=True)
                    media_urls.append(url)
                    logger.info(f"Video {i+1} saved and hosted for WhatsApp")
                except Exception as e:
                    logger.warning(f"Failed to process video {i+1} for WhatsApp: {str(e)}")

        # Send WhatsApp message
        message_params = {
            "to": f'whatsapp:{resolved_phone}',  # Use resolved phone number (handles keys and normalization)
            "from_": f'whatsapp:{config.whatsapp_from_number}',
            "body": request_data.message or ("Media from Observer AI" if media_urls else "Alert from Observer AI"),
            "status_callback": "https://api.observer-ai.com/webhooks/whatsapp-status"
        }

        # Add media URLs if we have any
        if media_urls:
            message_params["media_url"] = media_urls
            logger.info(f"Sending WhatsApp message with {len(media_urls)} media file(s) for user {current_user.id}")
        else:
            logger.info(f"Sending WhatsApp text message for user {current_user.id}")

        message = client.messages.create(**message_params)
        logger.info(f"WhatsApp message sent successfully to whitelisted number. SID: {message.sid}")
        return {"success": True, "message_sid": message.sid}

    except TwilioRestException as e:
        logger.error(f"WhatsApp message failed: {e.msg}")
        raise HTTPException(status_code=400, detail=f"Failed to send WhatsApp message: {e.msg}")
    except Exception as e:
        logger.error(f"Error sending WhatsApp message: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send WhatsApp message")

@messaging_router.post("/tools/is-whitelisted", tags=["Tools"])
async def check_is_whitelisted(
    request_data: IsWhitelistedRequest,
    current_user: AuthUser
):
    """Checks if a phone number is currently whitelisted for messaging."""
    whitelisted = await is_whitelisted(request_data.phone_number, channel=request_data.channel)
    logger.info(f"Whitelist check for user_id: {current_user.id}, number: {request_data.phone_number}, channel: {request_data.channel}, result: {whitelisted}")
    return {
        "phone_number": request_data.phone_number,
        "is_whitelisted": whitelisted,
        "channel": request_data.channel
    }

@messaging_router.post("/webhooks/whatsapp-incoming", tags=["Webhooks"], status_code=204)
async def whatsapp_incoming_webhook(
    request: Request,
    form_data: dict = Depends(validate_twilio_request)
):
    """
    Webhook endpoint for incoming WhatsApp messages.
    Automatically whitelists the sender's number for 24 hours.
    Validates Twilio signature to prevent spoofing attacks.
    """
    try:
        # Extract sender's phone number (form_data already validated and parsed)
        # Remove whatsapp: prefix
        from_number = form_data.get("From", "").replace("whatsapp:", "")
        message_body = form_data.get("Body", "")

        if not from_number:
            logger.warning("Received WhatsApp webhook without From number")
            return FastAPIResponse(status_code=204)

        # Add to WhatsApp whitelist - blazing fast in-memory operation
        # Store message body as key for friendly lookup if it's greater than 7 chars
        if (len(message_body)>7 and message_body!="Hi! I'd like to whitelist my phone number for Observer"):
            await add_to_whitelist(from_number, key=message_body, channel="whatsapp")
        else:
            await add_to_whitelist(from_number, channel="whatsapp")
            message_body = None

        # Get Twilio config to respond
        try:
            config = get_twilio_config()
            client = Client(config.account_sid, config.auth_token)

            # Send confirmation response - unified across all phone channels
            key_info = f" or use the key *'{message_body}'*" if message_body else ""
            response_message = f"🤖 This is the Observer Bot!\n\nYour number *{from_number}* is now whitelisted for 24 hours across SMS, WhatsApp, and voice calls!\n\nUse this number{key_info} in your Observer AI agents."

            client.messages.create(
                to=f"whatsapp:{from_number}",
                from_=f"whatsapp:{config.whatsapp_from_number}",
                body=response_message
            )

            logger.info(f"Auto-whitelisted and responded to {from_number}")

        except Exception as e:
            logger.error(f"Failed to send confirmation to {from_number}: {str(e)}")
            # Still return success since whitelist was added

        return FastAPIResponse(status_code=204)

    except Exception as e:
        logger.error(f"Error processing WhatsApp incoming webhook: {str(e)}")
        return FastAPIResponse(status_code=204)

@messaging_router.post("/webhooks/whatsapp-status", tags=["Webhooks"])
async def whatsapp_status_callback(
    request: Request,
    form_data: dict = Depends(validate_twilio_request)
):
    """
    Webhook endpoint to receive WhatsApp message delivery status from Twilio.
    Validates Twilio signature to prevent spoofing attacks.
    """
    try:
        # Extract values (form_data already validated and parsed)
        message_sid = form_data.get("MessageSid", "")
        message_status = form_data.get("MessageStatus", "")
        error_code = form_data.get("ErrorCode", "")
        to_number = form_data.get("To", "").replace("whatsapp:", "")

        logger.info(f"WhatsApp status callback: SID={message_sid}, Status={message_status}, ErrorCode={error_code}")

        # Log failed messages for monitoring
        if message_status in ["failed", "undelivered"] and error_code:
            logger.error(f"WhatsApp message FAILED - Phone: {to_number}, ErrorCode: {error_code}, SID: {message_sid}")

            # Provide specific error context
            if error_code == "63016":
                logger.error(f"Error 63016: User {to_number} is outside 24-hour window - needs to message first")
            elif error_code == "63112":
                logger.error(f"Error 63112: WhatsApp Business Account disabled by Meta - cannot send to {to_number}")
            else:
                logger.error(f"Unknown WhatsApp error {error_code} for {to_number}")

            return {"status": "logged", "error_code": error_code, "phone": to_number}

        # For other status updates, just acknowledge
        return {"status": "acknowledged"}

    except Exception as e:
        logger.error(f"Error processing WhatsApp status callback: {str(e)}")
        return {"status": "error", "detail": "Webhook processing failed"}

@messaging_router.post("/tools/make-call", tags=["Tools"])
async def make_voice_call(
    request_data: VoiceCallRequest,
    current_user: AuthUser,
    config: TwilioConfig = Depends(get_twilio_config)
):
    """Initiates an outbound voice call to whitelisted numbers only."""
    # 1. Whitelist Check - unified anti-spam protection
    if not await is_whitelisted(request_data.to_number):
        raise HTTPException(
            status_code=403,
            detail=f"Number {request_data.to_number} not whitelisted! Send an SMS or WhatsApp message to whatsapp:+1 (555) 783-4727 or call us first to receive messages"
        )

    # 1.25. Acceptable-use content screen — reject slurs/hate terms before we
    # consume quota or hand anything to Twilio.
    assert_content_allowed(request_data.message)

    # 1.5. Resolve key or phone number to normalized E.164 format
    resolved_phone = await resolve_to_phone(request_data.to_number)

    # 2. Quota Check (separate voice_call quota)
    allowed, _usage_count, _reason = await try_consume_for(current_user, "voice_call")
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Daily voice call quota has been exceeded.",
                "quota_type": "voice_call"
            }
        )
    logger.info(f"Processing voice call for user_id: {current_user.id} to {resolved_phone} (original input: {request_data.to_number})")

    # 4. Action: Initiate the call
    try:
        client = Client(config.account_sid, config.auth_token)

        # Stash the message first, so it is retrievable no matter how fast
        # Twilio comes back for the TwiML.
        token = await stash_voice_message(request_data.message or DEFAULT_VOICE_MESSAGE)

        # The callback URL that Twilio will request when the call connects
        callback_url = (
            f"https://api.observer-ai.com/webhooks/voice-callback?token={quote(token)}"
        )

        # Create the call
        call = client.calls.create(
            to=resolved_phone,  # Use resolved phone number (handles keys and normalization)
            from_=config.from_number,
            url=callback_url,
            method='POST',
            status_callback="https://api.observer-ai.com/webhooks/voice-status"
        )

        logger.info(f"Voice call initiated successfully. SID: {call.sid}")
        return {
            "success": True,
            "call_sid": call.sid,
            "to": resolved_phone,  # Return resolved phone number
            "status": call.status
        }

    except TwilioRestException as e:
        logger.error(f"Twilio voice call API error: {e.msg}")
        raise HTTPException(status_code=400, detail=f"Failed to initiate call: {e.msg}")
    except Exception as e:
        logger.error(f"Error initiating voice call: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initiate voice call")

@messaging_router.post("/webhooks/voice-callback", tags=["Webhooks"])
async def voice_callback(
    request: Request,
    form_data: dict = Depends(validate_twilio_request)
):
    """
    Webhook endpoint that Twilio calls when a voice call connects.
    Handles both incoming calls (whitelist them) and outgoing calls (speak message).
    Returns TwiML instructions.
    Validates Twilio signature to prevent spoofing attacks.
    """
    try:
        # Extract call info (form_data already validated and parsed)
        call_sid = form_data.get("CallSid", "")
        call_status = form_data.get("CallStatus", "")
        direction = form_data.get("Direction", "")
        from_number = form_data.get("From", "")

        logger.info(f"Voice callback received: SID={call_sid}, Status={call_status}, Direction={direction}")

        # Create TwiML response
        response = VoiceResponse()

        if direction == "inbound":
            # Someone called YOUR number - whitelist them for phone (SMS/Voice)
            await add_to_whitelist(from_number, channel="phone")
            logger.info(f"Auto-whitelisted caller {from_number} via incoming call")

            response.say(
                "Your number is now whitelisted for 24 hours across SMS, WhatsApp, and voice calls. You can now receive messages from Observer AI.",
                voice='alice',
                language='en-US'
            )
        else:
            # Outgoing call - speak the message from AI, looked up by the token
            # we minted when the call was created (see stash_voice_message).
            message_text = await pop_voice_message(request.query_params.get("token", ""))
            if message_text is None:
                logger.warning(f"No stashed message for call {call_sid}; using default")
                message_text = DEFAULT_VOICE_MESSAGE
            response.say(message_text, voice='alice', language='en-US')

        # Return TwiML as XML
        return FastAPIResponse(content=str(response), media_type="application/xml")

    except Exception as e:
        logger.error(f"Error processing voice callback: {str(e)}")
        # Return a fallback TwiML response
        response = VoiceResponse()
        response.say("An error occurred.", voice='alice')
        return FastAPIResponse(content=str(response), media_type="application/xml")

@messaging_router.post("/webhooks/voice-status", tags=["Webhooks"])
async def voice_status_callback(
    request: Request,
    form_data: dict = Depends(validate_twilio_request)
):
    """
    Webhook endpoint to receive voice call status updates from Twilio.
    Validates Twilio signature to prevent spoofing attacks.
    """
    try:
        # Extract call info (form_data already validated and parsed)
        call_sid = form_data.get("CallSid", "")
        call_status = form_data.get("CallStatus", "")
        call_duration = form_data.get("CallDuration", "0")
        to_number = form_data.get("To", "")

        logger.info(f"Voice status: SID={call_sid}, Status={call_status}, Duration={call_duration}s, To={to_number}")

        # Log failed calls
        if call_status in ["failed", "busy", "no-answer"]:
            error_code = form_data.get("ErrorCode", "")
            error_message = form_data.get("ErrorMessage", "")
            logger.warning(f"Voice call FAILED/INCOMPLETE: {call_status} - To: {to_number}, SID: {call_sid}, ErrorCode: {error_code}, ErrorMessage: {error_message}")

        return {"status": "acknowledged"}

    except Exception as e:
        logger.error(f"Error processing voice status callback: {str(e)}")
        return {"status": "error", "detail": "Webhook processing failed"}
