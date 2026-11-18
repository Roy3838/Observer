# remote.py
"""
Code pairing, and remote control over it.

Each channel has its own 4-word code (minted in the browser, see
app/src/utils/whitelistCode.ts), and a code is the only way to reach a user:

  SMS code      -> SMS and voice alerts to the paired phone
  WhatsApp code -> WhatsApp and voice alerts to the paired phone, and remote
                   control over WhatsApp
  Telegram code -> Telegram alerts to the paired chat, and remote control over Telegram

Each pairing is its own consent: texting the code to our SMS number is the
opt-in to SMS (and calls from that same number), sending it on WhatsApp opts in
to WhatsApp only. So a code pairs on exactly one channel, and nothing pairing on
one channel enables another channel's messages.

Remote control uses the code as the session id. The server is only a mailbox:
messages from a paired phone/chat are queued for the browser tab listening on
that code, which runs them through its MCP and answers through the reply route.
The conversation itself never leaves the browser.

The channels trust different things:
  SMS, WhatsApp - the code is owned by the first authenticated user to poll
                  /tools/is-whitelisted with it, and only binds inside the short
                  pairing window those polls keep open. Only the owner sends with
                  it or reads its inbox: those sends cost real money.
  Telegram      - the code itself is the credential, no account needed. Whoever
                  holds it can send to the paired chat and read its inbox, but never
                  redirect it: replies only ever go to the paired chat.
Either way the first address to send an unpaired code binds it, and no other
address can take it over. A paired chat/phone rotates by sending a new code, or
ends it with "disconnect"; both revoke the old code for good, so a leaked code can
never be re-paired by whoever holds it.

SMS pairs only: there is no remote control over it, since every reply would be a
paid message outside what the A2P campaign covers. A paired phone's other texts
are ignored (Twilio answers HELP itself).

WhatsApp adds one rule of Meta's: free-form messages can only be sent within 24h
of the user's last inbound message. Every inbound WhatsApp message reopens that
window here; voice to the same phone does not depend on it.

Dead-man's switch: a tab polling with deadman=1 arms its sessions. If an armed
session stops polling (the phone locked and the OS suspended Observer, the app was
killed, the connection dropped), the paired phone/chat is told once that monitoring
stopped. Polling with deadman=0 disarms it. No template: on WhatsApp the ping only
goes out inside Meta's 24h window, which the app opens when it arms.

This module only depends on Redis. The channel adapters (the Twilio webhook in
messaging.py, the Telegram webhook in tools_router.py) pass in how to reply,
which keeps the import graph acyclic.
"""

import asyncio
import json
import logging
import re
import time
from typing import Awaitable, Callable, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from auth import AuthUser, get_current_user
from redis_client import get_redis
from whitelist_words import WORDS

logger = logging.getLogger('remote')
remote_router = APIRouter()

Channel = Literal["whatsapp", "sms", "telegram"]
CHANNELS: tuple[Channel, ...] = ("whatsapp", "sms", "telegram")
REMOTE_CHANNELS = {"whatsapp", "telegram"}  # the ones that talk to an Observer session

PAIRING_TTL = 600    # 10 min: how long after the owner's last whitelist poll a WhatsApp code can still bind
ALIVE_TTL = 45       # a tab re-polls the inbox every ~25s, so this lapses ~20s after the last tab closes
INBOX_TTL = 600      # an unread inbox evaporates on its own
INBOX_MAX_AGE = 120  # older messages are dropped rather than run: stale commands are worse than lost ones
INBOX_WAIT = 25      # long-poll hold, comfortably under proxy idle timeouts
BLPOP_SLICE = 4      # each blocking read stays under redis-py's default 5s socket read timeout
SEEN_TTL = 3600      # provider retry dedupe
WA_WINDOW_TTL = 24 * 3600  # Meta's customer-service window: free-form WhatsApp sends only inside it
MAX_TEXT = 4000
MAX_NAME = 64

DEADMAN_KEY = "remote:deadman"  # sorted set of "{channel}:{code}", scored by last poll
DEADMAN_GRACE = 75   # seconds without a poll before an armed session counts as stopped.
                     # Above ALIVE_TTL, so one failed poll plus a backoff retry isn't an alarm.
DEADMAN_SWEEP = 15   # how often each worker looks for stopped sessions
DEADMAN_BATCH = 200
INTERRUPTED = (
    "⚠️ Your Observer session stopped reporting. The device may have been locked, "
    "closed, or lost its connection."
)

# Users are pointed at DISCONNECT first: Twilio treats STOP as a carrier opt-out,
# which counts against the sender's reputation. STOP is only named in the SMS
# opt-in confirmation, where A2P reviewers expect it. The carrier words are all
# honored as unlinks.
STOP_WORDS = {"disconnect", "stop", "/stop"}
# Every word Twilio opts a phone out of SMS on. Twilio confirms those itself and
# blocks anything we send after them, so only DISCONNECT gets our reply.
SMS_OPT_OUT_WORDS = {"stop", "stopall", "unsubscribe", "cancel", "end", "quit", "revoke", "optout"}

_WORDS = frozenset(WORDS)

# The phone texts are the user's notice of what pairing opts them into. The SMS one
# is also the A2P opt-in confirmation: brand, frequency, rates, HELP, how to leave.
# Plain GSM-7 (no emoji, no curly quotes) so it stays at 2 segments.
CONNECTED = {
    "sms": "Observer AI: Connected! Your Observer agents can now alert this number by SMS and phone call. Msg frequency varies. Msg & data rates may apply. Reply HELP for help, DISCONNECT to unlink, STOP to opt out.",
    "whatsapp": "✅ Connected! Your Observer agents can now alert this number by WhatsApp and phone call. Reply here any time to talk to your Observer session. Send DISCONNECT to unlink.",
    "telegram": "✅ Connected! Your Observer agents can now alert this chat. Reply here any time to talk to your Observer session. Send /stop to disconnect.",
}
LINKED_ELSEWHERE = "This code is already linked somewhere else. Generate a new one in Observer to link here."
NOT_PAIRING = "Got your code, but Observer isn't waiting for it. Open the connect QR in Observer and send it again."
REVOKED = "This code was disconnected. Generate a new one in Observer to connect again."
STOPPED = "Disconnected. Observer won't message you here anymore. Generate a new code in Observer to reconnect."
NO_SESSION = {
    "whatsapp": "Hi! Your phone is connected but no Observer session is open. Open Observer on your computer and try again.",
    "telegram": "Hi! This chat is linked but no Observer session is active. Open Observer on your computer and try again.",
}
TEXT_ONLY = "I can only read text messages for now."

Reply = Callable[[str], Awaitable[None]]


def normalize_code(text: str | None) -> str | None:
    """The canonical code if `text` is one ("Tree Book-shower golden" counts), else None."""
    if not text:
        return None
    parts = [p for p in re.split(r"[\s\-]+", text.strip().lower()) if p]
    if len(parts) == 4 and all(p in _WORDS for p in parts):
        return "-".join(parts)
    return None


def _owner_key(code: str) -> str: return f"remote:owner:{code}"
def _pairing_key(code: str) -> str: return f"remote:pairing:{code}"
def _revoked_key(code: str) -> str: return f"remote:revoked:{code}"
def _bind_key(code: str, channel: str) -> str: return f"remote:bind:{code}:{channel}"
def _name_key(code: str, channel: str) -> str: return f"remote:name:{code}:{channel}"
def _alive_key(code: str, channel: str) -> str: return f"remote:alive:{code}:{channel}"
def _inbox_key(code: str, channel: str) -> str: return f"remote:inbox:{code}:{channel}"
def _addr_key(channel: str, address: str) -> str: return f"remote:addr:{channel}:{address}"
def _wa_window_key(address: str) -> str: return f"remote:wawindow:{address}"


async def claim(code: str, user_id: str) -> bool:
    """
    Called from the authenticated whitelist poll. Claims an unowned code for
    user_id and, if they own it, (re)opens its pairing window. Returns ownership.
    """
    r = await get_redis()
    await r.set(_owner_key(code), user_id, nx=True)
    if await r.get(_owner_key(code)) != user_id:
        return False
    await r.setex(_pairing_key(code), PAIRING_TTL, "1")
    return True


async def owned_address(code: str, channel: str, user_id: str) -> str | None:
    """The address bound to `code` on `channel`, but only for the code's owner (SMS, WhatsApp)."""
    r = await get_redis()
    if await r.get(_owner_key(code)) != user_id:
        return None
    return await r.get(_bind_key(code, channel))


async def bound_address(code: str, channel: str) -> str | None:
    """The address bound to `code` on `channel`, for whoever holds the code (Telegram)."""
    r = await get_redis()
    return await r.get(_bind_key(code, channel))


async def open_whatsapp_window(address: str) -> None:
    """Record an inbound WhatsApp message from `address`: Meta's 24h window is open again."""
    r = await get_redis()
    await r.setex(_wa_window_key(address), WA_WINDOW_TTL, "1")


async def whatsapp_window_open(address: str) -> bool:
    """Whether a free-form WhatsApp message to `address` would be allowed right now."""
    r = await get_redis()
    return bool(await r.exists(_wa_window_key(address)))


async def first_delivery(msg_id: str | None) -> bool:
    """False if this provider message id was already processed (webhook retries)."""
    if not msg_id:
        return True
    r = await get_redis()
    return bool(await r.set(f"remote:seen:{msg_id}", "1", nx=True, ex=SEEN_TTL))


async def _revoke(code: str, channel: Channel) -> None:
    """Unpair `code` and mark it dead for good: agents still sending to it can't be picked up."""
    r = await get_redis()
    await r.set(_revoked_key(code), "1")
    await r.delete(_bind_key(code, channel), _name_key(code, channel),
                   _inbox_key(code, channel), _alive_key(code, channel))
    await r.zrem(DEADMAN_KEY, _deadman_member(code, channel))


async def _may_bind(code: str, channel: Channel) -> bool:
    # Telegram: holding the code is enough. SMS/WhatsApp: the owner's QR must be open.
    if channel == "telegram":
        return True
    r = await get_redis()
    return bool(await r.exists(_pairing_key(code)))


async def _bound_elsewhere(code: str, channel: Channel) -> bool:
    """Whether `code` is already paired on a channel other than `channel`."""
    r = await get_redis()
    others = [_bind_key(code, c) for c in CHANNELS if c != channel]
    return bool(await r.exists(*others))


async def route_inbound(channel: Channel, address: str, text: str, reply: Reply, name: str | None = None) -> bool:
    """
    Handle an inbound message if it is pairing business: a code, "disconnect", or anything
    from an address that is bound to one. Returns False otherwise, and the adapter
    tells the sender how to connect. `name` is the sender's display name, shown in
    the app once they pair.
    """
    r = await get_redis()
    current = await r.get(_addr_key(channel, address))

    word = text.strip().lower()
    if word in STOP_WORDS or (channel == "sms" and word in SMS_OPT_OUT_WORDS):
        if not current:
            return False
        await _revoke(current, channel)
        await r.delete(_addr_key(channel, address))
        logger.info(f"Remote: {channel} {address} stopped")
        if not (channel == "sms" and word in SMS_OPT_OUT_WORDS):
            await reply(STOPPED)
        return True

    code = normalize_code(text)
    if code:
        if await r.exists(_revoked_key(code)):
            await reply(REVOKED)
            return True

        bind_key = _bind_key(code, channel)
        bound = await r.get(bind_key)
        if bound is None and await _bound_elsewhere(code, channel):
            # One code, one channel: each one is its own consent.
            logger.warning(f"Remote: {channel} {address} sent a code paired on another channel")
            await reply(LINKED_ELSEWHERE)
            return True
        if bound is None and await _may_bind(code, channel):
            # SETNX: two chats racing for the same code, exactly one wins.
            await r.set(bind_key, address, nx=True)
            bound = await r.get(bind_key)

        if bound == address:
            # Rotation: a new code from a paired chat retires its old one.
            if current and current != code:
                await _revoke(current, channel)
            await r.set(_addr_key(channel, address), code)
            if name:
                await r.set(_name_key(code, channel), name[:MAX_NAME])
            logger.info(f"Remote: {channel} {address} connected")
            await reply(CONNECTED[channel])
        elif bound is not None:
            # Never let a second address take over the code's binding.
            logger.warning(f"Remote: {channel} {address} sent a code bound to another address")
            await reply(LINKED_ELSEWHERE)
        else:
            await reply(NOT_PAIRING)
        return True

    if not current:
        return False

    if channel not in REMOTE_CHANNELS:
        return True

    if not await r.exists(_alive_key(current, channel)):
        await reply(NO_SESSION[channel])
        return True

    if not text.strip():
        await reply(TEXT_ONLY)
        return True

    item = json.dumps({"text": text[:MAX_TEXT], "ts": time.time()})
    inbox = _inbox_key(current, channel)
    await r.rpush(inbox, item)
    await r.expire(inbox, INBOX_TTL)
    return True


# --- Dead-man's switch ---

def _deadman_member(code: str, channel: str) -> str: return f"{channel}:{code}"


# Claims every session that stopped polling before ARGV[1] and removes it, in one step:
# with several workers sweeping, each stopped session is announced exactly once, and a
# poll landing mid-sweep can't be removed after it refreshed its score.
_CLAIM_STOPPED = """
local due = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, ARGV[2])
if #due > 0 then redis.call('ZREM', KEYS[1], unpack(due)) end
return due
"""


async def _deadman_poll(sessions: dict[Channel, str], deadman: bool | None) -> bool:
    """
    Arm (True), disarm (False) or, for app builds that don't send the flag (None), just
    refresh whatever is already armed. Returns True when this poll disarmed something.
    """
    r = await get_redis()
    members = [_deadman_member(code, channel) for channel, code in sessions.items()]
    if deadman is False:
        return bool(await r.zrem(DEADMAN_KEY, *members))
    now = time.time()
    await r.zadd(DEADMAN_KEY, {m: now for m in members}, xx=deadman is None)
    return False


async def _announce_stopped(member: str) -> None:
    channel, code = member.split(":", 1)
    address = await bound_address(code, channel)
    if not address:
        return
    # Embrace Meta's window rather than work around it: outside it, no free-form send.
    if channel == "whatsapp" and not await whatsapp_window_open(address):
        logger.info(f"Dead-man: {member} stopped, but the WhatsApp window is closed; not sent")
        return
    logger.info(f"Dead-man: {member} stopped reporting; notifying")
    await _deliver(channel, address, INTERRUPTED, [])


async def deadman_sweeper() -> None:
    """
    Runs in every worker for the life of the process. Each sweep claims the armed
    sessions whose last poll is older than DEADMAN_GRACE and notifies their phone/chat.
    """
    # After a restart, every tab's poll was cut off; give them a grace period to reconnect
    # before reading their stale scores as stopped sessions.
    await asyncio.sleep(DEADMAN_GRACE)
    while True:
        try:
            r = await get_redis()
            stopped = await r.eval(_CLAIM_STOPPED, 1, DEADMAN_KEY, time.time() - DEADMAN_GRACE, DEADMAN_BATCH)
            for member in stopped:
                try:
                    await _announce_stopped(member)
                except Exception as e:
                    logger.error(f"Dead-man: failed to notify {member}: {e}")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Dead-man sweep failed")
        await asyncio.sleep(DEADMAN_SWEEP)


# --- Shared route bodies ---

def _require_code(code: str) -> str:
    normalized = normalize_code(code)
    if not normalized:
        raise HTTPException(status_code=400, detail="Not a valid Observer code.")
    return normalized


async def _require_owner(code: str, user_id: str) -> str:
    normalized = _require_code(code)
    r = await get_redis()
    if await r.get(_owner_key(normalized)) != user_id:
        raise HTTPException(status_code=403, detail="This code isn't linked to your account.")
    return normalized


async def _poll(sessions: dict[Channel, str], deadman: bool | None = None) -> dict:
    """
    Long-poll for messages sent from the paired phones/chats of `sessions` (channel ->
    code), all in one request. Every call also marks each session as active, which is
    what lets the webhooks tell "queue it" apart from "no Observer session is open",
    and feeds the dead-man's switch (see deadman_sweeper).
    """
    r = await get_redis()
    for channel, code in sessions.items():
        await r.setex(_alive_key(code, channel), ALIVE_TTL, "1")

    # A disarming poll answers at once: it may be the tab's last one (remote control
    # being turned off), and it must not take messages it won't stay to handle.
    if await _deadman_poll(sessions, deadman):
        return {"messages": []}

    # RPUSH + BLPOP = FIFO, and each message goes to exactly one polling tab. Each BLPOP
    # waits on every inbox at once, in short slices rather than one long block: redis-py
    # (>= 8) reads with a 5s socket timeout by default, and a longer block fails.
    inboxes = {_inbox_key(code, channel): channel for channel, code in sessions.items()}
    deadline = time.monotonic() + INBOX_WAIT
    popped = None
    while popped is None and time.monotonic() < deadline:
        popped = await r.blpop(list(inboxes), timeout=BLPOP_SLICE)
    if not popped:
        return {"messages": []}
    raw = [(inboxes[popped[0]], popped[1])]
    for inbox, channel in inboxes.items():
        raw += [(channel, entry) for entry in await r.lpop(inbox, 20) or []]

    now = time.time()
    messages = []
    for channel, entry in raw:
        item = json.loads(entry)
        if now - item["ts"] <= INBOX_MAX_AGE:
            messages.append({"channel": channel, "text": item["text"]})
    return {"messages": messages}


async def _status(code: str, channel: Channel) -> dict:
    """Whether the code is paired on `channel`, to whom, and whether it was revoked."""
    r = await get_redis()
    return {
        "code": code,
        "linked": bool(await r.get(_bind_key(code, channel))),
        "name": await r.get(_name_key(code, channel)),
        "revoked": bool(await r.exists(_revoked_key(code))),
    }


async def _deliver(channel: Channel, address: str, text: str, images: list[str]) -> None:
    """Send an MCP answer to the paired phone/chat. Image delivery is best-effort."""
    # Imported here: both modules import this one for their webhooks.
    from messaging import save_temp_image, send_whatsapp_text
    from tools_router import send_telegram_photo, send_telegram_text

    if channel == "whatsapp":
        media_urls = []
        for i, image_b64 in enumerate(images):
            try:
                media_urls.append(await save_temp_image(image_b64))
            except Exception as e:
                logger.warning(f"Remote reply: failed to host image {i+1} for WhatsApp: {e}")
        await send_whatsapp_text(address, text[:1600], media_urls or None)
    else:
        for i, image_b64 in enumerate(images):
            try:
                await send_telegram_photo(address, image_b64)
            except Exception as e:
                logger.warning(f"Remote reply: failed to send image {i+1} to Telegram: {e}")
        await send_telegram_text(address, text[:4096])


class RemoteReplyRequest(BaseModel):
    code: str = Field(..., description="The code the message arrived on.")
    text: str = Field(..., min_length=1, max_length=MAX_TEXT)
    images: list[str] | None = Field(None, description="Optional base64-encoded images (no data: prefix), e.g. from capture_screen.")


MAX_REPLY_IMAGES = 3  # keep MMS/Telegram sends bounded; mirrors useMCP's MAX_REMOTE_IMAGES


# --- Both channels in one long-poll ---

@remote_router.get("/remote/listen", tags=["Remote"])
async def remote_listen(
    request: Request,
    whatsapp: str | None = None,
    telegram: str | None = None,
    deadman: bool | None = None,
):
    """
    Long-poll the inboxes of a WhatsApp code and/or a Telegram code in one request.
    Each channel keeps its own trust rule: the WhatsApp code must be the caller's
    (so a token is required only when one is given); the Telegram code is enough
    on its own. deadman=1 arms the dead-man's switch for these sessions, deadman=0
    disarms it.
    """
    sessions: dict[Channel, str] = {}
    if whatsapp:
        current_user = await get_current_user(request)
        sessions["whatsapp"] = await _require_owner(whatsapp, current_user.id)
    if telegram:
        sessions["telegram"] = _require_code(telegram)
    if not sessions:
        raise HTTPException(status_code=400, detail="Give a whatsapp and/or telegram code.")
    return await _poll(sessions, deadman)


# --- WhatsApp: owner-gated ---

@remote_router.get("/remote/inbox", tags=["Remote"])
async def remote_inbox(code: str, current_user: AuthUser):
    """Long-poll the WhatsApp inbox of one of the caller's codes. Superseded by
    /remote/listen; kept for app builds that still poll per channel."""
    return await _poll({"whatsapp": await _require_owner(code, current_user.id)})


@remote_router.get("/remote/status", tags=["Remote"])
async def remote_status(code: str, current_user: AuthUser):
    """
    Whether the caller's code is paired on WhatsApp, and to whom, for the settings card.
    Answers the question the whitelist check can't: whitelisted means "we may message
    this number", linked means "this phone can talk to my Observer session".
    """
    return await _status(await _require_owner(code, current_user.id), "whatsapp")


@remote_router.post("/remote/reply", tags=["Remote"])
async def remote_reply(request_data: RemoteReplyRequest, current_user: AuthUser):
    """Send the MCP's answer back to the paired phone over WhatsApp, with any images the
    model captured along the way. Charges the WhatsApp quota once, regardless of image count."""
    from messaging import WHATSAPP_WINDOW_CLOSED, assert_content_allowed
    from quota_manager import try_consume_for

    code = await _require_owner(request_data.code, current_user.id)
    address = await bound_address(code, "whatsapp")
    if not address:
        raise HTTPException(status_code=404, detail="No WhatsApp linked to this code.")

    # Meta's 24h window, as in /tools/send-whatsapp: outside it Twilio accepts the
    # send and fails it later (63016), after we reported success and charged quota.
    if not await whatsapp_window_open(address):
        raise HTTPException(status_code=403, detail=WHATSAPP_WINDOW_CLOSED)

    assert_content_allowed(request_data.text)

    allowed, _usage_count, _reason = await try_consume_for(current_user, "whatsapp")
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={"message": "Daily whatsapp quota has been exceeded.", "quota_type": "whatsapp"},
        )

    await _deliver("whatsapp", address, request_data.text, (request_data.images or [])[:MAX_REPLY_IMAGES])
    return {"success": True}


# --- Telegram: the code is the credential ---

@remote_router.get("/remote/telegram/inbox", tags=["Remote"])
async def remote_telegram_inbox(code: str):
    """Long-poll the Telegram inbox of a code. No account: holding the code is enough.
    Superseded by /remote/listen; kept for app builds that still poll per channel."""
    return await _poll({"telegram": _require_code(code)})


@remote_router.get("/remote/telegram/status", tags=["Remote"])
async def remote_telegram_status(code: str):
    """Whether a code is paired on Telegram, to whom, and whether it was revoked."""
    return await _status(_require_code(code), "telegram")


@remote_router.post("/remote/telegram/reply", tags=["Remote"])
async def remote_telegram_reply(request_data: RemoteReplyRequest):
    """Send the MCP's answer back to the code's paired Telegram chat, with any captured images."""
    from messaging import assert_content_allowed

    address = await bound_address(_require_code(request_data.code), "telegram")
    if not address:
        raise HTTPException(status_code=404, detail="No Telegram chat linked to this code.")

    assert_content_allowed(request_data.text)
    await _deliver("telegram", address, request_data.text, (request_data.images or [])[:MAX_REPLY_IMAGES])
    return {"success": True}
