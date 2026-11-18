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

Remote control uses the code as the session id. The server is only a mailbox,
with one route each way:

  IN   phone/chat -> webhook -> inbox -> GET /remote/listen -> the tab's MCP
  OUT  every message of the tab's MCP conversation -> POST /remote/send -> phone/chat

/remote/send mirrors the whole conversation, whichever side started a turn (the
phone, the desktop, or an agent waking the MCP), so the phone reads what the
desktop shows. The conversation itself lives in the browser; the server only
relays it. While a turn runs, the phone shows "typing…" (see _keep_typing).

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
import hashlib
import json
import logging
import re
import time
from typing import Awaitable, Callable, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from auth import AuthenticatedUser, AuthUser, get_current_user
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

BUSY_TTL = 300       # safety net for a turn whose `done` never comes. Typing only shows while
                     # the tab polls, so this outlives the longest silent tool call (get_iteration: 2 min)
TYPING_EVERY = {"telegram": 3, "whatsapp": 20}  # each lasts ~5s (Telegram) / ~25s (Twilio)
TYPING_QUIET = 3     # no typing right after a message lands, so it can't outlive the turn's last one
TURN_TTL = 3600      # how long a turn's WhatsApp charge is remembered
DEPRECATED_LOG_EVERY = 3600

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
def _busy_key(code: str, channel: str) -> str: return f"remote:busy:{code}:{channel}"
def _typed_key(code: str, channel: str) -> str: return f"remote:typed:{code}:{channel}"
def _last_msg_key(code: str, channel: str) -> str: return f"remote:lastmsg:{code}:{channel}"


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
                   _inbox_key(code, channel), _alive_key(code, channel), _busy_key(code, channel),
                   _last_msg_key(code, channel))
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


async def route_inbound(
    channel: Channel, address: str, text: str, reply: Reply,
    name: str | None = None, msg_id: str | None = None,
) -> bool:
    """
    Handle an inbound message if it is pairing business: a code, "disconnect", or anything
    from an address that is bound to one. Returns False otherwise, and the adapter
    tells the sender how to connect. `name` is the sender's display name, shown in
    the app once they pair. `msg_id` is the provider's id for the message (WhatsApp's
    MessageSid), which Twilio's typing indicator has to answer.
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

    item = json.dumps({"text": text[:MAX_TEXT], "ts": time.time(), "id": msg_id})
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
    # The tab is gone mid-turn or not: whatever it was running won't finish.
    await (await get_redis()).delete(_busy_key(code, channel))
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


# --- Typing indicator ---
#
# A session is busy from the moment its tab picks up a message (or sends the first item of a
# turn it started itself) until the turn's `done`. While busy, the tab's own long-poll keeps
# "typing…" up on the phone, once per slice; so if the tab dies, the typing stops with it.

async def _set_busy(code: str, channel: str, busy: bool) -> None:
    r = await get_redis()
    if busy:
        await r.set(_busy_key(code, channel), "1", ex=BUSY_TTL)
    else:
        await r.delete(_busy_key(code, channel))


async def _keep_typing(sessions: dict[Channel, str]) -> None:
    """Show "typing…" on each busy session's phone/chat, at most every TYPING_EVERY seconds."""
    r = await get_redis()
    for channel, code in sessions.items():
        if not await r.exists(_busy_key(code, channel)):
            continue
        # Twilio's indicator answers an inbound message: the phone's latest one.
        msg_id = await r.get(_last_msg_key(code, channel)) if channel == "whatsapp" else None
        if channel == "whatsapp" and not msg_id:
            continue
        if not await r.set(_typed_key(code, channel), "1", nx=True, ex=TYPING_EVERY[channel]):
            continue
        address = await bound_address(code, channel)
        if not address:
            continue
        try:
            if channel == "whatsapp":
                from messaging import send_whatsapp_typing
                await send_whatsapp_typing(msg_id)
            else:
                from tools_router import send_telegram_typing
                await send_telegram_typing(address)
        except Exception as e:
            logger.info(f"Typing indicator failed on {channel}: {e}")


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
        await _keep_typing(sessions)
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
            # The tab is about to run it: typing starts now (and on WhatsApp, it shows as read).
            code = sessions[channel]
            await _set_busy(code, channel, True)
            if item.get("id"):
                await r.set(_last_msg_key(code, channel), item["id"], ex=WA_WINDOW_TTL)
    await _keep_typing(sessions)
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
        if text:
            await send_telegram_text(address, text[:4096])


async def _sessions(request: Request, whatsapp: str | None, telegram: str | None) -> tuple[dict[Channel, str], AuthenticatedUser | None]:
    """
    The sessions a /listen or /send request names, each under its own channel's trust rule:
    the WhatsApp code must be the caller's (so a token is required only when one is given);
    the Telegram code is enough on its own. Returns them with the caller, if signed in.
    """
    sessions: dict[Channel, str] = {}
    user = None
    if whatsapp:
        user = await get_current_user(request)
        sessions["whatsapp"] = await _require_owner(whatsapp, user.id)
    if telegram:
        sessions["telegram"] = _require_code(telegram)
    if not sessions:
        raise HTTPException(status_code=400, detail="Give a whatsapp and/or telegram code.")
    return sessions, user


async def _charge_turn(user: AuthenticatedUser, turn_id: str) -> bool:
    """WhatsApp quota is charged once per turn, however many messages the turn sends."""
    from quota_manager import try_consume_for

    r = await get_redis()
    key = f"remote:turn:{user.id}:{turn_id}"
    charged = await r.get(key)
    if charged is None:
        allowed, _usage_count, _reason = await try_consume_for(user, "whatsapp")
        charged = "ok" if allowed else "over"
        await r.set(key, charged, ex=TURN_TTL)
    return charged == "ok"


async def _deprecated(route: str, caller: str) -> None:
    """Log that an app build older than /remote/send used `route`, at most hourly per caller,
    so the logs show who still has to update before the old routes can go."""
    r = await get_redis()
    if await r.set(f"remote:deprecated:{route}:{caller}", "1", nx=True, ex=DEPRECATED_LOG_EVERY):
        logger.warning(f"Deprecated route {route} hit by {caller}: app build predates /remote/send")


def _anon(code: str) -> str:
    """A stable, non-secret tag for a Telegram code (the code itself is a credential)."""
    return "tg-" + hashlib.sha256(code.encode()).hexdigest()[:8]


# --- The two routes: IN and OUT ---

@remote_router.get("/remote/listen", tags=["Remote"])
async def remote_listen(
    request: Request,
    whatsapp: str | None = None,
    telegram: str | None = None,
    deadman: bool | None = None,
):
    """
    IN: long-poll the inboxes of a WhatsApp code and/or a Telegram code in one request.
    deadman=1 arms the dead-man's switch for these sessions, deadman=0 disarms it.
    """
    sessions, _user = await _sessions(request, whatsapp, telegram)
    return await _poll(sessions, deadman)


class RemoteSendRequest(BaseModel):
    whatsapp: str | None = Field(None, description="The WhatsApp code to send to (must be the caller's).")
    telegram: str | None = Field(None, description="The Telegram code to send to.")
    turn_id: str = Field(..., min_length=1, max_length=64, description="One id per MCP turn: WhatsApp quota is charged once per turn.")
    text: str | None = Field(None, max_length=MAX_TEXT)
    image: str | None = Field(None, description="One base64-encoded image (no data: prefix), sent with `text`.")
    done: bool = Field(False, description="The turn is over: stop showing typing.")


async def _send_item(channel: Channel, code: str, body: "RemoteSendRequest", user: AuthenticatedUser | None) -> str:
    """Deliver one item to one paired phone/chat. Returns what happened, for the caller's log."""
    address = await bound_address(code, channel)
    if not address:
        return "not_linked"
    if channel == "whatsapp":
        # Meta's 24h window: outside it, free-form sends fail later (63016). Dropped, not queued.
        if not await whatsapp_window_open(address):
            return "window_closed"
        if not await _charge_turn(user, body.turn_id):
            return "quota_exceeded"
    try:
        await _deliver(channel, address, body.text or "", [body.image] if body.image else [])
    except Exception as e:
        logger.warning(f"Remote send to {channel} failed: {e}")
        return "failed"
    # Hold typing back for a moment: a turn's `done` follows its last message right away.
    await (await get_redis()).set(_typed_key(code, channel), "1", ex=TYPING_QUIET)
    return "sent"


@remote_router.post("/remote/send", tags=["Remote"])
async def remote_send(body: RemoteSendRequest, request: Request):
    """
    OUT: one item of the tab's MCP conversation, to every session named. Items arrive in
    order, one request each: a message (text, optionally with one image), and finally
    `done`. Every item but `done` keeps the session busy, which keeps typing up between them.
    """
    from messaging import assert_content_allowed

    sessions, user = await _sessions(request, body.whatsapp, body.telegram)
    assert_content_allowed(body.text)
    sent: dict[str, str] = {}
    for channel, code in sessions.items():
        await _set_busy(code, channel, not body.done)
        if body.text or body.image:
            sent[channel] = await _send_item(channel, code, body, user)
    return {"sent": sent}


# --- Status, for the settings card ---

@remote_router.get("/remote/status", tags=["Remote"])
async def remote_status(code: str, current_user: AuthUser):
    """
    Whether the caller's code is paired on WhatsApp, and to whom, for the settings card.
    Answers the question the whitelist check can't: whitelisted means "we may message
    this number", linked means "this phone can talk to my Observer session".
    """
    return await _status(await _require_owner(code, current_user.id), "whatsapp")


@remote_router.get("/remote/telegram/status", tags=["Remote"])
async def remote_telegram_status(code: str):
    """Whether a code is paired on Telegram, to whom, and whether it was revoked."""
    return await _status(_require_code(code), "telegram")


# --- Deprecated: app builds from before /remote/listen and /remote/send ---
# Each one logs (see _deprecated). Delete them once the logs stay quiet.

class RemoteReplyRequest(BaseModel):
    code: str = Field(..., description="The code the message arrived on.")
    text: str = Field(..., min_length=1, max_length=MAX_TEXT)
    images: list[str] | None = Field(None, description="Optional base64-encoded images (no data: prefix), e.g. from capture_screen.")


MAX_REPLY_IMAGES = 3


@remote_router.get("/remote/inbox", tags=["Remote"], deprecated=True)
async def remote_inbox(code: str, current_user: AuthUser):
    """Deprecated: use /remote/listen. Long-poll the WhatsApp inbox of one of the caller's codes."""
    await _deprecated("/remote/inbox", current_user.id)
    return await _poll({"whatsapp": await _require_owner(code, current_user.id)})


@remote_router.get("/remote/telegram/inbox", tags=["Remote"], deprecated=True)
async def remote_telegram_inbox(code: str):
    """Deprecated: use /remote/listen. Long-poll the Telegram inbox of a code."""
    code = _require_code(code)
    await _deprecated("/remote/telegram/inbox", _anon(code))
    return await _poll({"telegram": code})


@remote_router.post("/remote/reply", tags=["Remote"], deprecated=True)
async def remote_reply(request_data: RemoteReplyRequest, current_user: AuthUser):
    """Deprecated: use /remote/send. Send the MCP's final answer to the paired phone over
    WhatsApp. Charges the WhatsApp quota once, regardless of image count."""
    from messaging import WHATSAPP_WINDOW_CLOSED, assert_content_allowed
    from quota_manager import try_consume_for

    await _deprecated("/remote/reply", current_user.id)
    code = await _require_owner(request_data.code, current_user.id)
    # An old build's reply is its turn's end.
    await _set_busy(code, "whatsapp", False)
    address = await bound_address(code, "whatsapp")
    if not address:
        raise HTTPException(status_code=404, detail="No WhatsApp linked to this code.")

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


@remote_router.post("/remote/telegram/reply", tags=["Remote"], deprecated=True)
async def remote_telegram_reply(request_data: RemoteReplyRequest):
    """Deprecated: use /remote/send. Send the MCP's final answer to the code's paired Telegram chat."""
    from messaging import assert_content_allowed

    code = _require_code(request_data.code)
    await _deprecated("/remote/telegram/reply", _anon(code))
    await _set_busy(code, "telegram", False)
    address = await bound_address(code, "telegram")
    if not address:
        raise HTTPException(status_code=404, detail="No Telegram chat linked to this code.")

    assert_content_allowed(request_data.text)
    await _deliver("telegram", address, request_data.text, (request_data.images or [])[:MAX_REPLY_IMAGES])
    return {"success": True}
