"""Telegram Worker: handles all Telegram I/O independently from Core.

Reads outbox queue and sends messages.  Receives user input and writes
to inbox queue for Core.  Telegram failures are retried with backoff
and never crash the Core worker.

Entry point:
    python -m src.bot.telegram_worker
"""
from __future__ import annotations

import asyncio
import atexit
import logging
import os
import signal
import time
from typing import Any, Dict, List, Optional

from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.bot.chat_router import chat_router
from src.bot.message_queue import ack_outbox, pop_outbox, push_inbox, push_outbox, recover_unacked
from src.core.settings import settings

log = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [TGRAM] %(levelname)s %(name)s: %(message)s",
)

PID_FILE = os.path.join(os.path.dirname(__file__), ".telegram_worker.pid")

# ── Per-chat circuit breaker ──────────────────────────────────
# Isolates failures per chat_id so one blocked/dead chat doesn't
# prevent sends to all other chats.

_CIRCUIT_THRESHOLD = 5   # open after 5 consecutive failures per chat
_CIRCUIT_COOLDOWN = 60   # seconds before retry

# {chat_id: {"failures": int, "open": bool, "reset_at": float}}
_chat_circuits: Dict[str, Dict[str, Any]] = {}


def _get_circuit(chat_id: str) -> Dict[str, Any]:
    if chat_id not in _chat_circuits:
        _chat_circuits[chat_id] = {"failures": 0, "open": False, "reset_at": 0.0}
    return _chat_circuits[chat_id]


async def _safe_send(bot, chat_id: str, text: str, **kwargs) -> bool:
    """Send a message with retry + exponential backoff + per-chat circuit breaker.

    Each chat_id has an independent circuit breaker. A blocked user or
    invalid chat_id will only trip its own breaker, not affect other chats.
    """
    circuit = _get_circuit(chat_id)

    if circuit["open"]:
        if time.time() < circuit["reset_at"]:
            log.warning("Circuit open for chat %s, skipping", chat_id)
            return False
        log.info("Circuit half-open for chat %s, attempting reset", chat_id)
        circuit["open"] = False

    delays = [1, 2, 4, 8]
    for attempt, delay in enumerate(delays):
        try:
            await bot.send_message(chat_id=chat_id, text=text[:4096], **kwargs)
            circuit["failures"] = 0
            return True
        except Exception as exc:
            circuit["failures"] += 1
            log.warning("Send to %s failed (attempt %d): %s", chat_id, attempt + 1, exc)
            if circuit["failures"] >= _CIRCUIT_THRESHOLD:
                circuit["open"] = True
                circuit["reset_at"] = time.time() + _CIRCUIT_COOLDOWN
                log.error("Circuit breaker OPEN for chat %s (retry in %ds)", chat_id, _CIRCUIT_COOLDOWN)
                return False
            if attempt < len(delays) - 1:
                await asyncio.sleep(delay)
    return False


async def _broadcast(bot, text: str, target: str = "broadcast", chat_ids: Optional[List[str]] = None):
    """Send to all applicable chat IDs based on routing target."""
    if chat_ids:
        ids = chat_ids
    elif target == "primary":
        ids = chat_router.primary_only_ids()
    else:
        ids = chat_router.broadcast_ids()

    for cid in ids:
        await _safe_send(bot, cid, text)


# ── Outbox consumer ──────────────────────────────────────────

async def _consume_outbox(bot):
    """Drain the outbox queue and send messages via Telegram.

    Uses reliable queue pattern: messages are moved to a processing queue
    on pop and only removed (acked) after successful delivery.
    """
    for _ in range(20):  # process up to 20 messages per tick
        msg = pop_outbox(timeout=0)
        if msg is None:
            break

        msg_type = msg.get("type", "text")
        payload = msg.get("payload", {})
        target = msg.get("target", "broadcast")
        chat_ids = msg.get("chat_ids")

        try:
            if msg_type == "text":
                await _broadcast(bot, payload.get("text", ""), target, chat_ids)

            elif msg_type == "signal_push":
                await _send_signal_push(bot, payload, target, chat_ids)

            elif msg_type == "combo_push":
                await _send_combo_push(bot, payload, target, chat_ids)

            else:
                # Generic text fallback
                text = payload.get("text", "")
                if text:
                    await _broadcast(bot, text, target, chat_ids)

            # Acknowledge successful delivery
            ack_outbox(msg)

        except Exception as exc:
            log.error("Outbox message processing failed: %s", exc)
            # Do NOT ack — message stays in processing queue for recovery


async def _send_signal_push(bot, payload: Dict, target: str, chat_ids: Optional[List[str]]):
    """Format and send signal push with strict execution + radar watchlist.

    Execution book = only PLAYABLE signals.
    Radar = close calls (positive EV but failed confidence gate).
    """
    from src.core.risk_guards import get_min_confidence
    from src.core.settings import settings
    from src.utils.signal_formatter import format_signal_card, format_summary_header

    signals = payload.get("signals", []) or []
    radar = payload.get("radar", []) or []
    ts = payload.get("ts", "")
    raw_count = payload.get("raw_signal_count", len(signals) + len(radar))
    status_counts = payload.get("status_counts", {})

    if ts:
        await _broadcast(bot, f"\U0001F4C5 Tages-Push | Datenstand: {ts[:16]}", target, chat_ids)

    # Build summary header
    conf_gates = f"Soccer={settings.min_confidence_soccer_h2h} Tennis={settings.min_confidence_tennis}"
    summary = format_summary_header(
        raw_count=raw_count,
        deduped_count=max(len(signals) + len(radar), len(signals)),
        statuses=status_counts,
        ev_cut=settings.min_ev_default,
        conf_gates=conf_gates,
    )
    await _broadcast(bot, summary, target, chat_ids)

    # Section 1: Execution book (live stakes)
    if signals:
        await _broadcast(bot, "\U0001F7E2 EXECUTION (Live Stakes)", target, chat_ids)
        for i, b in enumerate(signals[:10]):
            msg = format_signal_card(b, i, min(len(signals), 10))
            await _broadcast(bot, msg, target, chat_ids)
    else:
        await _broadcast(bot, "\U0001F7E2 EXECUTION (Live Stakes)\nKeine spielbaren Signale.", target, chat_ids)

    # Section 2: Radar watchlist (paper-only close calls)
    if radar:
        lines = ["\U0001F440 RADAR (Paper Only - Close Calls)"]
        for i, b in enumerate(radar[:3], start=1):
            sport = str(b.get("sport", ""))
            market = str(b.get("market", "h2h"))
            sel = str(b.get("selection", "?"))
            prob = float(b.get("model_probability", 0))
            ev = float(b.get("expected_value", 0))
            gate = get_min_confidence(sport, market)
            gap = prob - gate
            lines.append(
                f"{i}. {sport} | {market} | {sel}\n"
                f"   EV {ev:+.4f} | Model {prob:.1%} vs Gate {gate:.1%} (Δ {gap:+.1%})"
            )
        await _broadcast(bot, "\n".join(lines), target, chat_ids)


async def _send_combo_push(bot, payload: Dict, target: str, chat_ids: Optional[List[str]]):
    """Format and send combo push.

    Sends all combos in a single message (consolidated) since they're lottery-style
    and we want to show all options regardless of EV.
    """
    from src.bot.handlers import _format_combo_card
    combos = payload.get("combos", [])
    if not combos:
        await _broadcast(bot, "Keine Kombi-Vorschläge heute.", target, chat_ids)
        return

    # Build consolidated message with all combos (no EV filter - confidence only)
    lines = [f"🧩 {len(combos)} Lotto-Kombis (Confidence-only)"]
    for combo_data in combos:
        card = _format_combo_card(combo_data)
        if card:
            lines.append("\n" + card)

    await _broadcast(bot, "\n\n".join(lines), target, chat_ids)


# ── Outbox polling job (runs inside Telegram event loop) ──────

async def outbox_poller(context: ContextTypes.DEFAULT_TYPE):
    """Periodic job that drains the outbox queue."""
    await _consume_outbox(context.bot)


# ── Access control filter ─────────────────────────────────────

async def _check_access(update, context) -> bool:
    """Reject messages from unauthorized chats."""
    if not update.effective_chat:
        return False
    cid = str(update.effective_chat.id)
    if not chat_router.is_authorized(cid):
        log.warning("Unauthorized chat: %s", cid)
        return False
    return True


# ── Build & run ───────────────────────────────────────────────

def build_app() -> Application:
    """Build the Telegram application with handlers."""
    from src.bot.handlers import (
        start,
        menu_value_bets,
        balance,
        callback_handler,
        combo_suggestions,
        settings_menu,
        help_menu,
        refresh_data,
        agentic_chat,
    )
    from src.data.models import Base
    from src.data.postgres import engine as db_engine

    Base.metadata.create_all(bind=db_engine)

    app = Application.builder().token(settings.telegram_bot_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", balance))
    app.add_handler(CommandHandler("help", help_menu))
    app.add_handler(MessageHandler(filters.Regex("^Heutige Top 10 Einzelwetten$"), menu_value_bets))
    app.add_handler(MessageHandler(filters.Text(["Kombis"]), combo_suggestions))
    app.add_handler(MessageHandler(filters.Regex("^Daten aktualisieren$"), refresh_data))
    app.add_handler(MessageHandler(filters.Regex("^Kontostand$"), balance))
    app.add_handler(MessageHandler(filters.Regex("^Einstellungen$"), settings_menu))
    app.add_handler(MessageHandler(filters.Regex("^Hilfe$"), help_menu))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, agentic_chat))

    return app


def _write_pid():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def _remove_pid():
    try:
        if os.path.isfile(PID_FILE):
            with open(PID_FILE) as f:
                if f.read().strip() == str(os.getpid()):
                    os.remove(PID_FILE)
    except OSError:
        pass


async def error_handler(update, context):
    import traceback
    log.error("Telegram handler error: %s", context.error)
    traceback.print_exc()


def main():
    _write_pid()
    atexit.register(_remove_pid)

    # Recover any unacked messages from previous crash
    try:
        recovered = recover_unacked()
        if recovered:
            log.info("Recovered %d unacked messages from previous session", recovered)
    except Exception as exc:
        log.warning("Message recovery failed (Redis may be down): %s", exc)

    app = build_app()
    app.add_error_handler(error_handler)

    # Poll outbox every 5 seconds
    app.job_queue.run_repeating(outbox_poller, interval=5, first=5, name="outbox_poller")

    log.info("Telegram worker starting (pid=%d)", os.getpid())
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
