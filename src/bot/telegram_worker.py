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
from src.bot.message_queue import pop_outbox, push_inbox, push_outbox
from src.core.settings import settings

log = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [TGRAM] %(levelname)s %(name)s: %(message)s",
)

PID_FILE = os.path.join(os.path.dirname(__file__), ".telegram_worker.pid")

# ── Telegram send with retry + circuit breaker ────────────────

_send_failures = 0
_send_circuit_open = False
_circuit_reset_at = 0.0
_CIRCUIT_THRESHOLD = 5  # open after 5 consecutive failures
_CIRCUIT_COOLDOWN = 60  # seconds before retry


async def _safe_send(bot, chat_id: str, text: str, **kwargs) -> bool:
    """Send a message with retry + exponential backoff + circuit breaker."""
    global _send_failures, _send_circuit_open, _circuit_reset_at

    if _send_circuit_open:
        if time.time() < _circuit_reset_at:
            log.warning("Telegram circuit open, skipping send to %s", chat_id)
            return False
        log.info("Telegram circuit half-open, attempting reset")
        _send_circuit_open = False

    delays = [1, 2, 4, 8]
    for attempt, delay in enumerate(delays):
        try:
            await bot.send_message(chat_id=chat_id, text=text[:4096], **kwargs)
            _send_failures = 0
            return True
        except Exception as exc:
            _send_failures += 1
            log.warning("Telegram send failed (attempt %d): %s", attempt + 1, exc)
            if _send_failures >= _CIRCUIT_THRESHOLD:
                _send_circuit_open = True
                _circuit_reset_at = time.time() + _CIRCUIT_COOLDOWN
                log.error("Telegram circuit breaker OPEN (will retry in %ds)", _CIRCUIT_COOLDOWN)
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
    """Drain the outbox queue and send messages via Telegram."""
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

        except Exception as exc:
            log.error("Outbox message processing failed: %s", exc)


async def _send_signal_push(bot, payload: Dict, target: str, chat_ids: Optional[List[str]]):
    """Format and send signal push."""
    from src.bot.handlers import _format_signal_card, _progress_bar, _calibration_badge
    signals = payload.get("signals", [])
    ts = payload.get("ts", "")

    if ts:
        await _broadcast(bot, f"📅 Tages-Push | Datenstand: {ts[:16]}", target, chat_ids)

    if not signals:
        await _broadcast(bot, "Keine spielbaren Einzelwetten heute.", target, chat_ids)
        return

    await _broadcast(bot, f"🎯 {len(signals[:10])} Top Einzelwetten", target, chat_ids)
    for b in signals[:10]:
        sport = str(b.get("sport", "")).replace("_", " ").upper()
        model_p = float(b.get("model_probability", 0))
        badge = _calibration_badge(model_p)
        msg = (
            f"🎯 {sport} {badge}\n"
            f"Tipp: {b.get('selection', '?')}\n"
            f"Quote: {float(b.get('bookmaker_odds', 0)):.2f} | "
            f"Modell: {_progress_bar(model_p)}\n"
            f"EV: {float(b.get('expected_value', 0)):+.4f} | "
            f"Einsatz: {float(b.get('recommended_stake', 0)):.2f} EUR"
        )
        await _broadcast(bot, msg, target, chat_ids)


async def _send_combo_push(bot, payload: Dict, target: str, chat_ids: Optional[List[str]]):
    """Format and send combo push."""
    from src.bot.handlers import _format_combo_card
    combos = payload.get("combos", [])
    if not combos:
        await _broadcast(bot, "Keine Kombi-Vorschläge heute.", target, chat_ids)
        return

    sent = 0
    for combo_data in combos:
        if float(combo_data.get("expected_value", 0)) <= 0:
            continue
        card = _format_combo_card(combo_data)
        if card:
            if sent == 0:
                await _broadcast(bot, f"🧩 {len(combos)} Lotto-Kombis", target, chat_ids)
            await _broadcast(bot, card, target, chat_ids)
            sent += 1
    if sent == 0:
        await _broadcast(
            bot,
            "Keine Kombi-Vorschläge mit vollständigem Event-Kontext verfügbar.",
            target,
            chat_ids,
        )


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
    app.add_handler(MessageHandler(filters.Regex("^Heutige Top 10 Einzelwetten$"), menu_value_bets))
    app.add_handler(MessageHandler(filters.Regex("^10/20/30 Kombis$"), combo_suggestions))
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

    app = build_app()
    app.add_error_handler(error_handler)

    # Poll outbox every 5 seconds
    app.job_queue.run_repeating(outbox_poller, interval=5, first=5, name="outbox_poller")

    log.info("Telegram worker starting (pid=%d)", os.getpid())
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
