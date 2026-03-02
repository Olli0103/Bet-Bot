"""Telegram bot handlers with interactive inline keyboards, visual dashboards,
and agentic chat mode.

All sync DB calls are wrapped in asyncio.to_thread to avoid blocking the
event loop (the "gotcha" fix).
"""
import asyncio
import hashlib
import io
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, ReplyKeyboardMarkup
from telegram.ext import ContextTypes
from sqlalchemy import select

from src.core.bankroll import BankrollManager
from src.core.betting_engine import BettingEngine
from src.core.live_feed import (
    fetch_and_build_signals,
    get_cached_combo_legs,
    get_cached_meta,
    get_cached_signals,
)
from src.core.settings import settings
from src.data.postgres import SessionLocal
from src.data.models import PlacedBet
from src.data.redis_cache import cache

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (all DB calls wrapped for async safety)
# ---------------------------------------------------------------------------

def _get_bankroll() -> float:
    try:
        return BankrollManager().get_current_bankroll()
    except Exception:
        return settings.initial_bankroll


def _sync_get_all_bets() -> List[PlacedBet]:
    """Sync DB call — always call via asyncio.to_thread."""
    with SessionLocal() as db:
        return db.scalars(select(PlacedBet)).all()


def _sync_place_bet(payload: dict) -> None:
    """Sync DB call — always call via asyncio.to_thread."""
    with SessionLocal() as db:
        db.add(
            PlacedBet(
                event_id=str(payload["event_id"]),
                sport=str(payload["sport"]),
                market=str(payload["market"]),
                selection=str(payload["selection"]),
                odds=float(payload["odds"]),
                stake=float(payload["stake"]),
                status="open",
            )
        )
        db.commit()


MAIN_MENU = ReplyKeyboardMarkup(
    [
        ["Heutige Value Bets", "Kombi-Vorschläge"],
        ["Daten aktualisieren", "Kontostand"],
        ["Einstellungen", "Hilfe"],
    ],
    resize_keyboard=True,
)


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

def _progress_bar(value: float, width: int = 10) -> str:
    """Render a visual progress bar: [████████░░] 80%"""
    filled = int(round(value * width))
    empty = width - filled
    bar = "█" * filled + "░" * empty
    return f"[{bar}] {value:.0%}"


def _calibration_badge(model_prob: float) -> str:
    """Return a calibration quality badge based on reliability bin data."""
    try:
        from src.core.ml_trainer import get_reliability_adjustment
        adj = get_reliability_adjustment(model_prob)
        if 0.90 <= adj <= 1.10:
            return "🟢"  # well-calibrated
        elif 0.75 <= adj <= 1.25:
            return "🟡"  # moderate
        else:
            return "🟠"  # high variance
    except Exception:
        if 0.35 <= model_prob <= 0.65:
            return "🟢"
        elif 0.25 <= model_prob <= 0.75:
            return "🟡"
        return "🟠"


def _retail_trap_badge(bet: dict) -> str:
    """Return retail trap warning if public bias is significant."""
    bias = float(bet.get("public_bias", 0))
    if bias > 0.03:
        return " ⚠️ Retail Trap"
    elif bias > 0.02:
        return " 🔶 Public Bias"
    return ""


# ---------------------------------------------------------------------------
# Cache / state helpers
# ---------------------------------------------------------------------------

def _store_mark_payload(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True)
    mid = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    cache.set_json(f"mark:{mid}", payload, ttl_seconds=24 * 3600)
    return mid


def _bet_window_now():
    tz = ZoneInfo("Europe/Berlin")
    local = datetime.now(timezone.utc).astimezone(tz)
    start = local.replace(hour=7, minute=0, second=0, microsecond=0)
    if local < start:
        start = start - timedelta(days=1)
    end = start + timedelta(days=1)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def _placed_cache_key() -> str:
    start, _ = _bet_window_now()
    local_day = start.astimezone(ZoneInfo("Europe/Berlin")).date().isoformat()
    return f"placed:user:{local_day}"


def _placed_keys_today() -> set:
    return set(cache.get_json(_placed_cache_key()) or [])


def _fetched_within(hours: int = 4) -> bool:
    _, ts = get_cached_signals()
    if not ts:
        return False
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - t).total_seconds() < hours * 3600
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Value Bets: Inline Pagination
# ---------------------------------------------------------------------------

def _format_signal_card(b: dict, index: int, total: int) -> str:
    """Format a single signal as a rich card with progress bar and badges."""
    sport = str(b.get("sport", "")).replace("_", " ").upper()
    model_p = float(b.get("model_probability", 0))
    ev = float(b.get("expected_value", 0))
    odds = float(b.get("bookmaker_odds", 0))
    stake = float(b.get("recommended_stake", 0))
    conf = float(b.get("confidence", 0))
    source = b.get("source_mode", "n/a")
    ref = b.get("reference_book", "n/a")

    badge = _calibration_badge(model_p)
    trap = _retail_trap_badge(b)

    return (
        f"🎯 Signal {index + 1}/{total} | {sport}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Tipp: {b['selection']}\n"
        f"Quote: {odds:.2f}\n"
        f"Modell: {_progress_bar(model_p)} {badge}{trap}\n"
        f"EV: {ev:+.4f} | Einsatz: {stake:.2f} EUR\n"
        f"Conf: {conf:.0%} | {source} · {ref}"
    )


def _signal_nav_keyboard(index: int, total: int, bet_data: Optional[dict] = None) -> InlineKeyboardMarkup:
    """Build inline navigation keyboard for signal pagination."""
    nav_row = []
    if index > 0:
        nav_row.append(InlineKeyboardButton("⬅️ Prev", callback_data=f"sig_nav:{index - 1}"))
    nav_row.append(InlineKeyboardButton(f"{index + 1}/{total}", callback_data="sig_noop"))
    if index < total - 1:
        nav_row.append(InlineKeyboardButton("Next ➡️", callback_data=f"sig_nav:{index + 1}"))

    action_row = []
    if bet_data:
        mark_id = _store_mark_payload(bet_data)
        action_row.append(InlineKeyboardButton("✅ Als platziert", callback_data=f"markid:{mark_id}"))

    rows = [nav_row]
    if action_row:
        rows.append(action_row)
    return InlineKeyboardMarkup(rows)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Multi-Sport Signal Bot bereit.\n"
        "Tippe auf einen Button oder nutze /status.",
        reply_markup=MAIN_MENU,
    )


async def menu_value_bets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show value bets with inline pagination (one card at a time)."""
    items, ts = get_cached_signals()

    if not items:
        engine = BettingEngine(bankroll=await asyncio.to_thread(_get_bankroll))
        items = [
            x.model_dump()
            for x in engine.rank_value_bets([
                engine.make_signal("basketball", "nba_2026_001", "h2h",
                                   "Los Angeles Lakers vs Phoenix Suns", 2.10, 0.56),
                engine.make_signal("tennis", "atp_2026_044", "h2h",
                                   "Carlos Alcaraz vs Casper Ruud", 1.55, 0.71),
            ])
        ]

    placed = _placed_keys_today()
    items = [
        x for x in items
        if float(x.get("expected_value", 0)) > 0
        and float(x.get("recommended_stake", 0)) > 0
        and f"{x.get('event_id')}|{x.get('selection')}" not in placed
    ]

    if not items:
        await update.message.reply_text("Keine spielbaren Value Bets (EV<=0 oder bereits platziert).")
        return

    items = items[:10]
    context.user_data["signal_items"] = items

    header = f"📊 {len(items)} Value Bets"
    if ts:
        header += f" | Stand: {ts[:16]}"
    await update.message.reply_text(header)

    # Show first signal with navigation
    card = _format_signal_card(items[0], 0, len(items))
    bet_data = {
        "event_id": items[0].get("event_id", ""),
        "sport": items[0].get("sport", ""),
        "market": items[0].get("market", "h2h"),
        "selection": items[0].get("selection", ""),
        "odds": float(items[0].get("bookmaker_odds", 0)),
        "stake": float(items[0].get("recommended_stake", 0)),
    }
    keyboard = _signal_nav_keyboard(0, len(items), bet_data)
    await update.message.reply_text(card, reply_markup=keyboard)


# ---------------------------------------------------------------------------
# Combo Suggestions: Rich Display
# ---------------------------------------------------------------------------

def _format_combo_card(combo_data: dict) -> str:
    """Format a combo as a rich card with legs table and progress bar."""
    size = combo_data.get("size", 0)
    combo_type = combo_data.get("type", "lotto")
    stake = float(combo_data.get("stake", 1.00))
    combined_odds = float(combo_data.get("combined_odds", 0))
    combined_prob = float(combo_data.get("combined_probability", 0))
    ev = float(combo_data.get("expected_value", 0))
    legs = combo_data.get("legs", [])

    potential_payout = round(stake * combined_odds, 2)
    is_playable = ev > 0
    type_label = "EV-Optimal" if combo_type == "ev_optimal" else "Lotto"
    icon = "🎯" if combo_type == "ev_optimal" else "🧩"

    legs_lines = []
    for i, leg in enumerate(legs):
        sel = str(leg.get("selection", "?"))[:30]
        odds = float(leg.get("odds", 0))
        prob = float(leg.get("probability", 0))
        legs_lines.append(f" {i+1:2d}. {sel:<30s} {odds:5.2f}  {prob:.0%}")
    legs_txt = "\n".join(legs_lines)

    tax_badge = " 🏷️ Steuerfrei" if len(legs) >= 3 else ""

    return (
        f"{icon} KOMBI {size}er | {type_label}{tax_badge}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Status: {'✅ spielbar' if is_playable else '⚠️ watchlist'}\n"
        f"Gesamtquote: {combined_odds:.2f} ({len(legs)} Legs)\n"
        f"Wahrscheinlichkeit: {_progress_bar(combined_prob)}\n"
        f"💰 Einsatz: {stake:.2f} EUR\n"
        f"💎 Möglicher Gewinn: {potential_payout:.2f} EUR\n"
        f"EV: {ev:+.4f}\n"
        f"\nTipps:\n{legs_txt}"
    )


async def combo_suggestions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from src.core.live_feed import get_cached_combos
    combos = get_cached_combos()

    if not combos:
        try:
            await asyncio.to_thread(lambda: fetch_and_build_signals())
        except Exception:
            pass
        combos = get_cached_combos()

    if not combos:
        await update.message.reply_text("Keine Kombi-Daten vorhanden. Bitte später erneut versuchen.")
        return

    await update.message.reply_text(f"🧩 {len(combos)} Kombi-Vorschläge")
    for combo_data in combos:
        card = _format_combo_card(combo_data)
        await update.message.reply_text(card)


# ---------------------------------------------------------------------------
# Balance: Visual Dashboard
# ---------------------------------------------------------------------------

async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show balance with a visual PnL dashboard chart."""
    rows = await asyncio.to_thread(_sync_get_all_bets)
    settled = [r for r in rows if r.status in {"won", "lost"}]
    pnl = round(sum(float(r.pnl or 0) for r in settled), 2)
    open_n = sum(1 for r in rows if r.status == "open")
    won_n = sum(1 for r in settled if r.status == "won")
    lost_n = sum(1 for r in settled if r.status == "lost")
    current_bankroll = await asyncio.to_thread(_get_bankroll)
    initial = settings.initial_bankroll
    staked = sum(float(r.stake or 0) for r in settled)
    roi = pnl / max(1.0, staked) if staked > 0 else 0.0

    # Build equity curve from settled bets
    equity = [initial]
    running = initial
    for r in sorted(settled, key=lambda x: x.id if hasattr(x, "id") else 0):
        running += float(r.pnl or 0)
        equity.append(round(running, 2))

    # Try to generate and send chart
    chart_sent = False
    if len(equity) > 2:
        try:
            from src.utils.charts import generate_dashboard
            png_bytes = await asyncio.to_thread(
                generate_dashboard,
                equity_curve=equity,
                wins=won_n,
                losses=lost_n,
                open_bets=open_n,
                initial_bankroll=initial,
                pnl=pnl,
                roi=roi,
            )
            await update.message.reply_photo(
                photo=io.BytesIO(png_bytes),
                caption=(
                    f"📊 Dashboard | Bankroll: {current_bankroll:.2f} EUR\n"
                    f"PnL: {pnl:+.2f} EUR | ROI: {roi:.1%}\n"
                    f"W/L: {won_n}/{lost_n} | Open: {open_n}"
                ),
            )
            chart_sent = True
        except Exception as exc:
            log.warning("Chart generation failed: %s", exc)

    if not chart_sent:
        msg = (
            f"📊 Dashboard\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Bankroll: {current_bankroll:.2f} EUR\n"
            f"PnL: {pnl:+.2f} EUR | ROI: {roi:.1%}\n"
            f"Won: {won_n} | Lost: {lost_n} | Open: {open_n}\n"
            f"Hit Rate: {_progress_bar(won_n / max(1, won_n + lost_n))}"
        )
        await update.message.reply_text(msg)


# ---------------------------------------------------------------------------
# Callback Handler: Pagination + Mark + Agent Actions
# ---------------------------------------------------------------------------

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""

    # --- Signal pagination ---
    if data.startswith("sig_nav:"):
        try:
            idx = int(data.split(":")[1])
            items = context.user_data.get("signal_items", [])
            if not items or idx < 0 or idx >= len(items):
                await q.edit_message_text("Signal nicht mehr verfügbar.")
                return
            card = _format_signal_card(items[idx], idx, len(items))
            bet_data = {
                "event_id": items[idx].get("event_id", ""),
                "sport": items[idx].get("sport", ""),
                "market": items[idx].get("market", "h2h"),
                "selection": items[idx].get("selection", ""),
                "odds": float(items[idx].get("bookmaker_odds", 0)),
                "stake": float(items[idx].get("recommended_stake", 0)),
            }
            keyboard = _signal_nav_keyboard(idx, len(items), bet_data)
            await q.edit_message_text(card, reply_markup=keyboard)
        except Exception:
            await q.edit_message_text("Navigation fehlgeschlagen.")
        return

    if data == "sig_noop":
        return

    # --- Refresh ---
    if data.startswith("refresh:"):
        if _fetched_within(4):
            await q.edit_message_text("Letzter Fetch < 4h. Bitte 'Daten aktualisieren' nutzen.")
            return
        try:
            n = await asyncio.to_thread(lambda: len(fetch_and_build_signals()))
            await q.edit_message_text(f"Quoten aktualisiert. Signals={n}")
        except Exception:
            await q.edit_message_text("Aktualisierung fehlgeschlagen.")
        return

    # --- Mark as placed (async DB) ---
    if data.startswith("markid:"):
        mark_id = data.split(":", 1)[1]
        payload = cache.get_json(f"mark:{mark_id}")
        if not payload:
            await q.edit_message_text("Markierung abgelaufen. Bitte Signal neu laden.")
            return
        await asyncio.to_thread(_sync_place_bet, payload)
        key = f"{payload['event_id']}|{payload['selection']}"
        ck = _placed_cache_key()
        cur = set(cache.get_json(ck) or [])
        cur.add(key)
        cache.set_json(ck, sorted(cur), ttl_seconds=2 * 24 * 3600)
        await q.edit_message_text("Als platziert gespeichert ✅")
        return

    # --- Agent alert: Ask Analyst for deep dive ---
    if data.startswith("agent_analyze:"):
        alert_id = data.split(":", 1)[1]
        alert_data = cache.get_json(f"agent_alert:{alert_id}")
        if not alert_data:
            await q.edit_message_text("Alert abgelaufen.")
            return
        await q.edit_message_text("🔍 Analyst wird befragt...")
        try:
            from src.agents.analyst_agent import AnalystAgent
            analyst = AnalystAgent()
            analysis = await analyst.analyze_event(
                event_id=alert_data.get("event_id", ""),
                sport=alert_data.get("sport", ""),
                home=alert_data.get("home", ""),
                away=alert_data.get("away", ""),
                selection=alert_data.get("selection", ""),
                target_odds=float(alert_data.get("target_odds", 2.0)),
                sharp_odds=float(alert_data.get("sharp_odds", 2.0)),
                sharp_market={alert_data.get("selection", ""): float(alert_data.get("sharp_odds", 2.0))},
                trigger="user_deepdive",
            )
            reasoning = await analyst.reason_with_llm(analysis) or ""
            model_p = float(analysis.get("model_probability", 0))
            ev = float(analysis.get("expected_value", 0))
            badge = _calibration_badge(model_p)
            msg = (
                f"🔬 Deep Dive | {alert_data.get('home', '')} vs {alert_data.get('away', '')}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Modell: {_progress_bar(model_p)} {badge}\n"
                f"EV: {ev:+.4f}\n"
                f"Elo: {analysis.get('elo', {}).get('elo_diff', 0):+.0f}\n"
                f"Form: H={analysis.get('form', {}).get('home_wr', 0):.0%} "
                f"A={analysis.get('form', {}).get('away_wr', 0):.0%}\n"
                f"Poisson: {analysis.get('poisson_prob', 'n/a')}\n"
            )
            if reasoning:
                msg += f"\n💡 {reasoning}"
            msg += f"\n\nEmpfehlung: {analysis.get('recommendation', 'SKIP')}"
            await q.message.reply_text(msg)
        except Exception as exc:
            await q.message.reply_text(f"Analyse fehlgeschlagen: {type(exc).__name__}")
        return

    # --- Agent alert: Ghost Bet ---
    if data.startswith("agent_ghost:"):
        alert_id = data.split(":", 1)[1]
        alert_data = cache.get_json(f"agent_alert:{alert_id}")
        if not alert_data:
            await q.edit_message_text("Alert abgelaufen.")
            return
        try:
            from src.core.ghost_trading import place_virtual_bet
            await asyncio.to_thread(
                place_virtual_bet,
                event_id=str(alert_data.get("event_id", "")),
                sport=str(alert_data.get("sport", "")),
                market="h2h",
                selection=str(alert_data.get("selection", "")),
                odds=float(alert_data.get("target_odds", 2.0)),
                stake=float(alert_data.get("stake", 1.0)),
                features={},
            )
            await q.edit_message_text("💰 Ghost Bet platziert ✅")
        except Exception:
            await q.edit_message_text("Ghost Bet fehlgeschlagen.")
        return

    # --- Agent alert: Ignore ---
    if data.startswith("agent_ignore:"):
        await q.edit_message_text("🛑 Alert ignoriert.")
        return


# ---------------------------------------------------------------------------
# Agentic Chat Mode: Reply to any message to ask the LLM
# ---------------------------------------------------------------------------

async def agentic_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle free-text questions about betting decisions.

    Users can ask "Why is the model fading the Lakers tonight?" and the bot
    routes the question to the Analyst LLM for a natural language answer.
    """
    text = (update.message.text or "").strip()
    if not text or len(text) < 10:
        return

    question_markers = ["?", "why", "warum", "explain", "erkläre", "how", "wie", "what", "was"]
    if not any(m in text.lower() for m in question_markers):
        return

    await update.message.reply_text("🤔 Analysiere deine Frage...")

    try:
        from src.integrations.ollama_sentiment import OllamaSentimentClient
        nlp = OllamaSentimentClient()

        items, _ = get_cached_signals()
        context_str = ""
        if items:
            for b in items[:3]:
                context_str += (
                    f"- {b.get('selection')}: odds={b.get('bookmaker_odds')}, "
                    f"model_p={b.get('model_probability')}, ev={b.get('expected_value')}\n"
                )

        prompt = (
            f"The user asks about their betting bot: \"{text}\"\n\n"
            f"Current signals:\n{context_str}\n"
            f"Give a concise, helpful answer in 2-3 sentences. "
            f"Focus on the model features and EV reasoning."
        )

        result = await asyncio.to_thread(nlp.analyze, text=prompt, context="betting_qa")
        answer = f"💡 {result.label}" if hasattr(result, "label") else "Keine Antwort verfügbar."
        await update.message.reply_text(answer)
    except ImportError:
        await update.message.reply_text("LLM nicht verfügbar. Stelle sicher, dass Ollama läuft.")
    except Exception as exc:
        await update.message.reply_text(f"Analyse fehlgeschlagen: {type(exc).__name__}")


# ---------------------------------------------------------------------------
# Refresh / Settings / Help
# ---------------------------------------------------------------------------

async def _refresh_job(bot, chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    try:
        items = await asyncio.wait_for(
            asyncio.to_thread(lambda: fetch_and_build_signals()),
            timeout=120,
        )
        meta = get_cached_meta()
        await bot.send_message(
            chat_id=chat_id,
            text=(
                f"📡 Daten aktualisiert\n"
                f"Signals: {len(items)} | Sports: {meta.get('sports_expanded', 0)} "
                f"| Events: {meta.get('events_seen', 0)}"
            ),
        )
    except asyncio.TimeoutError:
        await bot.send_message(chat_id=chat_id, text="⏰ Datenupdate Timeout (120s).")
    except Exception as e:
        await bot.send_message(chat_id=chat_id, text=f"❌ Fehler: {type(e).__name__}")
    finally:
        context.application.bot_data["refresh_running"] = False


async def refresh_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.application.bot_data.get("refresh_running", False):
        await update.message.reply_text("⏳ Refresh läuft bereits...")
        return
    context.application.bot_data["refresh_running"] = True
    await update.message.reply_text("🔄 Aktualisiere Daten im Hintergrund...")
    context.application.create_task(_refresh_job(context.bot, update.effective_chat.id, context))


async def settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bankroll = await asyncio.to_thread(_get_bankroll)
    tax_mode = "Steuerfrei" if settings.tax_free_mode else f"{settings.tipico_tax_rate:.0%} Steuer"
    twitter_status = "✅ Aktiv" if settings.twitter_enabled else "❌ Inaktiv"
    msg = (
        f"⚙️ Einstellungen\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Bankroll: {bankroll:.2f} EUR\n"
        f"Steuer: {tax_mode}\n"
        f"Twitter/X: {twitter_status}\n"
        f"Enrichment: {'✅' if settings.enrichment_enabled else '❌'}\n"
        f"Sports: {settings.live_sports}\n"
        f"Fetch: 07:00 + 13:00 | Agent: 60s/5min\n"
        f"Auto-Grading: ✅ aktiv"
    )
    await update.message.reply_text(msg)


async def help_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "📖 Hilfe\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🎯 Heutige Value Bets — Top 10 mit Inline-Navigation\n"
        "🧩 Kombi-Vorschläge — 5/10/20/30er Kombis\n"
        "📊 Kontostand — PnL-Dashboard mit Chart\n"
        "🔄 Daten aktualisieren — Odds-Refresh\n"
        "⚙️ Einstellungen — Bot-Konfiguration\n\n"
        "💬 Frage stellen: Einfach eine Frage tippen!\n"
        "z.B. \"Warum empfiehlt das Modell diesen Tipp?\"\n\n"
        "🤖 Agent Alerts haben interaktive Buttons:\n"
        "  🔍 Deep Dive — Analyst-Analyse\n"
        "  💰 Ghost Bet — Virtuelle Wette\n"
        "  🛑 Ignorieren — Alert verwerfen"
    )
    await update.message.reply_text(msg)


# ---------------------------------------------------------------------------
# Push daily signals (called by scheduler)
# ---------------------------------------------------------------------------

async def push_daily_signals(bot, chat_id: str):
    items, ts = get_cached_signals()

    if ts:
        await bot.send_message(chat_id=chat_id, text=f"📅 Tages-Push | Datenstand: {ts[:16]}")

    placed = _placed_keys_today()
    singles = [
        x for x in items
        if float(x.get("expected_value", 0)) > 0
        and float(x.get("recommended_stake", 0)) > 0
        and f"{x.get('event_id')}|{x.get('selection')}" not in placed
    ]

    if not singles:
        await bot.send_message(chat_id=chat_id, text="Keine spielbaren Einzelwetten heute.")
    else:
        await bot.send_message(chat_id=chat_id, text=f"🎯 {len(singles[:10])} Value Bets")
        for b in singles[:10]:
            sport = str(b.get("sport", "")).replace("_", " ").upper()
            model_p = float(b.get("model_probability", 0))
            badge = _calibration_badge(model_p)
            msg = (
                f"🎯 {sport} {badge}\n"
                f"Tipp: {b['selection']}\n"
                f"Quote: {float(b['bookmaker_odds']):.2f} | "
                f"Modell: {_progress_bar(model_p)}\n"
                f"EV: {float(b['expected_value']):+.4f} | "
                f"Einsatz: {float(b['recommended_stake']):.2f} EUR"
            )
            await bot.send_message(chat_id=chat_id, text=msg)

    from src.core.live_feed import get_cached_combos
    cached_combos = get_cached_combos()

    if not cached_combos:
        await bot.send_message(chat_id=chat_id, text="Keine Kombi-Vorschläge heute.")
        return

    await bot.send_message(chat_id=chat_id, text=f"🧩 {len(cached_combos)} Kombi-Vorschläge")
    for combo_data in cached_combos:
        if float(combo_data.get("expected_value", 0)) <= 0:
            continue
        card = _format_combo_card(combo_data)
        await bot.send_message(chat_id=chat_id, text=card)
