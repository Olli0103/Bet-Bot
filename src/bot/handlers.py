import asyncio
import hashlib
import json
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ContextTypes
from sqlalchemy import select

from src.core.bankroll import BankrollManager
from src.core.betting_engine import BettingEngine
from src.core.live_feed import fetch_and_build_signals, get_cached_combo_legs, get_cached_meta, get_cached_signals
from src.core.settings import settings
from src.data.postgres import SessionLocal
from src.data.models import PlacedBet
from src.data.redis_cache import cache


def _get_bankroll() -> float:
    try:
        return BankrollManager().get_current_bankroll()
    except Exception:
        return settings.initial_bankroll

MAIN_MENU = ReplyKeyboardMarkup(
    [["Heutige Value Bets", "Kombi-Vorschläge"], ["Daten aktualisieren", "Kontostand"], ["Einstellungen", "Hilfe"]],
    resize_keyboard=True,
)


def sample_signals():
    engine = BettingEngine(bankroll=_get_bankroll())
    return engine.rank_value_bets(
        [
            engine.make_signal("basketball", "nba_2026_001", "h2h", "Los Angeles Lakers vs Phoenix Suns", 2.10, 0.56),
            engine.make_signal("tennis", "atp_2026_044", "h2h", "Carlos Alcaraz vs Casper Ruud", 1.55, 0.71),
        ]
    )


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
    # Only user-confirmed placed bets (not auto-ghost inserts)
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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Multi-Sport Signal Bot bereit.", reply_markup=MAIN_MENU)


async def menu_value_bets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    items, ts = get_cached_signals()
    if not items:
        items = [x.model_dump() for x in sample_signals()]

    placed = _placed_keys_today()
    # only meaningful singles + not already confirmed as placed today
    items = [
        x
        for x in items
        if float(x.get("expected_value", 0)) > 0
        and float(x.get("recommended_stake", 0)) > 0
        and f"{x.get('event_id')}|{x.get('selection')}" not in placed
    ]

    if not items:
        await update.message.reply_text("Keine spielbaren Value Bets (EV<=0 oder Stake=0).")
        return

    if ts:
        await update.message.reply_text(f"Datenstand: {ts}")

    for b in items[:10]:
        sport = str(b.get('sport', '')).replace('_', ' ').upper()
        msg = (
            f"🎯 {sport}\n"
            f"Tipp: {str(b['selection'])}\n"
            f"Quote: {float(b['bookmaker_odds']):.2f} | Modell: {float(b['model_probability']):.2%}\n"
            f"EV: {float(b['expected_value']):.4f} | Einsatz: {float(b['recommended_stake']):.2f} EUR\n"
            f"Quelle: {b.get('source_mode','n/a')} · Ref: {b.get('reference_book','n/a')} · Conf: {float(b.get('confidence',0)):.2f}"
        )

        await update.message.reply_text(msg)


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""

    if data.startswith("refresh:"):
        # Guard: avoid hidden fetches from non-primary buttons for 4h after last fetch.
        if _fetched_within(4):
            await q.edit_message_text("Letzter Fetch < 4h. Bitte den Button 'Daten aktualisieren' nutzen.")
            return
        try:
            n = await asyncio.to_thread(lambda: len(fetch_and_build_signals()))
            await q.edit_message_text(f"Quoten wurden aktualisiert. Signals={n}")
        except Exception:
            await q.edit_message_text("Aktualisierung fehlgeschlagen.")
        return

    if data.startswith("markid:"):
        mark_id = data.split(":", 1)[1]
        payload = cache.get_json(f"mark:{mark_id}")
        if not payload:
            await q.edit_message_text("Markierung abgelaufen. Bitte Signal neu laden.")
            return
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

        key = f"{payload['event_id']}|{payload['selection']}"
        ck = _placed_cache_key()
        cur = set(cache.get_json(ck) or [])
        cur.add(key)
        cache.set_json(ck, sorted(cur), ttl_seconds=2 * 24 * 3600)

        await q.edit_message_text("Als platziert gespeichert ✅")


async def combo_suggestions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    legs = get_cached_combo_legs()
    if not legs:
        # self-heal: build cache on demand
        try:
            await asyncio.to_thread(lambda: fetch_and_build_signals())
        except Exception:
            pass
        legs = get_cached_combo_legs()
    if not legs:
        await update.message.reply_text("Keine Kombi-Daten vorhanden. Bitte später erneut versuchen.")
        return

    engine = BettingEngine(bankroll=_get_bankroll())

    # one combo per sport, same-sport only, fixed 1 EUR stake (lotto mode)
    placed = _placed_keys_today()
    by_sport = {}
    for l in legs:
        if f"{l.get('event_id')}|{l.get('selection')}" in placed:
            continue
        s = str(l.get("sport", ""))
        by_sport.setdefault(s, []).append(l)

    for sport, sport_legs in sorted(by_sport.items()):
        # high probability first
        ranked = sorted(sport_legs, key=lambda x: float(x.get("probability", 0)), reverse=True)

        # practical target sizes
        target_size = 10 if len(ranked) >= 10 else 5 if len(ranked) >= 5 else 3 if len(ranked) >= 3 else 2 if len(ranked) >= 2 else 0
        if target_size == 0:
            sport_label = sport.replace("_", " ").upper()
            await update.message.reply_text(f"{sport_label}: nicht genug Legs für Kombi (mind. 2).")
            continue

        chosen = []
        seen_events = set()
        for l in ranked:
            if l["event_id"] in seen_events:
                continue
            chosen.append(l)
            seen_events.add(l["event_id"])
            if len(chosen) == target_size:
                break

        if len(chosen) < 2:
            sport_label = sport.replace("_", " ").upper()
            await update.message.reply_text(f"{sport_label}: nach Event-Filter nicht genug eindeutige Legs (mind. 2).")
            continue

        combo = engine.build_combo(chosen, correlation_penalty=0.70, kelly_frac=0.02)
        fixed_stake = 1.00

        sport_label = sport.replace("_", " ").upper()
        legs_txt = "\n".join(
            f"{i+1}. {leg.selection}  |  Quote {float(leg.odds):.2f}" for i, leg in enumerate(combo.legs)
        )
        is_playable = float(combo.expected_value) > 0
        status = "✅ spielbar" if is_playable else "⚠️ watchlist (kein +EV)"

        msg = (
            f"🧩 KOMBI • {sport_label}\n"
            f"Status: {status}\n"
            f"Legs: {len(combo.legs)}\n"
            f"Gesamtquote: {float(combo.combined_odds):.2f}\n"
            f"Modell-P: {float(combo.combined_probability):.2%}\n"
            f"EV: {float(combo.expected_value):.4f}\n"
            f"Einsatz: {fixed_stake:.2f} EUR\n\n"
            f"Tipps:\n{legs_txt}"
        )
        await update.message.reply_text(msg)


async def _refresh_job(bot, chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    try:
        items = await asyncio.wait_for(
            asyncio.to_thread(lambda: fetch_and_build_signals()),
            timeout=120,
        )
        meta = get_cached_meta()
        await bot.send_message(
            chat_id=chat_id,
            text=f"Daten aktualisiert: {len(items)} Signals | Sports: {meta.get('sports_expanded',0)} | Events: {meta.get('events_seen',0)}",
        )
    except asyncio.TimeoutError:
        await bot.send_message(chat_id=chat_id, text="Datenupdate Timeout (120s). Bitte später erneut versuchen.")
    except Exception as e:
        await bot.send_message(chat_id=chat_id, text=f"Datenupdate fehlgeschlagen: {type(e).__name__}")
    finally:
        context.application.bot_data["refresh_running"] = False


async def refresh_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.application.bot_data.get("refresh_running", False):
        await update.message.reply_text("Refresh läuft bereits. Bitte kurz warten…")
        return

    context.application.bot_data["refresh_running"] = True
    await update.message.reply_text("Aktualisiere Daten… (läuft im Hintergrund)")
    context.application.create_task(_refresh_job(context.bot, update.effective_chat.id, context))


async def settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Einstellungen\n"
        "• Fetch-Schedule: 07:00 und 13:00\n"
        "• Auto-Grading: aktiv\n"
        "• Modus: Live-API (Odds) + Fallback"
    )
    await update.message.reply_text(msg)


async def help_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Hilfe\n"
        "• Heutige Value Bets: Top 10 Einzelwetten\n"
        "• Kombi-Vorschläge: Top 1 für 5/10/20/30er (jeweils eigene Nachricht)\n"
        "• Daten aktualisieren: on-demand Fetch (rate-limit-schonend via Cache)\n"
        "• Als platziert markieren: speichert Wette lokal\n"
        "• Kontostand: zeigt Auto-Grading PnL"
    )
    await update.message.reply_text(msg)


async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with SessionLocal() as db:
        rows = db.scalars(select(PlacedBet)).all()
    pnl = round(sum(r.pnl for r in rows if r.status in {"won", "lost"}), 2)
    open_n = sum(1 for r in rows if r.status == "open")
    won_n = sum(1 for r in rows if r.status == "won")
    lost_n = sum(1 for r in rows if r.status == "lost")
    current_bankroll = _get_bankroll()

    msg = (
        "Auto-Grading Status\n"
        f"Bankroll: {current_bankroll:.2f} EUR\n"
        f"Open: {open_n}\n"
        f"Won: {won_n}\n"
        f"Lost: {lost_n}\n"
        f"PnL: {pnl:.2f} EUR"
    )
    await update.message.reply_text(msg)


async def push_daily_signals(bot, chat_id: str):
    items, ts = get_cached_signals()
    combos = get_cached_combo_legs()

    if ts:
        await bot.send_message(chat_id=chat_id, text=f"📅 Tages-Push | Datenstand: {ts}")

    placed = _placed_keys_today()
    singles = [
        x
        for x in items
        if float(x.get("expected_value", 0)) > 0
        and float(x.get("recommended_stake", 0)) > 0
        and f"{x.get('event_id')}|{x.get('selection')}" not in placed
    ]
    if not singles:
        await bot.send_message(chat_id=chat_id, text="Keine spielbaren Einzelwetten für heute.")
    else:
        await bot.send_message(chat_id=chat_id, text="🎯 Einzelwetten Top 10")
        for b in singles[:10]:
            sport = str(b.get('sport', '')).replace('_', ' ').upper()
            msg = (
                f"🎯 {sport}\n"
                f"Tipp: {str(b['selection'])}\n"
                f"Quote: {float(b['bookmaker_odds']):.2f} | Modell: {float(b['model_probability']):.2%}\n"
                f"EV: {float(b['expected_value']):.4f} | Einsatz: {float(b['recommended_stake']):.2f} EUR"
            )
            await bot.send_message(chat_id=chat_id, text=msg)

    # one combo per sport
    by_sport = {}
    for l in combos:
        if f"{l.get('event_id')}|{l.get('selection')}" in placed:
            continue
        s = str(l.get("sport", ""))
        by_sport.setdefault(s, []).append(l)

    if not by_sport:
        await bot.send_message(chat_id=chat_id, text="Keine Kombi-Legs für heute vorhanden.")
        return

    await bot.send_message(chat_id=chat_id, text="🧩 Kombi-Vorschläge je Sport")
    engine = BettingEngine(bankroll=_get_bankroll())
    for sport, sport_legs in sorted(by_sport.items()):
        ranked = sorted(sport_legs, key=lambda x: float(x.get("probability", 0)), reverse=True)
        target_size = 10 if len(ranked) >= 10 else 5 if len(ranked) >= 5 else 3 if len(ranked) >= 3 else 2 if len(ranked) >= 2 else 0
        if target_size == 0:
            sport_label = sport.replace("_", " ").upper()
            await bot.send_message(chat_id=chat_id, text=f"{sport_label}: nicht genug Legs (min 2).")
            continue

        chosen, seen = [], set()
        for l in ranked:
            if l['event_id'] in seen:
                continue
            chosen.append(l)
            seen.add(l['event_id'])
            if len(chosen) == target_size:
                break

        if len(chosen) < 2:
            sport_label = sport.replace("_", " ").upper()
            await bot.send_message(chat_id=chat_id, text=f"{sport_label}: nach Event-Filter nicht genug eindeutige Legs (mind. 2).")
            continue

        combo = engine.build_combo(chosen, correlation_penalty=0.70, kelly_frac=0.02)
        sport_label = sport.replace("_", " ").upper()
        legs_txt = "\n".join(
            f"{i+1}. {leg.selection}  |  Quote {float(leg.odds):.2f}" for i, leg in enumerate(combo.legs)
        )
        if float(combo.expected_value) <= 0:
            continue

        msg = (
            f"🧩 KOMBI • {sport_label}\n"
            f"Status: ✅ spielbar\n"
            f"Legs: {len(combo.legs)}\n"
            f"Gesamtquote: {combo.combined_odds:.2f}\n"
            f"Modell-P: {combo.combined_probability:.2%}\n"
            f"EV: {combo.expected_value:.4f} | Einsatz: 1.00 EUR\n\n"
            f"Tipps:\n{legs_txt}"
        )
        await bot.send_message(chat_id=chat_id, text=msg)
