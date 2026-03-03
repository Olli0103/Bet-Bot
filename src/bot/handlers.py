"""Telegram bot handlers with interactive inline keyboards, visual dashboards,
dynamic settings toggle dashboard, NLP intent routing, and agentic chat mode.

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
    get_all_ranked_signals,
    get_cached_combo_legs,
    get_cached_meta,
    get_cached_signals,
    run_enrichment_pass,
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
        ["Heutige Top 10 Einzelwetten", "10/20/30 Kombis"],
        ["Daten aktualisieren", "Kontostand"],
        ["Einstellungen", "Hilfe"],
    ],
    resize_keyboard=True,
)


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

def _progress_bar(value: float, width: int = 10) -> str:
    """Render a visual progress bar using ASCII: [########--] 80%"""
    filled = int(round(value * width))
    empty = width - filled
    bar = "#" * filled + "-" * empty
    return f"[{bar}] {value:.0%}"


def _calibration_badge(model_prob: float) -> str:
    """Return a calibration quality badge based on reliability bin data."""
    try:
        from src.core.ml_trainer import get_reliability_adjustment
        adj = get_reliability_adjustment(model_prob)
        if 0.90 <= adj <= 1.10:
            return "[OK]"  # well-calibrated
        elif 0.75 <= adj <= 1.25:
            return "[~]"  # moderate
        else:
            return "[!]"  # high variance
    except Exception:
        if 0.35 <= model_prob <= 0.65:
            return "[OK]"
        elif 0.25 <= model_prob <= 0.75:
            return "[~]"
        return "[!]"


def _retail_trap_badge(bet: dict) -> str:
    """Return retail trap warning if public bias is significant."""
    bias = float(bet.get("public_bias", 0))
    if bias > 0.03:
        return " [!TRAP]"
    elif bias > 0.02:
        return " [BIAS]"
    return ""


def _format_form_blocks(form_str: str) -> str:
    """Render a form string (e.g. 'WDWLW') as colored blocks."""
    mapping = {"W": "🟩", "D": "🟨", "L": "🟥"}
    return "".join(mapping.get(c.upper(), "⬜") for c in form_str)


def _format_stats_card(
    home: str,
    away: str,
    home_stats: dict,
    away_stats: dict,
    sport: str = "",
) -> str:
    """Format a Tipico-style stats overview card for a match.

    Shows form blocks, goals for/against, O/U rate, BTTS rate,
    attack/defense strength, and rest days.
    """
    # Form blocks
    h_form = home_stats.get("form_str", "")
    a_form = away_stats.get("form_str", "")
    h_form_blocks = _format_form_blocks(h_form) if h_form else "---"
    a_form_blocks = _format_form_blocks(a_form) if a_form else "---"

    # Win rate
    h_wr = home_stats.get("win_rate", 0)
    a_wr = away_stats.get("win_rate", 0)

    # Goals
    h_gf = home_stats.get("goals_scored_avg", 0.0)
    h_ga = home_stats.get("goals_conceded_avg", 0.0)
    a_gf = away_stats.get("goals_scored_avg", 0.0)
    a_ga = away_stats.get("goals_conceded_avg", 0.0)

    # O/U and BTTS rates
    h_o25 = home_stats.get("over25_rate", 0.0)
    a_o25 = away_stats.get("over25_rate", 0.0)
    h_btts = home_stats.get("btts_rate", 0.0)
    a_btts = away_stats.get("btts_rate", 0.0)

    # Attack/Defense strength
    h_atk = home_stats.get("attack_strength", 1.0)
    h_def = home_stats.get("defense_strength", 1.0)
    a_atk = away_stats.get("attack_strength", 1.0)
    a_def = away_stats.get("defense_strength", 1.0)

    # Rest days
    h_rest = home_stats.get("rest_days")
    a_rest = away_stats.get("rest_days")
    h_rest_str = f"{h_rest}d" if h_rest is not None else "?"
    a_rest_str = f"{a_rest}d" if a_rest is not None else "?"

    # League position
    h_pos = home_stats.get("league_position")
    a_pos = away_stats.get("league_position")
    h_pos_str = f"#{h_pos}" if h_pos else ""
    a_pos_str = f"#{a_pos}" if a_pos else ""

    # Home/Away splits
    h_home_wr = home_stats.get("home_win_rate")
    a_away_wr = away_stats.get("away_win_rate")

    lines = [
        f"📊 Match-Statistik",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"{'':>15} {'Heim':>8} {'Ausw.':>8}",
        f"{'Form:':>15} {h_form_blocks:>8} {a_form_blocks:>8}",
    ]

    if h_pos_str or a_pos_str:
        lines.append(f"{'Platz:':>15} {h_pos_str:>8} {a_pos_str:>8}")

    lines.extend([
        f"{'Siege:':>15} {h_wr:>7.0%} {a_wr:>7.0%}",
        f"{'Tore/Spiel:':>15} {h_gf:>7.1f} {a_gf:>7.1f}",
        f"{'Gegentore:':>15} {h_ga:>7.1f} {a_ga:>7.1f}",
        f"{'O2.5-Rate:':>15} {h_o25:>7.0%} {a_o25:>7.0%}",
        f"{'BTTS-Rate:':>15} {h_btts:>7.0%} {a_btts:>7.0%}",
        f"{'Angriff:':>15} {h_atk:>7.2f} {a_atk:>7.2f}",
        f"{'Abwehr:':>15} {h_def:>7.2f} {a_def:>7.2f}",
        f"{'Pause:':>15} {h_rest_str:>8} {a_rest_str:>8}",
    ])

    if h_home_wr is not None or a_away_wr is not None:
        h_hwr_str = f"{h_home_wr:.0%}" if h_home_wr is not None else "?"
        a_awr_str = f"{a_away_wr:.0%}" if a_away_wr is not None else "?"
        lines.append(f"{'Heim/Ausw.:':>15} {h_hwr_str:>8} {a_awr_str:>8}")

    return "\n".join(lines)


def build_stats_for_event(
    event_id: str,
    home: str,
    away: str,
    sport: str,
) -> str:
    """Build a stats card for a specific event using stored snapshots.

    Returns formatted text or a fallback message if no data.
    """
    try:
        from src.core.stats_ingester import get_event_snapshot
        home_snap = get_event_snapshot(event_id, home) or {}
        away_snap = get_event_snapshot(event_id, away) or {}
    except Exception:
        home_snap = {}
        away_snap = {}

    if not home_snap and not away_snap:
        return ""

    # Derive form string from recent results
    def _derive_form(snap: dict) -> str:
        w = snap.get("wins", 0)
        d = snap.get("draws", 0)
        l_ = snap.get("losses", 0)
        n = snap.get("matches_played", 0)
        if n == 0:
            return ""
        # Build approximate form: most recent first (W/D/L proportional)
        form = "W" * w + "D" * d + "L" * l_
        return form[:5]

    home_stats = {
        "form_str": _derive_form(home_snap),
        "win_rate": home_snap.get("wins", 0) / max(1, home_snap.get("matches_played", 1)),
        "goals_scored_avg": home_snap.get("goals_scored_avg", 0.0),
        "goals_conceded_avg": home_snap.get("goals_conceded_avg", 0.0),
        "over25_rate": home_snap.get("over25_rate", 0.0),
        "btts_rate": home_snap.get("btts_rate", 0.0),
        "attack_strength": home_snap.get("attack_strength", 1.0),
        "defense_strength": home_snap.get("defense_strength", 1.0),
        "rest_days": home_snap.get("rest_days"),
        "league_position": home_snap.get("league_position"),
        "home_win_rate": home_snap.get("home_win_rate"),
    }
    away_stats = {
        "form_str": _derive_form(away_snap),
        "win_rate": away_snap.get("wins", 0) / max(1, away_snap.get("matches_played", 1)),
        "goals_scored_avg": away_snap.get("goals_scored_avg", 0.0),
        "goals_conceded_avg": away_snap.get("goals_conceded_avg", 0.0),
        "over25_rate": away_snap.get("over25_rate", 0.0),
        "btts_rate": away_snap.get("btts_rate", 0.0),
        "attack_strength": away_snap.get("attack_strength", 1.0),
        "defense_strength": away_snap.get("defense_strength", 1.0),
        "rest_days": away_snap.get("rest_days"),
        "league_position": away_snap.get("league_position"),
        "away_win_rate": away_snap.get("away_win_rate"),
    }

    return _format_stats_card(home, away, home_stats, away_stats, sport)


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
# Value Bets: Inline Pagination — "Heutige Top 10 Einzelwetten"
# ---------------------------------------------------------------------------

def _format_signal_card(b: dict, index: int, total: int) -> str:
    """Format a single signal as a rich card with all transparency fields."""
    sport = str(b.get("sport", "")).replace("_", " ").upper()
    market = str(b.get("market", "h2h"))
    model_p = float(b.get("model_probability", 0))
    ev = float(b.get("expected_value", 0))
    odds = float(b.get("bookmaker_odds", 0))
    stake = float(b.get("recommended_stake", 0))
    conf = float(b.get("confidence", 0))
    source = b.get("source_mode", "n/a")
    ref = b.get("reference_book", "n/a")

    # Transparency: kelly_raw, stake_before_cap, cap status, trigger
    kelly_raw = float(b.get("kelly_raw", b.get("kelly_fraction", 0)))
    stake_before = float(b.get("stake_before_cap", stake))
    cap_applied = b.get("stake_cap_applied", False)
    trigger = b.get("trigger", "")

    badge = _calibration_badge(model_p)
    trap = _retail_trap_badge(b)

    # Show market type if not plain h2h
    market_tag = f" | {market}" if market != "h2h" else ""
    cap_tag = " [CAP]" if cap_applied else ""
    trigger_tag = f" | trigger={trigger}" if trigger else ""

    return (
        f"Signal {index + 1}/{total} | {sport}{market_tag}\n"
        f"--------------------\n"
        f"Tipp: {b['selection']}\n"
        f"Quote: {odds:.2f}\n"
        f"Modell: {_progress_bar(model_p)} {badge}{trap}\n"
        f"EV: {ev:+.4f} | Conf: {conf:.0%}\n"
        f"Kelly: {kelly_raw:.4f} | Stake: {stake_before:.2f} -> {stake:.2f} EUR{cap_tag}\n"
        f"{source} | {ref}{trigger_tag}"
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

    back_row = [InlineKeyboardButton("↩️ Zurück zur Sportauswahl", callback_data="top10:back")]

    rows = [nav_row]
    if action_row:
        rows.append(action_row)
    rows.append(back_row)
    return InlineKeyboardMarkup(rows)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Multi-Sport Signal Bot bereit.\n"
        "Tippe auf einen Button oder nutze /status.",
        reply_markup=MAIN_MENU,
    )


def _sport_filter_keyboard() -> InlineKeyboardMarkup:
    """Build inline sport filter keyboard for Top10 Einzelwetten."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Alle", callback_data="top10:all"),
            InlineKeyboardButton("⚽ Soccer", callback_data="top10:soccer"),
            InlineKeyboardButton("🏀 Basketball", callback_data="top10:basketball"),
        ],
        [
            InlineKeyboardButton("🎾 Tennis", callback_data="top10:tennis"),
            InlineKeyboardButton("🏈 NFL", callback_data="top10:americanfootball"),
            InlineKeyboardButton("🏒 NHL", callback_data="top10:icehockey"),
        ],
    ])


def _filter_items_by_sport(items: list, sport_filter: str) -> list:
    """Filter signal items by sport category prefix."""
    if sport_filter == "all":
        return items
    return [x for x in items if str(x.get("sport", "")).startswith(sport_filter)]


async def _show_top10_for_sport(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    sport_filter: str = "all",
    edit_message: bool = False,
) -> None:
    """Core logic for showing Top10 value bets, optionally filtered by sport.

    Pipeline: load ALL ranked → filter placed → filter sport → top 10 → paginate.
    Each sport filter operates independently on the full pool — no cross-view
    dedup or global cap.
    """
    # Load the FULL ranked pool (not just global top 10)
    items, ts = get_all_ranked_signals()
    if not items:
        # Fallback to legacy top-10 snapshot
        items, ts = get_cached_signals()

    total_candidates = len(items)

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

    # Filter out non-playable and already-placed bets
    placed = _placed_keys_today()
    playable = [
        x for x in items
        if float(x.get("expected_value", 0)) > 0
        and float(x.get("recommended_stake", 0)) > 0
        and f"{x.get('event_id')}|{x.get('selection')}" not in placed
    ]
    n_placed_filtered = len(items) - len(playable)

    # Apply sport filter AFTER playability check
    filtered = _filter_items_by_sport(playable, sport_filter)
    n_sport_filtered = len(playable) - len(filtered)

    # Re-sort per sport: confidence DESC -> EV DESC -> odds ASC
    filtered.sort(
        key=lambda x: (
            float(x.get("confidence", 0)),
            float(x.get("expected_value", 0)),
            -float(x.get("bookmaker_odds", 99)),
        ),
        reverse=True,
    )

    sport_label = {
        "all": "Alle Sportarten",
        "soccer": "⚽ Soccer",
        "basketball": "🏀 Basketball",
        "tennis": "🎾 Tennis",
        "americanfootball": "🏈 NFL",
        "icehockey": "🏒 NHL",
    }.get(sport_filter, sport_filter)

    log.info(
        "Top10 [%s]: pool=%d, playable=%d (placed_out=%d), "
        "after_sport_filter=%d (sport_out=%d), showing=%d",
        sport_filter, total_candidates, len(playable), n_placed_filtered,
        len(filtered), n_sport_filtered, min(len(filtered), 10),
    )

    if not filtered:
        msg = f"Keine spielbaren Value Bets für {sport_label}."
        if edit_message:
            q = update.callback_query
            await q.edit_message_text(msg, reply_markup=_sport_filter_keyboard())
        else:
            await update.message.reply_text(msg, reply_markup=_sport_filter_keyboard())
        return

    top_n = filtered[:10]
    context.user_data["signal_items"] = top_n
    context.user_data["top10_sport_filter"] = sport_filter

    header = f"🎯 Top {len(top_n)} Einzelwetten | {sport_label}"
    if ts:
        header += f"\nStand: {ts[:16]}"

    # Show first signal with navigation
    card = _format_signal_card(top_n[0], 0, len(top_n))
    bet_data = {
        "event_id": top_n[0].get("event_id", ""),
        "sport": top_n[0].get("sport", ""),
        "market": top_n[0].get("market", "h2h"),
        "selection": top_n[0].get("selection", ""),
        "odds": float(top_n[0].get("bookmaker_odds", 0)),
        "stake": float(top_n[0].get("recommended_stake", 0)),
    }
    keyboard = _signal_nav_keyboard(0, len(top_n), bet_data)

    if edit_message:
        q = update.callback_query
        await q.edit_message_text(f"{header}\n\n{card}", reply_markup=keyboard)
    else:
        await update.message.reply_text(header)
        await update.message.reply_text(card, reply_markup=keyboard)


async def menu_value_bets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show sport filter selector, then Top 10 value bets."""
    await update.message.reply_text(
        "🎯 Top 10 Einzelwetten\nWähle eine Sportart:",
        reply_markup=_sport_filter_keyboard(),
    )


# ---------------------------------------------------------------------------
# Combo Suggestions: "10/20/30 Kombis"
# ---------------------------------------------------------------------------

SPORT_EMOJI = {
    "soccer": "⚽", "football": "⚽",
    "basketball": "🏀",
    "americanfootball": "🏈",
    "icehockey": "🏒",
    "tennis": "🎾",
}


def _sport_emoji(sport_key: str) -> str:
    """Map sport key to emoji."""
    for prefix, emoji in SPORT_EMOJI.items():
        if sport_key.startswith(prefix):
            return emoji
    return "🎯"


def _league_short(sport_key: str) -> str:
    """Extract short league label from sport key."""
    try:
        from src.core.sport_mapping import api_key_to_display
        label = api_key_to_display(sport_key)
        if label != sport_key:
            return label
    except Exception:
        pass
    parts = sport_key.split("_", 1)
    return parts[1].replace("_", " ").upper() if len(parts) > 1 else sport_key.upper()


def _format_leg_line(i: int, leg: dict) -> str:
    """Format a single combo leg with FULL context. Never output naked selections."""
    sport_key = str(leg.get("sport", ""))
    emoji = _sport_emoji(sport_key)
    league = _league_short(sport_key)

    home = leg.get("home_team", "")
    away = leg.get("away_team", "")
    selection = str(leg.get("selection", "?"))
    market = str(leg.get("market", leg.get("market_type", "")))
    odds = float(leg.get("odds", 0))
    prob = float(leg.get("probability", 0))

    # Build event string
    if home and away:
        event = f"{home} vs {away}"
    else:
        event = ""

    # Build market + selection string
    market_type = str(leg.get("market_type", "h2h"))
    if market_type == "h2h":
        sel_display = selection
    elif market_type in ("totals", "spreads"):
        sel_display = f"{market} {selection}" if market != selection else selection
    elif market_type in ("double_chance", "draw_no_bet"):
        sel_display = selection
    else:
        sel_display = f"{market} {selection}" if market else selection

    # Validation: mark incomplete legs
    if not event and not home:
        sel_display = f"⚠ {sel_display} (Event?)"

    return f" {i+1:2d}. {emoji} {league} | {event} | {sel_display} | @{odds:.2f} | {prob:.0%}"


def _format_combo_card(combo_data: dict) -> Optional[str]:
    """Format a combo as a rich card with FULL context per leg.

    Returns None if no legs have valid event context (avoids sending
    confusing '0 legs' cards to the user).
    """
    size = combo_data.get("size", 0)
    stake = float(combo_data.get("stake", 1.00))
    combined_odds = float(combo_data.get("combined_odds", 0))
    combined_prob = float(combo_data.get("combined_probability", 0))
    ev = float(combo_data.get("expected_value", 0))
    legs = combo_data.get("legs", [])

    potential_payout = round(stake * combined_odds, 2)
    is_playable = ev > 0

    # Filter: reject legs without event context, mark as incomplete
    complete_legs = []
    incomplete_count = 0
    for leg in legs:
        if leg.get("home_team") or leg.get("away_team"):
            complete_legs.append(leg)
        else:
            incomplete_count += 1

    # No usable legs → return None so callers can skip this combo
    if not complete_legs:
        return None

    legs_lines = [_format_leg_line(i, leg) for i, leg in enumerate(complete_legs)]
    legs_txt = "\n".join(legs_lines)

    tax_badge = " 🏷️ Steuerfrei" if len(complete_legs) >= 3 else ""
    incomplete_note = f"\n⚠️ {incomplete_count} Legs ohne Event-Kontext verworfen" if incomplete_count > 0 else ""

    risk_note = ""
    if combined_prob < 0.01:
        risk_note = "\n⚠️ Sehr geringe Trefferwahrscheinlichkeit — Lotto-Charakter!"
    elif combined_prob < 0.05:
        risk_note = "\n⚠️ Geringe Trefferwahrscheinlichkeit — hohes Risiko"

    return (
        f"🧩 KOMBI {size}er | Lotto{tax_badge}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Status: {'✅ spielbar' if is_playable else '⚠️ watchlist'}\n"
        f"Gesamtquote: {combined_odds:.2f} ({len(complete_legs)} Legs)\n"
        f"Wahrscheinlichkeit: {_progress_bar(combined_prob)}\n"
        f"💰 Einsatz: {stake:.2f} EUR → 💎 {potential_payout:.2f} EUR\n"
        f"EV: {ev:+.4f}{risk_note}\n"
        f"\nTipps:\n{legs_txt}{incomplete_note}"
    )


async def combo_suggestions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show 10/20/30-leg Tipico-friendly Lotto combos."""
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

    sent = 0
    for combo_data in combos:
        card = _format_combo_card(combo_data)
        if card:
            if sent == 0:
                await update.message.reply_text(f"🧩 Lotto-Kombis (10/20/30 Legs)")
            await update.message.reply_text(card)
            sent += 1

    if sent == 0:
        await update.message.reply_text(
            "Keine Kombi-Vorschläge mit vollständigem Event-Kontext verfügbar."
        )


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
# Callback Handler: Pagination + Mark + Settings Toggles + Agent Actions
# ---------------------------------------------------------------------------

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data or ""

    # --- Top10 sport filter ---
    if data.startswith("top10:"):
        sport_filter = data.split(":", 1)[1]
        if sport_filter == "back":
            await q.edit_message_text(
                "🎯 Top 10 Einzelwetten\nWähle eine Sportart:",
                reply_markup=_sport_filter_keyboard(),
            )
            return
        try:
            await _show_top10_for_sport(update, context, sport_filter=sport_filter, edit_message=True)
        except Exception:
            await q.edit_message_text("Sport-Filter fehlgeschlagen.", reply_markup=_sport_filter_keyboard())
        return

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

    # --- Dynamic Settings Toggles (EPIC 2) ---
    if data.startswith("settings:"):
        from src.core.dynamic_settings import dynamic_settings
        parts = data.split(":", 2)
        if len(parts) < 3:
            return

        category, key = parts[1], parts[2]
        if category == "sport":
            dynamic_settings.toggle_sport(key)
        elif category == "market":
            dynamic_settings.toggle_market(key)
        elif category == "combo":
            try:
                dynamic_settings.toggle_combo_size(int(key))
            except ValueError:
                pass
        elif category == "min_odds":
            try:
                dynamic_settings.set_min_odds(float(key))
            except ValueError:
                pass

        # Re-render settings dashboard
        keyboard = _build_settings_keyboard()
        try:
            await q.edit_message_reply_markup(reply_markup=keyboard)
        except Exception:
            pass
        return

    # --- Agent alert: Deep Dive (show cached analysis + LLM reasoning) ---
    if data.startswith("agent_analyze:"):
        alert_id = data.split(":", 1)[1]
        alert_data = cache.get_json(f"agent_alert:{alert_id}")
        if not alert_data:
            await q.edit_message_text("Alert abgelaufen.")
            return
        await q.edit_message_text("🔍 Deep Dive wird geladen...")
        try:
            # Use the CACHED analysis data directly — no re-running the analyst.
            # This guarantees the Deep Dive shows the exact same numbers as the
            # alert card (model_p, EV, etc.), fixing the data mismatch bug.
            model_p = float(alert_data.get("model_probability", 0))
            ev = float(alert_data.get("expected_value", 0))
            badge = _calibration_badge(model_p)
            elo = alert_data.get("elo", {})
            form = alert_data.get("form", {})
            sentiment = alert_data.get("sentiment", {})
            injuries = alert_data.get("injuries", {})
            poisson_prob = alert_data.get("poisson_prob")
            public_bias = float(alert_data.get("public_bias", 0))
            momentum = float(alert_data.get("market_momentum", 0))
            trigger = alert_data.get("trigger", "")
            commence = str(alert_data.get("commence_time", ""))

            # Format event time
            event_time_str = ""
            if commence:
                try:
                    ct = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    local = ct.astimezone(ZoneInfo("Europe/Berlin"))
                    event_time_str = local.strftime("%d.%m. %H:%M")
                except Exception:
                    event_time_str = commence[:16] if len(commence) >= 16 else commence

            # Format poisson display
            poisson_display = f"{poisson_prob:.0%}" if poisson_prob is not None else "n/a"

            time_line = f"Anstoss: {event_time_str}\n" if event_time_str else ""

            msg = (
                f"🔬 Deep Dive | {alert_data.get('home', '')} vs {alert_data.get('away', '')}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"{time_line}"
                f"Tipp: {alert_data.get('selection', '')}\n"
                f"Quote: {float(alert_data.get('target_odds', 0)):.2f}\n"
                f"Modell: {_progress_bar(model_p)} {badge}\n"
                f"EV: {ev:+.4f}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"Elo: {elo.get('elo_diff', 0):+.0f}\n"
                f"Form: H={form.get('home_wr', 0):.0%} A={form.get('away_wr', 0):.0%}\n"
                f"Sentiment: H={sentiment.get('home', 0):.2f} A={sentiment.get('away', 0):.2f}\n"
                f"Verletzungen: H={injuries.get('home', 0)} A={injuries.get('away', 0)}\n"
                f"Poisson: {poisson_display}\n"
                f"Public Bias: {public_bias:.3f}\n"
                f"Momentum: {momentum:+.3f}\n"
                f"Trigger: {trigger}\n"
            )

            # Add stats card if data is available
            stats_text = build_stats_for_event(
                event_id=str(alert_data.get("event_id", "")),
                home=alert_data.get("home", ""),
                away=alert_data.get("away", ""),
                sport=str(alert_data.get("sport", "")),
            )
            if stats_text:
                msg += f"\n{stats_text}\n"

            # Generate fresh LLM reasoning (lightweight, uses cached data)
            try:
                from src.agents.analyst_agent import AnalystAgent
                analyst = AnalystAgent()
                reasoning = await analyst.reason_with_llm(alert_data) or ""
            except Exception:
                reasoning = ""

            if reasoning:
                msg += f"\n💡 {reasoning}"
            msg += f"\n\nEmpfehlung: {alert_data.get('recommendation', 'SKIP')}"
            await q.message.reply_text(msg)
        except Exception as exc:
            await q.message.reply_text(f"Deep Dive fehlgeschlagen: {type(exc).__name__}")
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
# Dynamic Settings Dashboard (EPIC 2)
# ---------------------------------------------------------------------------

def _build_settings_keyboard() -> InlineKeyboardMarkup:
    """Build the categorized InlineKeyboardMarkup toggle dashboard."""
    from src.core.dynamic_settings import (
        AVAILABLE_SPORTS,
        AVAILABLE_MARKETS,
        AVAILABLE_COMBO_SIZES,
        dynamic_settings,
    )
    ds = dynamic_settings.get_all()
    active_sports = set(ds.get("active_sports", []))
    active_markets = set(ds.get("active_markets", []))
    active_combos = set(ds.get("target_combo_sizes", []))
    min_odds = float(ds.get("min_odds_threshold", 1.20))

    rows = []

    # Sports row(s)
    sport_row: List[InlineKeyboardButton] = []
    for key, label in AVAILABLE_SPORTS.items():
        icon = "✅" if key in active_sports else "❌"
        sport_row.append(InlineKeyboardButton(f"{icon} {label}", callback_data=f"settings:sport:{key}"))
        if len(sport_row) == 3:
            rows.append(sport_row)
            sport_row = []
    if sport_row:
        rows.append(sport_row)

    # Markets row(s)
    mkt_row: List[InlineKeyboardButton] = []
    for key, label in AVAILABLE_MARKETS.items():
        icon = "✅" if key in active_markets else "❌"
        mkt_row.append(InlineKeyboardButton(f"{icon} {label}", callback_data=f"settings:market:{key}"))
        if len(mkt_row) == 3:
            rows.append(mkt_row)
            mkt_row = []
    if mkt_row:
        rows.append(mkt_row)

    # Min odds threshold row
    odds_row = []
    for val in [1.10, 1.20, 1.30, 1.50]:
        check = " ✓" if abs(min_odds - val) < 0.01 else ""
        odds_row.append(InlineKeyboardButton(f"{val:.2f}{check}", callback_data=f"settings:min_odds:{val}"))
    rows.append(odds_row)

    # Combo sizes row
    combo_row = []
    for size in AVAILABLE_COMBO_SIZES:
        icon = "✅" if size in active_combos else "❌"
        combo_row.append(InlineKeyboardButton(f"{icon} {size}er", callback_data=f"settings:combo:{size}"))
    rows.append(combo_row)

    return InlineKeyboardMarkup(rows)


async def settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Interactive settings dashboard with toggle buttons."""
    from src.core.dynamic_settings import dynamic_settings
    ds = dynamic_settings.get_all()

    msg = (
        f"⚙️ Einstellungen\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Bankroll: {await asyncio.to_thread(_get_bankroll):.2f} EUR\n"
        f"Min Quote: {ds.get('min_odds_threshold', 1.20):.2f}\n"
        f"Fetch: 07:00 + 13:00\n"
        f"LLM: {settings.ollama_model}\n\n"
        f"Tippe auf die Buttons um Einstellungen zu ändern:"
    )
    keyboard = _build_settings_keyboard()
    await update.message.reply_text(msg, reply_markup=keyboard)


# ---------------------------------------------------------------------------
# NLP Intent Router (EPIC 7): Free-text -> Structured Intent -> Action
# ---------------------------------------------------------------------------

async def _classify_intent(text: str) -> dict:
    """Send text to Gemma 3 4B for intent classification."""
    try:
        from src.integrations.ollama_sentiment import OllamaSentimentClient
        nlp = OllamaSentimentClient()

        prompt = (
            "Klassifiziere die Nutzer-Nachricht als JSON-Intent.\n"
            "Mögliche Intents:\n"
            '{"intent":"get_top_bets","limit":N} - Nutzer will Top Wetten sehen\n'
            '{"intent":"get_combos","size":N} - Nutzer will Kombis (10,20,30)\n'
            '{"intent":"explain_bet","query":"..."} - Nutzer stellt eine Frage\n\n'
            "Antworte NUR mit dem JSON. Keine Erklärung.\n\n"
            f"Nachricht: {text}"
        )

        result = await asyncio.to_thread(nlp.generate_json, prompt)
        if isinstance(result, dict) and "intent" in result:
            return result
    except Exception:
        pass

    return {"intent": "explain_bet", "query": text}


async def _handle_get_top_bets(update: Update, context: ContextTypes.DEFAULT_TYPE, limit: int = 5):
    """Handle get_top_bets intent from NLP."""
    items, _ = get_cached_signals()
    placed = _placed_keys_today()
    items = [
        x for x in items
        if float(x.get("expected_value", 0)) > 0
        and f"{x.get('event_id')}|{x.get('selection')}" not in placed
    ]

    if not items:
        await update.message.reply_text("Keine spielbaren Value Bets verfügbar.")
        return

    limit = min(limit, len(items), 10)
    items = items[:limit]

    await update.message.reply_text(f"🎯 Top {limit} Einzelwetten:")
    for i, b in enumerate(items):
        sport = str(b.get("sport", "")).replace("_", " ").upper()
        model_p = float(b.get("model_probability", 0))
        badge = _calibration_badge(model_p)
        msg = (
            f"🎯 {i+1}. {sport} {badge}\n"
            f"Tipp: {b['selection']}\n"
            f"Quote: {float(b['bookmaker_odds']):.2f} | "
            f"Modell: {_progress_bar(model_p)}\n"
            f"EV: {float(b['expected_value']):+.4f}"
        )
        await update.message.reply_text(msg)


async def _handle_get_combos(update: Update, context: ContextTypes.DEFAULT_TYPE, size: int = 10):
    """Handle get_combos intent from NLP."""
    from src.core.live_feed import get_cached_combos
    combos = get_cached_combos()

    if not combos:
        await update.message.reply_text("Keine Kombis verfügbar. Bitte 'Daten aktualisieren'.")
        return

    # Filter by requested size if specified
    if size in (10, 20, 30):
        filtered = [c for c in combos if c.get("size") == size]
        if filtered:
            combos = filtered

    sent = 0
    for combo_data in combos:
        card = _format_combo_card(combo_data)
        if card:
            await update.message.reply_text(card)
            sent += 1
    if sent == 0:
        await update.message.reply_text(
            "Keine Kombi-Vorschläge mit vollständigem Event-Kontext verfügbar."
        )


async def _handle_explain_bet(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str):
    """Handle explain_bet intent from NLP."""
    try:
        from src.integrations.ollama_sentiment import OllamaSentimentClient
        nlp = OllamaSentimentClient()

        items, _ = get_cached_signals()
        context_str = ""
        if items:
            for b in items[:3]:
                context_str += (
                    f"- {b.get('selection')}: quote={b.get('bookmaker_odds')}, "
                    f"modell_wk={b.get('model_probability')}, ev={b.get('expected_value')}\n"
                )

        prompt = (
            "Basiere deine Antwort STRIKT auf den Daten. "
            "Antworte in 2-3 Sätzen auf Deutsch.\n\n"
            f"Frage: {query}\n\n"
            f"Aktuelle Signals:\n{context_str}"
        )

        result = await asyncio.to_thread(nlp.generate_raw, prompt)
        answer = f"💡 {result}" if result else "Keine Antwort verfügbar."
        await update.message.reply_text(answer)
    except ImportError:
        await update.message.reply_text("LLM nicht verfügbar. Stelle sicher, dass Ollama läuft.")
    except Exception as exc:
        await update.message.reply_text(f"Analyse fehlgeschlagen: {type(exc).__name__}")


async def agentic_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """NLP Intent Router: classifies free-text messages and routes to handlers.

    Supported intents:
    - get_top_bets: Show top N value bets
    - get_combos: Show combo suggestions with specific size
    - explain_bet: Answer questions about betting decisions
    """
    text = (update.message.text or "").strip()
    if not text or len(text) < 3:
        return

    await update.message.reply_text("🤔 Analysiere deine Nachricht...")

    intent_data = await _classify_intent(text)
    intent = intent_data.get("intent", "explain_bet")

    if intent == "get_top_bets":
        limit = int(intent_data.get("limit", 5))
        await _handle_get_top_bets(update, context, limit=max(1, min(10, limit)))
    elif intent == "get_combos":
        size = int(intent_data.get("size", 10))
        await _handle_get_combos(update, context, size=size)
    elif intent == "explain_bet":
        query = str(intent_data.get("query", text))
        await _handle_explain_bet(update, context, query=query)
    else:
        await _handle_explain_bet(update, context, query=text)


# ---------------------------------------------------------------------------
# Refresh / Help
# ---------------------------------------------------------------------------

def _render_progress_display(info: dict) -> str:
    """Render a rich progress display for the fetch pipeline."""
    phase = info.get("phase", "init")
    pct = int(info.get("pct", 0))
    events = int(info.get("events", 0))
    signals = int(info.get("signals", 0))
    sport_idx = int(info.get("sport_idx", 0))
    sport_total = int(info.get("sport_total", 0))
    done_sports = info.get("done_sports", [])
    detail = info.get("detail", "")
    combos = int(info.get("combos", 0))

    # Progress bar (20 chars wide)
    filled = int(pct / 5)
    bar = "█" * filled + "░" * (20 - filled)

    # Phase labels
    phase_labels = {
        "init": "⚙️ Initialisierung",
        "resolve": "🔍 Sportarten auflösen",
        "fetch_odds": "📡 Odds abrufen",
        "enrichment": "🧠 Enrichment",
        "signals": "🎯 Signals berechnen",
        "combos": "🧩 Kombis bauen",
        "done": "✅ Fertig!",
    }
    phase_label = phase_labels.get(phase, phase)

    # Build sport status lines
    sport_lines = []
    for s in done_sports[-6:]:  # show last 6 completed sports
        short = s.replace("soccer_", "").replace("basketball_", "").replace(
            "americanfootball_", "").replace("icehockey_", "").replace(
            "tennis_", "").replace("_", " ").title()
        sport_lines.append(f"  ✅ {short}")

    if phase == "fetch_odds" and detail and detail not in done_sports:
        current = detail.replace("soccer_", "").replace("basketball_", "").replace(
            "americanfootball_", "").replace("icehockey_", "").replace(
            "tennis_", "").replace("_", " ").title()
        sport_lines.append(f"  ⏳ {current}...")

    sports_block = "\n".join(sport_lines) if sport_lines else ""

    # Stats line
    stats_parts = []
    if events > 0:
        stats_parts.append(f"Events: {events}")
    if signals > 0:
        stats_parts.append(f"Signals: {signals}")
    if combos > 0:
        stats_parts.append(f"Kombis: {combos}")
    stats_line = " | ".join(stats_parts) if stats_parts else ""

    # Sport counter
    counter = f" ({sport_idx}/{sport_total})" if sport_total > 0 else ""

    lines = [
        "🔄 Datenupdate",
        "━━━━━━━━━━━━━━━━━━━━",
        f"[{bar}] {pct}%",
        f"{phase_label}{counter}",
    ]
    if sports_block:
        lines.append("")
        lines.append(sports_block)
    if stats_line:
        lines.append("")
        lines.append(stats_line)

    return "\n".join(lines)


async def _enrichment_job(bot, chat_id: int):
    """Phase 2: run enrichment in background, then send a status update."""
    try:
        result = await asyncio.to_thread(run_enrichment_pass)
        if result.get("status") == "done":
            n_signals = result.get("signals", 0)
            n_events = result.get("events", 0)
            text = (
                f"🧠 Enrichment done: {n_signals} Signals "
                f"aus {n_events} Events angereichert"
            )
        else:
            text = f"🧠 Enrichment übersprungen: {result.get('reason', '?')}"
        await bot.send_message(chat_id=chat_id, text=text)
    except Exception as exc:
        log.warning("Background enrichment failed: %s", exc)
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=f"🧠 Enrichment fehlgeschlagen: {type(exc).__name__}",
            )
        except Exception:
            pass


async def _refresh_job(bot, chat_id: int, msg_id: int, context: ContextTypes.DEFAULT_TYPE):
    progress_queue = asyncio.Queue()
    last_text = ""

    def on_progress(info):
        """Thread-safe callback: pushes progress dict into the async queue."""
        try:
            progress_queue.put_nowait(info)
        except Exception:
            pass

    async def update_display():
        """Polls the progress queue and edits the Telegram message."""
        nonlocal last_text
        while True:
            try:
                info = await asyncio.wait_for(progress_queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            text = _render_progress_display(info)
            # Only edit if text actually changed (avoid Telegram rate limits)
            if text != last_text:
                try:
                    await bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text=text)
                    last_text = text
                except Exception:
                    pass  # message may have been deleted, rate limited, etc.

            if info.get("phase") == "done":
                break

    # Run fetch + display updater concurrently
    display_task = asyncio.create_task(update_display())

    try:
        # --- Phase 1: Core fetch (odds + signals + combos, NO enrichment) ---
        items = await asyncio.wait_for(
            asyncio.to_thread(
                fetch_and_build_signals,
                progress_callback=on_progress,
                skip_enrichment=True,
            ),
            timeout=120,
        )
        # Wait for display to catch the "done" message
        await asyncio.sleep(0.5)
        display_task.cancel()

        meta = get_cached_meta()
        core_text = (
            "✅ Data update done (core)\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 Signals: {len(items)}\n"
            f"📡 Sports: {meta.get('sports_expanded', 0)} | "
            f"Events: {meta.get('events_seen', 0)}\n"
            f"🧩 Kombis im Cache\n"
            f"🧠 Enrichment läuft im Hintergrund..."
        )
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text=core_text)
        except Exception:
            await bot.send_message(chat_id=chat_id, text=core_text)

        # --- Phase 2: Enrichment in background (non-blocking) ---
        context.application.create_task(
            _enrichment_job(bot, chat_id)
        )

    except asyncio.TimeoutError:
        display_task.cancel()
        try:
            await bot.edit_message_text(
                chat_id=chat_id, message_id=msg_id,
                text="⏰ Datenupdate Timeout (120s). Teilweise Daten im Cache."
            )
        except Exception:
            await bot.send_message(chat_id=chat_id, text="⏰ Datenupdate Timeout (120s).")
    except Exception as e:
        display_task.cancel()
        try:
            await bot.edit_message_text(
                chat_id=chat_id, message_id=msg_id,
                text=f"❌ Fehler beim Update: {type(e).__name__}"
            )
        except Exception:
            await bot.send_message(chat_id=chat_id, text=f"❌ Fehler: {type(e).__name__}")
    finally:
        context.application.bot_data["refresh_running"] = False


async def refresh_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.application.bot_data.get("refresh_running", False):
        await update.message.reply_text("⏳ Refresh läuft bereits...")
        return
    context.application.bot_data["refresh_running"] = True
    # Send initial progress message (will be edited in-place)
    msg = await update.message.reply_text(_render_progress_display({"phase": "init", "pct": 0}))
    context.application.create_task(
        _refresh_job(context.bot, update.effective_chat.id, msg.message_id, context)
    )


async def help_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "📖 Hilfe\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "🎯 Heutige Top 10 Einzelwetten — Beste Einzelwetten nach Trefferquote\n"
        "🧩 10/20/30 Kombis — Tipico-freundliche Lotto-Kombis\n"
        "📊 Kontostand — PnL-Dashboard mit Chart\n"
        "🔄 Daten aktualisieren — Odds-Refresh\n"
        "⚙️ Einstellungen — Sportarten, Märkte, Min-Quote umschalten\n\n"
        "💬 Natürliche Sprache (einfach schreiben):\n"
        '  • "Gib mir die Top 5 für heute"\n'
        '  • "Zeig mir 10er Kombis"\n'
        '  • "Erkläre mir den Lakers Tipp"\n'
        '  • "Zeig mir die besten 3 Fußball Wetten"\n\n'
        "🤖 Agent Alerts:\n"
        "  🔍 Deep Dive — Analyst-Analyse\n"
        "  💰 Ghost Bet — Virtuelle Wette\n"
        "  🛑 Ignorieren — Alert verwerfen\n\n"
        "📡 Datenquellen: Odds API (Pro), Ollama gemma:4b, NewsAPI, API-Sports, Rotowire RSS\n"
        "⏰ Automatische Fetches: 07:00 + 13:00"
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
        await bot.send_message(chat_id=chat_id, text=f"🎯 {len(singles[:10])} Top Einzelwetten")
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

    sent = 0
    for combo_data in cached_combos:
        if float(combo_data.get("expected_value", 0)) <= 0:
            continue
        card = _format_combo_card(combo_data)
        if card:
            if sent == 0:
                await bot.send_message(chat_id=chat_id, text=f"🧩 {len(cached_combos)} Lotto-Kombis")
            await bot.send_message(chat_id=chat_id, text=card)
            sent += 1
    if sent == 0:
        await bot.send_message(
            chat_id=chat_id,
            text="Keine Kombi-Vorschläge mit vollständigem Event-Kontext verfügbar.",
        )
