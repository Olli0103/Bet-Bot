"""Signal deduplication, canonical market grouping, and card formatting.

Provides:
- ``canonical_market_group(market)`` — maps any market string to one of
  five canonical groups: h2h, double_chance, draw_no_bet, spreads, totals.
- ``deduplicate_signals(signals)`` — keeps one best pick per
  (event_id, canonical_market_group).
- ``display_status(signal)`` — returns PLAYABLE / WATCHLIST / BLOCKED badge.
- ``format_signal_card(signal, index, total)`` — compact card-like format.
- ``format_summary_header(raw, deduped, statuses, ev_cut, conf_gates)``
  — top-level summary before cards.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ── Canonical market groups ──────────────────────────────────────────────

_MARKET_GROUP_MAP = {
    "h2h": "h2h",
    "moneyline": "h2h",
    "1x2": "h2h",
    "double_chance": "double_chance",
    "draw_no_bet": "draw_no_bet",
    "dnb": "draw_no_bet",
    "spreads": "spreads",
    "handicap": "spreads",
    "asian_handicap": "spreads",
    "totals": "totals",
    "over_under": "totals",
}

_MARKET_PREFIX_RE = re.compile(
    r"^(double_chance|draw_no_bet|spreads|totals|h2h)",
    re.IGNORECASE,
)


def canonical_market_group(market: str) -> str:
    """Map any market string (e.g. 'spreads +1.5', 'double_chance 1X') to its group."""
    m = market.strip().lower()

    # Direct match
    if m in _MARKET_GROUP_MAP:
        return _MARKET_GROUP_MAP[m]

    # Prefix match (e.g. "spreads +1.5" -> "spreads")
    match = _MARKET_PREFIX_RE.match(m)
    if match:
        prefix = match.group(1)
        return _MARKET_GROUP_MAP.get(prefix, prefix)

    return "h2h"  # default fallback


# ── Display status ───────────────────────────────────────────────────────

def display_status(signal: Dict[str, Any]) -> Tuple[str, str, str]:
    """Determine display status for a signal.

    Returns (badge, status_label, reason).
    """
    rejected = signal.get("rejected_reason", "")
    stake = float(signal.get("recommended_stake", 0))
    ev = float(signal.get("expected_value", 0))

    if rejected:
        if "confidence" in rejected.lower():
            return "\U0001F534", "BLOCKED", rejected
        if "ev" in rejected.lower():
            return "\U0001F534", "BLOCKED", rejected
        return "\U0001F534", "BLOCKED", rejected

    if stake <= 0:
        return "\U0001F534", "BLOCKED", "stake=0"

    if ev <= 0:
        return "\U0001F7E1", "WATCHLIST", f"EV={ev:+.4f}"

    return "\U0001F7E2", "PLAYABLE", ""


# ── Deduplication ────────────────────────────────────────────────────────

def _extract_event_id(signal: Dict[str, Any]) -> str:
    """Extract base event_id from signal."""
    return str(signal.get("event_id", ""))


def deduplicate_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep one best pick per (event_id, canonical_market_group).

    Selection rule: model_probability DESC -> expected_value DESC -> bookmaker_odds ASC.
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for sig in signals:
        eid = _extract_event_id(sig)
        mkt = canonical_market_group(sig.get("market", "h2h"))
        groups[(eid, mkt)].append(sig)

    deduped: List[Dict[str, Any]] = []
    for key, candidates in groups.items():
        best = max(
            candidates,
            key=lambda s: (
                float(s.get("model_probability", 0)),
                float(s.get("expected_value", 0)),
                -float(s.get("bookmaker_odds", 999)),
            ),
        )
        # Annotate with canonical group and dedup metadata
        best["canonical_market_group"] = key[1]
        badge, status, reason = display_status(best)
        best["display_status"] = status
        best["display_badge"] = badge
        best["display_reason"] = reason
        best["dedup_candidates"] = len(candidates)
        deduped.append(best)

    return deduped


def sort_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort signals: model_probability DESC -> expected_value DESC -> bookmaker_odds ASC."""
    return sorted(
        signals,
        key=lambda s: (
            float(s.get("model_probability", 0)),
            float(s.get("expected_value", 0)),
            -float(s.get("bookmaker_odds", 999)),
        ),
        reverse=True,
    )


# ── Card formatting ──────────────────────────────────────────────────────

def _progress_bar(value: float, width: int = 10) -> str:
    filled = int(round(value * width))
    empty = width - filled
    return f"[{'#' * filled}{'-' * empty}] {value:.0%}"


def _calibration_badge(model_prob: float) -> str:
    try:
        from src.core.ml_trainer import get_reliability_adjustment
        adj = get_reliability_adjustment(model_prob)
        if 0.90 <= adj <= 1.10:
            return "[OK]"
        elif 0.75 <= adj <= 1.25:
            return "[\u223C]"
        return "[!]"
    except Exception:
        if 0.35 <= model_prob <= 0.65:
            return "[OK]"
        elif 0.25 <= model_prob <= 0.75:
            return "[\u223C]"
        return "[!]"


def _sport_emoji(sport_key: str) -> str:
    mapping = {
        "soccer": "\u26BD",
        "basketball": "\U0001F3C0",
        "tennis": "\U0001F3BE",
        "icehockey": "\U0001F3D2",
        "americanfootball": "\U0001F3C8",
        "baseball": "\u26BE",
        "cricket": "\U0001F3CF",
        "mma": "\U0001F94A",
    }
    for prefix, emoji in mapping.items():
        if sport_key.startswith(prefix):
            return emoji
    return "\U0001F3AF"


def format_signal_card(b: Dict[str, Any], index: int, total: int) -> str:
    """Format a single signal as a compact, card-like message.

    All existing fields preserved (no information loss).
    """
    sport = str(b.get("sport", "")).replace("_", " ").upper()
    sport_key = str(b.get("sport", ""))
    market = str(b.get("market", "h2h"))
    model_p = float(b.get("model_probability", 0))
    ev = float(b.get("expected_value", 0))
    odds = float(b.get("bookmaker_odds", 0))
    stake = float(b.get("recommended_stake", 0))
    source = b.get("source_mode", "n/a")
    ref = b.get("reference_book", "n/a")
    src_q = float(b.get("source_quality", b.get("confidence", 0)))

    kelly_raw = float(b.get("kelly_raw", b.get("kelly_fraction", 0)))
    stake_before = float(b.get("stake_before_cap", stake))
    cap_applied = b.get("stake_cap_applied", False)
    trigger = b.get("trigger", "")

    badge = _calibration_badge(model_p)
    emoji = _sport_emoji(sport_key)

    # Status badge
    display_badge = b.get("display_badge", "")
    display_status_label = b.get("display_status", "")
    if not display_badge:
        display_badge, display_status_label, _ = display_status(b)

    # Market group tag
    mkt_group = b.get("canonical_market_group", canonical_market_group(market))
    market_tag = f" | {market}" if market != "h2h" else ""
    cap_tag = " [CAP]" if cap_applied else ""
    trigger_tag = f" | {trigger}" if trigger else ""

    # Retail trap badge
    bias = float(b.get("public_bias", 0))
    trap = " [!TRAP]" if bias > 0.03 else (" [BIAS]" if bias > 0.02 else "")

    explanation = b.get("explanation", "")
    why_line = f"\n\U0001F4AC {explanation}" if explanation else ""

    # Dedup info
    dedup_n = b.get("dedup_candidates", 0)
    dedup_tag = f" (best of {dedup_n})" if dedup_n > 1 else ""

    return (
        f"{display_badge} {emoji} {index + 1}/{total} | {sport}{market_tag}\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
        f"Status: {display_status_label}{dedup_tag}\n"
        f"Tipp: {b.get('selection', '?')}\n"
        f"Quote: {odds:.2f} | Modell: {_progress_bar(model_p)} {badge}{trap}\n"
        f"EV: {ev:+.4f} | SrcQ: {src_q:.0%} | {source} | {ref}\n"
        f"Kelly: {kelly_raw:.4f} | Einsatz: {stake_before:.2f} \u2192 {stake:.2f} EUR{cap_tag}{trigger_tag}"
        f"{why_line}"
    )


def format_summary_header(
    raw_count: int,
    deduped_count: int,
    statuses: Dict[str, int],
    ev_cut: float,
    conf_gates: str,
) -> str:
    """Top-level summary before signal cards."""
    playable = statuses.get("PLAYABLE", 0)
    watchlist = statuses.get("WATCHLIST", 0)
    blocked = statuses.get("BLOCKED", 0)

    return (
        f"\U0001F4CA Signal-\u00DCbersicht\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
        f"Roh: {raw_count} | Dedupliziert: {deduped_count}\n"
        f"\U0001F7E2 PLAYABLE: {playable} | \U0001F7E1 WATCHLIST: {watchlist} | \U0001F534 BLOCKED: {blocked}\n"
        f"EV-Cut: \u2265{ev_cut:.3f} | Conf-Gates: {conf_gates}"
    )
