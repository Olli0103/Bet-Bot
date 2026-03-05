"""Alert Manager: Priority scoring, dedup/debounce, routing, quality guards.

Replaces the naive "send every alert immediately" approach with a tiered
system that reduces spam and increases actionability per alert.

Priority Classes:
    CRITICAL  — immediate push (large edge, high confidence, fresh data)
    HIGH      — immediate push (solid edge, confirmed by multiple signals)
    MEDIUM    — bundled into digest (moderate edge, watchlist candidates)
    LOW       — logged internally only (weak signal, informational)

Every alert includes an Actionability Block:
    - Playability: PLAYABLE / WATCHLIST / BLOCKED
    - Top-3 reasons
    - Risk flags
    - EV breakdown (calibrated prob, implied prob, tax-adjusted EV)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.data.redis_cache import cache

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AlertPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Playability(str, Enum):
    PLAYABLE = "PLAYABLE"
    WATCHLIST = "WATCHLIST"
    BLOCKED = "BLOCKED"


# ---------------------------------------------------------------------------
# Alert Metrics (Redis-backed counters)
# ---------------------------------------------------------------------------

_METRICS_HASH_KEY = "alert_metrics:v2"
_METRICS_TS_KEY = "alert_metrics:timestamps"

# Field mapping for HINCRBY (atomic increments)
_COUNTER_FIELDS = {
    "total": "alerts_total",
    "sent_immediate": "alerts_sent_immediate",
    "digested": "alerts_digested",
    "suppressed_dedup": "alerts_suppressed_dedup",
    "suppressed_quality": "alerts_suppressed_quality",
}


class AlertMetrics:
    """Redis-backed alert metrics using atomic HINCRBY operations.

    Avoids the GET-modify-SET race condition by using Redis hash fields
    with atomic increments. Timestamps are stored in a separate sorted
    list with LPUSH/LTRIM.
    """

    @classmethod
    def record(cls, event: str, priority: str = "") -> None:
        r = cache.client
        field = _COUNTER_FIELDS.get(event)
        if field:
            r.hincrby(_METRICS_HASH_KEY, field, 1)
        if event == "sent_immediate":
            r.hset(_METRICS_HASH_KEY, "last_alert_ts", str(time.time()))
            # Append timestamp to list (capped at 200)
            r.lpush(_METRICS_TS_KEY, str(time.time()))
            r.ltrim(_METRICS_TS_KEY, 0, 199)
        if priority:
            r.hincrby(_METRICS_HASH_KEY, f"prio:{priority}:sent", 1)
        # Set TTL on the hash (refreshed on every write)
        r.expire(_METRICS_HASH_KEY, 7 * 24 * 3600)
        r.expire(_METRICS_TS_KEY, 7 * 24 * 3600)

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        r = cache.client
        raw = r.hgetall(_METRICS_HASH_KEY)
        if not raw:
            raw = {}
        # Decode bytes if needed
        data: Dict[str, str] = {}
        for k, v in raw.items():
            key = k.decode() if isinstance(k, bytes) else k
            val = v.decode() if isinstance(v, bytes) else v
            data[key] = val

        result = {
            "alerts_total": int(data.get("alerts_total", 0)),
            "alerts_sent_immediate": int(data.get("alerts_sent_immediate", 0)),
            "alerts_digested": int(data.get("alerts_digested", 0)),
            "alerts_suppressed_dedup": int(data.get("alerts_suppressed_dedup", 0)),
            "alerts_suppressed_quality": int(data.get("alerts_suppressed_quality", 0)),
            "last_alert_ts": float(data.get("last_alert_ts", 0)),
            "outcomes": {
                prio: {"sent": int(data.get(f"prio:{prio}:sent", 0))}
                for prio in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
            },
        }

        # Compute avg time between alerts from timestamp list
        ts_raw = r.lrange(_METRICS_TS_KEY, 0, 199) or []
        ts_list = sorted(float(t.decode() if isinstance(t, bytes) else t) for t in ts_raw)
        if len(ts_list) >= 2:
            deltas = [ts_list[i] - ts_list[i - 1] for i in range(1, len(ts_list))]
            result["avg_time_between_alerts"] = round(sum(deltas) / len(deltas), 1)
        else:
            result["avg_time_between_alerts"] = 0.0

        return result

    @classmethod
    def reset(cls) -> None:
        r = cache.client
        r.delete(_METRICS_HASH_KEY)
        r.delete(_METRICS_TS_KEY)


# ---------------------------------------------------------------------------
# Quality Guards
# ---------------------------------------------------------------------------

# Minimum data quality requirements for an alert to be sent
MIN_CALIBRATED_CONFIDENCE = 0.35
ODDS_GLITCH_MIN = 1.01
ODDS_GLITCH_MAX = 100.0
ODDS_GLITCH_MOVE_PCT = 30.0  # >30% implied prob move = likely glitch
MIN_DATA_FRESHNESS_SECONDS = 3600  # data older than 1h is stale


def check_data_quality(analysis: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (passes, reason) for data quality gate."""
    model_p = float(analysis.get("model_probability", 0))
    cal_source = str(analysis.get("calibration_source", ""))

    # Hard requirement: calibrated confidence
    if model_p < MIN_CALIBRATED_CONFIDENCE:
        return False, f"model_probability {model_p:.2f} < {MIN_CALIBRATED_CONFIDENCE}"

    # Must have calibration source (not raw passthrough for alerts)
    if cal_source == "raw_passthrough" or not cal_source:
        return False, f"no calibration: source={cal_source!r}"

    return True, ""


def check_odds_glitch(analysis: Dict[str, Any]) -> Tuple[bool, str]:
    """Detect obvious odds glitches (erroneous prices)."""
    current_odds = float(analysis.get("bookmaker_odds", 0) or analysis.get("current_odds", 0))
    prev_odds = float(analysis.get("sharp_odds", 0) or analysis.get("prev_odds", 0))

    if current_odds <= ODDS_GLITCH_MIN or current_odds >= ODDS_GLITCH_MAX:
        return False, f"odds out of range: {current_odds}"

    if prev_odds > 1.0 and current_odds > 1.0:
        ip_curr = 1.0 / current_odds
        ip_prev = 1.0 / prev_odds
        move = abs(ip_curr - ip_prev) * 100
        if move > ODDS_GLITCH_MOVE_PCT:
            return False, f"implied prob move {move:.1f}% > {ODDS_GLITCH_MOVE_PCT}% (likely glitch)"

    return True, ""


def check_steam_confirmation(analysis: Dict[str, Any]) -> Tuple[bool, str]:
    """A pure steam_move needs additional confirmation to alert."""
    trigger = str(analysis.get("trigger", ""))
    if trigger not in ("steam_move", "totals_steam"):
        return True, ""

    # Steam moves need at least one confirming signal
    ev = float(analysis.get("expected_value", 0))
    model_p = float(analysis.get("model_probability", 0))
    momentum = float(analysis.get("market_momentum", 0))

    confirmations = 0
    if ev > 0.02:
        confirmations += 1
    if model_p > 0.55:
        confirmations += 1
    if abs(momentum) > 0.01:
        confirmations += 1

    if confirmations < 1:
        return False, f"steam_move without confirmation (ev={ev:.3f}, p={model_p:.2f}, mom={momentum:.3f})"
    return True, ""


def run_quality_guards(analysis: Dict[str, Any]) -> Tuple[bool, str]:
    """Run all quality guards. Returns (passes, reason)."""
    for guard_fn in (check_data_quality, check_odds_glitch, check_steam_confirmation):
        passes, reason = guard_fn(analysis)
        if not passes:
            return False, reason
    return True, ""


# ---------------------------------------------------------------------------
# Priority Scoring
# ---------------------------------------------------------------------------

def compute_alert_score(analysis: Dict[str, Any]) -> float:
    """Compute a composite alert score from 0-100.

    Factors:
    - Calibrated probability edge vs implied (40% weight)
    - EV (30% weight)
    - Source quality / calibration (15% weight)
    - Market momentum stability (15% weight)
    """
    model_p = float(analysis.get("model_probability", 0))
    bookmaker_odds = float(analysis.get("bookmaker_odds", 0) or 2.0)
    implied_p = 1.0 / bookmaker_odds if bookmaker_odds > 1.0 else 0.5
    ev = float(analysis.get("expected_value", 0))
    cal_source = str(analysis.get("calibration_source", ""))
    momentum = abs(float(analysis.get("market_momentum", 0)))

    # Edge component: how much better is our prob than implied (0-40)
    edge = max(0, model_p - implied_p)
    edge_score = min(40, edge * 200)  # 0.20 edge = 40 points

    # EV component (0-30)
    ev_score = min(30, max(0, ev * 300))  # 0.10 EV = 30 points

    # Source quality (0-15)
    source_score = 15.0
    if cal_source == "raw_passthrough":
        source_score = 3.0
    elif cal_source == "global":
        source_score = 10.0
    # sport_market is best = 15

    # Momentum stability (0-15): moderate momentum is good, extreme is noise
    if 0.005 <= momentum <= 0.05:
        momentum_score = 15.0
    elif momentum < 0.005:
        momentum_score = 8.0  # no momentum = less urgent
    else:
        momentum_score = 5.0  # extreme momentum = uncertain

    return round(edge_score + ev_score + source_score + momentum_score, 1)


def classify_priority(score: float, analysis: Dict[str, Any]) -> AlertPriority:
    """Map score to priority class."""
    ev = float(analysis.get("expected_value", 0))
    model_p = float(analysis.get("model_probability", 0))

    if score >= 65 and ev > 0.05 and model_p > 0.65:
        return AlertPriority.CRITICAL
    if score >= 45 and ev > 0.02:
        return AlertPriority.HIGH
    if score >= 25:
        return AlertPriority.MEDIUM
    return AlertPriority.LOW


# ---------------------------------------------------------------------------
# Playability Assessment
# ---------------------------------------------------------------------------

def assess_playability(
    analysis: Dict[str, Any],
    priority: AlertPriority,
) -> Tuple[Playability, List[str], List[str]]:
    """Determine playability, top reasons, and risk flags.

    Returns (playability, reasons, risk_flags).
    """
    ev = float(analysis.get("expected_value", 0))
    model_p = float(analysis.get("model_probability", 0))
    bookmaker_odds = float(analysis.get("bookmaker_odds", 0) or 2.0)
    implied_p = 1.0 / bookmaker_odds if bookmaker_odds > 1.0 else 0.5
    trigger = str(analysis.get("trigger", ""))
    momentum = float(analysis.get("market_momentum", 0))
    public_bias = float(analysis.get("public_bias", 0))
    cal_source = str(analysis.get("calibration_source", ""))

    reasons: List[str] = []
    risk_flags: List[str] = []

    # Build reasons
    edge = model_p - implied_p
    if edge > 0.10:
        reasons.append(f"Starker Edge: +{edge * 100:.1f}pp vs Markt")
    elif edge > 0.03:
        reasons.append(f"Solider Edge: +{edge * 100:.1f}pp vs Markt")
    else:
        reasons.append(f"Kleiner Edge: +{edge * 100:.1f}pp")

    if ev > 0.05:
        reasons.append(f"Hoher EV: +{ev * 100:.1f}%")
    elif ev > 0.02:
        reasons.append(f"Positiver EV: +{ev * 100:.1f}%")

    if model_p > 0.65:
        reasons.append("Hohe Modell-Konfidenz")
    elif model_p > 0.55:
        reasons.append("Solide Modell-Konfidenz")

    if trigger in ("steam_move", "totals_steam"):
        reasons.append("Sharp Money Bewegung")
    if momentum > 0.02:
        reasons.append("Linie bewegt sich zugunsten")

    reasons = reasons[:3]

    # Risk flags
    if bookmaker_odds >= 3.5:
        risk_flags.append("high-odds")
    if abs(momentum) > 0.05:
        risk_flags.append("high-volatility")
    if cal_source in ("global", "raw_passthrough"):
        risk_flags.append("weak-calibration")
    if public_bias > 0.03:
        risk_flags.append("public-bias")
    if trigger in ("steam_move", "totals_steam") and ev < 0.03:
        risk_flags.append("steam-only-weak-ev")

    # Determine playability
    if priority in (AlertPriority.CRITICAL, AlertPriority.HIGH) and ev > 0.01 and not risk_flags:
        playability = Playability.PLAYABLE
    elif priority in (AlertPriority.CRITICAL, AlertPriority.HIGH) and ev > 0.01:
        playability = Playability.PLAYABLE  # playable but with flags
    elif ev > 0 and model_p > 0.50:
        playability = Playability.WATCHLIST
    else:
        playability = Playability.BLOCKED

    return playability, reasons, risk_flags


# ---------------------------------------------------------------------------
# Actionability Block
# ---------------------------------------------------------------------------

@dataclass
class ActionabilityBlock:
    """Structured actionability info attached to every alert."""
    playability: str
    reasons: List[str]
    risk_flags: List[str]
    confidence_source: str  # "calibrated" or "raw"
    model_probability: float
    implied_probability: float
    expected_value: float
    tax_adjusted_ev: float  # after vig/tax

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_actionability_block(
    analysis: Dict[str, Any],
    priority: AlertPriority,
) -> ActionabilityBlock:
    """Build the actionability block for an alert."""
    playability, reasons, risk_flags = assess_playability(analysis, priority)

    model_p = float(analysis.get("model_probability", 0))
    bookmaker_odds = float(analysis.get("bookmaker_odds", 0) or 2.0)
    implied_p = 1.0 / bookmaker_odds if bookmaker_odds > 1.0 else 0.5
    ev = float(analysis.get("expected_value", 0))
    cal_source = str(analysis.get("calibration_source", ""))

    return ActionabilityBlock(
        playability=playability.value,
        reasons=reasons,
        risk_flags=risk_flags,
        confidence_source="calibrated" if cal_source != "raw_passthrough" else "raw",
        model_probability=round(model_p, 4),
        implied_probability=round(implied_p, 4),
        expected_value=round(ev, 4),
        tax_adjusted_ev=round(ev, 4),  # EV already includes tax in our pipeline
    )


# ---------------------------------------------------------------------------
# Dedup + Debounce
# ---------------------------------------------------------------------------

_DEDUP_KEY_PREFIX = "alert_dedup:"
_DEBOUNCE_WINDOW_SECONDS = 600  # 10 minutes
_DELTA_THRESHOLD = 0.02  # suppress if movement_pct delta < 2%


def _dedup_key(event_id: str, market: str, selection: str) -> str:
    """Build a dedup key from event + market_group + selection."""
    raw = f"{event_id}:{market}:{selection}"
    return f"{_DEDUP_KEY_PREFIX}{hashlib.md5(raw.encode()).hexdigest()[:16]}"


def is_duplicate_alert(
    event_id: str,
    market: str,
    selection: str,
    movement_pct: float = 0.0,
) -> bool:
    """Check if a near-identical alert was sent recently.

    Uses two checks to prevent spam while catching creeping movements:
    1. Delta vs LAST alert: suppress if change < 2% since last push
    2. Delta vs BASELINE (first alert): allow if total movement since
       first alert exceeds 5%, even if each step is small (creeping line)

    Returns True if alert should be suppressed.
    """
    key = _dedup_key(event_id, market, selection)
    prev = cache.get_json(key)
    if prev is None:
        return False

    prev_ts = float(prev.get("ts", 0))
    if time.time() - prev_ts >= _DEBOUNCE_WINDOW_SECONDS:
        return False  # window expired, allow

    # Check delta vs last sent value
    prev_move = float(prev.get("movement_pct", 0))
    delta_vs_last = abs(movement_pct - prev_move)

    # Check delta vs baseline (first alert in this window)
    baseline_move = float(prev.get("baseline_movement_pct", prev_move))
    delta_vs_baseline = abs(movement_pct - baseline_move)

    # Allow if total creeping movement exceeds 5% even though each step is small
    _CREEP_THRESHOLD = 5.0
    if delta_vs_baseline >= _CREEP_THRESHOLD:
        return False  # significant aggregate movement, send alert

    if delta_vs_last < _DELTA_THRESHOLD:
        return True  # suppress: same alert, tiny delta

    return False


def mark_alert_sent(
    event_id: str,
    market: str,
    selection: str,
    movement_pct: float = 0.0,
) -> None:
    """Record that an alert was sent for dedup tracking.

    Preserves the baseline_movement_pct from the first alert in the
    window so that creeping line movements can be detected.
    """
    key = _dedup_key(event_id, market, selection)
    prev = cache.get_json(key)

    # Preserve baseline from first alert in window
    baseline = movement_pct
    if prev is not None:
        prev_ts = float(prev.get("ts", 0))
        if time.time() - prev_ts < _DEBOUNCE_WINDOW_SECONDS:
            baseline = float(prev.get("baseline_movement_pct", movement_pct))

    cache.set_json(key, {
        "ts": time.time(),
        "movement_pct": movement_pct,
        "baseline_movement_pct": baseline,
    }, ttl_seconds=_DEBOUNCE_WINDOW_SECONDS * 2)


# ---------------------------------------------------------------------------
# Digest Buffer (for MEDIUM alerts)
# ---------------------------------------------------------------------------

_DIGEST_KEY = "alert_digest:buffer"
_DIGEST_INTERVAL_SECONDS = 900  # 15 minutes
_LAST_DIGEST_KEY = "alert_digest:last_sent"


def add_to_digest(alert_data: Dict[str, Any]) -> None:
    """Add a MEDIUM priority alert to the digest buffer."""
    buffer = cache.get_json(_DIGEST_KEY) or []
    buffer.append(alert_data)
    # Keep max 20 alerts in buffer
    cache.set_json(_DIGEST_KEY, buffer[-20:], ttl_seconds=3600)


def should_send_digest() -> bool:
    """Check if enough time has passed to send a digest."""
    last_sent = cache.get_json(_LAST_DIGEST_KEY)
    if last_sent is None:
        return True
    return time.time() - float(last_sent.get("ts", 0)) >= _DIGEST_INTERVAL_SECONDS


def pop_digest() -> List[Dict[str, Any]]:
    """Pop all buffered digest alerts and mark as sent."""
    buffer = cache.get_json(_DIGEST_KEY) or []
    if not buffer:
        return []
    cache.set_json(_DIGEST_KEY, [], ttl_seconds=3600)
    cache.set_json(_LAST_DIGEST_KEY, {"ts": time.time()}, ttl_seconds=3600)
    return buffer


# ---------------------------------------------------------------------------
# Alert Card Formatting (mobile-friendly)
# ---------------------------------------------------------------------------

_PRIORITY_BADGES = {
    AlertPriority.CRITICAL: "\U0001f534",  # red circle
    AlertPriority.HIGH: "\U0001f7e0",       # orange circle
    AlertPriority.MEDIUM: "\U0001f7e1",     # yellow circle
    AlertPriority.LOW: "\u26aa",            # white circle
}

_PLAYABILITY_BADGES = {
    Playability.PLAYABLE: "\u2705",    # green check
    Playability.WATCHLIST: "\U0001f440",  # eyes
    Playability.BLOCKED: "\u26d4",     # no entry
}


def format_alert_card(
    analysis: Dict[str, Any],
    priority: AlertPriority,
    actionability: ActionabilityBlock,
    stake: float = 0.0,
) -> str:
    """Format a compact, mobile-friendly alert card.

    Structure:
    - Hero line (priority badge + sport + match)
    - 3-5 key metrics
    - Actionability block
    """
    sport = str(analysis.get("sport", "")).replace("_", " ").upper()
    home = analysis.get("home", "")
    away = analysis.get("away", "")
    selection = analysis.get("selection", "")
    trigger = analysis.get("trigger", "")
    model_p = actionability.model_probability
    ev = actionability.expected_value
    implied_p = actionability.implied_probability
    market = str(analysis.get("market", "h2h"))

    # Format event time
    commence = str(analysis.get("commence_time", ""))
    time_str = ""
    if commence:
        try:
            from datetime import datetime as dt
            from zoneinfo import ZoneInfo
            ct = dt.fromisoformat(commence.replace("Z", "+00:00"))
            local = ct.astimezone(ZoneInfo("Europe/Berlin"))
            time_str = local.strftime("%d.%m. %H:%M")
        except Exception:
            time_str = commence[:16] if len(commence) >= 16 else commence

    badge = _PRIORITY_BADGES.get(priority, "")
    play_badge = _PLAYABILITY_BADGES.get(
        Playability(actionability.playability), ""
    )

    # Hero line
    market_tag = f" | {market}" if market != "h2h" else ""
    lines = [
        f"{badge} {priority.value} | {sport}{market_tag}",
        f"{home} vs {away}",
    ]
    if time_str:
        lines.append(f"Anstoss: {time_str}")

    # Key metrics
    lines.append(f"")
    lines.append(f"Tipp: {selection}")

    # Compact probability bar
    filled = int(round(model_p * 10))
    bar = "\u2588" * filled + "\u2591" * (10 - filled)
    lines.append(f"Modell: [{bar}] {model_p:.0%} (Markt: {implied_p:.0%})")
    lines.append(f"EV: {ev:+.2%} | Edge: {(model_p - implied_p):+.1%}")

    if stake > 0:
        lines.append(f"Einsatz: {stake:.2f} EUR")

    # Actionability block
    lines.append(f"")
    lines.append(f"{play_badge} {actionability.playability}")
    for reason in actionability.reasons[:3]:
        lines.append(f"  \u2022 {reason}")

    if actionability.risk_flags:
        flags_str = ", ".join(actionability.risk_flags)
        lines.append(f"\u26a0\ufe0f Risiken: {flags_str}")

    lines.append(f"Quelle: {actionability.confidence_source}")

    return "\n".join(lines)


def format_digest_card(alerts: List[Dict[str, Any]]) -> str:
    """Format a digest summary for multiple MEDIUM alerts."""
    if not alerts:
        return ""

    lines = [
        f"\U0001f7e1 ALERT DIGEST | {len(alerts)} Watchlist-Signale",
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
    ]

    for i, alert in enumerate(alerts[:10], 1):
        home = alert.get("home", "?")
        away = alert.get("away", "?")
        selection = alert.get("selection", "?")
        ev = float(alert.get("expected_value", 0))
        model_p = float(alert.get("model_probability", 0))
        playability = alert.get("playability", "WATCHLIST")
        play_badge = "\u2705" if playability == "PLAYABLE" else "\U0001f440"

        lines.append(
            f"{i}. {play_badge} {home} vs {away} | {selection} "
            f"| {model_p:.0%} | EV {ev:+.2%}"
        )

    lines.append(f"\nAlle {len(alerts)} Signale in /status abrufbar")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Alert Pipeline
# ---------------------------------------------------------------------------

@dataclass
class ProcessedAlert:
    """Result of processing an alert through the pipeline."""
    should_send_immediate: bool
    should_digest: bool
    should_suppress: bool
    priority: AlertPriority
    score: float
    actionability: ActionabilityBlock
    suppress_reason: str = ""
    card_text: str = ""
    analysis: Dict[str, Any] = field(default_factory=dict)


def process_alert(
    analysis: Dict[str, Any],
    stake: float = 0.0,
) -> ProcessedAlert:
    """Process an analysis result through the full alert pipeline.

    Steps:
    1. Quality guards
    2. Score + classify priority
    3. Dedup/debounce
    4. Build actionability block
    5. Route (immediate / digest / suppress)
    6. Format card

    Returns a ProcessedAlert with routing decision and formatted card.
    """
    AlertMetrics.record("total")

    # 1. Quality guards
    passes, reason = run_quality_guards(analysis)
    if not passes:
        AlertMetrics.record("suppressed_quality")
        log.info("Alert suppressed by quality guard: %s", reason)
        return ProcessedAlert(
            should_send_immediate=False,
            should_digest=False,
            should_suppress=True,
            priority=AlertPriority.LOW,
            score=0.0,
            actionability=build_actionability_block(analysis, AlertPriority.LOW),
            suppress_reason=f"quality: {reason}",
            analysis=analysis,
        )

    # 2. Score + classify
    score = compute_alert_score(analysis)
    priority = classify_priority(score, analysis)

    # 3. Dedup/debounce
    event_id = str(analysis.get("event_id", ""))
    market = str(analysis.get("market", "h2h"))
    selection = str(analysis.get("selection", ""))
    movement_pct = float(analysis.get("movement_pct", 0))

    if is_duplicate_alert(event_id, market, selection, movement_pct):
        AlertMetrics.record("suppressed_dedup")
        log.info("Alert suppressed by dedup: event=%s selection=%s", event_id, selection)
        return ProcessedAlert(
            should_send_immediate=False,
            should_digest=False,
            should_suppress=True,
            priority=priority,
            score=score,
            actionability=build_actionability_block(analysis, priority),
            suppress_reason="dedup",
            analysis=analysis,
        )

    # 4. Build actionability
    actionability = build_actionability_block(analysis, priority)

    # 5. Route
    if priority in (AlertPriority.CRITICAL, AlertPriority.HIGH):
        card = format_alert_card(analysis, priority, actionability, stake)
        mark_alert_sent(event_id, market, selection, movement_pct)
        AlertMetrics.record("sent_immediate", priority.value)
        return ProcessedAlert(
            should_send_immediate=True,
            should_digest=False,
            should_suppress=False,
            priority=priority,
            score=score,
            actionability=actionability,
            card_text=card,
            analysis=analysis,
        )

    if priority == AlertPriority.MEDIUM:
        # Add to digest buffer
        digest_entry = {
            "home": analysis.get("home"),
            "away": analysis.get("away"),
            "selection": selection,
            "expected_value": float(analysis.get("expected_value", 0)),
            "model_probability": float(analysis.get("model_probability", 0)),
            "playability": actionability.playability,
            "sport": analysis.get("sport"),
            "event_id": event_id,
            "ts": time.time(),
        }
        add_to_digest(digest_entry)
        mark_alert_sent(event_id, market, selection, movement_pct)
        AlertMetrics.record("digested", priority.value)
        return ProcessedAlert(
            should_send_immediate=False,
            should_digest=True,
            should_suppress=False,
            priority=priority,
            score=score,
            actionability=actionability,
            analysis=analysis,
        )

    # LOW — log only
    log.debug("LOW priority alert suppressed: score=%.1f event=%s", score, event_id)
    return ProcessedAlert(
        should_send_immediate=False,
        should_digest=False,
        should_suppress=True,
        priority=AlertPriority.LOW,
        score=score,
        actionability=actionability,
        suppress_reason="low_priority",
        analysis=analysis,
    )


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_alert_quality_report() -> Dict[str, Any]:
    """Generate a quality report for alert performance."""
    metrics = AlertMetrics.get_all()
    total = metrics.get("alerts_total", 0)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": metrics,
        "summary": {
            "total_alerts": total,
            "sent_immediately": metrics.get("alerts_sent_immediate", 0),
            "digested": metrics.get("alerts_digested", 0),
            "suppressed_dedup": metrics.get("alerts_suppressed_dedup", 0),
            "suppressed_quality": metrics.get("alerts_suppressed_quality", 0),
            "avg_time_between_alerts_sec": metrics.get("avg_time_between_alerts", 0),
        },
        "priority_breakdown": metrics.get("outcomes", {}),
    }

    # Suppression rate
    if total > 0:
        suppressed = (
            metrics.get("alerts_suppressed_dedup", 0) +
            metrics.get("alerts_suppressed_quality", 0)
        )
        report["summary"]["suppression_rate_pct"] = round(suppressed / total * 100, 1)
    else:
        report["summary"]["suppression_rate_pct"] = 0.0

    return report


def write_alert_quality_artifacts() -> None:
    """Write alert quality report to artifacts/ and markdown."""
    import os
    report = generate_alert_quality_report()

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/alert_quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# Alert Quality Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    for k, v in report["summary"].items():
        lines.append(f"| {k} | {v} |")

    lines.extend([
        "",
        "## Priority Breakdown",
        "",
        "| Priority | Sent |",
        "|----------|------|",
    ])
    for prio, data in report["priority_breakdown"].items():
        lines.append(f"| {prio} | {data.get('sent', 0)} |")

    with open("ALERT_QUALITY_REPORT.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    log.info("Alert quality artifacts written")
