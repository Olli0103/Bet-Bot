"""Stateful Tip Publisher: State-graph flow for Tippprovider delivery.

Replaces the linear Scout -> Analyst pipeline with a
state machine that **re-validates** a tip if odds move while the
analysis is in progress.  This ensures that every published tip is
mathematically valid at the moment of delivery — critical for a
Tippprovider that provides actionable signals to German bettors.

State Flow:
    DISCOVERED -> ANALYZING -> VALIDATING -> PUBLISHED
                                  |
                                  +-> STALE (odds drifted, re-analyse)
                                  +-> REJECTED (EV lost)

As a Tippprovider, the system never places bets directly.  The
The Publisher is the final stage in the tip delivery path.

DSS Compliance:
    Every published tip is paired with a ``StatefulTip`` audit record
    that requires ``HumanReviewData`` before any bankroll update.
    Interactive Telegram inline keyboards allow the operator to
    [Mark as Placed], [Reject], or [Show Math] — satisfying the
    "meaningful human intervention" requirement under GDPR Art. 22.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.betting_math import (
    DEFAULT_GERMAN_TAX_RATE,
    calculate_mao,
    kelly_fraction,
    tax_adjusted_expected_value,
)
from src.models.compliance import (
    ConfidenceBreakdown,
    StatefulTip,
    TipAlert,
)

log = logging.getLogger(__name__)

# If odds drift more than this threshold (in implied probability) between
# discovery and publication, the tip must be re-validated.
ODDS_DRIFT_THRESHOLD = 0.03  # 3 percentage points of implied prob
MAX_REVALIDATION_ATTEMPTS = 2


class TipStatus(str, Enum):
    """Lifecycle states for a tip flowing through the publisher."""
    DISCOVERED = "DISCOVERED"
    ANALYZING = "ANALYZING"
    VALIDATING = "VALIDATING"
    PUBLISHED = "PUBLISHED"
    STALE = "STALE"
    REJECTED = "REJECTED"


class TipState:
    """Mutable state object for a single tip as it flows through the graph."""

    __slots__ = (
        "event_id", "sport", "home", "away", "selection", "market",
        "initial_odds", "current_odds", "model_probability",
        "expected_value", "analysis", "status", "revalidation_count",
        "created_at", "published_at", "rejection_reason",
    )

    def __init__(
        self,
        event_id: str,
        sport: str,
        home: str,
        away: str,
        selection: str,
        market: str,
        initial_odds: float,
    ) -> None:
        self.event_id = event_id
        self.sport = sport
        self.home = home
        self.away = away
        self.selection = selection
        self.market = market
        self.initial_odds = initial_odds
        self.current_odds = initial_odds
        self.model_probability: float = 0.0
        self.expected_value: float = 0.0
        self.analysis: Dict[str, Any] = {}
        self.status = TipStatus.DISCOVERED
        self.revalidation_count: int = 0
        self.created_at: float = time.time()
        self.published_at: Optional[float] = None
        self.rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sport": self.sport,
            "home": self.home,
            "away": self.away,
            "selection": self.selection,
            "market": self.market,
            "initial_odds": self.initial_odds,
            "current_odds": self.current_odds,
            "model_probability": self.model_probability,
            "expected_value": self.expected_value,
            "status": self.status.value,
            "revalidation_count": self.revalidation_count,
            "rejection_reason": self.rejection_reason,
        }


def has_odds_drifted(state: TipState) -> bool:
    """Check whether odds have drifted enough to require re-validation.

    Compares implied probabilities (not raw odds) because a 0.10 shift
    at odds 1.50 is much more significant than at odds 10.0.
    """
    if state.initial_odds <= 1.0 or state.current_odds <= 1.0:
        return True  # degenerate odds — always re-check
    initial_ip = 1.0 / state.initial_odds
    current_ip = 1.0 / state.current_odds
    return abs(current_ip - initial_ip) > ODDS_DRIFT_THRESHOLD


def validate_tip(state: TipState, tax_rate: float = DEFAULT_GERMAN_TAX_RATE) -> bool:
    """Validate that the tip is still profitable at current odds.

    Recalculates EV using the tax-adjusted formula and checks against
    the Minimum Acceptable Odds threshold.
    """
    ev = tax_adjusted_expected_value(
        state.model_probability, state.current_odds, tax_rate,
    )
    state.expected_value = ev

    mao = calculate_mao(
        state.model_probability, tax_rate=tax_rate, required_edge=0.01,
    )

    if ev <= 0:
        state.status = TipStatus.REJECTED
        state.rejection_reason = (
            f"Negative tax-adjusted EV: {ev:.4f} at odds {state.current_odds:.3f}"
        )
        return False

    if state.current_odds < mao:
        state.status = TipStatus.REJECTED
        state.rejection_reason = (
            f"Odds {state.current_odds:.3f} below MAO {mao:.3f}"
        )
        return False

    return True


async def tip_flow(
    state: TipState,
    analyst,
    get_current_odds,
    publish_fn,
    tax_rate: float = DEFAULT_GERMAN_TAX_RATE,
) -> TipState:
    """Execute the full state-graph flow for a single tip.

    Parameters
    ----------
    state : TipState
        Mutable tip state initialised from a Scout alert.
    analyst : AnalystAgent
        Agent that generates ML probability and qualitative context.
    get_current_odds : callable
        Async function ``(event_id, selection) -> float`` that returns
        the latest odds from the bookmaker/odds API.
    publish_fn : callable
        Async function ``(TipState) -> None`` that delivers the tip
        to the user (Telegram, webhook, etc.).
    tax_rate : float
        German Wettsteuer rate (default 5%).

    Returns
    -------
    TipState
        Final state after flow completion.
    """
    # --- PHASE 1: ANALYZE ---
    state.status = TipStatus.ANALYZING
    log.info(
        "Tip flow [%s]: %s %s @ %.3f → ANALYZING",
        state.event_id, state.sport, state.selection, state.initial_odds,
    )

    try:
        analysis = await analyst.analyze_event(
            event_id=state.event_id,
            sport=state.sport,
            home=state.home,
            away=state.away,
            selection=state.selection,
            target_odds=state.initial_odds,
            sharp_odds=state.initial_odds,
            sharp_market={state.selection: state.initial_odds},
            trigger="tip_flow",
            market_momentum=0.0,
        )
        state.analysis = analysis
        state.model_probability = float(analysis.get("model_probability", 0))
    except Exception as exc:
        state.status = TipStatus.REJECTED
        state.rejection_reason = f"Analysis failed: {exc}"
        log.error("Tip flow analysis failed for %s: %s", state.event_id, exc)
        return state

    # --- PHASE 2: VALIDATE (with re-entry on drift) ---
    while state.revalidation_count <= MAX_REVALIDATION_ATTEMPTS:
        state.status = TipStatus.VALIDATING

        # Refresh current odds
        try:
            fresh_odds = await get_current_odds(state.event_id, state.selection)
            if fresh_odds and fresh_odds > 1.0:
                state.current_odds = fresh_odds
        except Exception as exc:
            log.warning("Failed to refresh odds for %s: %s", state.event_id, exc)

        # Check for drift
        if has_odds_drifted(state) and state.revalidation_count < MAX_REVALIDATION_ATTEMPTS:
            state.revalidation_count += 1
            state.status = TipStatus.STALE
            log.info(
                "Tip flow [%s]: odds drifted %.3f → %.3f (revalidation %d/%d)",
                state.event_id, state.initial_odds, state.current_odds,
                state.revalidation_count, MAX_REVALIDATION_ATTEMPTS,
            )
            # Update initial_odds for next drift check
            state.initial_odds = state.current_odds
            continue

        # Validate EV at current odds
        if not validate_tip(state, tax_rate):
            log.info(
                "Tip flow [%s]: REJECTED — %s",
                state.event_id, state.rejection_reason,
            )
            return state

        break

    # --- PHASE 3: PUBLISH ---
    state.status = TipStatus.PUBLISHED
    state.published_at = time.time()
    log.info(
        "Tip flow [%s]: PUBLISHED — %s @ %.3f (EV=%.4f, revalidations=%d)",
        state.event_id, state.selection, state.current_odds,
        state.expected_value, state.revalidation_count,
    )

    try:
        await publish_fn(state)
    except Exception as exc:
        log.error("Tip publication failed for %s: %s", state.event_id, exc)

    return state


def build_dss_alert(state: TipState, reasoning: str = "") -> TipAlert:
    """Build a validated TipAlert from a published TipState.

    Includes the Tipico deep link, confidence breakdown, and net EV
    for the operator's decision-making.
    """
    from src.integrations.tipico_deeplink import event_link as tipico_event_link

    # Extract confidence breakdown from analysis if available
    cb_raw = state.analysis.get("confidence_breakdown")
    confidence_breakdown = None
    if cb_raw and isinstance(cb_raw, dict):
        try:
            confidence_breakdown = ConfidenceBreakdown.model_validate(cb_raw)
        except Exception:
            pass

    # Build Tipico deep link
    tipico_link = ""
    try:
        tipico_link = tipico_event_link(
            state.event_id, state.home, state.away,
        )
    except Exception:
        pass

    # Compute Kelly fraction for stake recommendation
    tax_rate = DEFAULT_GERMAN_TAX_RATE
    kf = kelly_fraction(
        state.model_probability, state.current_odds,
        frac=0.20, tax_rate=tax_rate,
    )

    mao = calculate_mao(state.model_probability, tax_rate=tax_rate, required_edge=0.01)

    risk_flags: List[str] = []
    if state.revalidation_count > 0:
        risk_flags.append(f"Odds drifted {state.revalidation_count}x during analysis")
    if state.current_odds < state.analysis.get("sharp_odds", state.current_odds):
        risk_flags.append("Below sharp opening odds")

    return TipAlert(
        event_id=state.event_id,
        match_name=f"{state.home} vs {state.away}",
        sport=state.sport,
        market=state.market,
        recommended_selection=state.selection,
        target_odds=state.current_odds,
        signal_odds=state.analysis.get("sharp_odds", state.current_odds),
        model_probability=state.model_probability,
        net_ev=state.expected_value,
        kelly_fraction=round(kf, 4),
        mao=round(mao, 3),
        confidence_breakdown=confidence_breakdown,
        ai_reasoning=reasoning or "Quantitatives Signal bestätigt durch Modell und Marktanalyse.",
        risk_flags=risk_flags,
        tipico_deeplink=tipico_link,
        commence_time=str(state.analysis.get("commence_time", "")),
    )


def cache_stateful_tip(state: TipState, alert: TipAlert) -> str:
    """Cache a StatefulTip for later human review, return the tip_id.

    The cached record awaits ``HumanReviewData`` before any bankroll
    update occurs — this is the audit trail required by German regulations.
    """
    from src.data.redis_cache import cache

    tip_id = f"{state.event_id}:{state.market}:{int(state.created_at)}"
    stateful = StatefulTip(
        tip_id=tip_id,
        ai_recommendation=alert,
    )
    cache.set_json(
        f"dss_tip:{tip_id}",
        stateful.model_dump(),
        ttl_seconds=12 * 3600,
    )
    return tip_id
