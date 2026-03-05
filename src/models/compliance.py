"""Compliance & Audit schemas for German DSS (Decision-Support System) model.

These schemas enforce the "Meaningful Human Intervention" requirement
under GDPR Article 22 and German gambling regulations (GlüStV 2021).

Every tip published by the system must eventually be paired with a
``HumanReviewData`` record — proving that:
1. A human operator reviewed the AI recommendation.
2. The operator had the ability to adjust stake, override odds, or reject.
3. The final decision was made by the human, not the system.

LUGAS/OASIS compliance:
The ``LUGAS_MONTHLY_DEPOSIT_LIMIT`` simulates the German player-protection
deposit cap (§6c GlüStV) within the DSS's own bankroll management so the
system never recommends stakes that would violate the operator's actual
bookmaker account limits.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# GlüStV 2021 §6c: Monthly deposit limit for online gambling in Germany
LUGAS_MONTHLY_DEPOSIT_LIMIT = 1000.0  # EUR


class HumanReviewData(BaseModel):
    """Record of the human operator's intervention on a published tip.

    Required for regulatory audit trail — proves meaningful human control
    over every bet placement decision.
    """
    model_config = ConfigDict(protected_namespaces=())

    operator_id: str = Field(
        ..., description="Telegram user ID or internal operator identifier"
    )
    confirmed_odds: float = Field(
        ..., gt=1.0,
        description="Actual odds the operator saw at placement time"
    )
    confirmed_stake: float = Field(
        ..., ge=0.0,
        description="Actual stake the operator chose (may differ from recommendation)"
    )
    action: str = Field(
        default="placed",
        description="What the operator did: 'placed', 'skipped', 'adjusted', 'rejected'"
    )
    reason_for_override: str = Field(
        default="",
        description="Why the operator deviated from AI recommendation (audit trail)"
    )
    reviewed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    @field_validator("action")
    @classmethod
    def normalize_action(cls, v: str) -> str:
        allowed = ("placed", "skipped", "adjusted", "rejected")
        v_lower = v.lower().strip()
        if v_lower not in allowed:
            return "skipped"
        return v_lower


class ConfidenceBreakdown(BaseModel):
    """XAI: Decomposition of model confidence into interpretable factors.

    Provides the operator with a transparent view of *why* the model
    believes a bet has edge — enabling "meaningful" intervention as
    required by German regulations.
    """
    model_config = ConfigDict(protected_namespaces=())

    statistical_weight: float = Field(
        ..., ge=0.0, le=1.0,
        description="Proportion of confidence from statistical/ML features"
    )
    market_signal_weight: float = Field(
        ..., ge=0.0, le=1.0,
        description="Proportion from market movement signals (steam, CLV)"
    )
    qualitative_weight: float = Field(
        ..., ge=0.0, le=1.0,
        description="Proportion from qualitative factors (news, injuries, weather)"
    )
    top_factors: List[str] = Field(
        default_factory=list,
        description="Top 3-5 human-readable factors driving the recommendation"
    )

    @field_validator("top_factors", mode="before")
    @classmethod
    def limit_factors(cls, v: list) -> list:
        if isinstance(v, list) and len(v) > 5:
            return v[:5]
        return v


class TipAlert(BaseModel):
    """Validated tip signal for Telegram delivery.

    Ensures the LLM cannot hallucinate data types when generating a tip.
    The ``net_ev`` field is post-5.3% German tax, preventing the operator
    from seeing misleadingly high "gross EV" numbers.
    """
    model_config = ConfigDict(protected_namespaces=())

    event_id: str
    match_name: str = Field(
        ..., min_length=3,
        description="'Home vs Away' display string"
    )
    sport: str
    market: str = "h2h"
    recommended_selection: str
    target_odds: float = Field(..., gt=1.0)
    signal_odds: float = Field(
        ..., gt=1.0,
        description="Odds at the time the signal was first detected"
    )
    model_probability: float = Field(..., gt=0.0, lt=1.0)
    net_ev: float = Field(
        ..., description="Expected Value AFTER 5.3% German Wettsteuer"
    )
    kelly_fraction: float = Field(default=0.0, ge=0.0)
    recommended_stake: float = Field(default=0.0, ge=0.0)
    mao: float = Field(
        default=0.0,
        description="Minimum Acceptable Odds — below this, edge is gone"
    )
    confidence_breakdown: Optional[ConfidenceBreakdown] = None
    ai_reasoning: str = Field(
        ..., min_length=10,
        description="LLM-generated qualitative reasoning for the tip"
    )
    risk_flags: List[str] = Field(default_factory=list)
    tipico_deeplink: str = Field(
        default="",
        description="One-tap deep link to Tipico bet slip for fast manual execution"
    )
    commence_time: str = Field(
        default="",
        description="ISO-8601 kickoff time for the event"
    )

    def format_for_telegram(self) -> str:
        """Format as a safe Telegram message string."""
        ev_pct = self.net_ev * 100
        prob_pct = self.model_probability * 100
        cb = self.confidence_breakdown

        lines = [
            f"\U0001f4ca TIPP | {self.match_name}",
            f"{'━' * 24}",
            f"Selection: {self.recommended_selection}",
            f"Odds: {self.target_odds:.2f} (Signal: {self.signal_odds:.2f})",
            f"Model: {prob_pct:.1f}% | Net EV: {ev_pct:+.2f}%",
            f"MAO: {self.mao:.3f}",
        ]

        if self.recommended_stake > 0:
            lines.append(f"Empf. Stake: {self.recommended_stake:.2f} EUR")

        if cb:
            lines.append(f"{'━' * 24}")
            lines.append(
                f"Konfidenz: Stats {cb.statistical_weight:.0%} | "
                f"Markt {cb.market_signal_weight:.0%} | "
                f"Qualitativ {cb.qualitative_weight:.0%}"
            )
            if cb.top_factors:
                lines.append("Faktoren: " + ", ".join(cb.top_factors[:3]))

        if self.risk_flags:
            lines.append(f"\u26a0\ufe0f Risiken: {', '.join(self.risk_flags)}")

        lines.append(f"{'━' * 24}")
        lines.append(f"\U0001f4a1 {self.ai_reasoning}")

        if self.tipico_deeplink:
            lines.append(f"\n\U0001f517 Tipico: {self.tipico_deeplink}")

        return "\n".join(lines)


class StatefulTip(BaseModel):
    """Full audit record linking AI recommendation to human intervention.

    This is the "golden record" for regulatory compliance — it proves
    the complete chain: signal → analysis → recommendation → human review.
    """
    model_config = ConfigDict(protected_namespaces=())

    tip_id: str = Field(
        ..., description="Unique identifier (event_id:market:timestamp)"
    )
    ai_recommendation: TipAlert
    human_intervention: Optional[HumanReviewData] = Field(
        default=None,
        description="Null until the operator reviews. Required before any bankroll update."
    )
    timestamp_published: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    timestamp_placed: Optional[str] = Field(
        default=None,
        description="When the operator actually placed (or skipped) the bet"
    )
    odds_at_placement: Optional[float] = Field(
        default=None,
        description="Live odds at the moment of placement (for slippage tracking)"
    )
    is_finalized: bool = Field(
        default=False,
        description="True once human_intervention is recorded"
    )

    def requires_human_review(self) -> bool:
        """Check whether this tip still needs operator intervention."""
        return self.human_intervention is None

    def finalize(self, review: HumanReviewData, live_odds: Optional[float] = None) -> None:
        """Record the human operator's decision (audit trail)."""
        self.human_intervention = review
        self.timestamp_placed = datetime.now(timezone.utc).isoformat()
        self.odds_at_placement = live_odds or review.confirmed_odds
        self.is_finalized = True
