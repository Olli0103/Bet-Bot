from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class BetSignal(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    sport: str
    event_id: str
    market: str
    market_type: str = "h2h"
    selection: str
    bookmaker_odds: float = Field(gt=1.0)
    model_probability: float = Field(gt=0.0, lt=1.0)
    expected_value: float
    kelly_fraction: float
    recommended_stake: float
    source_mode: str = "primary"
    reference_book: str = "pinnacle"
    # source_quality: how reliable is the odds source pair (1.0=primary, 0.8=fallback…)
    # NOT the model's prediction strength — that is model_probability.
    source_quality: float = 1.0
    confidence: float = 1.0  # kept for backward-compat serialization; == model_probability
    point: Optional[float] = None
    odds_age_minutes: float = 0.0
    is_stale: bool = False
    # Risk guard transparency fields
    kelly_raw: float = 0.0
    stake_before_cap: float = 0.0
    stake_cap_applied: bool = False
    trigger: str = ""
    rejected_reason: str = ""
    # Natural-language explanation for non-technical users ("Why this bet?")
    explanation: str = ""
    # Calibration fields
    model_probability_raw: float = 0.0  # raw model output before calibration
    model_probability_calibrated: float = 0.0  # calibrated probability
    calibration_source: str = ""  # "sport_market", "global", "raw_passthrough"
    # Paper-trading isolation: True = learning/exploration signal, not real money
    is_paper: bool = False


class ComboLeg(BaseModel):
    event_id: str
    selection: str
    odds: float
    probability: float
    sport: str = ""
    market_type: str = "h2h"
    home_team: str = ""
    away_team: str = ""
    market: str = ""  # full descriptor e.g. "totals 2.5", "spreads +3.5"


class ComboBet(BaseModel):
    legs: List[ComboLeg]
    combined_odds: float
    combined_probability: float
    correlation_penalty: float
    expected_value: float
    kelly_fraction: float
    recommended_stake: float
