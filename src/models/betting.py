from pydantic import BaseModel, Field, ConfigDict
from typing import List


class BetSignal(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    sport: str
    event_id: str
    market: str
    selection: str
    bookmaker_odds: float = Field(gt=1.0)
    model_probability: float = Field(gt=0.0, lt=1.0)
    expected_value: float
    kelly_fraction: float
    recommended_stake: float
    source_mode: str = "primary"
    reference_book: str = "pinnacle"
    confidence: float = 1.0


class ComboLeg(BaseModel):
    event_id: str
    selection: str
    odds: float
    probability: float


class ComboBet(BaseModel):
    legs: List[ComboLeg]
    combined_odds: float
    combined_probability: float
    correlation_penalty: float
    expected_value: float
    kelly_fraction: float
    recommended_stake: float
