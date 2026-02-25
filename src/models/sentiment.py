from pydantic import BaseModel, Field
from typing import Literal


class SentimentResult(BaseModel):
    label: Literal["positive", "neutral", "negative"] = "neutral"
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""


class TeamNewsSentiment(BaseModel):
    team: str
    title: str
    source: str
    published_at: str
    sentiment: SentimentResult
