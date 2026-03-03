"""SQLAlchemy ORM models for the Signal Bot database."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class PlacedBet(Base):
    """Tracks every bet (real, virtual, and historical import)."""

    __tablename__ = "placed_bets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(128), nullable=False, index=True)
    sport = Column(String(64), nullable=False)
    market = Column(String(32), nullable=False, default="h2h")
    selection = Column(String(256), nullable=False)

    odds = Column(Float, nullable=False)
    odds_open = Column(Float, nullable=True)
    odds_close = Column(Float, nullable=True)
    clv = Column(Float, nullable=True, default=0.0)

    stake = Column(Float, nullable=False, default=1.0)
    status = Column(String(16), nullable=False, default="open")  # open/won/lost/void
    pnl = Column(Float, nullable=True, default=0.0)

    # ML feature snapshot (for training)
    sharp_implied_prob = Column(Float, nullable=True)
    sentiment_delta = Column(Float, nullable=True)
    injury_delta = Column(Float, nullable=True)
    form_winrate_l5 = Column(Float, nullable=True)
    form_games_l5 = Column(Float, nullable=True)

    # Flexible JSON blob for sport-specific advanced stats extracted from CSVs.
    # Examples: shots, corners, ATP rank, quarter scores, line movement, etc.
    meta_features = Column(JSONB, nullable=True, default=dict)

    notes = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f"<PlacedBet id={self.id} event={self.event_id} "
            f"sel={self.selection} status={self.status}>"
        )
