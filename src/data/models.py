"""SQLAlchemy ORM models for the Signal Bot database."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class PlacedBet(Base):
    """Tracks every bet (real, virtual, and historical import)."""

    __tablename__ = "placed_bets"
    __table_args__ = (
        UniqueConstraint("event_id", "selection", "market", name="uq_event_selection_market"),
    )

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
    sharp_vig = Column(Float, nullable=True)
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


class TeamMatchStats(Base):
    """Per-team per-match statistics ingested from TheSportsDB / football-data.org.

    One row per team per match (so each match produces two rows: home + away).
    Used to compute rolling features (attack/defense strength, form trends,
    goals for/against, O/U rates, etc.).
    """

    __tablename__ = "team_match_stats"
    __table_args__ = (
        UniqueConstraint("source_match_id", "team", "source", name="uq_match_team_source"),
        Index("ix_tms_team_date", "team", "match_date"),
        Index("ix_tms_sport_date", "sport", "match_date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_match_id = Column(String(128), nullable=False, index=True)
    sport = Column(String(64), nullable=False)
    league = Column(String(128), nullable=True)
    season = Column(String(32), nullable=True)
    matchday = Column(Integer, nullable=True)
    match_date = Column(DateTime(timezone=True), nullable=False, index=True)

    # Teams
    team = Column(String(256), nullable=False)
    opponent = Column(String(256), nullable=False)
    is_home = Column(Boolean, nullable=False, default=True)

    # Scores
    goals_for = Column(Integer, nullable=True, default=0)
    goals_against = Column(Integer, nullable=True, default=0)
    result = Column(String(4), nullable=True)  # W / D / L

    # Extended stats (nullable — populated when available)
    shots = Column(Integer, nullable=True)
    shots_on_target = Column(Integer, nullable=True)
    possession_pct = Column(Float, nullable=True)
    corners = Column(Integer, nullable=True)
    fouls = Column(Integer, nullable=True)
    yellow_cards = Column(Integer, nullable=True)
    red_cards = Column(Integer, nullable=True)
    ht_goals_for = Column(Integer, nullable=True)
    ht_goals_against = Column(Integer, nullable=True)

    # Sport-specific extras (JSONB blob for flexibility)
    extra_stats = Column(JSONB, nullable=True, default=dict)

    # Source tracking
    source = Column(String(64), nullable=False, default="thesportsdb")

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<TeamMatchStats {self.team} vs {self.opponent} "
            f"{self.goals_for}-{self.goals_against} ({self.match_date})>"
        )


class EventStatsSnapshot(Base):
    """Pre-match feature snapshot for an upcoming event.

    Computed by the stats ingestion pipeline from TeamMatchStats.
    Stores rolling aggregates (attack/defense strength, form trend,
    O/U rates, rest days, etc.) at the time of signal generation.
    Prevents data leakage by only using data available before match.
    """

    __tablename__ = "event_stats_snapshots"
    __table_args__ = (
        UniqueConstraint("event_id", "team", name="uq_event_team_snapshot"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(128), nullable=False, index=True)
    sport = Column(String(64), nullable=False)
    team = Column(String(256), nullable=False)
    is_home = Column(Boolean, nullable=False, default=True)

    # Rolling stats (last N matches, typically 5-10)
    matches_played = Column(Integer, nullable=True, default=0)
    wins = Column(Integer, nullable=True, default=0)
    draws = Column(Integer, nullable=True, default=0)
    losses = Column(Integer, nullable=True, default=0)
    goals_scored_avg = Column(Float, nullable=True, default=0.0)
    goals_conceded_avg = Column(Float, nullable=True, default=0.0)
    clean_sheets = Column(Integer, nullable=True, default=0)

    # Computed features
    attack_strength = Column(Float, nullable=True, default=1.0)
    defense_strength = Column(Float, nullable=True, default=1.0)
    form_trend_slope = Column(Float, nullable=True, default=0.0)
    over25_rate = Column(Float, nullable=True, default=0.0)
    btts_rate = Column(Float, nullable=True, default=0.0)
    rest_days = Column(Integer, nullable=True)
    schedule_congestion = Column(Float, nullable=True, default=0.0)

    # Home/away split deltas
    home_win_rate = Column(Float, nullable=True)
    away_win_rate = Column(Float, nullable=True)
    home_goals_avg = Column(Float, nullable=True)
    away_goals_avg = Column(Float, nullable=True)

    # League position context
    league_position = Column(Integer, nullable=True)
    opponent_league_position = Column(Integer, nullable=True)

    snapshot_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<EventStatsSnapshot event={self.event_id} team={self.team} "
            f"atk={self.attack_strength} def={self.defense_strength}>"
        )
