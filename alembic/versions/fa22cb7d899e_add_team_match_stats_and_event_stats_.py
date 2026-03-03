"""add team_match_stats and event_stats_snapshots tables

Revision ID: fa22cb7d899e
Revises:
Create Date: 2026-03-03 12:29:33.404883

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = 'fa22cb7d899e'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create team_match_stats and event_stats_snapshots tables."""
    # --- team_match_stats ---
    op.create_table(
        "team_match_stats",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("source_match_id", sa.String(128), nullable=False, index=True),
        sa.Column("sport", sa.String(64), nullable=False),
        sa.Column("league", sa.String(128), nullable=True),
        sa.Column("season", sa.String(32), nullable=True),
        sa.Column("matchday", sa.Integer, nullable=True),
        sa.Column("match_date", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("team", sa.String(256), nullable=False),
        sa.Column("opponent", sa.String(256), nullable=False),
        sa.Column("is_home", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("goals_for", sa.Integer, nullable=True, server_default="0"),
        sa.Column("goals_against", sa.Integer, nullable=True, server_default="0"),
        sa.Column("result", sa.String(4), nullable=True),
        sa.Column("shots", sa.Integer, nullable=True),
        sa.Column("shots_on_target", sa.Integer, nullable=True),
        sa.Column("possession_pct", sa.Float, nullable=True),
        sa.Column("corners", sa.Integer, nullable=True),
        sa.Column("fouls", sa.Integer, nullable=True),
        sa.Column("yellow_cards", sa.Integer, nullable=True),
        sa.Column("red_cards", sa.Integer, nullable=True),
        sa.Column("ht_goals_for", sa.Integer, nullable=True),
        sa.Column("ht_goals_against", sa.Integer, nullable=True),
        sa.Column("extra_stats", JSONB, nullable=True),
        sa.Column("source", sa.String(64), nullable=False, server_default="thesportsdb"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("source_match_id", "team", "source", name="uq_match_team_source"),
    )
    op.create_index("ix_tms_team_date", "team_match_stats", ["team", "match_date"])
    op.create_index("ix_tms_sport_date", "team_match_stats", ["sport", "match_date"])

    # --- event_stats_snapshots ---
    op.create_table(
        "event_stats_snapshots",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("event_id", sa.String(128), nullable=False, index=True),
        sa.Column("sport", sa.String(64), nullable=False),
        sa.Column("team", sa.String(256), nullable=False),
        sa.Column("is_home", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("matches_played", sa.Integer, nullable=True, server_default="0"),
        sa.Column("wins", sa.Integer, nullable=True, server_default="0"),
        sa.Column("draws", sa.Integer, nullable=True, server_default="0"),
        sa.Column("losses", sa.Integer, nullable=True, server_default="0"),
        sa.Column("goals_scored_avg", sa.Float, nullable=True, server_default="0.0"),
        sa.Column("goals_conceded_avg", sa.Float, nullable=True, server_default="0.0"),
        sa.Column("clean_sheets", sa.Integer, nullable=True, server_default="0"),
        sa.Column("attack_strength", sa.Float, nullable=True, server_default="1.0"),
        sa.Column("defense_strength", sa.Float, nullable=True, server_default="1.0"),
        sa.Column("form_trend_slope", sa.Float, nullable=True, server_default="0.0"),
        sa.Column("over25_rate", sa.Float, nullable=True, server_default="0.0"),
        sa.Column("btts_rate", sa.Float, nullable=True, server_default="0.0"),
        sa.Column("rest_days", sa.Integer, nullable=True),
        sa.Column("schedule_congestion", sa.Float, nullable=True, server_default="0.0"),
        sa.Column("home_win_rate", sa.Float, nullable=True),
        sa.Column("away_win_rate", sa.Float, nullable=True),
        sa.Column("home_goals_avg", sa.Float, nullable=True),
        sa.Column("away_goals_avg", sa.Float, nullable=True),
        sa.Column("league_position", sa.Integer, nullable=True),
        sa.Column("opponent_league_position", sa.Integer, nullable=True),
        sa.Column("snapshot_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("event_id", "team", name="uq_event_team_snapshot"),
    )


def downgrade() -> None:
    """Drop team_match_stats and event_stats_snapshots tables."""
    op.drop_table("event_stats_snapshots")
    op.drop_index("ix_tms_sport_date", table_name="team_match_stats")
    op.drop_index("ix_tms_team_date", table_name="team_match_stats")
    op.drop_table("team_match_stats")
