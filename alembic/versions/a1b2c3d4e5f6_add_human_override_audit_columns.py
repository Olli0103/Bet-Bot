"""add human override audit columns to placed_bets

Revision ID: a1b2c3d4e5f6
Revises: f8a5b39e4d05
Create Date: 2026-03-05 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'f8a5b39e4d05'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add GDPR Art. 22 / GlüStV DSS compliance columns to placed_bets.

    These columns record the human operator's review decision for every
    AI-generated tip, satisfying the "meaningful human intervention"
    requirement under GDPR Article 22 and GlüStV 2021 §4(4).

    Columns:
        operator_id:     Telegram user who reviewed the tip
        confirmed_odds:  Odds at the moment of human confirmation
        confirmed_stake: Actual stake placed (may differ from AI recommendation)
        human_action:    placed / skipped / adjusted / rejected
        override_reason: Free-text reason when action != 'placed'
        reviewed_at:     UTC timestamp of the human decision
    """
    op.add_column('placed_bets', sa.Column('operator_id', sa.String(64), nullable=True))
    op.add_column('placed_bets', sa.Column('confirmed_odds', sa.Float(), nullable=True))
    op.add_column('placed_bets', sa.Column('confirmed_stake', sa.Float(), nullable=True))
    op.add_column('placed_bets', sa.Column('human_action', sa.String(16), nullable=True))
    op.add_column('placed_bets', sa.Column('override_reason', sa.Text(), nullable=True))
    op.add_column('placed_bets', sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column('placed_bets', 'reviewed_at')
    op.drop_column('placed_bets', 'override_reason')
    op.drop_column('placed_bets', 'human_action')
    op.drop_column('placed_bets', 'confirmed_stake')
    op.drop_column('placed_bets', 'confirmed_odds')
    op.drop_column('placed_bets', 'operator_id')
