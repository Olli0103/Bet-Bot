"""add is_training_data and data_source columns to placed_bets

Revision ID: e7f4a28d1c03
Revises: c4f2b83d9a12
Create Date: 2026-03-04 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e7f4a28d1c03'
down_revision: Union[str, Sequence[str], None] = 'c4f2b83d9a12'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('placed_bets', sa.Column(
        'is_training_data', sa.Boolean(), nullable=False, server_default='false',
    ))
    op.add_column('placed_bets', sa.Column(
        'data_source', sa.String(32), nullable=False, server_default='live_trade',
    ))
    # Index for fast filtering in dashboard/PnL queries
    op.create_index('ix_placed_bets_data_source', 'placed_bets', ['data_source'])
    op.create_index('ix_placed_bets_is_training', 'placed_bets', ['is_training_data'])


def downgrade() -> None:
    op.drop_index('ix_placed_bets_is_training', table_name='placed_bets')
    op.drop_index('ix_placed_bets_data_source', table_name='placed_bets')
    op.drop_column('placed_bets', 'data_source')
    op.drop_column('placed_bets', 'is_training_data')
