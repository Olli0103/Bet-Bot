"""add owner_chat_id to placed_bets + update unique constraint

Revision ID: f8a5b39e4d05
Revises: e7f4a28d1c03
Create Date: 2026-03-04 20:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f8a5b39e4d05'
down_revision: Union[str, Sequence[str], None] = 'e7f4a28d1c03'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add owner_chat_id column (nullable for backward compat)
    op.add_column('placed_bets', sa.Column(
        'owner_chat_id', sa.String(64), nullable=True,
    ))
    op.create_index('ix_placed_bets_owner_chat_id', 'placed_bets', ['owner_chat_id'])

    # 2. Drop old unique constraint and create new owner+source-scoped one
    #    The old constraint prevented different users/sources from having
    #    the same event+selection+market combination.
    try:
        op.drop_constraint('uq_event_selection_market', 'placed_bets', type_='unique')
    except Exception:
        pass  # Constraint may not exist in all environments

    op.create_unique_constraint(
        'uq_event_sel_market_owner_source',
        'placed_bets',
        ['event_id', 'selection', 'market', 'owner_chat_id', 'data_source'],
    )


def downgrade() -> None:
    try:
        op.drop_constraint('uq_event_sel_market_owner_source', 'placed_bets', type_='unique')
    except Exception:
        pass
    op.drop_index('ix_placed_bets_owner_chat_id', table_name='placed_bets')
    op.drop_column('placed_bets', 'owner_chat_id')

    # Restore original constraint (may fail if duplicates exist)
    try:
        op.create_unique_constraint(
            'uq_event_selection_market',
            'placed_bets',
            ['event_id', 'selection', 'market'],
        )
    except Exception:
        pass
