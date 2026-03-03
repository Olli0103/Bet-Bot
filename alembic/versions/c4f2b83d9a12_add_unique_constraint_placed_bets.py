"""add unique constraint on placed_bets (event_id, selection, market)

Revision ID: c4f2b83d9a12
Revises: b3e1a92c7f01
Create Date: 2026-03-03 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'c4f2b83d9a12'
down_revision: Union[str, Sequence[str], None] = 'b3e1a92c7f01'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_event_selection_market",
        "placed_bets",
        ["event_id", "selection", "market"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_event_selection_market", "placed_bets", type_="unique")
