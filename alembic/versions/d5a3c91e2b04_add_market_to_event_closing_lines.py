"""add market column to event_closing_lines

Revision ID: d5a3c91e2b04
Revises: b3e1a92c7f01
Create Date: 2026-03-04 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d5a3c91e2b04"
down_revision: Union[str, None] = "b3e1a92c7f01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add market column with default 'h2h' (all existing rows are h2h)
    op.add_column(
        "event_closing_lines",
        sa.Column("market", sa.String(32), nullable=False, server_default="h2h"),
    )

    # Drop old unique constraint and create new one including market
    op.drop_constraint("uq_closing_event_sel", "event_closing_lines", type_="unique")
    op.create_unique_constraint(
        "uq_closing_event_sel_mkt",
        "event_closing_lines",
        ["event_id", "selection", "market"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_closing_event_sel_mkt", "event_closing_lines", type_="unique")
    op.drop_column("event_closing_lines", "market")
    op.create_unique_constraint(
        "uq_closing_event_sel",
        "event_closing_lines",
        ["event_id", "selection"],
    )
