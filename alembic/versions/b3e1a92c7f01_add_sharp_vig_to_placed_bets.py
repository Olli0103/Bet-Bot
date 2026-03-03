"""add sharp_vig column to placed_bets

Revision ID: b3e1a92c7f01
Revises: fa22cb7d899e
Create Date: 2026-03-03 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3e1a92c7f01'
down_revision: Union[str, Sequence[str], None] = 'fa22cb7d899e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('placed_bets', sa.Column('sharp_vig', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('placed_bets', 'sharp_vig')
