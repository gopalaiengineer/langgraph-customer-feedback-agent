"""Restore the support tickets table if it was removed.

Revision ID: 20261123_01
Revises: 20261118_01
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20261123_01"
down_revision: Union[str, None] = "20261118_01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if not sa.inspect(op.get_bind()).has_table("support_tickets"):
        op.create_table(
            "support_tickets",
            sa.Column("thread_id", sa.String(length=255), nullable=False),
            sa.Column("ticket_id", sa.String(length=15), nullable=False),
            sa.Column(
                "ticket_data",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("thread_id"),
            sa.UniqueConstraint("ticket_id"),
        )


def downgrade() -> None:
    pass
