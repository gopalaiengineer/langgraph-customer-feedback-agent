"""Create support ticket and email delivery tables.

Revision ID: 20261118_01
Revises:
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20261118_01"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    existing_tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "support_tickets" not in existing_tables:
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
    if "email_deliveries" not in existing_tables:
        op.create_table(
            "email_deliveries",
            sa.Column(
                "idempotency_key",
                sa.String(length=512),
                nullable=False,
            ),
            sa.Column("status", sa.String(length=16), nullable=False),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.CheckConstraint(
                "status IN ('sending', 'sent', 'failed')",
                name="ck_email_deliveries_status",
            ),
            sa.PrimaryKeyConstraint("idempotency_key"),
        )


def downgrade() -> None:
    op.drop_table("email_deliveries")
    op.drop_table("support_tickets")
