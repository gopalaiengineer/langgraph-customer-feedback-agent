from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SupportTicketRecord(Base):
    __tablename__ = "support_tickets"

    thread_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    ticket_id: Mapped[str] = mapped_column(String(15), unique=True, nullable=False)
    ticket_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    gess: Mapped[str] = mapped_column(String(15), nullable=True)


class EmailDeliveryRecord(Base):
    __tablename__ = "email_deliveries"
    __table_args__ = (
        CheckConstraint(
            "status IN ('sending', 'sent', 'failed')",
            name="ck_email_deliveries_status",
        ),
    )

    idempotency_key: Mapped[str] = mapped_column(String(512), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
