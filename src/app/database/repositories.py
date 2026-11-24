from datetime import timedelta

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert

from app.database.connection import session_scope
from app.database.models import EmailDeliveryRecord, SupportTicketRecord


_DELIVERY_CLAIM_TIMEOUT = timedelta(minutes=10)


class SupportTicketRepository:
    def create_or_get(
        self,
        thread_id: str,
        ticket: dict,
    ) -> dict:
        with session_scope() as session:
            statement = (
                insert(SupportTicketRecord)
                .values(
                    thread_id=thread_id,
                    ticket_id=ticket["ticket_id"],
                    ticket_data=ticket,
                )
                .on_conflict_do_nothing(index_elements=["thread_id"])
            )
            session.execute(statement)
            record = session.scalar(
                select(SupportTicketRecord).where(
                    SupportTicketRecord.thread_id == thread_id
                )
            )
            if record is None:
                raise RuntimeError(
                    "Ticket persistence failed for the requested thread."
                )
            return record.ticket_data


class EmailDeliveryRepository:

    def claim(
        self,
        idempotency_key: str,
    ) -> tuple[bool, tuple[str, str | None] | None]:

        with session_scope() as session:

            # 1. Try to insert a new delivery claim.
            statement = (
                insert(EmailDeliveryRecord)
                .values(
                    idempotency_key=idempotency_key,
                    status="sending",
                    error=None,
                )
                .on_conflict_do_nothing(
                    index_elements=[
                        EmailDeliveryRecord.idempotency_key
                    ]
                )
                .returning(
                    EmailDeliveryRecord.idempotency_key
                )
            )

            inserted_key = session.execute(
                statement
            ).scalar_one_or_none()

            if inserted_key is not None:
                return True, None

            # 2. Existing claim found.
            # Try reclaiming an expired sending claim.
            reclaim_statement = (
                update(EmailDeliveryRecord)
                .where(
                    EmailDeliveryRecord.idempotency_key
                    == idempotency_key,
                    EmailDeliveryRecord.status == "sending",
                    EmailDeliveryRecord.updated_at
                    < func.now() - _DELIVERY_CLAIM_TIMEOUT,
                )
                .values(
                    status="sending",
                    error=None,
                    updated_at=func.now(),
                )
                .returning(
                    EmailDeliveryRecord.idempotency_key
                )
            )

            reclaimed_key = session.execute(
                reclaim_statement
            ).scalar_one_or_none()

            if reclaimed_key is not None:
                return True, None

            # 3. Read the existing delivery record.
            record = session.execute(
                select(
                    EmailDeliveryRecord.status,
                    EmailDeliveryRecord.error,
                ).where(
                    EmailDeliveryRecord.idempotency_key
                    == idempotency_key
                )
            ).one_or_none()

            if record is None:
                raise RuntimeError(
                    "Email delivery claim could not be loaded."
                )

            return False, (record.status, record.error)

    def mark_sent(self, idempotency_key: str) -> None:
        self._update_status(idempotency_key, "sent")

    def mark_failed(
        self,
        idempotency_key: str,
        error: str,
    ) -> None:
        self._update_status(
            idempotency_key,
            "failed",
            error,
        )

    @staticmethod
    def _update_status(
        idempotency_key: str,
        status: str,
        error: str | None = None,
    ) -> None:

        with session_scope() as session:

            statement = (
                update(EmailDeliveryRecord)
                .where(
                    EmailDeliveryRecord.idempotency_key
                    == idempotency_key
                )
                .values(
                    status=status,
                    error=error,
                    updated_at=func.now(),
                )
                .returning(
                    EmailDeliveryRecord.idempotency_key
                )
            )

            updated_key = session.execute(
                statement
            ).scalar_one_or_none()

            if updated_key is None:
                raise RuntimeError(
                    "Email delivery status could not be persisted."
                )
