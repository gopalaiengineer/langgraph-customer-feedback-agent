from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert

from app.database.connection import session_scope
from app.database.models import EmailDeliveryRecord, SupportTicketRecord


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
            statement = (
                insert(EmailDeliveryRecord)
                .values(idempotency_key=idempotency_key, status="sending")
                .on_conflict_do_nothing(
                    index_elements=["idempotency_key"]
                )
            )
            result = session.execute(statement)
            if result.rowcount == 1:
                return True, None

            record = session.get(EmailDeliveryRecord, idempotency_key)
            if record is None:
                raise RuntimeError(
                    "Email delivery claim could not be loaded."
                )
            return False, (record.status, record.error)

    def mark_sent(self, idempotency_key: str) -> None:
        self._update_status(idempotency_key, "sent")

    def mark_failed(self, idempotency_key: str, error: str) -> None:
        self._update_status(idempotency_key, "failed", error)

    @staticmethod
    def _update_status(
        idempotency_key: str,
        status: str,
        error: str | None = None,
    ) -> None:
        with session_scope() as session:
            result = session.execute(
                update(EmailDeliveryRecord)
                .where(
                    EmailDeliveryRecord.idempotency_key == idempotency_key
                )
                .values(status=status, error=error, updated_at=func.now())
            )
            if result.rowcount != 1:
                raise RuntimeError(
                    "Email delivery status could not be persisted."
                )
