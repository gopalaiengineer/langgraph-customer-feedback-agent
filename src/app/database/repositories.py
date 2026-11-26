from datetime import datetime, timedelta, timezone

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

    def list_all(self) -> list[dict]:
        with session_scope() as session:
            records = session.scalars(
                select(SupportTicketRecord).order_by(
                    SupportTicketRecord.created_at.desc()
                )
            ).all()
            return [
                {
                    **record.ticket_data,
                    "thread_id": record.thread_id,
                    "ticket_id": record.ticket_id,
                    "created_at": record.created_at.isoformat(),
                }
                for record in records
            ]

    def get_by_ticket_id(self, ticket_id: str) -> dict | None:
        with session_scope() as session:
            record = session.scalar(
                select(SupportTicketRecord).where(
                    SupportTicketRecord.ticket_id == ticket_id
                )
            )
            if record is None:
                return None
            return {
                "thread_id": record.thread_id,
                "ticket_id": record.ticket_id,
                "ticket_data": dict(record.ticket_data),
                "created_at": record.created_at.isoformat(),
            }

    def begin_resolution(
        self,
        thread_id: str,
        reason: str,
        email_subject: str,
        email_body: str,
    ) -> dict:
        with session_scope() as session:
            record = session.scalar(
                select(SupportTicketRecord)
                .where(SupportTicketRecord.thread_id == thread_id)
                .with_for_update()
            )
            if record is None:
                raise LookupError("The support ticket no longer exists.")

            ticket_data = dict(record.ticket_data)
            if ticket_data.get("status") in {"resolved", "rejected"}:
                raise ValueError(
                    f"This support ticket has already been "
                    f"{ticket_data['status']}."
                )

            pending = ticket_data.get("resolution_pending")
            if pending is not None:
                if pending.get("reason") != reason:
                    raise ValueError(
                        "A different resolution is already pending "
                        "for this ticket."
                    )
                return pending

            pending = {
                "reason": reason,
                "email_subject": email_subject,
                "email_body": email_body,
            }
            ticket_data["resolution_pending"] = pending
            record.ticket_data = ticket_data
            return pending

    def mark_resolved(
        self,
        thread_id: str,
        reason: str,
        admin_username: str,
        email_subject: str,
        email_body: str,
    ) -> dict:
        with session_scope() as session:
            record = session.scalar(
                select(SupportTicketRecord)
                .where(SupportTicketRecord.thread_id == thread_id)
                .with_for_update()
            )
            if record is None:
                raise LookupError("The support ticket no longer exists.")

            ticket_data = dict(record.ticket_data)
            if ticket_data.get("status") == "resolved":
                return ticket_data
            if ticket_data.get("status") == "rejected":
                raise ValueError("A rejected ticket cannot be resolved.")

            pending = ticket_data.get("resolution_pending")
            if pending != {
                "reason": reason,
                "email_subject": email_subject,
                "email_body": email_body,
            }:
                raise RuntimeError(
                    "The pending resolution changed before it could be saved."
                )

            history = ticket_data.get("resolution_history", [])
            if not isinstance(history, list):
                raise RuntimeError("The ticket resolution history is invalid.")
            history.append(
                {
                    "resolved_at": datetime.now(timezone.utc).isoformat(),
                    "resolved_by": admin_username,
                    "reason": reason,
                    "email_subject": email_subject,
                    "email_body": email_body,
                    "email_status": "sent",
                }
            )
            ticket_data["status"] = "resolved"
            ticket_data["resolution_reason"] = reason
            ticket_data["resolution_history"] = history
            ticket_data.pop("resolution_pending", None)
            record.ticket_data = ticket_data
            return ticket_data

    def mark_rejected(self, thread_id: str) -> dict:
        with session_scope() as session:
            record = session.scalar(
                select(SupportTicketRecord)
                .where(SupportTicketRecord.thread_id == thread_id)
                .with_for_update()
            )
            if record is None:
                raise LookupError("The support ticket no longer exists.")

            ticket_data = dict(record.ticket_data)
            if ticket_data.get("status") == "resolved":
                raise ValueError("A resolved ticket cannot be rejected.")
            if ticket_data.get("status") == "rejected":
                return ticket_data
            if ticket_data.get("resolution_pending") is not None:
                raise ValueError(
                    "A resolution email is already pending for this ticket."
                )

            ticket_data["status"] = "rejected"
            record.ticket_data = ticket_data
            return ticket_data


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
