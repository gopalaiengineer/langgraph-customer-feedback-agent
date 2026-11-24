import unittest
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sqlalchemy.dialects import postgresql

from app.database.repositories import EmailDeliveryRepository


class EmailDeliveryRepositoryTests(unittest.TestCase):
    def _claim_with_existing_record(self, reclaim_rowcount):
        session = Mock()
        session.execute.side_effect = [
            Mock(scalar_one_or_none=Mock(return_value=None)),
            Mock(
                scalar_one_or_none=Mock(
                    return_value="thread:customer" if reclaim_rowcount else None
                )
            ),
            Mock(
                one_or_none=Mock(
                    return_value=SimpleNamespace(
                        status="sending",
                        error=None,
                    )
                )
            ),
        ]

        @contextmanager
        def scoped_session():
            yield session

        with patch(
            "app.database.repositories.session_scope",
            side_effect=lambda: scoped_session(),
        ):
            result = EmailDeliveryRepository().claim("thread:customer")

        return result, session

    def test_claim_reclaims_a_stale_sending_record_atomically(self):
        result, session = self._claim_with_existing_record(reclaim_rowcount=1)

        self.assertEqual(result, (True, None))
        self.assertEqual(session.execute.call_count, 2)
        reclaim_statement = session.execute.call_args_list[1].args[0]
        compiled = reclaim_statement.compile(dialect=postgresql.dialect())
        self.assertIn("email_deliveries.updated_at <", str(compiled))
        self.assertIn(timedelta(minutes=10), compiled.params.values())

    def test_claim_does_not_reclaim_an_active_sending_record(self):
        result, session = self._claim_with_existing_record(reclaim_rowcount=0)

        self.assertEqual(result, (False, ("sending", None)))
        self.assertEqual(session.execute.call_count, 3)
        read_statement = session.execute.call_args_list[2].args[0]
        self.assertIn("email_deliveries.status", str(read_statement))

    def test_claim_creates_a_new_delivery_record(self):
        session = Mock()
        session.execute.return_value.scalar_one_or_none.return_value = (
            "thread:customer"
        )

        @contextmanager
        def scoped_session():
            yield session

        with patch(
            "app.database.repositories.session_scope",
            side_effect=lambda: scoped_session(),
        ):
            result = EmailDeliveryRepository().claim("thread:customer")

        self.assertEqual(result, (True, None))
        session.execute.assert_called_once()
        statement = session.execute.call_args.args[0]
        self.assertIn("ON CONFLICT", str(statement.compile(dialect=postgresql.dialect())))
        self.assertIn("RETURNING", str(statement.compile(dialect=postgresql.dialect())))


if __name__ == "__main__":
    unittest.main()
