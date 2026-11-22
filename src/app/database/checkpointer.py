import atexit
from threading import Lock

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.database.connection import get_checkpointer_url

_lock = Lock()
_pool: ConnectionPool | None = None
_checkpointer: PostgresSaver | None = None


def get_checkpointer() -> PostgresSaver:
    global _checkpointer, _pool
    if _checkpointer is not None:
        return _checkpointer

    with _lock:
        if _checkpointer is not None:
            return _checkpointer

        pool = ConnectionPool(
            conninfo=get_checkpointer_url(),
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
            min_size=1,
            max_size=5,
            open=False,
        )
        try:
            pool.open(wait=True)
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
        except Exception:
            pool.close()
            raise

        _pool = pool
        _checkpointer = checkpointer
        return checkpointer


def close_checkpointer_pool() -> None:
    global _checkpointer, _pool
    with _lock:
        if _pool is not None:
            _pool.close()
            _pool = None
            _checkpointer = None


atexit.register(close_checkpointer_pool)
