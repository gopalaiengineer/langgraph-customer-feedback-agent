import atexit
from contextlib import contextmanager
from functools import lru_cache
from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker

from app.configs.config import settings


def _database_url() -> URL:
    if settings.DATABASE_URL is None:
        raise RuntimeError(
            "DATABASE_URL must be configured to use PostgreSQL persistence."
        )

    url = make_url(settings.DATABASE_URL.get_secret_value())
    if url.drivername not in {
        "postgres",
        "postgresql",
        "postgresql+psycopg",
    }:
        raise ValueError("DATABASE_URL must use PostgreSQL.")
    return url.set(drivername="postgresql+psycopg")


def get_checkpointer_url() -> str:
    return _database_url().set(drivername="postgresql").render_as_string(
        hide_password=False
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(
        _database_url(),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
    )


@lru_cache(maxsize=1)
def _get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    with _get_session_factory().begin() as session:
        yield session


def close_database_connections() -> None:
    if get_engine.cache_info().currsize:
        get_engine().dispose()


atexit.register(close_database_connections)
