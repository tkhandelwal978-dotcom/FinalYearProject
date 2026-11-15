"""SQLAlchemy engine/session setup. Swap DATABASE_URL to move from SQLite to PostgreSQL."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def create_db_engine(database_url: str) -> Engine:
    connect_args = {}
    if database_url.startswith("sqlite"):
        # FastAPI may use a session from a different thread than the one that opened it.
        connect_args["check_same_thread"] = False
        db_file = database_url.split("///", 1)[-1]
        if db_file and db_file != ":memory:":
            Path(db_file).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)


class Database:
    def __init__(self, database_url: str) -> None:
        self.engine = create_db_engine(database_url)
        self.session_factory = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)

    def create_tables(self) -> None:
        from app.models import prediction  # noqa: F401  (registers the model on Base.metadata)

        Base.metadata.create_all(self.engine)

    def session(self) -> Iterator[Session]:
        db = self.session_factory()
        try:
            yield db
        finally:
            db.close()
