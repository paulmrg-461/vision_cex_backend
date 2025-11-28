from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.core.config.environment_config import EnvironmentConfig


class Base(DeclarativeBase):
    pass


class SQLAlchemySession:
    _engine = None
    _SessionLocal: Optional[sessionmaker] = None

    @classmethod
    def init(cls):
        if cls._engine is None:
            cfg = EnvironmentConfig()
            dsn = f"postgresql+psycopg2://{cfg.db_user}:{cfg.db_password}@{cfg.db_host}:{cfg.db_port}/{cfg.db_name}"
            cls._engine = create_engine(dsn, pool_pre_ping=True, future=True)
            cls._SessionLocal = sessionmaker(bind=cls._engine, autoflush=False, autocommit=False, future=True)
        return cls._engine, cls._SessionLocal

    @classmethod
    def session(cls):
        _, SessionLocal = cls.init()
        return SessionLocal()

