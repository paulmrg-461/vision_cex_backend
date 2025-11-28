import psycopg2
from psycopg2.extras import Json
from typing import Any

from app.core.config.environment_config import EnvironmentConfig
from app.core.utils.logger import get_logger

_logger = get_logger("db")


class PostgresClient:
    _conn = None

    @classmethod
    def get_conn(cls):
        if cls._conn is None or cls._conn.closed:
            cfg = EnvironmentConfig()
            cls._conn = psycopg2.connect(
                host=cfg.db_host,
                port=cfg.db_port,
                user=cfg.db_user,
                password=cfg.db_password,
                dbname=cfg.db_name,
            )
            cls._conn.autocommit = True
            cls._bootstrap(cls._conn)
        return cls._conn

    @staticmethod
    def _bootstrap(conn):
        """Ensure required extensions and tables exist."""
        with conn.cursor() as cur:
            # Enable extension for jsonb operations if needed
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
            except Exception:
                pass
            # Create table bus_reports
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS bus_reports (
                  id SERIAL PRIMARY KEY,
                  license_plate VARCHAR(6) NOT NULL CHECK (license_plate ~ '^[A-Z]{3}[0-9]{3}$'),
                  event_datetime TIMESTAMPTZ NOT NULL,
                  damages JSONB NOT NULL DEFAULT '[]'::jsonb
                );
                """
            )
            # Useful indexes
            try:
                cur.execute("CREATE INDEX IF NOT EXISTS idx_bus_reports_plate ON bus_reports (license_plate);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_bus_reports_event ON bus_reports (event_datetime);")
            except Exception:
                pass

    @classmethod
    def execute(cls, sql: str, params: tuple = ()):
        conn = cls.get_conn()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            try:
                return cur.fetchall()
            except Exception:
                return None
