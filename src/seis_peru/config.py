"""Configuración central (rutas de datos, conexión Postgres, HTTP).

Se puede sobrescribir por variables de entorno con prefijo ``SEIS_`` o un
archivo ``.env`` en la raíz del proyecto (ver ``.env.example``).
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# .../src/seis_peru/config.py -> raíz del repo
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEIS_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Rutas ----
    data_dir: Path = PROJECT_ROOT / "data"

    # ---- HTTP / FDSN ----
    http_timeout: float = 120.0
    http_max_retries: int = 5
    http_backoff: float = 1.5  # segundos base para el backoff exponencial
    user_agent: str = "SEIS-PERU/0.1 (research; martinlinohuaney@gmail.com)"

    # ---- PostgreSQL / PostGIS ----
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_db: str = "seis_peru"
    pg_user: str = "seis"
    pg_password: str = "seis"

    # ---- Rutas derivadas (data lake) ----
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    def dsn(self) -> str:
        """DSN estilo libpq para psycopg2."""
        return (
            f"host={self.pg_host} port={self.pg_port} dbname={self.pg_db} "
            f"user={self.pg_user} password={self.pg_password}"
        )

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.interim_dir, self.processed_dir, self.features_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
