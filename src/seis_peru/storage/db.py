"""Carga a PostgreSQL/PostGIS (opcional en Fase 1).

El artefacto principal de la Fase 1 es el Parquet canónico; la BD es para
consultas espaciales y las fases siguientes. Requiere el contenedor Docker
levantado (docker/docker-compose.yml).

Uso:
    from seis_peru.storage import db
    db.load_observations(obs_df)     # rellena core.observations (+ canonical_id)
    db.load_events(events_df)        # rellena core.events
    db.link_members()                # rellena core.event_members
"""
from __future__ import annotations

import logging
import math

import pandas as pd

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:  # pragma: no cover
    psycopg2 = None  # type: ignore

from ..config import settings

log = logging.getLogger("seis.db")


def _connect():
    if psycopg2 is None:
        raise RuntimeError("psycopg2 no instalado: pip install -r requirements.txt")
    return psycopg2.connect(settings.dsn())


def _clean(v):
    """Normaliza NaN/NaT -> None para psycopg2."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if pd.isna(v):
        return None
    return v


def ping() -> str:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT postgis_full_version();")
        return cur.fetchone()[0]


def load_observations(df: pd.DataFrame, page_size: int = 5000) -> int:
    """Upsert de observaciones normalizadas en core.observations.

    ``df`` debe incluir la columna ``canonical_id`` (de dedup.deduplicate).
    """
    cols = [
        "source", "source_event_id", "origin_time", "longitude", "latitude",
        "depth_km", "magnitude", "magnitude_type", "mw", "mw_method",
        "author", "place", "country", "canonical_id",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    records = [
        tuple(_clean(r[c]) for c in cols) for _, r in df[cols].iterrows()
    ]
    sql = """
        INSERT INTO core.observations
            (source, source_event_id, origin_time, geom, depth_km, magnitude,
             magnitude_type, mw, mw_method, author, place, country, canonical_id)
        VALUES %s
        ON CONFLICT (source, source_event_id) DO UPDATE SET
            canonical_id = EXCLUDED.canonical_id,
            mw           = EXCLUDED.mw,
            mw_method    = EXCLUDED.mw_method,
            country      = EXCLUDED.country;
    """
    # geom se construye desde lon/lat con ST_MakePoint.
    template = (
        "(%s,%s,%s, ST_SetSRID(ST_MakePoint(%s,%s),4326), "
        "%s,%s,%s,%s,%s,%s,%s,%s)"
    )
    with _connect() as conn, conn.cursor() as cur:
        execute_values(cur, sql, records, template=template, page_size=page_size)
        conn.commit()
    log.info("core.observations: %d filas cargadas", len(records))
    return len(records)


def load_events(df: pd.DataFrame, page_size: int = 5000) -> int:
    """Carga el catálogo canónico en core.events."""
    cols = [
        "canonical_id", "origin_time", "longitude", "latitude", "depth_km",
        "preferred_mw", "preferred_mag", "preferred_mag_type",
        "preferred_source", "mag_source", "n_sources", "country",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    records = [tuple(_clean(r[c]) for c in cols) for _, r in df[cols].iterrows()]
    sql = """
        INSERT INTO core.events
            (canonical_id, origin_time, geom, depth_km, preferred_mw,
             preferred_mag, preferred_mag_type, preferred_source, mag_source,
             n_sources, country)
        VALUES %s
        ON CONFLICT (canonical_id) DO NOTHING;
    """
    template = (
        "(%s,%s, ST_SetSRID(ST_MakePoint(%s,%s),4326), %s,%s,%s,%s,%s,%s,%s,%s)"
    )
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE core.events RESTART IDENTITY CASCADE;")
        execute_values(cur, sql, records, template=template, page_size=page_size)
        conn.commit()
    log.info("core.events: %d filas cargadas", len(records))
    return len(records)


def link_members() -> int:
    """Rellena core.event_members desde core.observations.canonical_id."""
    sql = """
        INSERT INTO core.event_members (canonical_id, obs_id)
        SELECT o.canonical_id, o.obs_id
        FROM core.observations o
        WHERE o.canonical_id IS NOT NULL
        ON CONFLICT DO NOTHING;
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql)
        n = cur.rowcount
        conn.commit()
    log.info("core.event_members: %d enlaces", n)
    return n
