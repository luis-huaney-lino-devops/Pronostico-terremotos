"""Orquestación de la ingesta (Fase 1).

    fuentes (USGS/EMSC/ISC vía FDSN + IGP vía CSV)
        -> normalizar a NormalizedEvent
        -> guardar RAW (inmutable)
        -> concatenar observaciones
        -> deduplicar (canonical_id)
        -> construir catálogo canónico
        -> Parquet (data/interim, data/processed)
        -> [opcional] cargar a PostGIS

Garantía anti-leakage: esto solo construye el CATÁLOGO (hechos observados).
El feature-engineering con corte temporal viene en fases posteriores.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from .config import settings
from .dedup import DedupConfig, build_canonical_events, deduplicate
from .ingestion import fdsn, gcmt, igp, iscgem
from .regions import STUDY_REGION, BBox
from .storage import raw_store

log = logging.getLogger("seis.pipeline")

FDSN_SOURCES = ("usgs", "emsc", "isc")
ALL_SOURCES = FDSN_SOURCES + ("igp",)
# 'gcmt' (FDSN vía ComCat) e 'iscgem' (CSV local) son fuentes Mw opcionales.


def _to_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _ingest_fdsn(source, start, end, bbox, min_magnitude) -> pd.DataFrame:
    native = fdsn.fetch_native(source, start, end, bbox, min_magnitude=min_magnitude)
    if native:
        raw_store.save_raw(source, native, start, end, bbox, min_magnitude)
    events = [fdsn.native_to_normalized(source, r) for r in native]
    rows = [e.to_row() for e in events if e is not None]
    return pd.DataFrame(rows)


def _ingest_igp(start, end, bbox, min_magnitude) -> pd.DataFrame:
    events = igp.fetch_igp()
    rows = [e.to_row() for e in events if e is not None]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # El CSV del IGP es histórico completo: filtramos a ventana + bbox aquí.
    df = df[(df["origin_time"] >= start) & (df["origin_time"] <= end)]
    df = df[df.apply(lambda r: bbox.contains(r["latitude"], r["longitude"]), axis=1)]
    if min_magnitude is not None:
        df = df[df["magnitude"].fillna(-99) >= min_magnitude]
    return df


def _ingest_catalog_file(events, start, end, bbox, min_magnitude) -> pd.DataFrame:
    """Filtro común para fuentes tipo-archivo (IGP/ISC-GEM/GCMT)."""
    rows = [e.to_row() for e in events if e is not None]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[(df["origin_time"] >= start) & (df["origin_time"] <= end)]
    df = df[df.apply(lambda r: bbox.contains(r["latitude"], r["longitude"]), axis=1)]
    if min_magnitude is not None:
        df = df[df["magnitude"].fillna(-99) >= min_magnitude]
    return df


def _ingest_iscgem(start, end, bbox, min_magnitude) -> pd.DataFrame:
    return _ingest_catalog_file(iscgem.fetch_iscgem(), start, end, bbox, min_magnitude)


def _ingest_gcmt(start, end, bbox, min_magnitude) -> pd.DataFrame:
    return _ingest_catalog_file(gcmt.fetch_gcmt(), start, end, bbox, min_magnitude)


def run(
    start: datetime,
    end: datetime,
    sources: tuple[str, ...] = ALL_SOURCES,
    bbox: BBox = STUDY_REGION,
    min_magnitude: float | None = None,
    dedup_config: DedupConfig | None = None,
    load_db: bool = False,
) -> dict:
    """Ejecuta la ingesta completa. Devuelve un resumen con conteos y rutas."""
    settings.ensure_dirs()
    start, end = _to_utc(start), _to_utc(end)
    log.info(
        "INGESTA %s .. %s | fuentes=%s | bbox=%s | Mmin=%s",
        start.date(), end.date(), sources, bbox.__dict__, min_magnitude,
    )

    frames = []
    per_source = {}
    for src in sources:
        try:
            if src == "igp":
                df = _ingest_igp(start, end, bbox, min_magnitude)
            elif src == "iscgem":
                df = _ingest_iscgem(start, end, bbox, min_magnitude)
            elif src == "gcmt":
                df = _ingest_gcmt(start, end, bbox, min_magnitude)
            elif src in fdsn.FDSN_ENDPOINTS:
                df = _ingest_fdsn(src, start, end, bbox, min_magnitude)
            else:
                log.warning("Fuente desconocida ignorada: %s", src)
                continue
        except Exception as exc:  # una fuente caída no debe tumbar todo
            log.error("Fuente %s falló: %s", src, exc)
            continue
        per_source[src] = len(df)
        if not df.empty:
            frames.append(df)
        log.info("  %s -> %d observaciones", src, len(df))

    if not frames:
        raise RuntimeError("Ninguna fuente devolvió datos.")

    obs = pd.concat(frames, ignore_index=True)
    # Filtro de seguridad de ventana temporal (todas las fuentes).
    obs = obs[(obs["origin_time"] >= start) & (obs["origin_time"] <= end)].reset_index(drop=True)
    log.info("Total observaciones (todas las fuentes): %d", len(obs))

    # --- Deduplicación -> canonical_id ---
    dedup_df = deduplicate(obs, dedup_config)
    obs_path = raw_store.save_observations(dedup_df)

    # --- Catálogo canónico ---
    canonical = build_canonical_events(dedup_df, dedup_config)
    canon_path = raw_store.save_canonical(canonical)

    # Vista Perú-only (útil para el foco del proyecto).
    peru = canonical[canonical["country"] == "Peru"].reset_index(drop=True)
    peru_path = raw_store.save_parquet(
        peru, settings.processed_dir / "catalog_canonical_peru.parquet"
    )

    summary = {
        "window": [start.isoformat(), end.isoformat()],
        "sources": per_source,
        "n_observations": int(len(dedup_df)),
        "n_canonical": int(len(canonical)),
        "n_canonical_peru": int(len(peru)),
        "paths": {
            "observations": str(obs_path),
            "canonical": str(canon_path),
            "canonical_peru": str(peru_path),
        },
    }

    if load_db:
        from .storage import db
        db.load_observations(dedup_df)
        db.load_events(canonical)
        db.link_members()
        summary["db_loaded"] = True

    log.info("RESUMEN: %s", summary)
    return summary


def setup_logging(level=logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
