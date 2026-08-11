"""Data lake en Parquet.

Estructura:
    data/
      raw/<source>/<source>_<ini>_<fin>_<fetch>.parquet   (crudo, inmutable)
      raw/<source>/<...>.manifest.json                      (parámetros/conteos)
      interim/observations.parquet     (normalizado, todas las fuentes)
      processed/catalog_canonical.parquet  (eventos canónicos deduplicados)

Regla de oro: los ficheros en raw/ NUNCA se modifican (append-only). Eso
garantiza que cualquier experimento sea reproducible desde el dato original.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import settings
from ..regions import BBox

log = logging.getLogger("seis.store")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_raw(
    source: str,
    rows: list[dict],
    start: datetime,
    end: datetime,
    bbox: BBox,
    min_magnitude: float | None,
) -> Path:
    """Guarda filas nativas crudas en Parquet + manifiesto. Devuelve la ruta."""
    d = settings.raw_dir / source
    d.mkdir(parents=True, exist_ok=True)
    base = f"{source}_{start:%Y%m%d}_{end:%Y%m%d}_{_stamp()}"
    pq_path = d / f"{base}.parquet"

    df = pd.DataFrame(rows)
    df.to_parquet(pq_path, index=False)

    manifest = {
        "source": source,
        "query_start": start.isoformat(),
        "query_end": end.isoformat(),
        "bbox": bbox.__dict__,
        "min_magnitude": min_magnitude,
        "n_events": len(rows),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "parquet": pq_path.name,
    }
    (d / f"{base}.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("[%s] guardado raw: %s (%d filas)", source, pq_path.name, len(rows))
    return pq_path


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Guardado %s (%d filas)", path, len(df))
    return path


def save_observations(df: pd.DataFrame) -> Path:
    return save_parquet(df, settings.interim_dir / "observations.parquet")


def save_canonical(df: pd.DataFrame) -> Path:
    return save_parquet(df, settings.processed_dir / "catalog_canonical.parquet")


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
