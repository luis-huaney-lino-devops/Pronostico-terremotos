#!/usr/bin/env python
"""CLI de ingesta SEIS-PERU (Fase 1).

Ejemplos:
    # Smoke test rápido (1 año, M>=4.5, todas las fuentes):
    python scripts/run_ingest.py --smoke

    # Catálogo histórico completo del Perú+vecinos (M>=4.0):
    python scripts/run_ingest.py --start 1960-01-01 --end 2026-08-10 --min-magnitude 4.0

    # Solo USGS+IGP, y además cargar a PostGIS:
    python scripts/run_ingest.py --sources usgs,igp --load-db
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seis_peru import pipeline  # noqa: E402
from seis_peru.dedup import DedupConfig  # noqa: E402
from seis_peru.regions import PERU, STUDY_REGION  # noqa: E402


def _date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main() -> int:
    p = argparse.ArgumentParser(description="Ingesta de catálogos sísmicos SEIS-PERU")
    p.add_argument("--start", type=_date, default=_date("1960-01-01"))
    p.add_argument("--end", type=_date, default=datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0))
    p.add_argument("--sources", default="usgs,emsc,isc,igp",
                   help="lista separada por comas: usgs,emsc,isc,igp")
    p.add_argument("--min-magnitude", type=float, default=None)
    p.add_argument("--bbox", choices=["study", "peru"], default="study",
                   help="'study' = Perú + vecinos (default); 'peru' = solo Perú")
    p.add_argument("--dt-max-s", type=float, default=None, help="ventana dedup (s)")
    p.add_argument("--dist-max-km", type=float, default=None, help="distancia dedup (km)")
    p.add_argument("--load-db", action="store_true", help="cargar a PostGIS")
    p.add_argument("--smoke", action="store_true",
                   help="prueba rápida: 2023, M>=4.5")
    args = p.parse_args()

    pipeline.setup_logging()

    if args.smoke:
        args.start = _date("2023-01-01")
        args.end = _date("2024-01-01")
        if args.min_magnitude is None:
            args.min_magnitude = 4.5

    bbox = PERU if args.bbox == "peru" else STUDY_REGION
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())

    dcfg = DedupConfig()
    if args.dt_max_s is not None:
        dcfg.dt_max_s = args.dt_max_s
    if args.dist_max_km is not None:
        dcfg.dist_max_km = args.dist_max_km

    summary = pipeline.run(
        start=args.start,
        end=args.end,
        sources=sources,
        bbox=bbox,
        min_magnitude=args.min_magnitude,
        dedup_config=dcfg,
        load_db=args.load_db,
    )
    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
