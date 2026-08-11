#!/usr/bin/env python
"""Construye el dataset de features/target para ML (Fase 3).

    python scripts/build_features.py                       # 0.1°, M6+/30d
    python scripts/build_features.py --res 0.2 --step-days 15
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from seis_peru.features.build import FeatureConfig, build_feature_matrix  # noqa: E402


def _date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default=str(ROOT / "data/processed/catalog_canonical.parquet"),
                   help="catálogo canónico (usa el completo; la malla recorta a Perú)")
    p.add_argument("--res", type=float, default=0.1)
    p.add_argument("--feature-min-mag", type=float, default=4.5)
    p.add_argument("--horizon-days", type=int, default=30)
    p.add_argument("--step-days", type=int, default=30)
    p.add_argument("--start", type=_date, default=_date("2000-01-01"))
    p.add_argument("--min-active-events", type=int, default=5)
    p.add_argument("--etas", action="store_true", help="añadir feature de intensidad ETAS")
    p.add_argument("--out", default=str(ROOT / "data/features/features_peru.parquet"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                        datefmt="%H:%M:%S")

    cat = pd.read_parquet(args.catalog)
    cfg = FeatureConfig(
        res=args.res, feature_min_mag=args.feature_min_mag,
        horizon_days=args.horizon_days, step_days=args.step_days,
        start=args.start, min_active_events=args.min_active_events,
    )
    if args.etas:
        from seis_peru.models.etas import ETASParams
        # Parámetros ETAS ya ajustados al Perú (scripts/run_etas.py, M0=4.5).
        # Se reusan para no re-ajustar (el MLE es lento) al construir la feature.
        cfg.etas = ETASParams(mu=0.481, K=0.171, alpha=0.866, c=0.398, p=1.534,
                              M0=args.feature_min_mag)
        cfg.etas_window_days = 365
        logging.info("ETAS feature con params ajustados: %s", cfg.etas)
    df = build_feature_matrix(cat, cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nDataset -> {out}  ({len(df):,} filas, {df.shape[1]} columnas)")
    print("Positivos:",
          {t: int(df[t].sum()) for t in ("y_m6_30d", "y_m5_30d", "y_m5_7d")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
