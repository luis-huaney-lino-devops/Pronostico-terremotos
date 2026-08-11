#!/usr/bin/env python
"""Genera el forecast MÁS RECIENTE por celda (probabilidad calibrada).

Usa el baseline honesto (Poisson-GR suavizado, calibrado isotónicamente con la
historia) — el que definió el listón. Salida para la API y el dashboard.

    python scripts/build_forecast.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from seis_peru.models import baselines as B  # noqa: E402

MC, B_VALUE, H = 4.5, 1.0, 30


def _calibrated(df_hist, df_now, target, m0):
    p_hist = B.poisson_smooth(df_hist, B_VALUE, m0, MC, H)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(p_hist, df_hist[target].to_numpy())
    return iso.transform(B.poisson_smooth(df_now, B_VALUE, m0, MC, H))


def main() -> int:
    df = pd.read_parquet(ROOT / "data/features/features_peru.parquet")
    latest = df["t"].max()
    now = df[df["t"] == latest].copy()
    hist = df[df["t"] < latest]

    now["p_m5_30d"] = _calibrated(hist, now, "y_m5_30d", 5.0)
    now["p_m6_30d"] = _calibrated(hist, now, "y_m6_30d", 6.0)

    out_cols = ["cell_id", "cen_lat", "cen_lon", "cnt_30d", "cnt_365d",
                "maxmag_365d", "days_since_last", "p_m5_30d", "p_m6_30d"]
    snap = now[out_cols].sort_values("p_m5_30d", ascending=False).reset_index(drop=True)
    snap.to_parquet(ROOT / "data/features/forecast_latest.parquet", index=False)

    meta = {
        "forecast_time": pd.Timestamp(latest).isoformat(),
        "horizon_days": H, "mc": MC, "b_value": B_VALUE,
        "n_cells": int(len(snap)),
        "model": "Poisson-Gutenberg-Richter suavizado (calibrado isotónico)",
        "top_m5_cells": snap.head(10)[["cen_lat", "cen_lon", "p_m5_30d"]].round(4).to_dict("records"),
    }
    (ROOT / "data/features/forecast_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Forecast {latest} | {len(snap)} celdas")
    print(f"P(M5+/30d) máx: {snap['p_m5_30d'].max():.3%} en "
          f"lat={snap.iloc[0]['cen_lat']:.2f}, lon={snap.iloc[0]['cen_lon']:.2f}")
    print("-> data/features/forecast_latest.parquet + forecast_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
