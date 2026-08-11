#!/usr/bin/env python
"""Ensambla todos los resultados en un JSON para el dashboard (Artifact)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from seis_peru.analysis.completeness import fmd, mc_maxc  # noqa: E402


def main() -> int:
    fc = pd.read_parquet(ROOT / "data/features/forecast_latest.parquet")
    meta = json.loads((ROOT / "data/features/forecast_meta.json").read_text("utf-8"))
    cat = pd.read_parquet(ROOT / "data/processed/catalog_canonical_peru.parquet").dropna(subset=["preferred_mw"])
    cat["year"] = cat["origin_time"].dt.year

    # FMD moderna (para la curva Gutenberg-Richter).
    modern = cat[cat.year >= 2000]["preferred_mw"].to_numpy()
    centers, inc, cum = fmd(modern, 0.1)

    # Mc por década.
    mc_dec = []
    for d in range(1960, 2030, 10):
        s = cat[(cat.year >= d) & (cat.year < d + 10)]["preferred_mw"].to_numpy()
        if len(s) >= 100:
            mc_dec.append({"decade": f"{d}s", "mc": round(mc_maxc(s, 0.1), 2), "n": int(len(s))})

    top = cat.nlargest(12, "preferred_mw")
    admin = json.loads((ROOT / "reports/admin_forecast.json").read_text("utf-8"))
    ctx = json.loads((ROOT / "reports/context.json").read_text("utf-8"))

    data = {
        "horizons": admin["horizons"],
        "peru_total": admin["peru_total"],
        "levels": admin["levels"],
        "seasonality": ctx["seasonality"],
        "neighbors": ctx["neighbors"],
        "meta": {
            "forecast_time": meta["forecast_time"],
            "model": "Poisson-Gutenberg-Richter (calibrado)",
            "disclaimer": "Forecast probabilístico de investigación. NO es alerta oficial ni predicción determinista.",
            "n_events": int(len(cat)),
            "date_min": str(cat["origin_time"].min().date()),
            "date_max": str(cat["origin_time"].max().date()),
        },
        "big_events": [{"date": str(r.origin_time.date()), "lat": round(r.latitude, 2),
                        "lon": round(r.longitude, 2), "mw": round(float(r.preferred_mw), 1),
                        "depth": round(float(r.depth_km), 0) if pd.notna(r.depth_km) else None}
                       for r in top.itertuples()],
        "fmd": {"m": [round(x, 2) for x in centers.tolist()],
                "cum": cum.tolist(), "b": 0.99, "mc": 4.5},
        "mc_by_decade": mc_dec,
        "b_value": {"typed_scordilis": 0.99, "direct_mw": 0.94, "note": "b real del Perú ≈0.9–1.0"},
        "models": [
            {"name": "Climatología", "roc": 0.529, "bss": 0.000, "kind": "ref"},
            {"name": "Poisson-Gutenberg-Richter", "roc": 0.639, "bss": 0.0012, "kind": "best"},
            {"name": "XGBoost", "roc": 0.618, "bss": -0.0014, "kind": "ml"},
            {"name": "LightGBM", "roc": 0.590, "bss": -0.0013, "kind": "ml"},
        ],
        "etas": {"mu": 0.481, "alpha": 0.866, "p": 1.534, "n": 0.840},
        "hypotheses": [
            {"id": "H1", "text": "Actividad reciente mejora vs tasa constante",
             "verdict": "Parcial: señal ×40 en celdas activas, pero skill global bajo"},
            {"id": "H2", "text": "Sismos fuertes modifican la tasa posterior (ETAS)",
             "verdict": "Sí: ETAS n=0.84 (84% aftershocks)"},
            {"id": "H3", "text": "Chile aporta información al Perú",
             "verdict": "NO: la señal aparente es campo cercano; remoto es nulo (RR≈1.1)"},
            {"id": "H5", "text": "El ML supera a los baselines",
             "verdict": "NO: XGBoost no supera al Poisson-GR (resultado negativo honesto)"},
        ],
    }
    out = ROOT / "reports" / "dashboard_data.json"
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"-> {out}  ({out.stat().st_size/1024:.0f} KB, "
          f"{len(data['levels']['dist'])} distritos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
