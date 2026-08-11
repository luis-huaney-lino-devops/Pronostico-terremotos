#!/usr/bin/env python
"""Agrega el forecast por DEPARTAMENTO del Perú (mapa real) y por horizonte.

- Carga el GeoJSON real de departamentos, lo simplifica para embeber.
- Asigna cada celda del forecast a su departamento (point-in-polygon; las celdas
  offshore van al departamento costero más cercano).
- Suma las tasas de Poisson por región (superposición): Λ = Σ λ_celda.
  P(≥1 M5+ en H días) = 1 − exp(−Λ·H/365).  Recurrencia = 1/Λ años.

Esto da el "DÓNDE" (ranking de regiones) y el "CUÁNDO" honesto (probabilidad por
horizonte + recurrencia esperada), sin fingir una fecha/lugar determinista.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from shapely.geometry import Point, shape  # noqa: E402
from shapely.prepared import prep  # noqa: E402

YEAR = 365.25
HORIZONS = [7, 30, 90, 365]


def _rings(geom, ndp=3):
    """Anillos exteriores (lon,lat) redondeados, para dibujar y embeber."""
    out = []
    geoms = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for g in geoms:
        ring = [[round(x, ndp), round(y, ndp)] for x, y in g.exterior.coords]
        if len(ring) > 3:
            out.append(ring)
    return out


def main() -> int:
    geo = json.loads((ROOT / "data/raw/geo/peru_dep.geojson").read_text("utf-8"))
    regions = []
    for f in geo["features"]:
        name = f["properties"]["NOMBDEP"].title()
        g = shape(f["geometry"]).buffer(0)               # corrige geometrías inválidas
        gs = g.simplify(0.02, preserve_topology=True)     # ~2 km
        regions.append({"name": name, "geom": gs, "prep": prep(gs),
                        "cx": gs.centroid.x, "cy": gs.centroid.y,
                        "lam5": 0.0, "lam6": 0.0, "ncells": 0})

    fc = pd.read_parquet(ROOT / "data/features/forecast_latest.parquet")
    H0 = 30 / YEAR
    p5 = np.clip(fc["p_m5_30d"].to_numpy(), 1e-9, 0.999)
    p6 = np.clip(fc["p_m6_30d"].to_numpy(), 1e-12, 0.999)
    lam5 = -np.log(1 - p5) / H0    # tasa anual M5+ por celda
    lam6 = -np.log(1 - p6) / H0

    # Asignar cada celda a un departamento (o al más cercano si es offshore).
    n_off = 0
    for i, r in fc.reset_index(drop=True).iterrows():
        pt = Point(r["cen_lon"], r["cen_lat"])
        hit = next((reg for reg in regions if reg["prep"].contains(pt)), None)
        if hit is None:
            hit = min(regions, key=lambda reg: reg["geom"].distance(pt))
            n_off += 1
        hit["lam5"] += float(lam5[i]); hit["lam6"] += float(lam6[i]); hit["ncells"] += 1

    def P(lam, H):
        return float(1 - np.exp(-lam * H / YEAR))

    out_regions = []
    for reg in regions:
        L5, L6 = reg["lam5"], reg["lam6"]
        out_regions.append({
            "name": reg["name"],
            "rings": _rings(reg["geom"]),
            "cx": round(reg["cx"], 3), "cy": round(reg["cy"], 3),
            "ncells": reg["ncells"],
            "lam5_yr": round(L5, 4), "lam6_yr": round(L6, 5),
            "p5": {str(h): round(P(L5, h), 5) for h in HORIZONS},
            "p6": {str(h): round(P(L6, h), 6) for h in HORIZONS},
            "recur5_yr": round(1 / L5, 2) if L5 > 0 else None,
            "recur6_yr": round(1 / L6, 1) if L6 > 0 else None,
        })
    out_regions.sort(key=lambda x: -x["p5"]["30"])

    tot5 = sum(r["lam5_yr"] for r in out_regions)
    tot6 = sum(r["lam6_yr"] for r in out_regions)
    summary = {
        "horizons": HORIZONS,
        "peru_total": {
            "p5": {str(h): round(P(tot5, h), 4) for h in HORIZONS},
            "p6": {str(h): round(P(tot6, h), 5) for h in HORIZONS},
            "recur5_yr": round(1 / tot5, 3), "recur6_yr": round(1 / tot6, 2),
            "lam5_yr": round(tot5, 2), "lam6_yr": round(tot6, 3),
        },
        "regions": out_regions,
    }
    out = ROOT / "reports" / "regions_forecast.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, separators=(",", ":")), "utf-8")

    print(f"{len(out_regions)} departamentos · {n_off} celdas offshore→costa")
    print(f"regions_forecast.json: {out.stat().st_size/1024:.0f} KB")
    print("\nTop 5 regiones por P(M5+/30d):")
    for r in out_regions[:5]:
        print(f"  {r['name']:16s} P30d={r['p5']['30']*100:5.1f}%  P1año={r['p5']['365']*100:5.1f}%  "
              f"recurrencia≈{r['recur5_yr']}años  ({r['ncells']} celdas)")
    print(f"\nPerú total M5+: P30d={summary['peru_total']['p5']['30']*100:.1f}% "
          f"P1año={summary['peru_total']['p5']['365']*100:.1f}% | "
          f"M6+ P1año={summary['peru_total']['p6']['365']*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
