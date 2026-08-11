#!/usr/bin/env python
"""Forecast por unidad administrativa a 3 niveles: departamento / provincia / distrito.

Método honesto de integración: el forecast es un campo de TASA por celda 0.1°
(~11 km). Para cada distrito se integra esa densidad de tasa sobre su área:
    ρ_celda = λ_celda / A_celda        (tasa por km²)
    Λ_distrito = ρ_celda(centroide) · A_distrito
Provincias y departamentos = suma jerárquica de sus distritos (consistente).
    P(≥1 en H días) = 1 − exp(−Λ·H/365) ;  recurrencia = 1/Λ años.

CAVEAT (se muestra en el dashboard): la resolución sísmica nativa es 0.1° (~11 km);
a nivel distrital el color es esa malla proyectada sobre límites reales — el
detalle es del MAPA, no una resolución sísmica sub-11 km.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from shapely.geometry import shape  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

YEAR = 365.25
HORIZONS = [7, 30, 90, 365]
KM_DEG = 111.32
SIMP = {"dep": 0.02, "prov": 0.011, "dist": 0.006}


def area_km2(geom):
    return geom.area * KM_DEG * KM_DEG * np.cos(np.radians(geom.centroid.y))


def rings(geom, ndp=3):
    out = []
    for g in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
        r = [[round(x, ndp), round(y, ndp)] for x, y in g.exterior.coords]
        if len(r) > 3:
            out.append(r)
    return out


def punit(lam5, lam6):
    def Pf(l, h):
        return round(float(1 - np.exp(-l * h / YEAR)), 6)
    return ({str(h): Pf(lam5, h) for h in HORIZONS},
            {str(h): Pf(lam6, h) for h in HORIZONS})


def main() -> int:
    # --- campo de tasa por celda (0.1°) ---
    fc = pd.read_parquet(ROOT / "data/features/forecast_latest.parquet")
    H0 = 30 / YEAR
    lam5 = -np.log(1 - np.clip(fc["p_m5_30d"].to_numpy(), 1e-9, .999)) / H0
    lam6 = -np.log(1 - np.clip(fc["p_m6_30d"].to_numpy(), 1e-12, .999)) / H0
    clat = fc["cen_lat"].to_numpy(); clon = fc["cen_lon"].to_numpy()
    A_cell = (0.1 * KM_DEG) ** 2 * np.cos(np.radians(clat))
    rho5 = lam5 / A_cell; rho6 = lam6 / A_cell
    tree = cKDTree(np.column_stack([clon, clat]))

    def density_at(lon, lat):
        d, i = tree.query([lon, lat])
        return (rho5[i], rho6[i]) if d < 0.08 else (0.0, 0.0)  # 0.08° ~ media celda

    # --- distritos (fuente única con jerarquía) ---
    geo = json.loads((ROOT / "data/raw/geo/peru_distrital.geojson").read_text("utf-8"))
    dists = []
    by_prov = defaultdict(list); by_dep = defaultdict(list)
    for f in geo["features"]:
        p = f["properties"]
        if not f.get("geometry"):
            continue
        dep = p["NOMBDEP"].title(); prov = p["NOMBPROV"].title(); name = p["NOMBDIST"].title()
        g = shape(f["geometry"]).buffer(0)
        if g.is_empty:
            continue
        A = area_km2(g)
        r5, r6 = density_at(g.centroid.x, g.centroid.y)
        L5, L6 = r5 * A, r6 * A
        gs = g.simplify(SIMP["dist"], preserve_topology=True)
        d = {"name": name, "dep": dep, "prov": prov, "rings": rings(gs),
             "cx": round(g.centroid.x, 3), "cy": round(g.centroid.y, 3),
             "lam5": L5, "lam6": L6, "geom": g}
        dists.append(d); by_prov[(dep, prov)].append(d); by_dep[dep].append(d)

    def pack(name, parent, members, geom, extra=None):
        L5 = sum(m["lam5"] for m in members); L6 = sum(m["lam6"] for m in members)
        p5, p6 = punit(L5, L6)
        u = {"name": name, "rings": rings(geom.simplify(SIMP[extra], preserve_topology=True)),
             "cx": round(geom.centroid.x, 3), "cy": round(geom.centroid.y, 3),
             "p5": p5, "p6": p6, "n": len(members),
             "recur5": round(1/L5, 2) if L5 > 0 else None,
             "recur6": round(1/L6, 1) if L6 > 0 else None}
        if parent:
            u["parent"] = parent
        return u

    print("Disolviendo provincias y departamentos...")
    prov_units, dep_units = [], []
    for (dep, prov), ms in by_prov.items():
        prov_units.append(pack(prov, dep, ms, unary_union([m["geom"] for m in ms]), "prov"))
    for dep, ms in by_dep.items():
        dep_units.append(pack(dep, None, ms, unary_union([m["geom"] for m in ms]), "dep"))

    dist_units = []
    for d in dists:
        p5, p6 = punit(d["lam5"], d["lam6"])
        dist_units.append({"name": d["name"], "parent": d["dep"], "prov": d["prov"],
                           "rings": d["rings"], "cx": d["cx"], "cy": d["cy"],
                           "p5": p5, "p6": p6, "n": 1,
                           "recur5": round(1/d["lam5"], 1) if d["lam5"] > 0 else None,
                           "recur6": round(1/d["lam6"], 1) if d["lam6"] > 0 else None})
    for u in (dep_units, prov_units, dist_units):
        u.sort(key=lambda x: -x["p5"]["30"])

    tot5 = sum(d["lam5"] for d in dists); tot6 = sum(d["lam6"] for d in dists)
    p5t, p6t = punit(tot5, tot6)
    out = {"horizons": HORIZONS,
           "peru_total": {"p5": p5t, "p6": p6t,
                          "recur5": round(1/tot5, 3), "recur6": round(1/tot6, 2)},
           "levels": {"dep": dep_units, "prov": prov_units, "dist": dist_units}}
    path = ROOT / "reports" / "admin_forecast.json"
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), "utf-8")

    print(f"dep={len(dep_units)} prov={len(prov_units)} dist={len(dist_units)}")
    print(f"admin_forecast.json: {path.stat().st_size/1024:.0f} KB")
    print("\nTop 6 distritos por P(M5+/30d):")
    for d in dist_units[:6]:
        print(f"  {d['name']:18s} ({d['prov']}, {d['parent']})  P30d={d['p5']['30']*100:5.1f}%  recur≈{d['recur5']}a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
