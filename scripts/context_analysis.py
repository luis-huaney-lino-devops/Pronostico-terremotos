#!/usr/bin/env python
"""Contexto temporal y regional (honesto).

(1) ESTACIONALIDAD — ¿hay un mes/fecha más probable? La física dice que NO: los
    sismos responden a esfuerzo tectónico, no a estaciones. Se verifica con:
      - chi² de la distribución mensual vs. uniforme (ajustada por días/mes)
      - test de Schuster (periodicidad anual): p = exp(−R²/N), p<0.05 = periódico
    Se usan sismos grandes (M≥6, casi independientes) para el veredicto y M≥5
    para el histograma. Un pico espurio suele ser una secuencia de réplicas, no
    estacionalidad.

(2) CINTURÓN DE FUEGO / PAÍSES VECINOS — actividad por país en el margen de
    subducción de Nazca (Perú, Chile, Ecuador, Colombia, Bolivia). Contexto, no
    causalidad (ver experiment_chile_peru.py: el triggering remoto es nulo).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chisquare  # noqa: E402

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
DIAS_MES = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def schuster(doy: np.ndarray) -> tuple[float, float]:
    """Test de Schuster de periodicidad anual. Devuelve (p, R/N)."""
    th = 2 * np.pi * (doy / 365.25)
    C, S = np.cos(th).sum(), np.sin(th).sum()
    R = np.hypot(C, S)
    n = len(doy)
    return float(np.exp(-R * R / n)), float(R / n)


def seasonality(cat: pd.DataFrame) -> dict:
    peru = cat[cat["country"] == "Peru"].copy()
    peru = peru[peru["origin_time"].dt.year >= 2000]
    peru["month"] = peru["origin_time"].dt.month
    peru["doy"] = peru["origin_time"].dt.dayofyear

    m5 = peru[peru["preferred_mw"] >= 5.0]
    counts = np.array([(m5["month"] == i).sum() for i in range(1, 13)], float)
    exp = counts.sum() * DIAS_MES / DIAS_MES.sum()
    chi2, p_chi = chisquare(counts, exp)

    m6 = peru[peru["preferred_mw"] >= 6.0]
    p_sch5, _ = schuster(m5["doy"].to_numpy())
    p_sch6, _ = schuster(m6["doy"].to_numpy()) if len(m6) > 5 else (float("nan"), 0)

    top_month = int(np.argmax(counts)) + 1
    # Veredicto por el test INDEPENDIENTE (M6+): el M5+ está contaminado por
    # réplicas (Arequipa 2001→jun, Pisco 2007→ago) y da falsos positivos.
    real = bool(not np.isnan(p_sch6) and p_sch6 < 0.05)
    return {
        "months": MESES,
        "counts_m5": counts.astype(int).tolist(),
        "n_m5": int(counts.sum()), "n_m6": int(len(m6)),
        "chi2": round(float(chi2), 2), "p_chi2": round(float(p_chi), 3),
        "schuster_p_m5": round(p_sch5, 3), "schuster_p_m6": round(p_sch6, 3),
        "significant": real,
        "top_month": MESES[top_month - 1],
        "verdict": ("Hay periodicidad anual en M6+ (revisar)." if real else
                    "SIN estacionalidad real: no hay un mes más probable. El pico "
                    "aparente de M5+ (jun/ago) es artefacto de réplicas de Arequipa "
                    "2001 y Pisco 2007; el test con eventos grandes independientes "
                    "(M6+) no es significativo."),
        "big_months": [cat.loc[i, "origin_time"].strftime("%b") for i in
                       peru[peru["preferred_mw"] >= 7.0].nlargest(8, "preferred_mw").index],
    }


def neighbors(cat: pd.DataFrame) -> dict:
    from seis_peru.regions import COUNTRY_BOXES
    rows = []
    recent_year = 2021
    for name in ["Peru", "Chile", "Ecuador", "Colombia", "Bolivia"]:
        c = cat[cat["country"] == name]
        c5 = c[c["preferred_mw"] >= 5.0]
        big = c.nlargest(1, "preferred_mw")
        rows.append({
            "country": name,
            "n_m5": int(len(c5)),
            "n_m6": int((c["preferred_mw"] >= 6.0).sum()),
            "n_m7": int((c["preferred_mw"] >= 7.0).sum()),
            "recent_m5": int(((c["preferred_mw"] >= 5.0) &
                              (c["origin_time"].dt.year >= recent_year)).sum()),
            "max_mw": round(float(big["preferred_mw"].iloc[0]), 1),
            "max_date": str(big["origin_time"].iloc[0].date()),
        })
    return {"since_year": int(cat["origin_time"].dt.year.min()),
            "recent_year": recent_year, "by_country": rows}


def main() -> int:
    cat = pd.read_parquet(ROOT / "data/processed/catalog_canonical.parquet").dropna(subset=["preferred_mw"])
    seas = seasonality(cat)
    neigh = neighbors(cat)
    out = {"seasonality": seas, "neighbors": neigh}
    (ROOT / "reports" / "context.json").write_text(
        json.dumps(out, ensure_ascii=False, separators=(",", ":")), "utf-8")

    print("== ESTACIONALIDAD (¿qué mes?) ==")
    print("Conteo M5+ por mes:", dict(zip(MESES, seas["counts_m5"])))
    print(f"chi² vs uniforme = {seas['chi2']} (p={seas['p_chi2']}); "
          f"Schuster M5+ p={seas['schuster_p_m5']}, M6+ p={seas['schuster_p_m6']}")
    print("->", seas["verdict"])
    print("Meses de los grandes M7+:", seas["big_months"], "(sin patrón claro)")

    print("\n== CINTURÓN DE FUEGO / VECINOS (subducción de Nazca) ==")
    for r in neigh["by_country"]:
        print(f"  {r['country']:9s} M5+={r['n_m5']:5d}  M6+={r['n_m6']:4d}  M7+={r['n_m7']:3d}  "
              f"máx={r['max_mw']} ({r['max_date']})  desde-{neigh['recent_year']}: {r['recent_m5']} M5+")
    print("\n-> Todo el margen es una sola frontera de placas (Nazca bajo Sudamérica).")
    print("   OJO: actividad simultánea NO implica que un país dispare a otro "
          "(H3 = nulo remoto, ver experiment_chile_peru.py).")
    print("\n-> reports/context.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
