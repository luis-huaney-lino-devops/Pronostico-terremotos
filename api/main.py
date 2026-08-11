"""API REST de SEIS-PERU (FastAPI).

Sirve el catálogo, el forecast probabilístico más reciente, la comparación de
modelos (baselines vs ML) y el resultado del experimento Chile→Perú. Consume los
artefactos Parquet/JSON generados por los scripts.

Levantar:
    uvicorn api.main:app --reload --port 8000
Docs interactivas: http://localhost:8000/docs

IMPORTANTE: es un forecast PROBABILÍSTICO de investigación, NO una alerta
oficial ni una predicción determinista.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FEAT = ROOT / "data" / "features"

app = FastAPI(
    title="SEIS-PERU API",
    description="Pronóstico sísmico probabilístico del Perú (investigación, no alerta oficial).",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Resultados clave (Fases 4-5, ETAS, H3) — resumen honesto para la plataforma.
RESULTS = {
    "baselines_ml": {
        "target": "M≥5.0 en 30 días · celda 0.25° · test 2015–2026",
        "metric": "ROC-AUC / BSS calibrado",
        "models": [
            {"name": "Climatología", "roc_auc": 0.529, "bss": 0.000},
            {"name": "Poisson-Gutenberg-Richter (listón)", "roc_auc": 0.639, "bss": 0.0012},
            {"name": "XGBoost", "roc_auc": 0.618, "bss": -0.0014},
            {"name": "LightGBM", "roc_auc": 0.590, "bss": -0.0013},
        ],
        "verdict_H5": "El ML NO supera al baseline físico (resultado negativo honesto).",
    },
    "etas": {"mu": 0.481, "K": 0.171, "alpha": 0.866, "c": 0.398, "p": 1.534,
             "branching_ratio": 0.840, "note": "84% aftershocks; α bajo típico de subducción."},
    "chile_peru_H3": {
        "verdict": "Sin evidencia de triggering remoto Chile→Perú (M≥4.5).",
        "detail": "La señal aparente (RR≈11.5) es fuga de aftershocks de campo cercano; "
                  "con Chile remoto >600km el efecto desaparece (RR≈1.1, p>0.1).",
    },
}


@lru_cache
def _catalog() -> pd.DataFrame:
    return pd.read_parquet(PROC / "catalog_canonical_peru.parquet").dropna(subset=["preferred_mw"])


@lru_cache
def _forecast() -> pd.DataFrame:
    return pd.read_parquet(FEAT / "forecast_latest.parquet")


@lru_cache
def _forecast_meta() -> dict:
    return json.loads((FEAT / "forecast_meta.json").read_text(encoding="utf-8"))


@app.get("/")
def root():
    return {"name": "SEIS-PERU API", "version": "0.1.0",
            "disclaimer": "Forecast probabilístico de investigación. NO es alerta oficial.",
            "endpoints": ["/health", "/catalog/summary", "/catalog/events",
                          "/forecast/latest", "/forecast/grid", "/models/comparison",
                          "/experiments/chile-peru", "/docs"]}


@app.get("/health")
def health():
    ok = (PROC / "catalog_canonical_peru.parquet").exists()
    return {"status": "ok" if ok else "sin datos", "catalog": ok,
            "forecast": (FEAT / "forecast_latest.parquet").exists()}


@app.get("/catalog/summary")
def catalog_summary():
    df = _catalog()
    return {
        "n_events": int(len(df)),
        "date_min": str(df["origin_time"].min()),
        "date_max": str(df["origin_time"].max()),
        "mw_min": float(df["preferred_mw"].min()),
        "mw_max": float(df["preferred_mw"].max()),
        "by_decade": {str(int(d)): int(c) for d, c in
                      df.groupby((df["origin_time"].dt.year // 10 * 10)).size().items()},
    }


@app.get("/catalog/events")
def catalog_events(min_mag: float = Query(5.0, ge=0), limit: int = Query(100, le=2000)):
    df = _catalog()
    df = df[df["preferred_mw"] >= min_mag].nlargest(limit, "origin_time")
    return [{"time": str(r.origin_time), "lat": float(r.latitude), "lon": float(r.longitude),
             "depth_km": float(r.depth_km) if pd.notna(r.depth_km) else None,
             "mw": float(r.preferred_mw), "sources": r.sources}
            for r in df.itertuples()]


@app.get("/forecast/latest")
def forecast_latest():
    return _forecast_meta()


@app.get("/forecast/grid")
def forecast_grid(target: str = Query("m5", pattern="^(m5|m6)$")):
    df = _forecast()
    col = "p_m5_30d" if target == "m5" else "p_m6_30d"
    return {"forecast_time": _forecast_meta()["forecast_time"], "target": target,
            "cells": [{"lat": float(r.cen_lat), "lon": float(r.cen_lon),
                       "p": float(getattr(r, col)), "cnt_365d": int(r.cnt_365d)}
                      for r in df.itertuples()]}


@app.get("/models/comparison")
def models_comparison():
    return RESULTS["baselines_ml"] | {"etas": RESULTS["etas"]}


@app.get("/experiments/chile-peru")
def experiment_chile_peru():
    return RESULTS["chile_peru_H3"]
