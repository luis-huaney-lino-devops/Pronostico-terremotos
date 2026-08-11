"""Backtesting temporal (walk-forward) — evita el leakage de un split aleatorio.

    2000 ───────────── 2014   |   2015 ──────────── 2026
           TRAIN (ajusta b,     |      TEST (evaluación
           climatología)        |      out-of-sample)

Los baselines usan tasas por celda que se expanden con el tiempo (features ≤ t),
así que evaluar en el periodo de test ES walk-forward por construcción. La
función ``walk_forward_years`` permite además re-evaluar año por año.
"""
from __future__ import annotations

import pandas as pd


def split_by_year(df: pd.DataFrame, test_start_year: int):
    """Divide en train (< año) y test (≥ año) por el instante de predicción t."""
    yr = df["t"].dt.year
    return df[yr < test_start_year].copy(), df[yr >= test_start_year].copy()


def walk_forward_years(df: pd.DataFrame, first_test_year: int):
    """Genera (año, train, test_del_año) con ventana de entrenamiento expansiva."""
    yr = df["t"].dt.year
    last = int(yr.max())
    for y in range(first_test_year, last + 1):
        train = df[yr < y]
        test = df[yr == y]
        if len(test):
            yield y, train.copy(), test.copy()
