"""Baselines sismológicos (el 'listón' que el ML deberá superar).

Todos producen P(M≥M0 en H | información hasta t) por celda, usando SOLO
features del pasado (≤ t). Idea física:

  Poisson:  P(≥1 evento en H) = 1 − exp(−λ·H)
  Gutenberg-Richter:  λ(≥M0) = λ(≥Mc)·10^(−b·(M0−Mc))

Es decir: contamos sismos por encima de la completitud Mc (abundantes) y
extrapolamos hacia la magnitud objetivo con la ley GR. Eso es mucho más robusto
para eventos raros (M6) que contar M6 directamente (casi nunca hay).

Baselines:
  - ``climatology``  : tasa base constante (la referencia más tonta).
  - ``poisson_lt``   : Poisson-GR de LARGO PLAZO (tasa desde 1960).
  - ``poisson_1y``   : Poisson-GR de tasa RECIENTE (últimos 365 d).
  - ``poisson_smooth``: reciente + suavizado espacial (vecindario 3×3).
La comparación poisson_1y vs poisson_lt vs climatology PRUEBA la hipótesis H1
(¿la actividad reciente mejora sobre una tasa histórica constante?).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

YEAR_DAYS = 365.25
CATALOG_T0 = pd.Timestamp("1960-01-01", tz="UTC")  # inicio para tasa de largo plazo


def poisson_prob(rate_per_year, horizon_days: float):
    """P(≥1 evento en H) para una tasa anual (Poisson)."""
    H = horizon_days / YEAR_DAYS
    return 1.0 - np.exp(-np.asarray(rate_per_year, dtype=float) * H)


def gr_rate(count, window_years, b: float, m0: float, mc: float):
    """Tasa anual de M≥M0 vía Gutenberg-Richter desde conteos de M≥Mc."""
    lam_mc = np.asarray(count, dtype=float) / np.asarray(window_years, dtype=float)
    return lam_mc * 10.0 ** (-b * (m0 - mc))


def climatology(y_train: np.ndarray, n: int) -> np.ndarray:
    """Probabilidad constante = tasa base del periodo de entrenamiento."""
    return np.full(n, float(np.mean(y_train)))


def _elapsed_years(t: pd.Series) -> np.ndarray:
    return np.maximum((t - CATALOG_T0).dt.total_seconds().to_numpy() / (YEAR_DAYS * 86400), 1.0)


def poisson_lt(df: pd.DataFrame, b: float, m0: float, mc: float, horizon_days: float):
    """Poisson-GR de largo plazo: tasa = cnt_all / (t − 1960)."""
    rate = gr_rate(df["cnt_all"], _elapsed_years(df["t"]), b, m0, mc)
    return poisson_prob(rate, horizon_days)


def poisson_1y(df: pd.DataFrame, b: float, m0: float, mc: float, horizon_days: float):
    """Poisson-GR reciente: tasa = cnt_365d / 1 año."""
    rate = gr_rate(df["cnt_365d"], 1.0, b, m0, mc)
    return poisson_prob(rate, horizon_days)


def poisson_smooth(df: pd.DataFrame, b: float, m0: float, mc: float, horizon_days: float):
    """Reciente + suavizado espacial: mezcla la tasa de la celda y del vecindario."""
    cell_rate = gr_rate(df["cnt_365d"], 1.0, b, m0, mc)
    nb_rate = gr_rate(df["nb_cnt_365d"] / 9.0, 1.0, b, m0, mc)  # ~9 celdas 3×3
    return poisson_prob(0.5 * cell_rate + 0.5 * nb_rate, horizon_days)


# Registro de baselines dependientes de features (excluye climatología, que
# necesita el target de entrenamiento).
FEATURE_BASELINES = {
    "poisson_lt": poisson_lt,
    "poisson_1y": poisson_1y,
    "poisson_smooth": poisson_smooth,
}
