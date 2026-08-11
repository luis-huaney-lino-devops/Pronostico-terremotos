"""Distribución frecuencia-magnitud (FMD) y magnitud de completitud (Mc).

La FMD sigue Gutenberg-Richter:  log10 N(≥M) = a − b·M.  Por debajo de la
magnitud de completitud **Mc** el catálogo deja de registrar todos los sismos
(la curva se dobla). Estimar Mc es OBLIGATORIO: ajustar a/b por debajo de Mc
sesga el b-value.

Estimadores implementados (verificados contra la literatura):
  - **MAXC** (máxima curvatura): Mc = magnitud del pico de la FMD incremental,
    con corrección +0.2 (Woessner & Wiemer 2005) porque MAXC subestima.
  - **GFT** (goodness-of-fit test, Wiemer & Wyss 2000): el menor corte Mco cuyo
    ajuste GR reproduce la FMD observada a un nivel R (90% operativo, 95% ideal).

Implementación propia (no dependemos de SeismoStats, pre-1.0). Ver tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clean(mags) -> np.ndarray:
    m = np.asarray(mags, dtype=float)
    return m[~np.isnan(m)]


def fmd(mags, mbin: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distribución frecuencia-magnitud.

    Devuelve (m_centros, incremental, acumulada) donde:
      - m_centros: niveles de magnitud (centrados en múltiplos de mbin)
      - incremental: nº de eventos en cada bin [m−mbin/2, m+mbin/2)
      - acumulada: N(≥m)
    """
    m = _clean(mags)
    if m.size == 0:
        return np.array([]), np.array([]), np.array([])
    mmin = np.round(np.floor(m.min() / mbin) * mbin, 5)
    mmax = np.round(np.ceil(m.max() / mbin) * mbin, 5)
    centers = np.round(np.arange(mmin, mmax + mbin, mbin), 5)
    edges = np.append(centers - mbin / 2, centers[-1] + mbin / 2)
    inc, _ = np.histogram(m, bins=edges)
    cum = np.cumsum(inc[::-1])[::-1]  # N(≥m)
    return centers, inc, cum


def mc_maxc(mags, mbin: float = 0.1, correction: float = 0.2) -> float:
    """Mc por máxima curvatura (magnitud del pico incremental) + corrección."""
    centers, inc, _ = fmd(mags, mbin)
    if centers.size == 0:
        return float("nan")
    return float(centers[int(np.argmax(inc))] + correction)


@dataclass
class GFTResult:
    mc: float
    R: float          # goodness-of-fit alcanzado en Mc (%)
    level: int        # nivel usado (95 o 90); 0 si se cayó al máximo R
    b: float


def mc_gft(mags, mbin: float = 0.1, min_events: int = 25) -> GFTResult:
    """Mc por goodness-of-fit test (Wiemer & Wyss 2000).

    R = 100·(1 − Σ|N_obs − N_synth| / Σ N_obs). Se toma el menor corte con
    R ≥ 95%; si ninguno lo alcanza, R ≥ 90%; si tampoco, el corte de mayor R.
    """
    from .gutenberg_richter import b_value_aki

    centers, _, cum = fmd(mags, mbin)
    m = _clean(mags)
    if centers.size == 0:
        return GFTResult(float("nan"), float("nan"), 0, float("nan"))

    best_r = -np.inf
    best = None
    for mco in centers:
        sub = m[m >= mco - mbin / 2]
        if sub.size < min_events:
            continue
        bv = b_value_aki(sub, mco, mbin)
        mrange = centers[centers >= mco - mbin / 2]
        obs = cum[centers >= mco - mbin / 2]
        synth = 10 ** (bv.a - bv.b * mrange)
        denom = obs.sum()
        if denom == 0:
            continue
        R = 100.0 * (1 - np.sum(np.abs(obs - synth)) / denom)
        if R > best_r:
            best_r, best = R, (float(mco), R, bv.b)
        if R >= 95.0:
            return GFTResult(float(mco), float(R), 95, float(bv.b))
    # ningún corte llegó a 95%: probar 90%, si no el mejor R.
    # (recorremos otra vez para respetar 'el menor corte' con R>=90)
    for mco in centers:
        sub = m[m >= mco - mbin / 2]
        if sub.size < min_events:
            continue
        bv = b_value_aki(sub, mco, mbin)
        mrange = centers[centers >= mco - mbin / 2]
        obs = cum[centers >= mco - mbin / 2]
        synth = 10 ** (bv.a - bv.b * mrange)
        denom = obs.sum()
        if denom == 0:
            continue
        R = 100.0 * (1 - np.sum(np.abs(obs - synth)) / denom)
        if R >= 90.0:
            return GFTResult(float(mco), float(R), 90, float(bv.b))
    if best is None:
        return GFTResult(float("nan"), float("nan"), 0, float("nan"))
    return GFTResult(best[0], best[1], 0, best[2])
