"""Gutenberg-Richter: b-value (Aki MLE) y ajuste completo a/b/Mc.

    log10 N(≥M) = a − b·M          (solo válido para M ≥ Mc)

- **b-value** por máxima verosimilitud (Aki 1965) con corrección de binning
  (Utsu): b = log10(e) / (⟨M⟩ − (Mc − Δm/2)).
  Interpretación: b≈1 global; b alto → predominan pequeños; b bajo → relativamente
  más grandes. Es EXTREMADAMENTE sensible al extrapolar hacia arriba (un error de
  ~0.05 en b cambia la tasa de M7 en ~40%).
- **incertidumbre** de Shi & Bolt (1982):
  σ_b = 2.30·b²·√( Σ(Mᵢ−⟨M⟩)² / (n·(n−1)) ).
- **a-value** normalizado para que N(≥Mc) = n_observados: a = log10(n) + b·Mc.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .completeness import _clean, mc_maxc

LOG10_E = np.log10(np.e)  # 0.4342944819...


@dataclass
class BValue:
    b: float
    sigma_b: float     # incertidumbre Shi & Bolt
    a: float
    mc: float
    n: int             # nº de eventos usados (M ≥ Mc)
    mean_mag: float


def b_value_aki(mags, mc: float, mbin: float = 0.1) -> BValue:
    """b-value por MLE de Aki con corrección de binning e incertidumbre S&B."""
    m = _clean(mags)
    m = m[m >= mc - mbin / 2 - 1e-9]  # incluye el bin de Mc
    n = int(m.size)
    if n < 2:
        return BValue(float("nan"), float("nan"), float("nan"), mc, n,
                      float(m.mean()) if n else float("nan"))
    mean_m = float(m.mean())
    denom = mean_m - (mc - mbin / 2)
    if denom <= 0:
        return BValue(float("nan"), float("nan"), float("nan"), mc, n, mean_m)
    b = LOG10_E / denom
    sigma_b = 2.30 * b ** 2 * np.sqrt(np.sum((m - mean_m) ** 2) / (n * (n - 1)))
    a = np.log10(n) + b * mc
    return BValue(float(b), float(sigma_b), float(a), float(mc), n, mean_m)


@dataclass
class GRFit:
    a: float
    b: float
    sigma_b: float
    mc: float
    n: int
    mean_mag: float

    def rate_ge(self, m0: float, duration_years: float) -> float:
        """Tasa anual λ(≥M0) implícita por el ajuste GR (para forecasting)."""
        n_ge = 10 ** (self.a - self.b * m0)
        return n_ge / duration_years


def gr_fit(mags, mc: float | None = None, mbin: float = 0.1) -> GRFit:
    """Ajuste GR completo. Si ``mc`` es None, se estima con MAXC+0.2."""
    m = _clean(mags)
    if mc is None:
        mc = mc_maxc(m, mbin)
    bv = b_value_aki(m, mc, mbin)
    return GRFit(bv.a, bv.b, bv.sigma_b, bv.mc, bv.n, bv.mean_mag)
