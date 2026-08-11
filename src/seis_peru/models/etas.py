"""Modelo ETAS temporal (Epidemic-Type Aftershock Sequence, Ogata 1988).

Idea: cada sismo aumenta TEMPORALMENTE la tasa de sismos posteriores
(triggering de Omori-Utsu). La tasa condicional es:

    λ(t) = μ + Σ_{ti<t}  K · exp(α·(mi − M0)) · (t − ti + c)^(−p)

con μ = tasa de fondo, K = productividad, α = acoplamiento magnitud-productividad,
(c, p) = decaimiento de Omori. M0 = magnitud de referencia (= Mc). Tiempos en días.

Log-verosimilitud sobre [T0, T]:
    LL = Σ_i log λ(ti) − ∫_{T0}^{T} λ(t) dt
    ∫ = μ·(T−T0) + Σ_i K·e^{α(mi−M0)}·[ (T−ti+c)^{1−p} − c^{1−p} ] / (1−p)

Se ajusta por máxima verosimilitud. Incluye simulación por ramificación para
recuperar parámetros conocidos (ver tests). Es el componente sismológico
central y un baseline físico mucho más apropiado que meter una LSTM a ciegas.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger("seis.etas")
LN10 = np.log(10.0)


@dataclass
class ETASParams:
    mu: float       # tasa de fondo (eventos/día)
    K: float        # productividad
    alpha: float    # acoplamiento magnitud-productividad
    c: float        # Omori (días)
    p: float        # Omori (exponente)
    M0: float       # magnitud de referencia (Mc)

    def as_vector(self):
        return np.array([self.mu, self.K, self.alpha, self.c, self.p])

    def branching_ratio(self, b: float) -> float:
        """n = nº medio de aftershocks directos por evento (n<1 = subcrítico).
        Requiere p>1 y α<β (β=b·ln10)."""
        beta = b * LN10
        if self.p <= 1 or self.alpha >= beta:
            return float("nan")
        return self.K * (beta / (beta - self.alpha)) * (self.c ** (1 - self.p) / (self.p - 1))


def _omori_integral(dt, c, p):
    """∫_0^{dt} (s+c)^{-p} ds."""
    if abs(p - 1.0) < 1e-8:
        return np.log((dt + c) / c)
    return ((dt + c) ** (1 - p) - c ** (1 - p)) / (1 - p)


def triggering_sum_at_events(times, mags, mu, K, alpha, c, p, M0, window_days=3000.0):
    """λ(ti) para cada evento (solo eventos ANTERIORES; ventana de truncado)."""
    n = len(times)
    lam = np.full(n, mu, dtype=float)
    prod = K * np.exp(alpha * (mags - M0))
    for i in range(1, n):
        lo = np.searchsorted(times, times[i] - window_days, side="left")
        if lo >= i:
            continue
        dt = times[i] - times[lo:i]
        lam[i] += np.sum(prod[lo:i] * (dt + c) ** (-p))
    return lam


def neg_loglik(theta, times, mags, T0, T, M0, window_days=3000.0):
    mu, K, alpha, c, p = theta
    if mu <= 0 or K < 0 or c <= 0 or p <= 0 or alpha < 0:
        return 1e12
    lam = triggering_sum_at_events(times, mags, mu, K, alpha, c, p, M0, window_days)
    if np.any(lam <= 0):
        return 1e12
    sum_log = np.sum(np.log(lam))
    integral = mu * (T - T0) + np.sum(
        K * np.exp(alpha * (mags - M0)) * _omori_integral(T - times, c, p)
    )
    ll = sum_log - integral
    return -ll if np.isfinite(ll) else 1e12


def fit_etas(times, mags, T0, T, M0, init=None, window_days=3000.0) -> ETASParams:
    """Ajuste MLE (Nelder-Mead sobre log-parámetros para positividad)."""
    times = np.asarray(times, float)
    mags = np.asarray(mags, float)
    order = np.argsort(times)
    times, mags = times[order], mags[order]

    if init is None:
        rate = len(times) / max(T - T0, 1.0)
        init = np.array([0.5 * rate, 0.1, 1.5, 0.05, 1.1])

    def obj(log_theta):
        theta = np.exp(log_theta)
        return neg_loglik(theta, times, mags, T0, T, M0, window_days)

    res = minimize(obj, np.log(init), method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-4})
    mu, K, alpha, c, p = np.exp(res.x)
    log.info("ETAS ajustado: mu=%.4f K=%.4f alpha=%.3f c=%.4f p=%.3f (LL=%.1f)",
             mu, K, alpha, c, p, -res.fun)
    return ETASParams(mu, K, alpha, c, p, M0)


# --------------------------- Simulación (para tests) ---------------------------
def _gr_mags(n, b, M0, rng):
    return M0 - np.log10(rng.random(n)) / b


def _omori_wait(n, c, p, rng):
    """Muestra tiempos de espera de la densidad Omori normalizada (p>1)."""
    u = rng.random(n)
    return c * ((1 - u) ** (-1 / (p - 1)) - 1)


def simulate_etas(params: ETASParams, b, T, rng=None, max_gen=100):
    """Simula un catálogo ETAS por ramificación en [0, T]. Devuelve (times, mags)."""
    rng = rng or np.random.default_rng(0)
    mu, K, alpha, c, p, M0 = (params.mu, params.K, params.alpha, params.c, params.p, params.M0)
    # Inmigrantes (fondo Poisson).
    n_bg = rng.poisson(mu * T)
    times = list(rng.uniform(0, T, n_bg))
    mags = list(_gr_mags(n_bg, b, M0, rng))
    # Ramificación.
    queue = list(zip(times, mags))
    gen = 0
    while queue and gen < max_gen:
        new = []
        for ti, mi in queue:
            expected = K * np.exp(alpha * (mi - M0)) * _omori_integral(T - ti, c, p)
            k = rng.poisson(expected)
            if k == 0:
                continue
            waits = _omori_wait(k, c, p, rng)
            child_t = ti + waits
            child_t = child_t[child_t < T]
            if len(child_t) == 0:
                continue
            child_m = _gr_mags(len(child_t), b, M0, rng)
            for tt, mm in zip(child_t, child_m):
                times.append(tt); mags.append(mm); new.append((tt, mm))
        queue = new
        gen += 1
    times = np.array(times); mags = np.array(mags)
    order = np.argsort(times)
    return times[order], mags[order]


def intensity_at(t, hist_times, hist_mags, params: ETASParams):
    """λ(t) dado el historial (eventos < t)."""
    m = hist_times < t
    if not np.any(m):
        return params.mu
    dt = t - hist_times[m]
    prod = params.K * np.exp(params.alpha * (hist_mags[m] - params.M0))
    return params.mu + np.sum(prod * (dt + params.c) ** (-params.p))
