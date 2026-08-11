"""Tests del ETAS: fórmulas + recuperación de parámetros conocidos por MLE."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from seis_peru.models.etas import (
    ETASParams,
    _omori_integral,
    fit_etas,
    intensity_at,
    simulate_etas,
)

LN10 = np.log(10.0)


def test_omori_integral():
    assert _omori_integral(10.0, 0.05, 1.0) == pytest.approx(np.log((10.05) / 0.05))
    # p != 1 forma cerrada
    got = _omori_integral(10.0, 0.05, 1.2)
    exp = ((10.05) ** (-0.2) - 0.05 ** (-0.2)) / (-0.2)
    assert got == pytest.approx(exp)


def test_branching_ratio_analytic():
    pr = ETASParams(mu=0.1, K=0.02, alpha=1.5, c=0.05, p=1.2, M0=4.0)
    beta = 1.0 * LN10
    exp = 0.02 * (beta / (beta - 1.5)) * (0.05 ** (-0.2) / 0.2)
    assert pr.branching_ratio(1.0) == pytest.approx(exp)
    assert pr.branching_ratio(1.0) < 1.0  # subcrítico


def test_intensity_monotone_after_event():
    hist_t = np.array([0.0, 1.0])
    hist_m = np.array([5.0, 5.0])
    pr = ETASParams(0.1, 0.05, 1.5, 0.05, 1.1, 4.0)
    # justo después de un evento la intensidad es mayor que mucho después
    assert intensity_at(1.001, hist_t, hist_m, pr) > intensity_at(50.0, hist_t, hist_m, pr)


def test_simulate_and_recover_params():
    true = ETASParams(mu=0.3, K=0.03, alpha=1.6, c=0.05, p=1.15, M0=4.0)
    rng = np.random.default_rng(7)
    times, mags = simulate_etas(true, b=1.0, T=1500, rng=rng)
    assert len(times) > 300
    fit = fit_etas(times, mags, 0.0, 1500.0, M0=4.0, window_days=600.0)
    # recuperación (ETAS es ruidoso -> tolerancias amplias pero informativas)
    assert 0.4 * true.mu < fit.mu < 2.5 * true.mu
    assert abs(fit.p - true.p) < 0.35
    assert abs(fit.alpha - true.alpha) < 0.8
    assert 0.0 < fit.branching_ratio(1.0) < 1.0
