"""Tests de baselines sismológicos y métricas probabilísticas."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from seis_peru.models import baselines as B
from seis_peru.models.metrics import evaluate, reliability


def test_poisson_prob_bounds_and_monotonic():
    assert B.poisson_prob(0.0, 30) == pytest.approx(0.0)
    p = B.poisson_prob([0.0, 1.0, 10.0], 30)
    assert np.all((p >= 0) & (p <= 1))
    assert p[0] < p[1] < p[2]


def test_gr_rate_extrapolation():
    # 10 sismos M>=5 en 1 año, b=1 -> tasa M>=6 = 10 * 10^-1 = 1.0/año
    assert B.gr_rate(10, 1.0, b=1.0, m0=6.0, mc=5.0) == pytest.approx(1.0)
    # b=1, un salto de 2 magnitudes -> factor 10^-2
    assert B.gr_rate(100, 1.0, b=1.0, m0=6.0, mc=4.0) == pytest.approx(1.0)


def test_evaluate_perfect_predictor():
    y = np.array([0, 0, 1, 0, 1, 0, 0, 1])
    m = evaluate(y, y.astype(float))  # p = y (perfecto, tras clip)
    assert m["brier"] < 1e-10
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)


def test_bss_zero_vs_self():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.05).astype(int)
    p_clima = np.full(1000, y.mean())
    m = evaluate(y, p_clima, p_ref=p_clima)
    assert m["bss"] == pytest.approx(0.0, abs=1e-9)


def test_ranking_metrics_nan_when_one_class():
    y = np.zeros(10, dtype=int)
    m = evaluate(y, np.full(10, 0.01))
    assert np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"])


def test_reliability_shapes():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.1).astype(int)
    p = rng.random(2000)
    mp, of, cnt = reliability(y, p, n_bins=10)
    assert len(mp) == len(of) == len(cnt)
    assert cnt.sum() == 2000
