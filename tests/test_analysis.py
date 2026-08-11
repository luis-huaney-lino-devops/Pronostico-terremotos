"""Tests de análisis: recuperar b-value y Mc conocidos de un catálogo sintético."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from seis_peru.analysis.completeness import fmd, mc_gft, mc_maxc
from seis_peru.analysis.gutenberg_richter import b_value_aki, gr_fit


def _synthetic_gr(b_true=1.0, mc=3.0, dM=0.1, n=300_000, seed=0):
    """Catálogo GR DISCRETO (geométrico) con b y Mc conocidos."""
    rng = np.random.default_rng(seed)
    k = np.arange(60)
    w = 10.0 ** (-b_true * dM * k)
    w /= w.sum()
    draws = rng.choice(k, size=n, p=w)
    return np.round(mc + draws * dM, 1)


def test_fmd_shapes():
    mags = _synthetic_gr()
    centers, inc, cum = fmd(mags, 0.1)
    assert centers.size == inc.size == cum.size
    # La acumulada es monótona decreciente y empieza en n total.
    assert cum[0] == len(mags)
    assert np.all(np.diff(cum) <= 0)


def test_recover_b_value():
    mags = _synthetic_gr(b_true=1.0, mc=3.0)
    bv = b_value_aki(mags, mc=3.0, mbin=0.1)
    assert bv.b == pytest.approx(1.0, abs=0.03)
    assert bv.n > 250_000
    assert bv.sigma_b < 0.01  # muchos eventos -> incertidumbre pequeña


def test_recover_b_value_08():
    mags = _synthetic_gr(b_true=0.8, mc=3.0, seed=1)
    bv = b_value_aki(mags, mc=3.0, mbin=0.1)
    assert bv.b == pytest.approx(0.8, abs=0.03)


def test_maxc_finds_completeness():
    # Catálogo completo en 3.0 + ruido incompleto por debajo.
    rng = np.random.default_rng(2)
    mags = _synthetic_gr(mc=3.0)
    noise = np.round(rng.uniform(2.0, 2.99, 5000), 1)
    allm = np.concatenate([mags, noise])
    # sin corrección, el pico incremental está en 3.0
    assert mc_maxc(allm, mbin=0.1, correction=0.0) == pytest.approx(3.0, abs=0.05)
    # con corrección estándar +0.2
    assert mc_maxc(allm, mbin=0.1) == pytest.approx(3.2, abs=0.05)


def test_gft_finds_completeness():
    mags = _synthetic_gr(b_true=1.0, mc=3.0)
    res = mc_gft(mags, mbin=0.1)
    assert 2.9 <= res.mc <= 3.3
    assert res.b == pytest.approx(1.0, abs=0.05)
    assert res.R >= 90.0


def test_gr_fit_rate():
    mags = _synthetic_gr(b_true=1.0, mc=3.0)
    fit = gr_fit(mags, mc=3.0)
    # N(>=Mc) reconstruido debe ~igualar el nº de eventos usados.
    n_ge_mc = 10 ** (fit.a - fit.b * fit.mc)
    assert n_ge_mc == pytest.approx(fit.n, rel=0.02)
