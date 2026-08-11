"""Tests de homogenización de magnitud (Scordilis 2006)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from seis_peru.magnitude import mb_to_mw, ms_to_mw, to_mw


def test_ms_to_mw_low_segment():
    # Mw = 0.67*Ms + 2.07 para Ms <= 6.1
    assert ms_to_mw(5.0) == pytest.approx(0.67 * 5.0 + 2.07)


def test_ms_to_mw_high_segment():
    # Mw = 0.99*Ms + 0.08 para Ms >= 6.2
    assert ms_to_mw(7.0) == pytest.approx(0.99 * 7.0 + 0.08)


def test_mb_to_mw():
    assert mb_to_mw(5.0) == pytest.approx(0.85 * 5.0 + 1.03)


def test_direct_mw_not_converted():
    assert to_mw(6.3, "mww") == (6.3, "direct")
    assert to_mw(5.1, "Mw") == (5.1, "direct")


def test_conversions_dispatch():
    mw, method = to_mw(5.0, "Ms")
    assert method == "scordilis_ms" and mw == pytest.approx(5.42)
    mw, method = to_mw(4.0, "mb")
    assert method == "scordilis_mb" and mw == pytest.approx(4.43)


def test_local_and_generic_are_flagged():
    assert to_mw(3.2, "ML")[1] == "assume_ml"
    assert to_mw(2.9, "Md")[1] == "assume_md"
    # "m" genérico de EMSC NO debe tratarse como Mw directa.
    assert to_mw(4.1, "m")[1] == "assume_unknown"


def test_none_magnitude():
    assert to_mw(None, "mb") == (None, "none")
