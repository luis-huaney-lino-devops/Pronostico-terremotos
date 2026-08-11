"""Tests de la malla y del contrato ANTI-LEAKAGE del feature builder."""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from seis_peru.features.build import FeatureConfig, build_feature_matrix
from seis_peru.features.grid import Grid, assign_cells
from seis_peru.regions import PERU

UTC = timezone.utc


def test_grid_assignment_and_centroid():
    g = Grid(res=0.1, bbox=PERU)
    i, j = g.cell_index(-12.05, -77.05)
    assert (int(i), int(j)) == (64, 44)
    lat, lon = g.centroid(64, 44)
    assert lat == pytest.approx(-12.05) and lon == pytest.approx(-77.05)


def _make_catalog(extra=None):
    base = [
        (datetime(2010, 1, 1), 4.6),
        (datetime(2010, 3, 1), 4.8),
        (datetime(2011, 1, 1), 5.2),
        (datetime(2011, 5, 20), 4.7),
    ]
    if extra:
        base += extra
    rows = [
        {"preferred_mw": m, "origin_time": d.replace(tzinfo=UTC),
         "latitude": -12.05, "longitude": -77.05}
        for d, m in base
    ]
    # evento en celda vecina
    rows.append({"preferred_mw": 4.9, "origin_time": datetime(2011, 2, 1, tzinfo=UTC),
                 "latitude": -12.15, "longitude": -77.05})
    return pd.DataFrame(rows)


def _cfg():
    # Un único instante de predicción: 2011-06-01.
    t = datetime(2011, 6, 1, tzinfo=UTC)
    return FeatureConfig(res=0.1, feature_min_mag=4.0, start=t, end=t,
                         step_days=30, min_active_events=1)


def test_no_leakage():
    """Un evento FUTURO (posterior a t) no debe alterar ninguna feature,
    pero SÍ debe reflejarse en el target."""
    cfg = _cfg()
    base = build_feature_matrix(_make_catalog(), cfg)
    # mismo dataset + un M6.5 dos días DESPUÉS del instante de predicción
    withfut = build_feature_matrix(
        _make_catalog(extra=[(datetime(2011, 6, 3), 6.5)]), cfg
    )

    # fila de la celda de Lima (la de más eventos)
    def lima_row(df):
        return df.sort_values("cnt_all", ascending=False).iloc[0]

    r0, r1 = lima_row(base), lima_row(withfut)

    feature_cols = [c for c in base.columns if not c.startswith("y_")]
    for col in feature_cols:
        assert r0[col] == r1[col], f"LEAKAGE: la feature {col} cambió por un evento futuro"

    # El target sí ve el futuro:
    assert r0["y_m6_30d"] == 0
    assert r1["y_m6_30d"] == 1


def test_feature_values_make_sense():
    cfg = _cfg()
    df = build_feature_matrix(_make_catalog(), cfg)
    lima = df.sort_values("cnt_all", ascending=False).iloc[0]
    # eventos <= 2011-06-01 en la celda de Lima: 4 (dos de 2010, dos de 2011)
    assert lima["cnt_all"] == 4
    # últimos 365 días (2010-06-01..2011-06-01): 2011-01-01 y 2011-05-20 -> 2
    assert lima["cnt_365d"] == 2
    # días desde el último evento (2011-05-20 -> 2011-06-01) = 12
    assert lima["days_since_last"] == pytest.approx(12.0, abs=1.0)
