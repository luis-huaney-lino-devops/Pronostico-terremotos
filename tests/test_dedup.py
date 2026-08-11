"""Tests de deduplicación / asociación de eventos."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from seis_peru.dedup import build_canonical_events, deduplicate, haversine_km


def test_haversine_one_degree_lat():
    # 1 grado de latitud ~ 111 km
    d = float(haversine_km(0.0, 0.0, 1.0, 0.0))
    assert d == pytest.approx(111.19, abs=1.0)


def _obs(source, sid, t, lat, lon, mw, method="direct", magtype="mww"):
    return {
        "source": source, "source_event_id": sid, "origin_time": t,
        "latitude": lat, "longitude": lon, "depth_km": 50.0,
        "magnitude": mw, "magnitude_type": magtype, "mw": mw,
        "mw_method": method, "country": "Peru",
    }


def test_three_sources_one_event():
    t0 = datetime(2023, 12, 20, 12, 11, 21, tzinfo=timezone.utc)
    # Mismo sismo visto por 3 fuentes (dentro de 5 s y ~2 km).
    df = pd.DataFrame([
        _obs("usgs", "us1", t0, -15.8585, -72.5207, 6.2),
        _obs("emsc", "em1", t0 + timedelta(seconds=2), -15.86, -72.52, 6.2, "assume_unknown", "m"),
        _obs("isc", "is1", t0 + timedelta(seconds=1), -15.9088, -72.5166, 6.26),
        # Evento distinto, lejos y otro día.
        _obs("usgs", "us2", t0 + timedelta(days=1), -8.50, -74.56, 5.0),
    ])
    dd = deduplicate(df)
    assert dd["canonical_id"].nunique() == 2

    canon = build_canonical_events(dd)
    assert len(canon) == 2
    big = canon.sort_values("n_obs", ascending=False).iloc[0]
    assert big["n_sources"] == 3
    # La magnitud preferida debe venir de una Mw directa (usgs o isc), no del "m".
    assert big["mag_source"] in {"usgs", "isc"}
    assert big["preferred_mw"] == pytest.approx(6.2, abs=0.1)


def test_far_events_not_merged():
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    df = pd.DataFrame([
        _obs("usgs", "a", t0, -12.0, -77.0, 5.0),
        _obs("usgs", "b", t0 + timedelta(seconds=3), -12.0, -70.0, 5.0),  # ~760 km
    ])
    dd = deduplicate(df)
    assert dd["canonical_id"].nunique() == 2
