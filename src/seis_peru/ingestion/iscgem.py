"""Cliente del catálogo ISC-GEM (Global Instrumental, Mw homogéneo 1904–2021).

ISC-GEM es el mejor backbone de largo plazo: magnitud momento Mw homogénea
(≈46% GCMT, ≈43% proxy-Ms, ≈8% proxy-mb) para toda la era instrumental.

⚠️ NO es descargable por GET anónimo: está detrás de un formulario con CAPTCHA.
    Descarga MANUAL (una vez):
      1. Abrir  http://www.isc.ac.uk/iscgem/request_catalogue.php
      2. Rellenar Nombre/Email/Afiliación/País + CAPTCHA y enviar.
      3. Guardar el CSV principal en  data/raw/iscgem/isc-gem-cat.csv
    Versión actual: v12.1 (2025-11-27), DOI 10.31905/D808B825, CC-BY-SA 3.0.

Formato verificado (24 columnas, comas, cabecera con '#'):
    1 date, 2 lat, 3 lon, 4 smajax, 5 sminax, 6 strike, 7 q, 8 depth, 9 unc,
    10 q, 11 mw, 12 unc(mw), 13 q(mw), 14 s, 15 mo, 16 fac, 17 mo_auth,
    18 mpp, 19 mpr, 20 mrr, 21 mrt, 22 mtp, 23 mtt, 24 eventid
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import settings
from ..models import NormalizedEvent

log = logging.getLogger("seis.iscgem")

DEFAULT_PATH = "data/raw/iscgem/isc-gem-cat.csv"
COLS = [
    "date", "lat", "lon", "smajax", "sminax", "strike", "q_loc", "depth",
    "depth_unc", "q_depth", "mw", "mw_unc", "q_mw", "s", "mo", "fac",
    "mo_auth", "mpp", "mpr", "mrr", "mrt", "mtp", "mtt", "eventid",
]


def fetch_iscgem(local_path: Optional[Path] = None) -> list[NormalizedEvent]:
    """Carga el catálogo ISC-GEM desde un CSV local (descarga manual).

    Devuelve [] con un aviso si el archivo no existe (para no romper el pipeline).
    """
    path = Path(local_path) if local_path else (settings.data_dir.parent / DEFAULT_PATH)
    if not path.is_absolute():
        path = Path(DEFAULT_PATH)
    if not path.exists():
        # también probar bajo data/raw/iscgem/
        alt = settings.raw_dir / "iscgem" / "isc-gem-cat.csv"
        path = alt if alt.exists() else path
    if not path.exists():
        log.warning(
            "ISC-GEM no encontrado en %s. Descárgalo manualmente "
            "(http://www.isc.ac.uk/iscgem/request_catalogue.php) y colócalo ahí.",
            path,
        )
        return []

    raw = pd.read_csv(path, comment="#", header=None, names=COLS,
                      skipinitialspace=True, engine="python")
    events: list[NormalizedEvent] = []
    n_bad = 0
    ts = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    for i, row in raw.iterrows():
        ot = ts.iloc[i]
        try:
            lat, lon = float(row["lat"]), float(row["lon"])
            mw = float(row["mw"])
        except (ValueError, TypeError):
            n_bad += 1
            continue
        if pd.isna(ot):
            n_bad += 1
            continue
        try:
            events.append(NormalizedEvent(
                source="iscgem",
                source_event_id=str(row["eventid"]).strip(),
                origin_time=ot.to_pydatetime(),
                latitude=lat, longitude=lon,
                depth_km=_num(row["depth"]),
                magnitude=mw, magnitude_type="Mw",  # ya es Mw -> mw_method='direct'
                author=str(row["mo_auth"]).strip() or "ISC-GEM",
                catalog="ISC-GEM",
            ))
        except Exception:
            n_bad += 1
    if n_bad:
        log.warning("ISC-GEM: %d filas descartadas", n_bad)
    log.info("ISC-GEM: %d eventos cargados", len(events))
    return events


def _num(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None
