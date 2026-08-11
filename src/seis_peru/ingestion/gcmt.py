"""Cliente GCMT (Global Centroid Moment Tensor) vía NDK directo.

GCMT publica Mw para casi todos los sismos M≳5 desde 1976. NO se sirve por
FDSN (ComCat lista 'gcmt' pero no devuelve orígenes), pero el catálogo NDK SÍ
es un GET directo y anónimo desde globalcmt.org / LDEO.

Formato NDK: 5 líneas de 80 caracteres por evento.
    L1  hipocentro PDE:  cat  YYYY/MM/DD  HH:MM:SS.s  lat  lon  depth  mb  MS  región
    L2  nombre CMT (ej. C202312010018A) + info de inversión
    L3  CENTROID (offset de tiempo, lat, lon, depth del centroide)
    L4  EXPONENTE (col 0) + 6 elementos del tensor
    L5  Vnn + 3 ejes principales + MOMENTO ESCALAR (token 10) + planos nodales

Mw se DERIVA del momento escalar (Kanamori 1977, verificado en globalcmt.org):
    M0 = mantisa(L5) · 10^exponente(L4)   [dyne·cm]
    Mw = (2/3)·(log10(M0) − 16.1)

Usamos el HIPOCENTRO PDE (L1) como ubicación —no el centroide— para que
deduplique limpio con USGS/IGP/ISC (evita el offset del centroide).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import requests
from dateutil import parser as dtparser

from ..config import settings
from ..models import NormalizedEvent

log = logging.getLogger("seis.gcmt")

# Catálogo combinado completo (1976–2025). El sufijo de fecha cambia con el
# tiempo; si falla, re-resolver desde la página del catálogo.
CATALOG_URL = "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/jan76_dec25.ndk"
QUICK_URL = "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_QUICK/qcmt.ndk"


def download_gcmt_ndk(dest: Optional[Path] = None, url: str = CATALOG_URL) -> Path:
    dest = dest or (settings.raw_dir / "gcmt" / "gcmt_catalog.ndk")
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=settings.http_timeout,
                     headers={"User-Agent": settings.user_agent})
    r.raise_for_status()
    dest.write_bytes(r.content)
    log.info("GCMT: descargado %s (%.1f MB)", dest.name, len(r.content) / 1e6)
    return dest


def _mw_from_ndk(line4: str, line5: str) -> Optional[float]:
    try:
        exp = int(line4.split()[0])
        mant = float(line5.split()[10])  # momento escalar (tras Vnn + 9 de ejes)
        if mant <= 0:
            return None
        return round((2.0 / 3.0) * (math.log10(mant) + exp - 16.1), 2)
    except (ValueError, IndexError):
        return None


def parse_ndk(text: str) -> list[NormalizedEvent]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    events: list[NormalizedEvent] = []
    n_bad = 0
    for k in range(0, len(lines) - 4, 5):
        l1, l2, l3, l4, l5 = lines[k:k + 5]
        p1 = l1.split()
        if len(p1) < 6:
            n_bad += 1
            continue
        try:
            ot = dtparser.parse(f"{p1[1]} {p1[2]}")  # 'YYYY/MM/DD HH:MM:SS.s' UTC
            lat, lon = float(p1[3]), float(p1[4])
            depth = float(p1[5])
        except (ValueError, IndexError):
            n_bad += 1
            continue
        mw = _mw_from_ndk(l4, l5)
        if mw is None:
            n_bad += 1
            continue
        cmt_name = l2.split()[0] if l2.split() else f"gcmt_{k}"
        try:
            events.append(NormalizedEvent(
                source="gcmt",
                source_event_id=cmt_name,
                origin_time=ot,
                latitude=lat, longitude=lon, depth_km=depth,
                magnitude=mw, magnitude_type="Mw",  # -> mw_method='direct'
                author="GCMT", catalog="GCMT",
            ))
        except Exception:
            n_bad += 1
    if n_bad:
        log.warning("GCMT: %d registros NDK descartados", n_bad)
    log.info("GCMT: %d eventos parseados", len(events))
    return events


def fetch_gcmt(local_path: Optional[Path] = None, download: bool = True) -> list[NormalizedEvent]:
    """Carga GCMT desde NDK local o descargándolo (catálogo completo)."""
    if local_path and Path(local_path).exists():
        text = Path(local_path).read_text(encoding="utf-8", errors="replace")
    else:
        cached = settings.raw_dir / "gcmt" / "gcmt_catalog.ndk"
        if cached.exists():
            text = cached.read_text(encoding="utf-8", errors="replace")
        elif download:
            path = download_gcmt_ndk()
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            log.warning("GCMT: sin NDK local y download=False")
            return []
    return parse_ndk(text)
