"""Cliente del catálogo del IGP (Instituto Geofísico del Perú).

A diferencia de USGS/ISC/EMSC, el IGP NO expone un servicio FDSN. Su catálogo
nacional (1960-presente, la 'verdad de campo' para el Perú) se distribuye como
un ÚNICO CSV en la Plataforma Nacional de Datos Abiertos.  (Verificado en vivo,
2026-08: 24,289 eventos, 1960-01-13 .. 2025-12-31.)

Formato del CSV (verificado):
    - separador ';'  ·  UTF-8
    - columnas: ID;FECHA_UTC;HORA_UTC;LATITUD;LONGITUD;PROFUNDIDAD;MAGNITUD;FECHA_CORTE
    - FECHA_UTC = YYYYMMDD  ·  HORA_UTC = HHMMSS pero SIN ceros a la izquierda
      (ej. 93024 -> 09:30:24)  ·  todo en UTC.
    - MAGNITUD: valor único SIN tipo (mezcla ML/mb/Mw...) -> mw_method='assume_unknown'.

Trampas conocidas:
    - El nombre del archivo cambia con cada actualización de cobertura
      (1960_2021 -> 2023 -> 2025); si el GET directo falla, se re-resuelve el
      enlace desde la página del dataset.
    - No hay API de consulta filtrada: se descarga todo y se filtra localmente.
"""
from __future__ import annotations

import io
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from ..config import settings
from ..models import NormalizedEvent

log = logging.getLogger("seis.igp")

# URL directa conocida (2025). Puede cambiar; ver _resolve_csv_url().
IGP_CSV_URL = (
    "https://www.datosabiertos.gob.pe/sites/default/files/"
    "IGP_catalogo_sismico_1960_%202025_Dataset.csv"
)
IGP_DATASET_PAGE = "https://www.datosabiertos.gob.pe/dataset/catalogo-sismico"


def _resolve_csv_url() -> Optional[str]:
    """Intenta encontrar el enlace al CSV en la página del dataset (fallback)."""
    try:
        r = requests.get(IGP_DATASET_PAGE, timeout=settings.http_timeout,
                         headers={"User-Agent": settings.user_agent})
        r.raise_for_status()
        m = re.findall(r'href="([^"]+IGP_catalogo_sismico[^"]+\.csv)"', r.text)
        if m:
            url = m[0]
            if url.startswith("/"):
                url = "https://www.datosabiertos.gob.pe" + url
            log.info("IGP: URL de CSV re-resuelta -> %s", url)
            return url
    except requests.RequestException as exc:
        log.warning("IGP: no se pudo re-resolver la URL del CSV: %s", exc)
    return None


def download_igp_csv(dest: Optional[Path] = None, url: Optional[str] = None) -> Path:
    """Descarga el CSV crudo del IGP a data/raw/igp/. Devuelve la ruta."""
    dest = dest or (settings.raw_dir / "igp" / "IGP_catalogo_sismico.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = url or IGP_CSV_URL

    r = requests.get(url, timeout=settings.http_timeout,
                     headers={"User-Agent": settings.user_agent})
    if r.status_code == 404:
        alt = _resolve_csv_url()
        if not alt:
            raise RuntimeError("IGP: CSV no encontrado (404) y sin URL alternativa.")
        r = requests.get(alt, timeout=settings.http_timeout,
                         headers={"User-Agent": settings.user_agent})
    r.raise_for_status()
    dest.write_bytes(r.content)
    log.info("IGP: descargado %s (%.1f KB)", dest.name, len(r.content) / 1024)
    return dest


def _parse_igp_frame(raw: pd.DataFrame) -> list[NormalizedEvent]:
    events: list[NormalizedEvent] = []
    n_bad = 0
    # Construye timestamp UTC desde FECHA_UTC (YYYYMMDD) + HORA_UTC (HHMMSS, sin pad).
    fecha = raw["FECHA_UTC"].astype(str).str.strip().str.zfill(8)
    hora = raw["HORA_UTC"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    ts = pd.to_datetime(fecha + hora, format="%Y%m%d%H%M%S", utc=True, errors="coerce")

    for i, row in raw.iterrows():
        ot = ts.iloc[i]
        if pd.isna(ot):
            n_bad += 1
            continue
        try:
            lat = float(row["LATITUD"])
            lon = float(row["LONGITUD"])
        except (ValueError, TypeError):
            n_bad += 1
            continue

        def _fnum(x):
            try:
                return float(x)
            except (ValueError, TypeError):
                return None

        try:
            events.append(
                NormalizedEvent(
                    source="igp",
                    source_event_id=str(row.get("ID", i)),
                    origin_time=ot.to_pydatetime(),
                    latitude=lat,
                    longitude=lon,
                    depth_km=_fnum(row.get("PROFUNDIDAD")),
                    magnitude=_fnum(row.get("MAGNITUD")),
                    magnitude_type=None,  # el IGP no publica el tipo -> assume_unknown
                    author="IGP",
                    catalog="IGP-CENSIS",
                    place=None,
                )
            )
        except Exception:
            n_bad += 1

    if n_bad:
        log.warning("IGP: %d filas descartadas (fecha/coords inválidas)", n_bad)
    log.info("IGP: %d eventos normalizados", len(events))
    return events


def fetch_igp(
    local_path: Optional[Path] = None, url: Optional[str] = None, download: bool = True
) -> list[NormalizedEvent]:
    """Carga el catálogo del IGP como lista de ``NormalizedEvent``.

    - Si ``local_path`` existe, lo usa.
    - Si no y ``download`` es True, descarga el CSV a data/raw/igp/.
    """
    if local_path and Path(local_path).exists():
        raw = pd.read_csv(local_path, sep=";", dtype=str, encoding="utf-8")
    else:
        path = download_igp_csv(url=url) if download else None
        if path is None:
            raise FileNotFoundError("IGP: sin local_path y download=False")
        raw = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8")

    raw.columns = [c.strip().upper() for c in raw.columns]
    return _parse_igp_frame(raw)
