"""Cliente FDSN genérico (USGS / ISC / IRIS / EMSC).

Todas estas fuentes exponen el mismo estándar **FDSN event web service**, así
que un único cliente sirve para las cuatro. Usamos ``format=text`` porque es el
formato tabular estándar soportado por TODAS (columnas separadas por '|').

Problema clásico: cada servicio limita la respuesta (USGS: 20 000 eventos). Para
descargar décadas de historia usamos **bisección temporal adaptativa**: si una
ventana devuelve el máximo de filas (truncada), se parte en dos mitades y se
reintenta cada una, recursivamente, hasta que ninguna venga truncada.

Formato FDSN text (cabecera '#'):
    EventID | Time | Latitude | Longitude | Depth/km | Author | Catalog |
    Contributor | ContributorID | MagType | Magnitude | MagAuthor |
    EventLocationName
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional

import requests
from dateutil import parser as dtparser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import settings
from ..models import NormalizedEvent
from ..regions import BBox

log = logging.getLogger("seis.fdsn")

# Endpoints FDSN event query por fuente. Verificados EN VIVO (2026-08):
#   - USGS / EMSC / ISC: los tres emiten format=text (pipe-delimited).
#     ISC añade una 14ª columna EventType y líneas de comentario '#' al final;
#     el parser lo tolera.
#   - IRIS/EarthScope FUE RETIRADO (HTTP 410 Gone desde 2026-06-01). No usar.
#     EarthScope redirige a ISC y USGS. Ver PLAN_FASE1.md.
FDSN_ENDPOINTS: dict[str, str] = {
    "usgs": "https://earthquake.usgs.gov/fdsnws/event/1/query",
    "emsc": "https://www.seismicportal.eu/fdsnws/event/1/query",
    "isc": "https://www.isc.ac.uk/fdsnws/event/1/query",
}
# NOTA: GCMT NO se sirve por FDSN/ComCat (catalog=gcmt existe pero no devuelve
# orígenes: aporta tensores como productos). GCMT se ingiere por NDK directo
# (globalcmt.org) en ingestion/gcmt.py.

# Parámetro 'catalog' de FDSN por fuente (subcatálogo dentro del endpoint).
SOURCE_CATALOG: dict[str, str] = {}

# Tope de filas por ventana según la fuente (para la bisección adaptativa).
#   USGS = 20000 (tope duro), EMSC = 20000, ISC = 40000 (default del servicio).
SOURCE_MAX_ROWS: dict[str, int] = {"usgs": 20000, "emsc": 20000, "isc": 20000}

# Máximo de filas que pedimos por ventana (USGS tope duro = 20000).
DEFAULT_MAX_ROWS = 20000
# Mínima ventana antes de rendirnos al bisecar (evita recursión infinita).
MIN_WINDOW = timedelta(seconds=2)


def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=settings.http_max_retries,
        backoff_factor=settings.http_backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": settings.user_agent})
    return s


_SESSION: Optional[requests.Session] = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = _make_session()
    return _SESSION


def _fmt(dt: datetime) -> str:
    """FDSN espera 'YYYY-MM-DDThh:mm:ss' en UTC."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _parse_text(body: str) -> list[dict]:
    """Parsea la respuesta FDSN text a filas nativas (dict)."""
    rows: list[dict] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 13:
            continue

        def _f(x: str) -> Optional[float]:
            x = x.strip()
            return float(x) if x else None

        rows.append(
            {
                "event_id": parts[0].strip(),
                "time": parts[1].strip(),
                "latitude": _f(parts[2]),
                "longitude": _f(parts[3]),
                "depth_km": _f(parts[4]),
                "author": parts[5].strip() or None,
                "catalog": parts[6].strip() or None,
                "contributor": parts[7].strip() or None,
                "contributor_id": parts[8].strip() or None,
                "mag_type": parts[9].strip() or None,
                "magnitude": _f(parts[10]),
                "mag_author": parts[11].strip() or None,
                "place": parts[12].strip() or None,
            }
        )
    return rows


def _request_window(
    source: str,
    start: datetime,
    end: datetime,
    bbox: BBox,
    min_magnitude: Optional[float],
    max_rows: int,
) -> tuple[list[dict], bool]:
    """Una petición a una ventana. Devuelve (filas, truncada?)."""
    url = FDSN_ENDPOINTS[source]
    params = {
        "starttime": _fmt(start),
        "endtime": _fmt(end),
        "minlatitude": bbox.min_lat,
        "maxlatitude": bbox.max_lat,
        "minlongitude": bbox.min_lon,
        "maxlongitude": bbox.max_lon,
        "format": "text",
        "orderby": "time-asc",
        "limit": max_rows,
        "nodata": 204,
    }
    if min_magnitude is not None:
        params["minmagnitude"] = min_magnitude
    if source in SOURCE_CATALOG:
        params["catalog"] = SOURCE_CATALOG[source]

    resp = _session().get(url, params=params, timeout=settings.http_timeout)

    if resp.status_code == 204:  # sin datos
        return [], False
    if resp.status_code == 400:
        # Algunos servicios responden 400 cuando se excede el límite de filas.
        low = resp.text.lower()
        if any(k in low for k in ("limit", "exceed", "too many", "20000")):
            return [], True
        raise RuntimeError(f"[{source}] HTTP 400: {resp.text[:300]}")
    resp.raise_for_status()

    rows = _parse_text(resp.text)
    truncated = len(rows) >= max_rows
    return rows, truncated


def fetch_native(
    source: str,
    start: datetime,
    end: datetime,
    bbox: BBox = None,  # type: ignore[assignment]
    min_magnitude: Optional[float] = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    polite_delay: float = 0.3,
) -> list[dict]:
    """Descarga TODOS los eventos [start, end] con bisección adaptativa.

    Devuelve filas nativas (dict) deduplicadas por ``event_id`` dentro de la
    misma fuente.
    """
    from ..regions import STUDY_REGION

    if source not in FDSN_ENDPOINTS:
        raise ValueError(f"Fuente FDSN desconocida: {source!r}")
    if bbox is None:
        bbox = STUDY_REGION

    out: dict[str, dict] = {}
    stack: list[tuple[datetime, datetime]] = [(start, end)]
    n_requests = 0

    while stack:
        s, e = stack.pop()
        rows, truncated = _request_window(source, s, e, bbox, min_magnitude, max_rows)
        n_requests += 1
        # Progreso visible en descargas largas (ISC/históricos con bisección).
        if n_requests % 10 == 0:
            log.info(
                "[%s] %d peticiones, %d eventos acumulados, %d ventanas en cola "
                "(procesando ~%s)",
                source, n_requests, len(out), len(stack), s.date(),
            )
        if polite_delay:
            time.sleep(polite_delay)

        if truncated and (e - s) > MIN_WINDOW:
            mid = s + (e - s) / 2
            stack.append((mid, e))
            stack.append((s, mid))
            log.debug("[%s] ventana %s..%s truncada -> bisecando", source, s, e)
            continue

        if truncated:
            log.warning(
                "[%s] ventana mínima %s..%s aún truncada; puede faltar data",
                source, s, e,
            )
        for r in rows:
            eid = r["event_id"]
            if eid:  # dedup por id nativo
                out[eid] = r

    log.info("[%s] %d eventos únicos en %d peticiones", source, len(out), n_requests)
    return list(out.values())


def native_to_normalized(source: str, row: dict) -> Optional[NormalizedEvent]:
    """Convierte una fila nativa FDSN a ``NormalizedEvent`` (o None si inválida)."""
    if row.get("latitude") is None or row.get("longitude") is None:
        return None
    try:
        ot = dtparser.isoparse(row["time"])
    except (ValueError, TypeError):
        return None
    try:
        return NormalizedEvent(
            source=source,
            source_event_id=row["event_id"],
            origin_time=ot,
            latitude=row["latitude"],
            longitude=row["longitude"],
            depth_km=row.get("depth_km"),
            magnitude=row.get("magnitude"),
            magnitude_type=row.get("mag_type"),
            author=row.get("author"),
            catalog=row.get("catalog"),
            contributor=row.get("contributor"),
            place=row.get("place"),
        )
    except Exception as exc:  # validación Pydantic (lat/lon fuera de rango, etc.)
        log.debug("[%s] evento descartado (%s): %s", source, exc, row.get("event_id"))
        return None


def iter_time_chunks(
    start: datetime, end: datetime, step_days: int
) -> Iterator[tuple[datetime, datetime]]:
    """Genera sub-ventanas [a, b] de ``step_days`` para descargas por lotes."""
    cur = start
    step = timedelta(days=step_days)
    while cur < end:
        nxt = min(cur + step, end)
        yield cur, nxt
        cur = nxt
