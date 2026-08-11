"""Deduplicación / asociación de eventos entre catálogos.

El mismo terremoto físico es reportado por varias fuentes (IGP, USGS, ISC,
EMSC...) con tiempo, hipocentro y magnitud ligeramente distintos:

            EVENTO REAL
                 │
      ┌──────────┼──────────┐
     IGP        USGS       EMSC
   15:31:21   15:31:22   15:31:20
    M5.2       M5.1       M5.2

No son tres sismos: es UNO. Este módulo los agrupa en un ``canonical_id``.

Método (estándar en la literatura, p.ej. asociación ANSS/ComCat, fusión
ISC-GEM): dos observaciones son el mismo evento si
    |Δt| ≤ dt_max_s   Y   distancia_epicentral ≤ dist_max_km
(con tolerancias opcionales de profundidad y magnitud). Se construye un grafo
de coincidencias y se toman las componentes conexas (union-find). El barrido
va ordenado por tiempo, comparando solo vecinos dentro de la ventana temporal,
así que es eficiente incluso con cientos de miles de eventos.

El ORIGEN preferido (hipocentro/tiempo) se elige por prioridad de fuente
(IGP local manda para el Perú); la MAGNITUD preferida por calidad del tipo de
magnitud (Mw directa manda) y luego prioridad de fuente. NUNCA se descartan
observaciones: solo se enlazan.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .magnitude import method_priority

log = logging.getLogger("seis.dedup")

EARTH_R_KM = 6371.0088


@dataclass
class DedupConfig:
    # Umbrales avalados por la literatura de fusión multi-catálogo:
    #  - local vs teleseísmico (IGP vs USGS/ISC) tiene dispersión ~5 s / ~12-16 km.
    #  - ventana 16 s / 100 km cubre ese caso sin ser tan laxa como 60 s / 1°.
    # (Zuñiga 2019; Vorobieva 2022 σ_T≈5s, σ_epi≈15km; estudio de fusión 2025.)
    dt_max_s: float = 16.0          # ventana temporal
    dist_max_km: float = 100.0      # distancia epicentral máx (haversine)
    depth_max_km: float | None = None   # tolerancia de profundidad (opcional)
    mag_tol: float | None = None    # tolerancia de magnitud (opcional; None = ignorar)
    # Prioridad de fuente para el ORIGEN (hipocentro/tiempo). Menor índice = mejor.
    # IGP primero: red LOCAL = mejores epicentros para sismos DEL Perú; para
    # eventos fuera del Perú IGP no los tiene y cae a ISC/USGS.
    origin_priority: tuple[str, ...] = ("igp", "isc", "usgs", "emsc")
    # Umbral para avisar de posibles fusiones falsas (enjambres): si un evento
    # canónico absorbe más de este nº de observaciones, se marca para revisión.
    warn_component_size: int = 12


def _src_rank(source: str, priority: tuple[str, ...]) -> int:
    s = (source or "").lower()
    return priority.index(s) if s in priority else len(priority)


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distancia de gran círculo en km (acepta escalares o arrays numpy)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # compresión de camino
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def deduplicate(obs: pd.DataFrame, config: DedupConfig | None = None) -> pd.DataFrame:
    """Asigna ``canonical_id`` a cada observación.

    Entrada: DataFrame de observaciones normalizadas (columnas: source,
    source_event_id, origin_time [tz-aware], latitude, longitude, depth_km,
    magnitude, magnitude_type, mw, mw_method, ...).

    Devuelve el mismo DataFrame con una columna entera ``canonical_id`` añadida.
    """
    cfg = config or DedupConfig()
    if obs.empty:
        return obs.assign(canonical_id=pd.Series(dtype="int64"))

    df = obs.sort_values("origin_time", kind="mergesort").reset_index(drop=True)
    n = len(df)
    uf = _UnionFind(n)

    t = df["origin_time"].astype("int64").to_numpy()  # ns epoch
    lat = df["latitude"].to_numpy(dtype=float)
    lon = df["longitude"].to_numpy(dtype=float)
    depth = df["depth_km"].to_numpy(dtype=float)
    mw = df["mw"].to_numpy(dtype=float)
    dt_max_ns = int(cfg.dt_max_s * 1e9)

    for i in range(n):
        j = i + 1
        while j < n and (t[j] - t[i]) <= dt_max_ns:
            same = haversine_km(lat[i], lon[i], lat[j], lon[j]) <= cfg.dist_max_km
            if same and cfg.depth_max_km is not None:
                if not (np.isnan(depth[i]) or np.isnan(depth[j])):
                    same = same and abs(depth[i] - depth[j]) <= cfg.depth_max_km
            if same and cfg.mag_tol is not None:
                if not (np.isnan(mw[i]) or np.isnan(mw[j])):
                    same = same and abs(mw[i] - mw[j]) <= cfg.mag_tol
            if same:
                uf.union(i, j)
            j += 1

    roots = np.array([uf.find(i) for i in range(n)])
    # Renumera raíces a IDs canónicos densos 1..K (por primera aparición temporal).
    _, first_idx = np.unique(roots, return_index=True)
    order = np.argsort(first_idx)
    remap = {roots[first_idx[k]]: cid for cid, k in enumerate(order, start=1)}
    df["canonical_id"] = [remap[r] for r in roots]

    n_events = df["canonical_id"].nunique()
    log.info(
        "Dedup: %d observaciones -> %d eventos canónicos (%.1f%% de fusión)",
        n, n_events, 100 * (1 - n_events / n) if n else 0,
    )
    return df


def build_canonical_events(
    dedup_df: pd.DataFrame, config: DedupConfig | None = None
) -> pd.DataFrame:
    """A partir de observaciones con ``canonical_id``, construye el catálogo
    canónico: una fila por evento físico con origen y magnitud preferidos.
    """
    cfg = config or DedupConfig()
    df = dedup_df.copy()
    df["_src_rank"] = df["source"].map(lambda s: _src_rank(s, cfg.origin_priority))
    df["_mag_rank"] = df["mw_method"].map(method_priority)
    df["_has_depth"] = df["depth_km"].notna().astype(int)
    df["_has_mw"] = df["mw"].notna().astype(int)

    events = []
    for cid, g in df.groupby("canonical_id", sort=True):
        # --- Origen preferido: mejor prioridad de fuente, luego más completo ---
        og = g.sort_values(
            ["_src_rank", "_has_depth", "_has_mw"], ascending=[True, False, False]
        ).iloc[0]
        # --- Magnitud preferida: mejor tipo (Mw directa), luego prioridad fuente ---
        mg = g.sort_values(
            ["_mag_rank", "_src_rank"], ascending=[False, True]
        ).iloc[0]

        events.append(
            {
                "canonical_id": int(cid),
                "origin_time": og["origin_time"],
                "latitude": og["latitude"],
                "longitude": og["longitude"],
                "depth_km": og["depth_km"],
                "preferred_mw": mg["mw"],
                "preferred_mag": mg["magnitude"],
                "preferred_mag_type": mg["magnitude_type"],
                "preferred_source": og["source"],
                "mag_source": mg["source"],
                "n_sources": g["source"].nunique(),
                "n_obs": len(g),
                "country": og.get("country"),
                "sources": ",".join(sorted(g["source"].unique())),
            }
        )

    out = pd.DataFrame(events).sort_values("origin_time").reset_index(drop=True)
    log.info("Catálogo canónico: %d eventos", len(out))

    # Aviso de posibles fusiones falsas en enjambres (single-link puede encadenar).
    big = out[out["n_obs"] > cfg.warn_component_size]
    if len(big):
        log.warning(
            "%d eventos canónicos absorbieron >%d observaciones (posible fusión "
            "falsa en enjambre; revisar). Máx=%d obs.",
            len(big), cfg.warn_component_size, int(out["n_obs"].max()),
        )
    return out
