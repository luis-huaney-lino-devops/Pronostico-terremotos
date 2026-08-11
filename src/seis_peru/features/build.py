"""Construcción del dataset de ML con corte temporal ESTRICTO (anti-leakage).

Para cada par (celda, instante de predicción t):
    - FEATURES  ← eventos con origin_time <= t        (pasado y presente)
    - TARGET    ← eventos con origin_time ∈ (t, t+H]   (futuro, nunca en features)

Formalmente  X(t) → Y(t+H).  Jamás  X(t+H) → Y(t+H).
El test `tests/test_features.py::test_no_leakage` inyecta eventos futuros y
verifica que NINGUNA feature cambia.

Nota de completitud: las features usan eventos con Mw ≥ ``feature_min_mag``
(por defecto 4.5 ≈ Mc moderna), para no contaminar los conteos con la parte
incompleta del catálogo. Los targets (M≥5, M≥6) son un subconjunto de ese set.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .grid import Grid, active_cells, assign_cells

log = logging.getLogger("seis.features")

DAY_NS = 86_400 * 1_000_000_000
CAP_DAYS = 99_999.0  # "hace mucho / nunca"


@dataclass
class FeatureConfig:
    res: float = 0.1
    feature_min_mag: float = 4.5
    horizon_days: int = 30
    step_days: int = 30
    start: datetime = field(default_factory=lambda: datetime(2000, 1, 1, tzinfo=timezone.utc))
    end: datetime | None = None  # por defecto: max(origin_time) - horizonte
    min_active_events: int = 5
    count_windows_days: tuple[int, ...] = (7, 30, 90, 365)
    etas: object | None = None        # ETASParams -> feature de intensidad ETAS
    etas_window_days: int = 730


def _merge_sorted(times_list, mags_list):
    if not times_list:
        return np.empty(0, dtype="int64"), np.empty(0, dtype=float)
    ts = np.concatenate(times_list)
    mg = np.concatenate(mags_list)
    order = np.argsort(ts, kind="mergesort")
    return ts[order], mg[order]


def build_feature_matrix(catalog: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    cfg = config or FeatureConfig()
    grid = Grid(res=cfg.res)

    cat = catalog.dropna(subset=["preferred_mw"]).copy()
    cat = cat[cat["preferred_mw"] >= cfg.feature_min_mag]
    cat = assign_cells(cat, grid)
    cat = cat.sort_values("origin_time").reset_index(drop=True)
    if cat.empty:
        raise RuntimeError("Catálogo vacío tras filtros de magnitud/malla.")

    t_ns_all = cat["origin_time"].astype("int64").to_numpy()
    mw_all = cat["preferred_mw"].to_numpy()
    cell_all = cat["cell_id"].to_numpy()
    i_all = cat["i"].to_numpy()
    j_all = cat["j"].to_numpy()

    # Arrays por celda (ya ordenados en tiempo porque cat está ordenado).
    from collections import defaultdict
    pos_by_cell: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(cat)):
        pos_by_cell[int(cell_all[idx])].append(idx)
    cell_times: dict[int, np.ndarray] = {}
    cell_mags: dict[int, np.ndarray] = {}
    cell_ij: dict[int, tuple[int, int]] = {}
    for c, pos in pos_by_cell.items():
        p = np.asarray(pos)
        cell_times[c] = t_ns_all[p]
        cell_mags[c] = mw_all[p]
        cell_ij[c] = (int(i_all[p[0]]), int(j_all[p[0]]))

    active = active_cells(cat, cfg.min_active_events)
    log.info("Malla %.2f° | %d celdas activas (≥%d eventos) de %d con actividad",
             cfg.res, len(active), cfg.min_active_events, len(pos_by_cell))

    # Instantes de predicción.
    end = cfg.end or (cat["origin_time"].max().to_pydatetime() -
                      pd.Timedelta(days=cfg.horizon_days))
    times = pd.date_range(cfg.start, end, freq=f"{cfg.step_days}D", tz="UTC")
    t_pred = times.asi8  # ns int64 UTC
    H_ns = cfg.horizon_days * DAY_NS
    win_ns = {w: w * DAY_NS for w in cfg.count_windows_days}
    log.info("%d instantes de predicción (%s → %s, paso %dd, H=%dd)",
             len(t_pred), times.min().date(), times.max().date(),
             cfg.step_days, cfg.horizon_days)

    # --- Features regionales (todo el Perú): iguales para todas las celdas en t ---
    reg_cnt30 = (np.searchsorted(t_ns_all, t_pred, "right")
                 - np.searchsorted(t_ns_all, t_pred - win_ns[30], "right"))
    reg_maxmag30 = np.zeros(len(t_pred))
    for k, t in enumerate(t_pred):
        lo = np.searchsorted(t_ns_all, t - win_ns[30], "right")
        hi = np.searchsorted(t_ns_all, t, "right")
        if hi > lo:
            reg_maxmag30[k] = mw_all[lo:hi].max()

    def _win(ts, mags, t, w):
        hi = np.searchsorted(ts, t, "right")
        lo = np.searchsorted(ts, t - w, "right")
        if hi > lo:
            seg = mags[lo:hi]
            return hi - lo, float(seg.max()), float(np.log10(1.0 + np.sum(10.0 ** (1.5 * seg))))
        return 0, 0.0, 0.0

    et = cfg.etas
    et_win = cfg.etas_window_days * DAY_NS

    def _etas_rate(ts, mags, t):
        """Intensidad de triggering ETAS (leakage-safe: solo eventos < t)."""
        if et is None:
            return 0.0
        lo = np.searchsorted(ts, t - et_win, "right")
        hi = np.searchsorted(ts, t, "right")
        if hi <= lo:
            return 0.0
        dt = (t - ts[lo:hi]) / DAY_NS
        return float(np.sum(et.K * np.exp(et.alpha * (mags[lo:hi] - et.M0)) * (dt + et.c) ** (-et.p)))

    rows = []
    for c in active:
        i0, j0 = cell_ij[c]
        cen_lat, cen_lon = grid.centroid(i0, j0)
        cts, cmg = cell_times[c], cell_mags[c]
        m5_times = cts[cmg >= 5.0]  # para days_since_M5

        # Vecindario 3x3 (celda + adyacentes que existan con actividad).
        nb_ids = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                nid = int(grid.cell_id(i0 + di, j0 + dj))
                if nid in cell_times:
                    nb_ids.append(nid)
        nts, nmg = _merge_sorted([cell_times[n] for n in nb_ids],
                                 [cell_mags[n] for n in nb_ids])

        for k, t in enumerate(t_pred):
            # --- conteos por ventana (celda) ---
            c7 = np.searchsorted(cts, t, "right") - np.searchsorted(cts, t - win_ns[7], "right")
            cnt30, mx30, le30 = _win(cts, cmg, t, win_ns[30])
            cnt90, mx90, _ = _win(cts, cmg, t, win_ns[90])
            cnt365, mx365, le365 = _win(cts, cmg, t, win_ns[365])
            cnt_all = int(np.searchsorted(cts, t, "right"))

            # --- tiempo desde el último / último M5 ---
            hi = np.searchsorted(cts, t, "right")
            dsl = (t - cts[hi - 1]) / DAY_NS if hi > 0 else CAP_DAYS
            hi5 = np.searchsorted(m5_times, t, "right")
            dsl5 = (t - m5_times[hi5 - 1]) / DAY_NS if hi5 > 0 else CAP_DAYS

            # --- vecindario ---
            nb30 = int(np.searchsorted(nts, t, "right") - np.searchsorted(nts, t - win_ns[30], "right"))
            nb365, nbmx365, _ = _win(nts, nmg, t, win_ns[365])

            # --- Intensidad ETAS (celda y vecindario) ---
            etas_cell = np.log1p(_etas_rate(cts, cmg, t))
            etas_nb = np.log1p(_etas_rate(nts, nmg, t))

            # --- TARGET: eventos en (t, t+H] en la celda ---
            f_lo = np.searchsorted(cts, t, "right")
            f_hi = np.searchsorted(cts, t + H_ns, "right")
            fut = cmg[f_lo:f_hi]
            y_m6_30 = int((fut >= 6.0).any())
            y_m5_30 = int((fut >= 5.0).any())
            # M5 en 7 días
            f_hi7 = np.searchsorted(cts, t + win_ns[7], "right")
            y_m5_7 = int((cmg[f_lo:f_hi7] >= 5.0).any())

            rows.append((
                int(c), i0, j0, float(cen_lat), float(cen_lon), int(t),
                int(c7), cnt30, cnt90, cnt365, cnt_all,
                mx30, mx90, mx365, le30, le365, dsl, dsl5,
                nb30, nb365, nbmx365,
                etas_cell, etas_nb,
                int(reg_cnt30[k]), float(reg_maxmag30[k]),
                y_m6_30, y_m5_30, y_m5_7,
            ))

    cols = [
        "cell_id", "i", "j", "cen_lat", "cen_lon", "t_ns",
        "cnt_7d", "cnt_30d", "cnt_90d", "cnt_365d", "cnt_all",
        "maxmag_30d", "maxmag_90d", "maxmag_365d", "logE_30d", "logE_365d",
        "days_since_last", "days_since_M5",
        "nb_cnt_30d", "nb_cnt_365d", "nb_maxmag_365d",
        "etas_cell", "etas_nb",
        "reg_cnt_30d", "reg_maxmag_30d",
        "y_m6_30d", "y_m5_30d", "y_m5_7d",
    ]
    df = pd.DataFrame.from_records(rows, columns=cols)
    df["t"] = pd.to_datetime(df["t_ns"], utc=True)
    df = df.drop(columns=["t_ns"])
    log.info("Dataset: %d filas (%d celdas × %d instantes)",
             len(df), len(active), len(t_pred))
    for tgt in ("y_m6_30d", "y_m5_30d", "y_m5_7d"):
        pos = df[tgt].mean()
        log.info("  positivos %s: %d (%.3f%%)", tgt, int(df[tgt].sum()), 100 * pos)
    return df
