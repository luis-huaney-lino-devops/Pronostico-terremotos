"""Malla espacial del Perú y asignación de eventos a celdas.

Celdas cuadradas de ``res`` grados (por defecto 0.1°). Cada celda tiene un
índice (i, j) y un id entero estable ``cell_id = i * n_lon + j``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..regions import PERU, BBox


@dataclass(frozen=True)
class Grid:
    res: float = 0.1
    bbox: BBox = PERU

    @property
    def n_lat(self) -> int:
        return int(round((self.bbox.max_lat - self.bbox.min_lat) / self.res))

    @property
    def n_lon(self) -> int:
        return int(round((self.bbox.max_lon - self.bbox.min_lon) / self.res))

    def cell_index(self, lat, lon):
        """(i, j) enteros de la celda. Vectorizado (acepta arrays)."""
        i = np.floor((np.asarray(lat) - self.bbox.min_lat) / self.res).astype(int)
        j = np.floor((np.asarray(lon) - self.bbox.min_lon) / self.res).astype(int)
        return i, j

    def cell_id(self, i, j):
        return np.asarray(i) * self.n_lon + np.asarray(j)

    def centroid(self, i, j):
        return (
            self.bbox.min_lat + (np.asarray(i) + 0.5) * self.res,
            self.bbox.min_lon + (np.asarray(j) + 0.5) * self.res,
        )

    def in_bounds(self, i, j):
        return (i >= 0) & (i < self.n_lat) & (j >= 0) & (j < self.n_lon)


def assign_cells(df: pd.DataFrame, grid: Grid) -> pd.DataFrame:
    """Añade columnas i, j, cell_id. Descarta eventos fuera de la malla."""
    i, j = grid.cell_index(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    out = df.copy()
    out["i"], out["j"] = i, j
    mask = grid.in_bounds(i, j)
    out = out[mask].copy()
    out["cell_id"] = grid.cell_id(out["i"].to_numpy(), out["j"].to_numpy())
    return out


def active_cells(df_assigned: pd.DataFrame, min_events: int = 3) -> np.ndarray:
    """cell_id de celdas con al menos ``min_events`` eventos (para acotar el dataset)."""
    counts = df_assigned.groupby("cell_id").size()
    return counts[counts >= min_events].index.to_numpy()
