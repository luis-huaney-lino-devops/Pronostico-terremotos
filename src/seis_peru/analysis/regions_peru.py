"""Macro-regiones del Perú para análisis regional de b-value.

Bandas latitudinales usadas habitualmente en estudios de sismicidad peruana.
La subducción de Nazca cambia de geometría (flat-slab en el centro-norte,
subducción normal en el sur), así que el b-value suele variar norte↔sur.
"""
from __future__ import annotations

from ..regions import BBox

PERU_NORTE = BBox(min_lat=-8.0, max_lat=0.5, min_lon=-81.5, max_lon=-68.0)
PERU_CENTRO = BBox(min_lat=-14.0, max_lat=-8.0, min_lon=-81.5, max_lon=-68.0)
PERU_SUR = BBox(min_lat=-18.5, max_lat=-14.0, min_lon=-76.0, max_lon=-68.0)

PERU_MACRO = {
    "Perú-Norte": PERU_NORTE,
    "Perú-Centro": PERU_CENTRO,
    "Perú-Sur": PERU_SUR,
}
