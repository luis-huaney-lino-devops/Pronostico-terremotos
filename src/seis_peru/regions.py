"""Regiones geográficas y bounding boxes.

- ``STUDY_REGION``: caja MAESTRA de ingesta (Perú + países vecinos + zona
  offshore de la placa de Nazca). Todo lo que descargamos cae aquí.
- ``COUNTRY_BOXES``: cajas por país para etiquetado rápido en Python
  (sin necesidad de PostGIS). Se refinará con polígonos reales luego.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def contains(self, lat: float, lon: float) -> bool:
        return (
            self.min_lat <= lat <= self.max_lat
            and self.min_lon <= lon <= self.max_lon
        )


# Perú un poco holgado (incluye margen costero/offshore).
PERU = BBox(min_lat=-18.5, max_lat=0.5, min_lon=-81.5, max_lon=-68.0)
ECUADOR = BBox(min_lat=-5.0, max_lat=2.0, min_lon=-82.0, max_lon=-75.0)
CHILE = BBox(min_lat=-56.0, max_lat=-17.0, min_lon=-76.0, max_lon=-66.0)
BOLIVIA = BBox(min_lat=-23.0, max_lat=-9.0, min_lon=-70.0, max_lon=-57.5)
COLOMBIA = BBox(min_lat=-4.5, max_lat=13.0, min_lon=-82.0, max_lon=-66.0)

# Caja maestra de ingesta: generosa, cubre toda la subducción de Nazca
# adyacente al Perú y los vecinos que entran en la hipótesis de propagación.
STUDY_REGION = BBox(min_lat=-60.0, max_lat=15.0, min_lon=-85.0, max_lon=-58.0)

# Orden = prioridad de asignación (Perú primero: es nuestro foco).
COUNTRY_BOXES: dict[str, BBox] = {
    "Peru": PERU,
    "Ecuador": ECUADOR,
    "Chile": CHILE,
    "Bolivia": BOLIVIA,
    "Colombia": COLOMBIA,
}


def tag_country(lat: float, lon: float) -> str:
    """Etiqueta país por bbox (aprox). Perú tiene prioridad ante solapes."""
    for name, box in COUNTRY_BOXES.items():
        if box.contains(lat, lon):
            return name
    return "Other"
