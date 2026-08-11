"""Esquema normalizado de un evento sísmico (Pydantic v2).

Toda fuente (FDSN o IGP) se transforma a ``NormalizedEvent`` antes de
deduplicar o cargar. Valida rangos físicos y homogeniza la magnitud a Mw.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .magnitude import to_mw
from .regions import tag_country


class NormalizedEvent(BaseModel):
    # --- Identidad de la observación ---
    source: str                      # 'usgs' | 'isc' | 'emsc' | 'iris' | 'igp'
    source_event_id: str

    # --- Origen ---
    origin_time: datetime
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    depth_km: Optional[float] = None

    # --- Magnitud ---
    magnitude: Optional[float] = None
    magnitude_type: Optional[str] = None
    mw: Optional[float] = None
    mw_method: Optional[str] = None

    # --- Metadatos ---
    author: Optional[str] = None
    catalog: Optional[str] = None
    contributor: Optional[str] = None
    place: Optional[str] = None
    country: Optional[str] = None
    quality: Optional[str] = None

    @field_validator("origin_time")
    @classmethod
    def _tz_aware_utc(cls, v: datetime) -> datetime:
        # Todo en UTC. Si viene naive, se asume UTC.
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @field_validator("depth_km")
    @classmethod
    def _depth_sane(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        # Profundidades negativas pequeñas (sobre el nivel del mar) -> 0.
        if v < 0:
            return 0.0
        if v > 800:  # más profundo que cualquier sismo conocido
            return None
        return v

    @model_validator(mode="after")
    def _fill_derived(self) -> "NormalizedEvent":
        if self.mw is None:
            self.mw, self.mw_method = to_mw(self.magnitude, self.magnitude_type)
        if self.country is None:
            self.country = tag_country(self.latitude, self.longitude)
        return self

    def to_row(self) -> dict:
        """Fila plana para Parquet / carga en BD."""
        return {
            "source": self.source,
            "source_event_id": self.source_event_id,
            "origin_time": self.origin_time,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "depth_km": self.depth_km,
            "magnitude": self.magnitude,
            "magnitude_type": self.magnitude_type,
            "mw": self.mw,
            "mw_method": self.mw_method,
            "author": self.author,
            "catalog": self.catalog,
            "contributor": self.contributor,
            "place": self.place,
            "country": self.country,
            "quality": self.quality,
        }
