"""Homogenización de magnitudes a Mw (magnitud momento).

Distintas fuentes reportan tipos distintos (mb, Ms, ML, Md, Mw...). Para
comparar y modelar necesitamos una escala común: Mw. Se conserva SIEMPRE la
magnitud original; ``mw`` y ``mw_method`` son derivadas y trazables.

ESQUEMAS de conversión disponibles (todos con coeficientes VERIFICADOS y citados):

- ``scordilis``  (por defecto) — Scordilis (2006), J. Seismology 10:225.
  Es el estándar de facto y **el que usa la PSHA nacional del Perú (IGP
  2014/2015 → SENCICO / Norma E.030)**, así que maximiza comparabilidad.
      Ms→Mw: 0.67·Ms+2.07 (3.0–6.1); 0.99·Ms+0.08 (6.2–8.2)
      mb→Mw: 0.85·mb+1.03 (3.5–6.2)

- ``digiacomo`` — Di Giacomo et al. (2015), PEPI 239:33 (base de ISC-GEM):
      Ms→Mw: exp(−0.222+0.233·Ms)+2.863
      mb→Mw: exp(−4.664+0.859·mb)+4.555

- ``peru`` — Cahuari Begazo (2008), tesis UNSA/IGP, 112 sismos del Perú
  1990–2005. GREY LITERATURE (OLS, rango 4.5–6.8, correlación moderada);
  usar con cautela y solo en rango. Es la ÚNICA relación ML(d)→Mw peruana:
      mb→Mw:    0.9588·mb+0.458   (5.1–6.8)
      Ms→Mw:    0.7044·Ms+1.702   (4.5–6.8)
      ML(d)→Mw: 0.9879·ML+0.3316  (4.5–6.8)  [ML(d)=magnitud de duración del IGP]

- ``igp_national`` — flujo oficial IGP: mb→Ms (GSHAP Andes-Norte) y luego
  Ms→Mw (Scordilis):  Ms=1.644·mb−3.753 (mb<5.9); Ms=2.763·mb−10.301 (mb≥5.9).

NOTA HONESTA: no existe una relación ML→Mw peruana peer-reviewed robusta. Lo
más defensible es preferir Mw DIRECTA (GCMT / ISC-GEM / USGS-W-phase) vía la
deduplicación, y convertir solo mb/Ms. Ver PLAN_FASE1.md.
"""
from __future__ import annotations

import math
from typing import Optional


def _norm_type(magtype: Optional[str]) -> str:
    return (magtype or "").strip().lower()


# ---------------------------------------------------------------- Scordilis
def ms_to_mw(ms: float) -> float:
    return 0.67 * ms + 2.07 if ms <= 6.1 else 0.99 * ms + 0.08


def mb_to_mw(mb: float) -> float:
    return 0.85 * mb + 1.03


def _scheme_scordilis(mag: float, t: str) -> tuple[float, str]:
    if t.startswith("ms"):
        return ms_to_mw(mag), "scordilis_ms"
    if t.startswith("mb") or t == "mblg":
        return mb_to_mw(mag), "scordilis_mb"
    if t.startswith("ml") or t in {"mlv", "ml(v)"}:
        return mag, "assume_ml"
    if t.startswith("md") or t.startswith("mc"):
        return mag, "assume_md"
    return mag, "assume_unknown"


# ---------------------------------------------------------------- Di Giacomo 2015
def _scheme_digiacomo(mag: float, t: str) -> tuple[float, str]:
    if t.startswith("ms"):
        return math.exp(-0.222 + 0.233 * mag) + 2.863, "digiacomo_ms"
    if t.startswith("mb") or t == "mblg":
        return math.exp(-4.664 + 0.859 * mag) + 4.555, "digiacomo_mb"
    return _scheme_scordilis(mag, t)  # sin relación ML/Md -> cae a assume


# ---------------------------------------------------------------- Perú (Cahuari 2008)
def _scheme_peru(mag: float, t: str) -> tuple[float, str]:
    if t.startswith("mb") or t == "mblg":
        return 0.9588 * mag + 0.458, "peru_cahuari_mb"
    if t.startswith("ms"):
        return 0.7044 * mag + 1.702, "peru_cahuari_ms"
    if t.startswith("ml") or t.startswith("md") or t.startswith("mc") or t in {"mlv", "ml(v)"}:
        # ML(d) = magnitud de duración del IGP (grey lit, rango 4.5–6.8).
        return 0.9879 * mag + 0.3316, "peru_cahuari_mld"
    return _scheme_scordilis(mag, t)


# ---------------------------------------------------------------- IGP nacional
def _scheme_igp_national(mag: float, t: str) -> tuple[float, str]:
    if t.startswith("mb") or t == "mblg":
        ms = 1.644 * mag - 3.753 if mag < 5.9 else 2.763 * mag - 10.301
        return ms_to_mw(ms), "igp_mb_ms_mw"
    if t.startswith("ms"):
        return ms_to_mw(mag), "scordilis_ms"
    return _scheme_scordilis(mag, t)


_SCHEMES = {
    "scordilis": _scheme_scordilis,
    "digiacomo": _scheme_digiacomo,
    "peru": _scheme_peru,
    "igp_national": _scheme_igp_national,
}


def to_mw(magnitude: Optional[float], magtype: Optional[str],
          scheme: str = "scordilis") -> tuple[Optional[float], str]:
    """Devuelve (mw, metodo). Mw directa nunca se convierte."""
    if magnitude is None:
        return None, "none"
    t = _norm_type(magtype)
    # Familia de magnitudes-momento -> directa (el "m" genérico de EMSC NO).
    if t.startswith("mw"):
        return magnitude, "direct"
    fn = _SCHEMES.get(scheme, _scheme_scordilis)
    return fn(magnitude, t)


# Prioridad para elegir la magnitud "preferida" (mayor = mejor).
def method_priority(method: str) -> int:
    m = method or ""
    if m == "direct":
        return 100
    if "ms" in m:
        return 80
    if "mb" in m:
        return 70
    if "ml" in m or "mld" in m:
        return 40
    if "md" in m:
        return 30
    if m.startswith("assume"):
        return 10
    return 0


# Compatibilidad hacia atrás (algunos módulos lo importan).
MAG_METHOD_PRIORITY = {
    "direct": 100, "scordilis_ms": 80, "digiacomo_ms": 80, "peru_cahuari_ms": 80,
    "scordilis_mb": 70, "digiacomo_mb": 70, "peru_cahuari_mb": 70, "igp_mb_ms_mw": 75,
    "peru_cahuari_mld": 40, "assume_ml": 40, "assume_md": 30,
    "assume_unknown": 10, "none": 0,
}
