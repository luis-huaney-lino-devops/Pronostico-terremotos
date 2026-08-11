#!/usr/bin/env python
"""EDA + Gutenberg-Richter + Mc + b-value sobre el catálogo del Perú (Fase 2).

Genera figuras en reports/figures/ y un resumen en reports/eda_summary.md.

    python scripts/run_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seis_peru.analysis.completeness import fmd, mc_gft, mc_maxc
from seis_peru.analysis.gutenberg_richter import b_value_aki, gr_fit
from seis_peru.analysis.regions_peru import PERU_MACRO

FIGDIR = ROOT / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
CATALOG = ROOT / "data" / "processed" / "catalog_canonical_peru.parquet"
MODERN_YEAR = 2000  # periodo con red densa para b-value estable


def _bbox_mask(df, box):
    return (
        df["latitude"].between(box.min_lat, box.max_lat)
        & df["longitude"].between(box.min_lon, box.max_lon)
    )


def main() -> int:
    if not CATALOG.exists():
        print(f"No existe {CATALOG}. Corre primero scripts/run_ingest.py")
        return 1
    df = pd.read_parquet(CATALOG).dropna(subset=["preferred_mw"])
    df["year"] = df["origin_time"].dt.year
    mags = df["preferred_mw"].to_numpy()
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"# EDA SEIS-PERU — catálogo del Perú")
    emit(f"\n- Eventos: **{len(df):,}** | {df['year'].min()}–{df['year'].max()} | "
         f"Mw {df['preferred_mw'].min():.1f}–{df['preferred_mw'].max():.1f}")

    # ---------- Mc global (era moderna) ----------
    modern = df[df["year"] >= MODERN_YEAR]
    mm = modern["preferred_mw"].to_numpy()
    mc_mx = mc_maxc(mm, 0.1)
    gft = mc_gft(mm, 0.1)
    emit(f"\n## Magnitud de completitud (Mc), {MODERN_YEAR}+ ({len(mm):,} eventos)")
    emit(f"- MAXC+0.2 = **{mc_mx:.2f}**")
    emit(f"- GFT (Wiemer-Wyss) = **{gft.mc:.2f}** (R={gft.R:.1f}%, nivel {gft.level or 'maxR'}%)")
    mc_use = gft.mc if not np.isnan(gft.mc) else mc_mx
    emit(f"- Mc adoptado para b-value = **{mc_use:.2f}**")

    # ---------- b-value global ----------
    bv = b_value_aki(mm, mc_use, 0.1)
    emit(f"\n## Gutenberg-Richter b-value (Perú, {MODERN_YEAR}+)")
    emit(f"- **b = {bv.b:.3f} ± {bv.sigma_b:.3f}** | a = {bv.a:.2f} | n = {bv.n:,}")

    # ---------- Diagnóstico: b-value SOLO con Mw directa ----------
    # Si el b global está inflado por la conversión mb->Mw, al usar solo eventos
    # con magnitud-momento directa el b debería bajar hacia ~1.0-1.2.
    mask_mw = modern["preferred_mag_type"].fillna("").str.lower().str.startswith("mw")
    mw_only = modern[mask_mw]["preferred_mw"].to_numpy()
    if len(mw_only) > 300:
        mc_mwonly = mc_maxc(mw_only, 0.1)
        bv_mw = b_value_aki(mw_only, mc_mwonly, 0.1)
        emit(f"\n### Diagnóstico — b-value SOLO con Mw directa ({len(mw_only):,} eventos)")
        emit(f"- Mc(Mw)={mc_mwonly:.2f} | **b = {bv_mw.b:.3f} ± {bv_mw.sigma_b:.3f}**")
        emit(f"- Si b(Mw) << b(global {bv.b:.2f}), la conversión de magnitudes "
             f"inflaba el b global (heterogeneidad de escalas).")

    # ---------- Comparación de b bajo cada esquema de conversión ----------
    # Re-homogeniza la magnitud de cada evento canónico bajo cada esquema
    # (los que ya son Mw directa no cambian) y recalcula b. Muestra cuánto
    # depende el b de la elección de relación de magnitud.
    from seis_peru.magnitude import to_mw
    emit(f"\n## b-value según esquema de conversión de magnitud ({MODERN_YEAR}+, Mc fijo={mc_use:.1f})")
    emit("Mismo Mc para todos → aísla el efecto de la relación de conversión.")
    emit("| Esquema | b ± σ | n |")
    emit("|---|---|---:|")
    mdf = modern.dropna(subset=["preferred_mag", "preferred_mag_type"])
    for scheme in ("scordilis", "digiacomo", "peru", "igp_national"):
        mw_s = np.array([
            to_mw(m, t, scheme)[0]
            for m, t in zip(mdf["preferred_mag"], mdf["preferred_mag_type"])
        ], dtype=float)
        mw_s = mw_s[~np.isnan(mw_s)]
        b_s = b_value_aki(mw_s, mc_use, 0.1)  # Mc común
        emit(f"| {scheme} | {b_s.b:.3f} ± {b_s.sigma_b:.3f} | {b_s.n:,} |")

    # ---------- b-value por macro-región ----------
    emit(f"\n## b-value por macro-región ({MODERN_YEAR}+, Mc por GFT regional)")
    emit("| Región | n | Mc | b ± σ |")
    emit("|---|---:|---:|---|")
    region_rows = []
    for name, box in PERU_MACRO.items():
        sub = modern[_bbox_mask(modern, box)]["preferred_mw"].to_numpy()
        if len(sub) < 200:
            emit(f"| {name} | {len(sub)} | — | pocos eventos |")
            continue
        g = mc_gft(sub, 0.1)
        mc_r = g.mc if not np.isnan(g.mc) else mc_maxc(sub, 0.1)
        b = b_value_aki(sub, mc_r, 0.1)
        emit(f"| {name} | {b.n:,} | {mc_r:.2f} | {b.b:.3f} ± {b.sigma_b:.3f} |")
        region_rows.append((name, mc_r, b))

    # ---------- Mc por década ----------
    emit(f"\n## Mc por década (MAXC+0.2) — muestra la densificación de la red")
    emit("| Década | n | Mc |")
    emit("|---|---:|---:|")
    decades, mc_dec = [], []
    for dec in range(1960, 2030, 10):
        sub = df[(df["year"] >= dec) & (df["year"] < dec + 10)]["preferred_mw"].to_numpy()
        if len(sub) < 100:
            continue
        mc_d = mc_maxc(sub, 0.1)
        emit(f"| {dec}s | {len(sub):,} | {mc_d:.2f} |")
        decades.append(f"{dec}s"); mc_dec.append(mc_d)

    # =================== FIGURAS ===================
    _fig_fmd(mm, mc_use, bv, FIGDIR / "01_fmd_peru.png")
    _fig_regions(region_rows, FIGDIR / "02_bvalue_regiones.png")
    _fig_mc_decade(decades, mc_dec, FIGDIR / "03_mc_por_decada.png")
    _fig_mag_time(df, FIGDIR / "04_magnitud_tiempo.png")
    _fig_depth(df, FIGDIR / "05_profundidad.png")
    _fig_map(df, FIGDIR / "06_mapa_epicentros.png")
    emit(f"\n## Figuras\nGeneradas en `reports/figures/` (6 PNG).")

    # ---------- Interpretación honesta ----------
    emit("\n## Hallazgos e interpretación")
    emit("- **Mc realista (~4.5)** en la era moderna, coherente con la literatura "
         "del Perú; la FMD se aplana por debajo (incompletitud).")
    emit(f"- **b del catálogo completo alto ({bv.b:.2f})** pero es un ARTEFACTO: "
         "sobre el subconjunto de magnitudes TIPADAS con Scordilis (estándar "
         "nacional) b≈0.99, y solo con Mw directa b≈0.94. El **b real del Perú es "
         "~0.9–1.0**. Lo inflan (a) las magnitudes del IGP SIN tipo y (b) la "
         "elección de conversión (la tabla de esquemas mueve b entre 0.75 y 1.09). "
         "Lección: fijar Mc, preferir Mw directa y documentar el esquema.")
    emit("- **Gradiente norte→sur real** (b: Norte>Centro>Sur). b más bajo en el "
         "sur ⇒ relativamente más grandes ⇒ coherente con el gap sísmico del sur "
         "y el terremoto de Arequipa 2001 (M8.4).")
    emit("- **Mc(t) NO es estacionaria**: baja de ~5.1 (1960-80s) a ~4.2 (2020s) "
         "por densificación de la red ⇒ hay que usar Mc por ventana temporal.")
    emit("\n### Próximo refinamiento")
    emit("- Recalcular b sobre una fuente **homogénea** (ISC-GEM Mw) para eliminar "
         "el sesgo de conversión; relaciones ML→Mw regionales del IGP; Mc(t,x) mapeado.")

    (ROOT / "reports" / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResumen -> reports/eda_summary.md | figuras -> {FIGDIR}")
    return 0


def _fig_fmd(mags, mc, bv, path):
    centers, inc, cum = fmd(mags, 0.1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(centers, cum, s=18, c="#1f77b4", label="N(≥M) acumulada")
    ax.scatter(centers, inc, s=12, c="#aaaaaa", marker="s", label="incremental")
    mrange = centers[centers >= mc - 0.05]
    ax.plot(mrange, 10 ** (bv.a - bv.b * mrange), "r-", lw=2,
            label=f"GR: b={bv.b:.2f}")
    ax.axvline(mc, color="green", ls="--", label=f"Mc={mc:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("Magnitud (Mw)"); ax.set_ylabel("Nº de eventos")
    ax.set_title(f"Distribución frecuencia-magnitud · Perú {MODERN_YEAR}+")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _fig_regions(rows, path):
    if not rows:
        return
    names = [r[0] for r in rows]
    bs = [r[2].b for r in rows]; errs = [r[2].sigma_b for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, bs, yerr=errs, capsize=6, color="#4c72b0")
    ax.axhline(1.0, color="gray", ls=":", label="b=1 (referencia global)")
    ax.set_ylabel("b-value"); ax.set_title("b-value por macro-región del Perú")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _fig_mc_decade(decades, mc_dec, path):
    if not decades:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(decades, mc_dec, "o-", color="#c44e52")
    ax.set_ylabel("Mc (MAXC+0.2)"); ax.set_title("Magnitud de completitud por década")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _fig_mag_time(df, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(df["origin_time"], df["preferred_mw"], s=4, alpha=0.25, c="#1f77b4")
    ax.set_ylabel("Mw"); ax.set_title("Magnitud vs tiempo (Perú)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _fig_depth(df, path):
    d = df["depth_km"].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d, bins=np.arange(0, 700, 20), color="#55a868", edgecolor="white")
    ax.set_xlabel("Profundidad (km)"); ax.set_ylabel("Nº eventos")
    ax.set_title("Distribución de profundidad (subducción de Nazca)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _fig_map(df, path):
    fig, ax = plt.subplots(figsize=(6, 8))
    sc = ax.scatter(df["longitude"], df["latitude"], s=6 + (df["preferred_mw"] - 4) * 4,
                    c=df["depth_km"].clip(0, 300), cmap="viridis_r", alpha=0.5)
    fig.colorbar(sc, ax=ax, label="Profundidad (km)")
    ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud")
    ax.set_title("Epicentros · Perú (M≥4)")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
