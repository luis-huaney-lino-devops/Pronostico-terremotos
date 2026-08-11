#!/usr/bin/env python
"""Ajusta el ETAS temporal al catálogo del Perú y reporta parámetros.

    python scripts/run_etas.py --mc 4.5 --start-year 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from seis_peru.models.etas import fit_etas, intensity_at  # noqa: E402

FIGDIR = ROOT / "reports" / "figures"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(ROOT / "data/processed/catalog_canonical_peru.parquet"))
    ap.add_argument("--mc", type=float, default=4.5)
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--window-days", type=float, default=1200.0)
    args = ap.parse_args()

    df = pd.read_parquet(args.catalog).dropna(subset=["preferred_mw"])
    df = df[(df["preferred_mw"] >= args.mc) &
            (df["origin_time"].dt.year >= args.start_year)].sort_values("origin_time")
    t0 = df["origin_time"].min()
    times = (df["origin_time"] - t0).dt.total_seconds().to_numpy() / 86400.0
    mags = df["preferred_mw"].to_numpy()
    T = float(times.max())
    lines = []

    def emit(s=""):
        print(s); lines.append(s)

    emit(f"# ETAS temporal — Perú (M≥{args.mc}, {args.start_year}+)")
    emit(f"\nEventos: {len(df):,} | periodo: {T/365.25:.1f} años | b={args.b}")

    emit("\nAjustando MLE (puede tardar ~1–3 min)...")
    fit = fit_etas(times, mags, 0.0, T, M0=args.mc, window_days=args.window_days)
    n = fit.branching_ratio(args.b)

    emit("\n## Parámetros ETAS ajustados")
    emit(f"- μ (fondo)      = {fit.mu:.4f} eventos/día  ({fit.mu*365.25:.1f}/año)")
    emit(f"- K (productiv.) = {fit.K:.4f}")
    emit(f"- α              = {fit.alpha:.3f}  (α/ln10 = {fit.alpha/np.log(10):.2f})")
    emit(f"- c (Omori)      = {fit.c:.4f} días")
    emit(f"- p (Omori)      = {fit.p:.3f}")
    emit(f"- **razón de ramificación n = {n:.3f}** "
         f"(fracción de aftershocks ≈ {n:.0%}; fondo ≈ {1-n:.0%})")
    emit(f"- fondo directo: μ·T = {fit.mu*T:.0f} de {len(df)} eventos "
         f"({100*fit.mu*T/len(df):.0f}% ~ inmigrantes)")

    # Validación de rangos típicos (subducción/global).
    emit("\n## Chequeo de plausibilidad (rangos típicos de la literatura)")
    checks = [
        ("p", fit.p, 1.0, 1.3), ("c (días)", fit.c, 0.003, 0.5),
        ("α/ln10", fit.alpha/np.log(10), 0.6, 1.1), ("n", n, 0.2, 0.95),
    ]
    for name, val, lo, hi in checks:
        ok = "✓" if lo <= val <= hi else "⚠"
        emit(f"- {name} = {val:.3f}  [{lo}–{hi}]  {ok}")

    # Figura: intensidad ETAS vs eventos (últimos ~6 años, muestreada).
    _fig_intensity(times, mags, fit, T, FIGDIR / "10_etas_intensidad.png")
    emit("\nFigura -> reports/figures/10_etas_intensidad.png")

    (ROOT / "reports" / "etas_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nResumen -> reports/etas_summary.md")
    return 0


def _fig_intensity(times, mags, fit, T, path):
    grid = np.linspace(max(0, T - 6 * 365.25), T, 1500)
    lam = np.array([intensity_at(t, times, mags, fit) for t in grid])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grid / 365.25, lam, color="#c44e52", lw=0.9, label="λ(t) ETAS")
    ax.axhline(fit.mu, color="gray", ls="--", lw=1, label=f"μ fondo={fit.mu:.3f}")
    m = times >= grid[0]
    ax.scatter(times[m] / 365.25, np.full(m.sum(), fit.mu / 2),
               s=6 + (mags[m] - fit.M0) * 8, c="#333", alpha=0.4, label="sismos")
    ax.set_yscale("log")
    ax.set_xlabel("Años desde el inicio"); ax.set_ylabel("λ (eventos/día)")
    ax.set_title("Intensidad condicional ETAS del Perú (últimos ~6 años)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
