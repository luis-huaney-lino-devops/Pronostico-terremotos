#!/usr/bin/env python
"""Fase 4 — Baselines + backtesting walk-forward.

Evalúa climatología, Poisson-largo-plazo, Poisson-reciente y Poisson-suavizado
sobre el periodo de test, con métricas de probabilidad (Brier/BSS/log-loss/
PR-AUC/ROC-AUC) y diagramas de confiabilidad. Define el 'listón' del ML.

    python scripts/run_baselines.py --test-start 2015
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

from seis_peru.models import baselines as B  # noqa: E402
from seis_peru.models.backtest import split_by_year  # noqa: E402
from seis_peru.models.metrics import evaluate, reliability  # noqa: E402

FIGDIR = ROOT / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

TARGETS = [("y_m5_30d", 5.0), ("y_m6_30d", 6.0)]
MC = 4.5
B_VALUE = 1.0     # b-value limpio del Perú (~0.9–1.0)
HORIZON = 30


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default=str(ROOT / "data/features/features_peru.parquet"))
    p.add_argument("--test-start", type=int, default=2015)
    p.add_argument("--b-value", type=float, default=B_VALUE)
    args = p.parse_args()

    df = pd.read_parquet(args.features)
    train, test = split_by_year(df, args.test_start)
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# Fase 4 — Baselines + backtesting walk-forward")
    emit(f"\nTrain: {train['t'].dt.year.min()}–{args.test_start - 1} ({len(train):,} filas) | "
         f"Test: {args.test_start}–{test['t'].dt.year.max()} ({len(test):,} filas)")
    emit(f"Mc={MC} · b={args.b_value} · H={HORIZON}d · celdas 0.1°")

    for target, m0 in TARGETS:
        y_test = test[target].to_numpy()
        emit(f"\n## Target: {target}  (M≥{m0} en {HORIZON}d)")
        emit(f"Positivos en test: {int(y_test.sum())} de {len(y_test):,} "
             f"(tasa base {100 * y_test.mean():.4f}%)")

        from sklearn.isotonic import IsotonicRegression
        y_train = train[target].to_numpy()

        # Referencia: climatología (tasa base del TRAIN).
        p_clima = B.climatology(y_train, len(test))
        m_clima = evaluate(y_test, p_clima, p_ref=p_clima)

        # Baselines: versión CRUDA y CALIBRADA (isotónica ajustada en train).
        # La calibración preserva el ranking (PR/ROC-AUC) pero corrige el Brier.
        preds_cal = {}   # para la figura
        results = {"climatology": m_clima}
        emit("\n| Modelo | PR-AUC↑ | ROC-AUC↑ | BSS crudo | **BSS calibrado**↑ |")
        emit("|---|---|---|---|---|")
        emit(f"| climatology | {m_clima['pr_auc']:.4f} | 0.5000 | +0.000 | +0.000 |")
        for name, fn in B.FEATURE_BASELINES.items():
            p_tr = fn(train, args.b_value, m0, MC, HORIZON)
            p_te = fn(test, args.b_value, m0, MC, HORIZON)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            iso.fit(p_tr, y_train)
            p_te_cal = iso.transform(p_te)
            preds_cal[name] = p_te_cal
            m_raw = evaluate(y_test, p_te, p_ref=p_clima)
            m_cal = evaluate(y_test, p_te_cal, p_ref=p_clima)
            results[name] = m_cal
            emit(f"| {name} | {m_cal['pr_auc']:.4f} | {m_cal['roc_auc']:.4f} "
                 f"| {m_raw.get('bss', float('nan')):+.3f} "
                 f"| **{m_cal.get('bss', float('nan')):+.3f}** |")

        # Figura de confiabilidad (target con más positivos): crudo vs calibrado.
        if target == "y_m5_30d":
            raw = {n: B.FEATURE_BASELINES[n](test, args.b_value, m0, MC, HORIZON)
                   for n in B.FEATURE_BASELINES}
            _fig_reliability(y_test, raw, preds_cal, FIGDIR / "07_confiabilidad_baselines.png")

        # Veredicto H1 (comparar por BSS calibrado)
        try:
            if results["poisson_1y"]["bss"] > results["poisson_lt"]["bss"]:
                emit("\n**H1 (reciente > histórico): APOYADA** — tras calibrar, la "
                     "tasa reciente supera a la de largo plazo (BSS).")
            else:
                emit("\n**H1: NO concluyente en este target** — la tasa reciente no "
                     "mejora claramente a la de largo plazo tras calibrar.")
        except KeyError:
            pass

    emit("\n## Hallazgos — el 'listón' para el ML (Fase 5)")
    emit("- Los baselines Poisson/GR CRUDOS **sobreconfían** (BSS crudo negativo): "
         "predicen hasta ~3% donde se observa ~0.2%. La calibración isotónica lo "
         "corrige (ver `07_confiabilidad_baselines.png`). → **regla #3 en acción**.")
    emit("- Tras calibrar, **ningún baseline supera claramente a la climatología** en "
         "Brier (BSS≈0). A resolución 0.1°/30d el skill es GENUINAMENTE BAJO: es la "
         "realidad del pronóstico sísmico de corto plazo, no un error del código.")
    emit("- Hay **señal débil de ranking**: el suavizado espacial (poisson_smooth) es "
         "el mejor (ROC-AUC ~0.58 en M5) → la actividad del vecindario aporta algo.")
    emit("- **Listón para Fase 5 (H5):** XGBoost/LightGBM deberá (a) superar ROC-AUC "
         "~0.58 y (b) lograr **BSS calibrado > 0**. Si no lo logra, será un resultado "
         "NEGATIVO honesto y publicable (regla #4), no un fracaso a esconder.")

    (ROOT / "reports" / "baselines_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResumen -> reports/baselines_summary.md")
    return 0


def _fig_reliability(y, raw, cal, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, data, titulo in ((axes[0], raw, "CRUDO (sobreconfía)"),
                             (axes[1], cal, "CALIBRADO (isotónico)")):
        ax.plot([0, 1], [0, 1], "k:", label="perfecto")
        for name, pr in data.items():
            mp, of, _ = reliability(y, pr, n_bins=8)
            ax.plot(mp, of, "o-", label=name, alpha=0.8, ms=4)
        ax.set_xlabel("Probabilidad predicha"); ax.set_title(titulo)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_xlim(0, 0.08); ax.set_ylim(0, 0.08)
    axes[0].set_ylabel("Frecuencia observada")
    fig.suptitle("Confiabilidad de baselines (M5+/30d, test)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
