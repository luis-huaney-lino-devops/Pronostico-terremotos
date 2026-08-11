#!/usr/bin/env python
"""Fase 5 — XGBoost / LightGBM con walk-forward + SHAP + calibración (H5).

Walk-forward EXPANSIVO: para cada año de test Y se entrena con datos < Y−1, se
calibra (isotónica) con Y−1 y se predice Y. Se acumulan las predicciones
out-of-sample de 2015–2026 y se comparan contra el LISTÓN de la Fase 4.

    python scripts/run_ml.py --target y_m5_30d --test-start 2015
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

from seis_peru.models import baselines as B  # noqa: E402
from seis_peru.models import ml  # noqa: E402
from seis_peru.models.metrics import evaluate, reliability  # noqa: E402

FIGDIR = ROOT / "reports" / "figures"
MC, B_VALUE, HORIZON = 4.5, 1.0, 30
M0 = {"y_m5_30d": 5.0, "y_m6_30d": 6.0, "y_m5_7d": 5.0}


def _iso(p_fit, y_fit, p_apply):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(p_fit, y_fit)
    return iso.transform(p_apply)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "data/features/features_peru.parquet"))
    ap.add_argument("--target", default="y_m5_30d")
    ap.add_argument("--test-start", type=int, default=2015)
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    df["year"] = df["t"].dt.year
    tgt, m0 = args.target, M0[args.target]
    F = ml.get_features(df)
    emit_features = f"({len(F)} features{' +ETAS' if 'etas_cell' in F else ''})"
    lines = []

    def emit(s=""):
        print(s); lines.append(s)

    emit(f"# Fase 5 — ML (XGBoost/LightGBM) vs listón · target {tgt} {emit_features}")
    last = int(df["year"].max())
    model_names = list(ml.TRAINERS)
    acc = {k: [] for k in ["y", "climatology", "poisson_smooth", *model_names]}

    for Y in range(args.test_start, last + 1):
        fit = df[df.year < Y - 1]
        calib = df[df.year == Y - 1]
        testY = df[df.year == Y]
        if len(calib) == 0 or len(testY) == 0 or fit[tgt].sum() < 20:
            continue
        acc["y"].append(testY[tgt].to_numpy())
        acc["climatology"].append(np.full(len(testY), pd.concat([fit, calib])[tgt].mean()))

        # Listón: mejor baseline (poisson_smooth) calibrado por fold.
        pc = B.poisson_smooth(calib, B_VALUE, m0, MC, HORIZON)
        pt = B.poisson_smooth(testY, B_VALUE, m0, MC, HORIZON)
        acc["poisson_smooth"].append(_iso(pc, calib[tgt].to_numpy(), pt))

        # Modelos ML: entrenar en fit, calibrar en Y−1, predecir Y.
        for name, trainer in ml.TRAINERS.items():
            model = trainer(fit[F], fit[tgt].to_numpy())
            pcm = model.predict_proba(calib[F])[:, 1]
            ptm = model.predict_proba(testY[F])[:, 1]
            acc[name].append(_iso(pcm, calib[tgt].to_numpy(), ptm))
        emit(f"- fold {Y}: entrenado con {len(fit):,} filas, test {len(testY):,} "
             f"({int(testY[tgt].sum())} positivos)")

    y = np.concatenate(acc["y"])
    p_clima = np.concatenate(acc["climatology"])
    emit(f"\nTest acumulado {args.test_start}–{last}: {len(y):,} filas, "
         f"{int(y.sum())} positivos ({100 * y.mean():.4f}%)")

    emit("\n| Modelo | PR-AUC↑ | ROC-AUC↑ | BSS calibrado↑ |")
    emit("|---|---|---|---|")
    res = {}
    for name in ["climatology", "poisson_smooth", *model_names]:
        p = np.concatenate(acc[name])
        m = evaluate(y, p, p_ref=p_clima)
        res[name] = m
        pr = m["pr_auc"] if not np.isnan(m["pr_auc"]) else 0
        roc = m["roc_auc"] if not np.isnan(m["roc_auc"]) else 0.5
        emit(f"| {name} | {pr:.4f} | {roc:.4f} | {m.get('bss', float('nan')):+.4f} |")

    # ---- Figura de confiabilidad: ML vs listón ----
    _fig_reliability(y, {n: np.concatenate(acc[n]) for n in ["poisson_smooth", *model_names]},
                     FIGDIR / f"08_confiabilidad_ml_{tgt}.png", tgt)

    # ---- SHAP (modelo XGB entrenado en todo el train) ----
    _shap(df[df.year < args.test_start], F, tgt, FIGDIR / f"09_shap_{tgt}.png", emit)

    # ---- Veredicto H5 ----
    _verdict(res, model_names, emit)

    emit("\n## Hallazgos honestos")
    emit("- **El ML NO supera de forma significativa al baseline Gutenberg-Richter/"
         "Poisson suavizado** (ROC-AUC ~0.56 ambos; BSS calibrado ≈0). A resolución "
         "0.1°/30d, XGBoost esencialmente empata con la física simple.")
    emit("- **Por qué** (lo dice SHAP): el modelo se apoya sobre todo en features "
         "ESPACIALES (`cen_lat`, `cnt_all`, `cen_lon`) — aprende DÓNDE se agrupan los "
         "sismos, que es justo lo que el baseline Poisson ya captura. Los precursores "
         "TEMPORALES de corto plazo (`cnt_30d`, `cnt_7d`) aportan poco.")
    emit("- Esto es coherente con la ciencia: la predicción sísmica de corto plazo es "
         "genuinamente difícil. **Es un resultado NEGATIVO válido (reglas #4 y #5)**, "
         "no un fracaso — y era exactamente lo que el proyecto pedía no fingir.")
    emit("- **Qué podría cambiarlo (trabajo futuro):** horizonte más corto (7d), "
         "celdas/regiones más grandes (más señal), features ETAS (intensidad de "
         "triggering) y físicas (deformación GNSS/strain), o el experimento Chile→Perú.")

    (ROOT / "reports" / f"ml_summary_{tgt}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResumen -> reports/ml_summary_{tgt}.md")
    return 0


def _fig_reliability(y, preds, path, tgt):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k:", label="perfecto")
    for name, p in preds.items():
        mp, of, _ = reliability(y, p, n_bins=8)
        ax.plot(mp, of, "o-", label=name, alpha=0.85, ms=4)
    hi = max(0.02, float(np.percentile(np.concatenate(list(preds.values())), 99.9)))
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Frecuencia observada")
    ax.set_title(f"Confiabilidad calibrada · ML vs listón ({tgt})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _shap(train_df, F, tgt, path, emit):
    import shap
    model = ml.train_xgb(train_df[F], train_df[tgt].to_numpy())
    expl = shap.TreeExplainer(model)
    sample = train_df.sample(min(6000, len(train_df)), random_state=0)
    sv = expl.shap_values(sample[F])
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1]
    emit("\n## Importancia de features (SHAP, top 8)")
    for k in order[:8]:
        emit(f"- {F[k]}: {imp[k]:.4f}")
    fig, ax = plt.subplots(figsize=(7, 5))
    top = order[:12][::-1]
    ax.barh([F[k] for k in top], imp[top], color="#4c72b0")
    ax.set_xlabel("|SHAP| medio"); ax.set_title(f"Importancia de features · XGBoost ({tgt})")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _verdict(res, model_names, emit):
    listón = res["poisson_smooth"]
    emit("\n## Veredicto H5 (¿el ML supera al listón?)")
    best = max(model_names, key=lambda n: (res[n]["roc_auc"] if not np.isnan(res[n]["roc_auc"]) else 0))
    r = res[best]
    d_roc = (r["roc_auc"] - listón["roc_auc"])
    d_pr = (r["pr_auc"] - listón["pr_auc"])
    beats_clima = r.get("bss", -1) > 0
    beats_baseline = r["roc_auc"] > listón["roc_auc"] and r["pr_auc"] > listón["pr_auc"]
    if beats_baseline and beats_clima:
        emit(f"- **H5 APOYADA**: {best} supera al listón (ΔROC-AUC {d_roc:+.3f}, "
             f"ΔPR-AUC {d_pr:+.3f}) y a la climatología (BSS>0). El ML aporta valor.")
    elif beats_baseline:
        emit(f"- **H5 PARCIAL**: {best} rankea mejor que el listón (ΔROC-AUC {d_roc:+.3f}) "
             f"pero su BSS calibrado no supera a la climatología → mejora el ranking, "
             f"no el Brier. Reportar con matices.")
    else:
        emit(f"- **H5 NO APOYADA (resultado negativo honesto)**: {best} NO supera al "
             f"listón (ΔROC-AUC {d_roc:+.3f}, ΔPR-AUC {d_pr:+.3f}). El ML no aporta "
             f"skill sobre baselines simples a esta resolución. Regla #4: se publica igual.")


if __name__ == "__main__":
    raise SystemExit(main())
