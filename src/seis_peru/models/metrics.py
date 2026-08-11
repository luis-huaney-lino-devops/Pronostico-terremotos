"""Métricas para forecasts PROBABILÍSTICOS de eventos raros.

No usamos accuracy (inútil con 0.01% de positivos). Usamos:
  - **Brier score**: error cuadrático medio de la probabilidad (menor = mejor).
  - **Brier Skill Score (BSS)**: 1 − BS/BS_ref (vs climatología). >0 = mejor que
    la referencia; 0 = igual; <0 = peor.
  - **Log-loss**: penaliza fuerte las probabilidades muy equivocadas.
  - **PR-AUC** (average precision): clave con desbalance extremo (mejor que ROC-AUC).
  - **ROC-AUC**: capacidad de ranking (útil pero insuficiente).
  - **Confiabilidad/calibración**: ¿cuando digo 10% ocurre ~10%?
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

EPS = 1e-7


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def evaluate(y, p, p_ref=None) -> dict:
    y = np.asarray(y, dtype=int)
    p = _clip(p)
    n_pos = int(y.sum())
    out = {
        "n": int(len(y)),
        "n_pos": n_pos,
        "base_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
    }
    # ranking metrics necesitan ambas clases
    if 0 < n_pos < len(y):
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    if p_ref is not None:
        bs_ref = brier_score_loss(y, _clip(p_ref))
        out["bss"] = float(1 - out["brier"] / bs_ref) if bs_ref > 0 else float("nan")
    return out


def reliability(y, p, n_bins: int = 10, strategy: str = "quantile"):
    """Curva de confiabilidad: (prob_media_predicha, frecuencia_observada, n)."""
    y = np.asarray(y, dtype=int)
    p = _clip(p)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(p.min(), p.max(), n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    mean_pred, obs_freq, counts = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        mean_pred.append(p[m].mean())
        obs_freq.append(y[m].mean())
        counts.append(int(m.sum()))
    return np.array(mean_pred), np.array(obs_freq), np.array(counts)
