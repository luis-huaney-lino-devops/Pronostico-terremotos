"""Modelos de ML (XGBoost / LightGBM) para el pronóstico sísmico.

Usan SOLO features leakage-safe (todas ≤ t). El desbalance extremo se maneja
con ``scale_pos_weight``; eso desregula la escala de probabilidad, por lo que
la CALIBRACIÓN posterior (isotónica, por fold) es obligatoria.

NO se incluye ``cell_id`` como feature (evita memorizar celdas concretas); sí
``cen_lat``/``cen_lon`` como prior espacial físico (zonas sismogénicas).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "cnt_7d", "cnt_30d", "cnt_90d", "cnt_365d", "cnt_all",
    "maxmag_30d", "maxmag_90d", "maxmag_365d",
    "logE_30d", "logE_365d",
    "days_since_last", "days_since_M5",
    "nb_cnt_30d", "nb_cnt_365d", "nb_maxmag_365d",
    "reg_cnt_30d", "reg_maxmag_30d",
    "cen_lat", "cen_lon",
]
# Features ETAS opcionales (solo si el dataset se construyó con ETAS).
ETAS_COLS = ["etas_cell", "etas_nb"]


def get_features(df) -> list[str]:
    """Columnas de features presentes en el dataset (incluye ETAS si existen)."""
    return [c for c in FEATURE_COLS + ETAS_COLS if c in df.columns]


def _spw(y) -> float:
    y = np.asarray(y)
    pos = max(int(y.sum()), 1)
    return (len(y) - pos) / pos


def train_xgb(X: pd.DataFrame, y, seed: int = 0):
    import xgboost as xgb
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=_spw(y), eval_metric="logloss",
        tree_method="hist", n_jobs=-1, random_state=seed,
    )
    m.fit(X, y)
    return m


def train_lgbm(X: pd.DataFrame, y, seed: int = 0):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=31, max_depth=-1, learning_rate=0.05,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=50, scale_pos_weight=_spw(y),
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    m.fit(X, y)
    return m


TRAINERS = {"xgboost": train_xgb, "lightgbm": train_lgbm}


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[get_features(X)])[:, 1]
