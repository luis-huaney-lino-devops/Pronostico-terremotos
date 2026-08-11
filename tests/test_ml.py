"""Test de sanidad de los modelos ML (aprenden señal, predicen probabilidades)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from seis_peru.models import ml


def _synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.normal(size=n) for c in ml.FEATURE_COLS})
    # target con señal real en cnt_30d + nb_cnt_30d
    logit = 1.2 * X["cnt_30d"] + 0.8 * X["nb_cnt_30d"] - 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


@pytest.mark.parametrize("trainer", ["xgboost", "lightgbm"])
def test_model_learns_signal(trainer):
    X, y = _synthetic()
    model = ml.TRAINERS[trainer](X, y)
    p = ml.predict_proba(model, X)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert roc_auc_score(y, p) > 0.70  # recupera la señal sintética
