"""Feature engineering para el pronóstico sísmico (Fase 3).

CONTRATO ANTI-LEAKAGE: para un instante de predicción t, TODAS las features se
calculan con eventos de origin_time <= t, y el target vive en (t, t+H]. Ver
`build.py` y el test `tests/test_features.py::test_no_leakage`.
"""
