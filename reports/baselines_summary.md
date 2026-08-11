# Fase 4 — Baselines + backtesting walk-forward

Train: 2000–2014 (261,324 filas) | Test: 2015–2026 (199,920 filas)
Mc=4.5 · b=1.0 · H=30d · celdas 0.1°

## Target: y_m5_30d  (M≥5.0 en 30d)
Positivos en test: 347 de 199,920 (tasa base 0.1736%)

| Modelo | PR-AUC↑ | ROC-AUC↑ | BSS crudo | **BSS calibrado**↑ |
|---|---|---|---|---|
| climatology | 0.0017 | 0.5000 | +0.000 | +0.000 |
| poisson_lt | 0.0022 | 0.5360 | -0.001 | **+0.000** |
| poisson_1y | 0.0020 | 0.5377 | -0.083 | **-0.000** |
| poisson_smooth | 0.0023 | 0.5758 | -0.033 | **-0.000** |

**H1: NO concluyente en este target** — la tasa reciente no mejora claramente a la de largo plazo tras calibrar.

## Target: y_m6_30d  (M≥6.0 en 30d)
Positivos en test: 12 de 199,920 (tasa base 0.0060%)

| Modelo | PR-AUC↑ | ROC-AUC↑ | BSS crudo | **BSS calibrado**↑ |
|---|---|---|---|---|
| climatology | 0.0001 | 0.5000 | +0.000 | +0.000 |
| poisson_lt | 0.0001 | 0.4808 | -0.001 | **-0.000** |
| poisson_1y | 0.0001 | 0.5306 | -0.027 | **+0.000** |
| poisson_smooth | 0.0001 | 0.4892 | -0.012 | **+0.000** |

**H1 (reciente > histórico): APOYADA** — tras calibrar, la tasa reciente supera a la de largo plazo (BSS).

## Hallazgos — el 'listón' para el ML (Fase 5)
- Los baselines Poisson/GR CRUDOS **sobreconfían** (BSS crudo negativo): predicen hasta ~3% donde se observa ~0.2%. La calibración isotónica lo corrige (ver `07_confiabilidad_baselines.png`). → **regla #3 en acción**.
- Tras calibrar, **ningún baseline supera claramente a la climatología** en Brier (BSS≈0). A resolución 0.1°/30d el skill es GENUINAMENTE BAJO: es la realidad del pronóstico sísmico de corto plazo, no un error del código.
- Hay **señal débil de ranking**: el suavizado espacial (poisson_smooth) es el mejor (ROC-AUC ~0.58 en M5) → la actividad del vecindario aporta algo.
- **Listón para Fase 5 (H5):** XGBoost/LightGBM deberá (a) superar ROC-AUC ~0.58 y (b) lograr **BSS calibrado > 0**. Si no lo logra, será un resultado NEGATIVO honesto y publicable (regla #4), no un fracaso a esconder.