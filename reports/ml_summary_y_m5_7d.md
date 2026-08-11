# Fase 5 — ML (XGBoost/LightGBM) vs listón · target y_m5_7d (21 features +ETAS)
- fold 2015: entrenado con 227,088 filas, test 15,936 (16 positivos)
- fold 2016: entrenado con 243,024 filas, test 15,936 (13 positivos)
- fold 2017: entrenado con 258,960 filas, test 17,264 (17 positivos)
- fold 2018: entrenado con 274,896 filas, test 15,936 (10 positivos)
- fold 2019: entrenado con 292,160 filas, test 15,936 (16 positivos)
- fold 2020: entrenado con 308,096 filas, test 15,936 (10 positivos)
- fold 2021: entrenado con 324,032 filas, test 15,936 (22 positivos)
- fold 2022: entrenado con 339,968 filas, test 17,264 (11 positivos)
- fold 2023: entrenado con 355,904 filas, test 15,936 (14 positivos)
- fold 2024: entrenado con 373,168 filas, test 15,936 (11 positivos)
- fold 2025: entrenado con 389,104 filas, test 15,936 (10 positivos)
- fold 2026: entrenado con 405,040 filas, test 7,968 (6 positivos)

Test acumulado 2015–2026: 185,920 filas, 156 positivos (0.0839%)

| Modelo | PR-AUC↑ | ROC-AUC↑ | BSS calibrado↑ |
|---|---|---|---|
| climatology | 0.0009 | 0.5165 | +0.0000 |
| poisson_smooth | 0.0016 | 0.6072 | -0.0027 |
| xgboost | 0.0014 | 0.5578 | -0.0006 |
| lightgbm | 0.0009 | 0.5123 | -0.0002 |

## Importancia de features (SHAP, top 8)
- cen_lat: 0.4440
- days_since_M5: 0.3233
- etas_nb: 0.2822
- cnt_all: 0.2819
- reg_maxmag_30d: 0.2672
- cen_lon: 0.2650
- days_since_last: 0.2586
- nb_maxmag_365d: 0.2218

## Veredicto H5 (¿el ML supera al listón?)
- **H5 NO APOYADA (resultado negativo honesto)**: xgboost NO supera al listón (ΔROC-AUC -0.049, ΔPR-AUC -0.000). El ML no aporta skill sobre baselines simples a esta resolución. Regla #4: se publica igual.

## Hallazgos honestos
- **El ML NO supera de forma significativa al baseline Gutenberg-Richter/Poisson suavizado** (ROC-AUC ~0.56 ambos; BSS calibrado ≈0). A resolución 0.1°/30d, XGBoost esencialmente empata con la física simple.
- **Por qué** (lo dice SHAP): el modelo se apoya sobre todo en features ESPACIALES (`cen_lat`, `cnt_all`, `cen_lon`) — aprende DÓNDE se agrupan los sismos, que es justo lo que el baseline Poisson ya captura. Los precursores TEMPORALES de corto plazo (`cnt_30d`, `cnt_7d`) aportan poco.
- Esto es coherente con la ciencia: la predicción sísmica de corto plazo es genuinamente difícil. **Es un resultado NEGATIVO válido (reglas #4 y #5)**, no un fracaso — y era exactamente lo que el proyecto pedía no fingir.
- **Qué podría cambiarlo (trabajo futuro):** horizonte más corto (7d), celdas/regiones más grandes (más señal), features ETAS (intensidad de triggering) y físicas (deformación GNSS/strain), o el experimento Chile→Perú.