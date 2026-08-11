# SEIS-PERU — Reporte final (Fases 1–7)

Sistema de pronóstico sísmico probabilístico del Perú, construido con rigor de
investigación: catálogo multi-fuente, features sin fuga temporal, baselines
físicos, ML, ETAS y evaluación walk-forward calibrada. **Las 5 reglas de oro se
respetaron en cada fase** — incluida la más importante: reportar los resultados
negativos.

## Veredictos de las hipótesis

| ID | Hipótesis | Veredicto | Evidencia |
|----|-----------|-----------|-----------|
| **H1** | La actividad reciente mejora vs. tasa histórica constante | **Parcial** | Señal ×40 en celdas activas (3–5 sismos/30d → P(M5+)≈8%), pero el skill global es bajo |
| **H2** | Los sismos fuertes modifican la tasa posterior | **Sí** | ETAS ajustado: n=0.84 (84% aftershocks), α=0.87 (subducción) |
| **H3** | Chile aporta información al pronóstico del Perú | **No** | La señal aparente (RR≈11.5) es fuga de aftershocks de campo cercano; con Chile remoto >600 km el efecto desaparece (RR≈1.1, p>0.1) → coincide con Parsons & Velasco (2011) |
| **H4** | Las features espaciales mejoran el pronóstico | **Sí (pero...)** | El prior espacial es la señal dominante (SHAP), pero el baseline ya lo captura |
| **H5** | El ML supera a los baselines sismológicos | **No** | XGBoost/LightGBM **no superan** al Poisson-GR en 3 configuraciones (0.1°/30d, 0.25°/30d, 0.25°/7d, con y sin ETAS). ΔROC-AUC ≤ 0 siempre |

**Conclusión científica honesta:** a resolución de celda/30 días, el pronóstico
sísmico de corto plazo tiene skill genuinamente bajo, y **un baseline físico
simple (Poisson-Gutenberg-Richter calibrado) es el mejor modelo**. El ML no
aporta valor porque aprende sobre todo *dónde* ocurren los sismos (prior
espacial), que la física ya encapsula. Esto coincide con el consenso: predecir
terremotos a corto plazo es muy difícil. **No fingimos que el ML funciona.**

## Números clave

- **Catálogo:** 6 fuentes (IGP, USGS, ISC, EMSC, GCMT, ISC-GEM) → 103,501 obs →
  69,661 eventos canónicos (26,459 en el Perú), 1960–2026. Valida Arequipa 2001
  (M8.4), Áncash 1970 (M7.9), Pisco 2007 (M8.0).
- **Sismología:** Mc moderna ≈4.5; **b real ≈0.9–1.0** (el 1.46 crudo era
  artefacto de magnitudes IGP sin tipo); gradiente b Norte>Centro>Sur.
- **ETAS:** μ=0.48/día, α=0.87, p=1.53, c=0.40, n=0.84.
- **Mejor forecaster (M5+/30d, 0.25°, test 2015–26):** Poisson-GR calibrado,
  ROC-AUC **0.639**, BSS **+0.0012**. XGBoost 0.618, LightGBM 0.590.
- **Forecast vigente:** máx P(M5+/30d) ≈ 0.74% (costa central).

## Cómo reproducir todo

```bash
python scripts/run_ingest.py --start 1960-01-01 --end 2026-08-10 --min-magnitude 4.0 --sources usgs,emsc,igp,gcmt
python scripts/run_eda.py                 # EDA + Gutenberg-Richter + Mc + b-value
python scripts/run_etas.py                # ajuste ETAS
python scripts/build_features.py --res 0.25 --etas --out data/features/features_peru_v2.parquet
python scripts/run_baselines.py           # baselines + backtesting (listón)
python scripts/run_ml.py --features data/features/features_peru_v2.parquet --target y_m5_30d
python scripts/experiment_chile_peru.py   # H3 con control de distancia
python scripts/build_forecast.py && python scripts/build_dashboard_data.py
uvicorn api.main:app --port 8000          # API REST
# Dashboard: dashboard/index.html (Artifact) — mapa + forecast + modelos + hipótesis
python -m pytest -q                       # 31 tests
```

## Reportes y figuras
`reports/eda_summary.md`, `baselines_summary.md`, `ml_summary_*.md`,
`etas_summary.md`, `chile_peru_summary.md` + 11 figuras en `reports/figures/`.

## Trabajo futuro
Features físicas (deformación GNSS/strain), ETAS espacio-temporal completo,
agregación regional, y el frontend Angular + MapLibre sobre la API.
