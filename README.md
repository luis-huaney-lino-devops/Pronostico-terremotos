<div align="center">

# 🌎 SEIS-PERU

### Sistema de pronóstico sísmico probabilístico para el Perú

*Catálogo multi-fuente · Gutenberg-Richter · ETAS · Machine Learning · backtesting walk-forward*

**Python 3.11+** · **FastAPI** · **PostGIS** · **XGBoost / LightGBM** · **25 dep · 197 prov · 1 826 distritos**

</div>

---

> **Estima** `P(M ≥ M₀ | información hasta hoy, horizonte H)` por unidad administrativa —
> p. ej. *"probabilidad de un M≥5 en Arequipa en los próximos 30 días"*.
>
> ⚠️ **Es un forecast probabilístico de investigación. NO es una alerta oficial ni una
> predicción determinista.** Nadie puede predecir la fecha y el lugar exactos de un
> terremoto; este sistema da probabilidades por región y horizonte, con rigor científico.

**Documentación:** [`METODOLOGIA.md`](METODOLOGIA.md) · [`PLAN_FASE1.md`](PLAN_FASE1.md) · [`REPORTE_FINAL.md`](REPORTE_FINAL.md)
**Dashboard interactivo:** consola con mapa real del Perú, forecast por horizonte, estacionalidad y Cinturón de Fuego.

---

## 📊 Resultados clave (honestos)

| Hipótesis | Veredicto |
|-----------|-----------|
| **H1** · La actividad reciente mejora el pronóstico | Parcial (señal ×40 local, skill global bajo) |
| **H2** · Los sismos fuertes modifican la tasa (ETAS) | ✅ Sí (ramificación n = 0.84) |
| **H3** · Chile dispara sismos en el Perú | ❌ **No** (la señal aparente es campo cercano; remoto = nulo) |
| **H5** · El ML supera a los baselines físicos | ❌ **No** (XGBoost ≈ Poisson-GR; resultado negativo honesto) |
| *Extra* · ¿Hay un mes más probable? | ❌ **No** (el pico jun/ago es artefacto de réplicas) |

> El mejor forecaster es un **baseline físico simple** (Poisson-Gutenberg-Richter calibrado).
> Reportamos los resultados negativos: esto es investigación, no adivinación.

---

## ⚙️ Requisitos

| | Mínimo | Recomendado |
|---|---|---|
| **Python** | 3.11 | 3.12+ |
| **RAM** | 4 GB | 8 GB |
| **Disco** | ~2 GB | — |
| **Docker** | opcional (PostGIS) | Docker Desktop |
| **SO** | Windows / Linux / macOS | — |
| **Node.js** | opcional (validar dashboard) | — |

Sin GPU. Conexión a internet para descargar los catálogos sísmicos.

---

## 🚀 Instalación

```bash
git clone <URL-del-repo> seis-peru
cd seis-peru

python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/mac:  source .venv/bin/activate

python -m pip install -r requirements.txt              # núcleo (ingesta)
python -m pip install -r requirements-analysis.txt     # EDA / modelado
python -m pip install xgboost lightgbm shap fastapi "uvicorn[standard]"  # ML + API

cp .env.example .env    # ajustar si hace falta (opcional)
```

---

## ▶️ Cómo levantarlo (pipeline completo)

Ejecuta en orden. Cada paso genera artefactos en `data/` y `reports/`.

```bash
# 1) INGESTA — catálogo multi-fuente 1960–hoy (~5 min, descarga de IGP/USGS/ISC/EMSC/GCMT)
python scripts/run_ingest.py --start 1960-01-01 --end 2026-08-10 \
       --min-magnitude 4.0 --sources usgs,emsc,igp,gcmt

# 2) EDA — Gutenberg-Richter, Mc, b-value (+ 6 figuras)
python scripts/run_eda.py

# 3) ETAS — ajuste del modelo epidémico de réplicas
python scripts/run_etas.py

# 4) FEATURES — dataset ML anti-leakage (celda 0.25° + feature ETAS)
python scripts/build_features.py --res 0.25 --etas \
       --out data/features/features_peru_v2.parquet

# 5) BASELINES — backtesting walk-forward (el "listón")
python scripts/run_baselines.py

# 6) ML — XGBoost/LightGBM vs. listón (+ SHAP)
python scripts/run_ml.py --features data/features/features_peru_v2.parquet --target y_m5_30d

# 7) EXPERIMENTOS — Chile→Perú (H3) y estacionalidad
python scripts/experiment_chile_peru.py
python scripts/context_analysis.py

# 8) FORECAST + DASHBOARD — snapshot, agregación por región y datos del tablero
python scripts/build_forecast.py
python scripts/build_regions.py      # forecast por departamento (rápido)
python scripts/build_admin.py        # forecast por dep/prov/distrito (mapa detallado)
python scripts/build_dashboard_data.py
python scripts/_inject_dashboard.py  # embebe los datos en dashboard/index.html
```

### API REST + Dashboard

```bash
# API (docs interactivas en http://localhost:8000/docs)
uvicorn api.main:app --reload --port 8000

# Dashboard: abre dashboard/index.html en el navegador (self-contained)
```

**Endpoints:** `/catalog/summary` · `/catalog/events` · `/forecast/grid` · `/models/comparison` · `/experiments/chile-peru`

### PostgreSQL + PostGIS (opcional)

```bash
docker compose -f docker/docker-compose.yml up -d
python scripts/init_db.py --check
python scripts/run_ingest.py --load-db
```

### Tests

```bash
python -m pytest -q        # 31 tests: magnitud, dedup, Mc/b, features, ML, ETAS, baselines
```

---

## 🗂️ Estructura del proyecto

```
seis-peru/
├── src/seis_peru/
│   ├── config.py · regions.py · models.py · magnitude.py
│   ├── ingestion/     fdsn.py · igp.py · gcmt.py · iscgem.py
│   ├── dedup.py                       # asociación de eventos → canonical_id
│   ├── analysis/      completeness.py · gutenberg_richter.py · regions_peru.py
│   ├── features/      grid.py · build.py     # feature engineering anti-leakage
│   ├── models/        baselines.py · metrics.py · backtest.py · etas.py · ml.py
│   └── storage/       raw_store.py · db.py
├── scripts/           run_ingest · run_eda · run_etas · build_features ·
│                      run_baselines · run_ml · experiment_chile_peru ·
│                      context_analysis · build_forecast · build_admin · …
├── api/               main.py            # FastAPI
├── dashboard/         index.html         # consola self-contained (mapa + forecast)
├── sql/               001..005           # PostGIS: raw / core / spatial
├── docker/            docker-compose.yml (postgis/postgis:16-3.4)
├── tests/             8 archivos, 31 tests
├── reports/           *.md + figures/*.png + *_forecast.json
├── data/              raw · interim · processed · features  (git-ignored)
├── notebooks/         01_eda_gutenberg_richter.py  (celdas # %%)
├── README.md · METODOLOGIA.md · PLAN_FASE1.md · REPORTE_FINAL.md
└── requirements*.txt · pyproject.toml
```

---

## 🌐 Fuentes de datos (verificadas)

| Fuente | Acceso | Rol |
|--------|--------|-----|
| **IGP** (Inst. Geofísico del Perú) | CSV `datosabiertos.gob.pe` | Verdad de campo nacional (1960–) |
| **USGS / ComCat** | FDSN `earthquake.usgs.gov` | Backbone global |
| **ISC** | FDSN `isc.ac.uk` | Backbone revisado |
| **EMSC** | FDSN `seismicportal.eu` | Complemento tiempo-real |
| **GCMT** | NDK `globalcmt.org` | Mw directa (M≳5, 1976–) |
| **ISC-GEM** | CSV manual (CAPTCHA) | Mw homogéneo 1904–2021 |
| **Límites INEI** | GeoJSON `juaneladio/peru-geojson` | Mapa dep/prov/distrito |

*(IRIS/EarthScope FDSN fue retirado — HTTP 410 desde 2026-06-01.)*
Detalles, endpoints exactos y coeficientes en [`METODOLOGIA.md`](METODOLOGIA.md).

---

## 🧭 Las 5 reglas de oro

1. **Cero leakage temporal** — las features usan solo datos ≤ t; el target vive en (t, t+H].
2. **Siempre comparar contra baselines** (histórico, Poisson, GR, ETAS) antes de creer en el ML.
3. **Evaluar probabilidades** (Brier, log-loss, PR-AUC, calibración), no accuracy.
4. **Publicar también los resultados negativos.**
5. Si un vecino o el ML **no** mejora el pronóstico, ese resultado negativo vale más que una relación inventada.

---

## ⚖️ Aviso legal

Este software es para **investigación y educación**. Los pronósticos son **probabilísticos**
y **no constituyen una alerta oficial** ni una predicción determinista de terremotos. Para
información oficial del Perú, consulta al **IGP** (igp.gob.pe) e **INDECI**.

## 📄 Licencia

MIT (código). Los catálogos sísmicos y límites administrativos conservan las licencias de
sus fuentes (IGP CC-BY, ISC-GEM CC-BY-SA 3.0, GeoJSON INEI, etc.).
