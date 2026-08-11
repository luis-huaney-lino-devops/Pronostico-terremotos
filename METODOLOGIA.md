# SEIS-PERU · Metodología, investigación y fuentes

Documento técnico-científico del sistema de pronóstico sísmico probabilístico del Perú.
Describe el problema, los datos, cada método (con sus ecuaciones), los experimentos y las
referencias. Todas las fuentes y endpoints fueron **verificados en vivo** (2026-08).

---

## 1. Problema y objetivo

Para una región `r` y un instante `t`, estimar la probabilidad condicional

> **P( M ≥ M₀ | Xₜ , H )**

donde `M₀` es la magnitud objetivo, `H` el horizonte temporal y `Xₜ` la información
disponible **hasta** `t`. Ejemplo: *P(M≥6.0 en los próximos 30 días | información de hoy)*.

**Regla inviolable (anti-leakage):** las features se calculan con eventos de `origin_time ≤ t`
y el target vive en `(t, t+H]`. Formalmente `X(t) → Y(t+H)`, nunca `X(t+H) → Y(t+H)`.

### Hipótesis

| ID | Hipótesis | Prueba |
|----|-----------|--------|
| H1 | La actividad reciente mejora vs. tasa histórica constante | baseline vs. Poisson/ETAS |
| H2 | Los sismos fuertes modifican temporalmente la tasa | ETAS (triggering de Omori) |
| H3 | La actividad de países vecinos informa el pronóstico del Perú | superposed-epoch + control |
| H4 | Las features espaciales mejoran el pronóstico | ablación / SHAP |
| H5 | Un modelo ML supera a los sismológicos tradicionales | backtesting walk-forward |

---

## 2. Fuentes de datos

### 2.1 Catálogos sísmicos

| Fuente | Endpoint / acceso | Formato | Cobertura | Rol |
|--------|-------------------|---------|-----------|-----|
| **IGP / CENSIS** | `datosabiertos.gob.pe/.../IGP_catalogo_sismico_1960_%202025_Dataset.csv` | CSV `;` | 1960– (24 289 ev.) | Verdad de campo nacional |
| **USGS / ComCat** | `earthquake.usgs.gov/fdsnws/event/1/query` | FDSN text/geojson | 1900– | Backbone global |
| **ISC** | `isc.ac.uk/fdsnws/event/1/query` | FDSN text/QuakeML | 1964– (rev. ~24 m atrás) | Backbone revisado |
| **EMSC** | `seismicportal.eu/fdsnws/event/1/query` | FDSN text/json | tiempo-real | Complemento |
| **GCMT** | `ldeo.columbia.edu/~gcmt/.../jan76_dec25.ndk` | NDK | 1976– | Mw directa |
| **ISC-GEM** | `isc.ac.uk/iscgem` (form + CAPTCHA) | CSV | 1904–2021 | Mw homogéneo |

**Notas verificadas:**
- **IRIS/EarthScope FDSN retirado** (HTTP 410 desde 2026-06-01) → eliminado del pipeline.
- El **IGP no expone FDSN**; su catálogo es un CSV único (`HORA_UTC` sin ceros a la izquierda,
  `MAGNITUD` sin tipo → tratada como `assume_unknown`).
- **GCMT no se sirve por ComCat** (`catalog=gcmt` no devuelve orígenes); se ingiere por NDK
  y se usa el **hipocentro PDE** (no el centroide) para deduplicar limpio. Mw = ⅔·(log₁₀M₀ − 16.1)
  con M₀ en dyne·cm (Kanamori 1977).
- **Bounding box del Perú:** lat −18.5…0.5, lon −81.5…−68.0 (incluye zona offshore de la fosa).

### 2.2 Límites administrativos

GeoJSON de **departamentos (25), provincias (197) y distritos (1826)** — repositorio
`juaneladio/peru-geojson` (base INEI). Simplificados (Douglas-Peucker, ~0.6–2 km) para el mapa.

### 2.3 Ingesta y troceado

Cliente FDSN genérico con **bisección temporal adaptativa**: si una ventana devuelve el máximo
de filas (truncada), se parte en dos y se reintenta — así se descargan décadas sorteando el tope
de 20 000 eventos/consulta. Almacenamiento inmutable (data lake Parquet) + manifiestos.

---

## 3. Deduplicación / asociación de eventos

El mismo sismo aparece en varias fuentes con tiempo/hipocentro/magnitud distintos. Se agrupan
en un `canonical_id` por **matching espacio-temporal + componentes conexas** (union-find, barrido
ordenado por tiempo).

- **Umbrales por defecto:** `|Δt| ≤ 16 s` **Y** `distancia_haversine ≤ 100 km`. Profundidad y
  magnitud NO se usan como criterio duro (poco fiables entre catálogos).
- **Origen canónico** por prioridad de fuente `IGP > ISC > USGS > EMSC` (red local mejor para el Perú).
- **Magnitud canónica** por calidad del tipo: **Mw directa** > Ms→Mw > mb→Mw > ML/Md.
- **Salvaguarda anti-enjambre:** se avisa si un evento absorbe > 12 observaciones.

**Base:** Zúñiga et al. (2019, Nature Sci. Data) usan Δt<1 min, Δpos<1°; Vorobieva et al. (2022)
reportan σ_T≈5 s, σ_epi≈15 km; framework de vecino cercano de Zaliapin & Ben-Zion (2013).

---

## 4. Homogenización de magnitud a Mw

Toda magnitud no-Mw se convierte con relaciones empíricas **citadas** (`magnitude.py`), conservando
siempre la magnitud original:

- **Scordilis (2006)** — estándar de facto, y el que usa la PSHA nacional del Perú (IGP 2014/2015 → SENCICO/E.030):
  - Ms→Mw: `0.67·Ms + 2.07` (3.0–6.1); `0.99·Ms + 0.08` (6.2–8.2)
  - mb→Mw: `0.85·mb + 1.03` (3.5–6.2)
- **Di Giacomo et al. (2015)** — base de ISC-GEM (exponencial): Ms→Mw `exp(−0.222+0.233·Ms)+2.863`; mb→Mw `exp(−4.664+0.859·mb)+4.555`.
- **Perú (Cahuari Begazo 2008, UNSA/IGP)** — literatura gris, rango 4.5–6.8: mb→Mw `0.9588·mb+0.458`; ML(d)→Mw `0.9879·ML+0.3316`.

**Hallazgo honesto:** no existe una relación ML→Mw peruana peer-reviewed robusta → se **prefiere
Mw directa** (GCMT/ISC-GEM) vía deduplicación y solo se convierte mb/Ms.

---

## 5. Magnitud de completitud (Mc) y b-value

FMD de Gutenberg-Richter: `log₁₀ N(≥M) = a − b·M` (válida para M ≥ Mc). Estimadores propios,
testeados contra catálogos sintéticos (`analysis/`):

- **Mc — MAXC** (máxima curvatura) + 0.2 (Woessner & Wiemer 2005) y **GFT** (Wiemer & Wyss 2000, nivel 90%).
- **b — Aki (1965) MLE** con corrección de binning (Utsu): `b = log₁₀e / (⟨M⟩ − (Mc − ΔM/2))`;
  incertidumbre de **Shi & Bolt (1982)**: `σ_b = 2.30·b²·√(Σ(Mᵢ−⟨M⟩)²/(n(n−1)))`.
- **Mc(t) no estacionaria** (densificación de la red) e incompletitud post-mainshock
  `Mc(t,M)=M−4.5−0.75·log₁₀(t)` (Helmstetter, Kagan & Jackson 2006).

**Resultado Perú:** Mc moderna ≈ 4.5; **b real ≈ 0.9–1.0** (con Mw directa/tipada). El b=1.46 del
catálogo crudo es un **artefacto** de las magnitudes IGP sin tipo + la elección de conversión
(la elección mueve b entre 0.75 y 1.09). Gradiente Norte(1.58) > Centro > Sur(1.37).

---

## 6. Feature engineering (anti-leakage)

Malla de **0.1°/0.25°** sobre el Perú. Para cada par (celda, instante t), con corte estricto ≤ t:

- **Temporales:** conteos 7/30/90/365 d y total; magnitud máxima 30/90/365 d; energía log
  (Σ10^(1.5·Mw)); días desde el último sismo / último M5.
- **Espaciales:** actividad del vecindario 3×3; actividad regional; centroide (lat, lon).
- **ETAS:** intensidad de triggering `Σ K·e^{α(mᵢ−Mc)}·(t−tᵢ+c)^{−p}` (celda y vecindario).
- **Targets:** `M≥6/30d`, `M≥5/30d`, `M≥5/7d`.

El test `tests/test_features.py::test_no_leakage` inyecta un evento futuro y verifica que
**ninguna feature cambia** (pero el target sí). Señal observada: con 3–5 sismos/30 d en una celda,
P(M5+/30d) sube ~40× respecto a la base.

---

## 7. Modelos

### 7.1 Baselines (el "listón")

- **Climatología:** tasa base constante (referencia para el Brier Skill Score).
- **Poisson:** `P(≥1 en H) = 1 − e^{−λ·H}`.
- **Gutenberg-Richter:** `λ(≥M₀) = λ(≥Mc)·10^{−b(M₀−Mc)}` → Poisson. Variantes: largo plazo,
  reciente (365 d) y **suavizado espacial** (vecindario).

### 7.2 ETAS (Ogata 1988) — `models/etas.py`

Intensidad condicional temporal:

> **λ(t) = μ + Σ_{tᵢ<t} K·exp(α·(mᵢ−M₀)) · (t − tᵢ + c)^{−p}**

Log-verosimilitud con compensador cerrado; inversión por MLE (Nelder-Mead sobre log-parámetros);
simulación por ramificación para tests de recuperación de parámetros. Razón de ramificación
`n = K·(β/(β−α))·(c^{1−p}/(p−1))`, β = b·ln10.

**Ajuste Perú (M≥4.5, 2000+):** μ=0.48/día, K=0.17, α=0.87 (α/ln10=0.38, bajo → típico de
subducción), c=0.40 d, p=1.53, **n=0.84** (84% réplicas; fondo 16%). Ancla de validación: Chile
(Nicolis et al. 2015) K≈0.04, α=2.3, c=0.03, p=1.21. *Caveat honesto:* c, p y K están débilmente
restringidos y correlacionados; los robustos son α y la razón de ramificación.

### 7.3 Machine Learning — `models/ml.py`

**XGBoost** y **LightGBM** con `scale_pos_weight` (desbalance extremo). Solo features leakage-safe;
se excluye `cell_id` (evita memorizar celdas) y se incluye `cen_lat/cen_lon` (prior espacial físico).
**Calibración isotónica por fold** (la escala de probabilidad se desregula con el peso de clase).

---

## 8. Evaluación

- **Métricas:** Brier, **Brier Skill Score** (vs. climatología), log-loss, **PR-AUC** (>ROC-AUC por
  el desbalance extremo), ROC-AUC, diagrama de **confiabilidad/calibración**.
- **Backtesting walk-forward expansivo:** para cada año Y de test se entrena con datos <Y−1, se
  calibra con Y−1 y se predice Y; se acumulan las predicciones out-of-sample 2015–2026. Nunca
  `random_split` (produce leakage temporal).
- **Marco CSEP** (Collaboratory for the Study of Earthquake Predictability): N/S/M/L-tests y
  `paired_t_test` vía **pyCSEP** son la referencia estándar para forecasts gridados (trabajo futuro).

**Resultado H5 (3 configuraciones: 0.1°/30d, 0.25°/30d, 0.25°/7d, con y sin ETAS):** el mejor
forecaster es el **Poisson-GR calibrado** (ROC-AUC 0.639, BSS +0.0012); **XGBoost no lo supera**
(0.618). SHAP: el modelo se apoya en features **espaciales** (cen_lat, cnt_all) — aprende *dónde*,
que la física ya captura. **H5 refutada de forma robusta.**

---

## 9. Experimentos regionales y temporales

### 9.1 ¿Chile dispara sismos en el Perú? (H3) — `experiment_chile_peru.py`

**Método:** superposed-epoch analysis con **null de Monte Carlo** (3000 permutaciones de tiempos
aleatorios); rate-ratio, beta-statistic (Matthews & Reasenberg 1988) y p-valor por permutación,
para ventanas de 1/7/30/90 días tras cada gran sismo de Chile (M≥7, M≥7.5).

**Resultado:** con TODOS los sismos de Chile aparece una señal fuerte en el sur del Perú
(RR≈11.5 a 1 día). Pero el **control de distancia** (solo Chile remoto >600 km, donde únicamente
cabe triggering dinámico) la **anula** (RR≈1.1, p>0.1). → La señal era **fuga de réplicas de
campo cercano** de eventos del norte de Chile, no triggering remoto. **H3 = nulo**, coincidiendo
con Parsons & Velasco (2011). *Sin el control habríamos afirmado una causalidad falsa.*

**Física:** el triggering estático (Coulomb) decae ~1/r³ y se confina a ~1–2 longitudes de ruptura
(King, Stein & Lin 1994); el dinámico (ondas) puede actuar a distancia teleseísmica pero afecta
sobre todo **microsismicidad** en zonas susceptibles (Velasco et al. 2008), y NO eleva el peligro
de grandes sismos.

### 9.2 ¿Qué mes es más probable? (estacionalidad) — `context_analysis.py`

**Método:** χ² de la distribución mensual vs. uniforme (ajustada por días/mes) + **test de Schuster**
de periodicidad anual `p = exp(−R²/N)`. Se usa M≥6 (casi independiente) para el veredicto.

**Resultado:** M5+ muestra un pico aparente en jun/ago (χ² p≈0), pero es **artefacto de las réplicas
de Arequipa 2001 (jun) y Pisco 2007 (ago)**; el test con eventos grandes independientes (M6+,
Schuster p=0.15) **no es significativo**, y los M7+ caen en meses dispersos. → **No hay un mes más
probable.** Los sismos no son estacionales.

### 9.3 Cinturón de Fuego / países vecinos

Todo el margen (Perú, Chile, Ecuador, Colombia, Bolivia) es **una sola frontera de placas** — la
subducción de Nazca bajo Sudamérica. Contexto (no causa): Chile máx **9.5** (Valdivia 1960, el
mayor jamás registrado), Perú 8.4 (Arequipa 2001), Ecuador 8.1, Bolivia 8.2 (profundo). La
actividad simultánea es lo esperado en un mismo margen — no implica triggering entre países (§9.1).

---

## 10. Limitaciones y trabajo futuro

- **Resolución nativa** del forecast: celda 0.1° (~11 km). Los valores por provincia/distrito son
  esa malla integrada sobre límites reales — el detalle es del mapa, no una resolución sísmica sub-11 km.
- Skill de pronóstico de corto plazo **genuinamente bajo** (realidad física, no un bug).
- **Futuro:** ETAS espacio-temporal completo; features físicas (deformación GNSS/strain); ISC-GEM
  como backbone; evaluación formal con pyCSEP; relaciones ML→Mw regionales; frontend Angular + MapLibre.

---

## 11. Referencias

- Aki, K. (1965). *Maximum likelihood estimate of b in the formula logN=a−bM.* Bull. Earthq. Res. Inst. 43:237.
- Cahuari Begazo, A. Y. (2008). *Cálculo de la magnitud local (ML)…* Tesis, UNSA / IGP.
- Di Giacomo, D. et al. (2015). *ISC-GEM: proxy Mw.* Phys. Earth Planet. Inter. 239:33. DOI 10.1016/j.pepi.2014.06.005.
- Helmstetter, A., Kagan, Y., Jackson, D. (2006). *Comparison of short-term forecasts…* BSSA 96:90.
- Kanamori, H. (1977). *The energy release in great earthquakes.* JGR 82:2981.
- King, G., Stein, R., Lin, J. (1994). *Static stress changes and the triggering of earthquakes.* BSSA 84:935.
- Matthews, M., Reasenberg, P. (1988). *Statistical methods for investigating quiescence.* PAGEOPH 126:357.
- Nicolis, O. et al. (2015). *Windowed ETAS models… Chilean catalogs.* Spatial Statistics.
- Ogata, Y. (1988). *Statistical models for earthquake occurrences (ETAS).* JASA 83:9.
- Parsons, T., Velasco, A. (2011). *Absence of remotely triggered large earthquakes…* Nat. Geosci. 4:312. DOI 10.1038/ngeo1110.
- Scordilis, E. (2006). *Empirical global relations converting Ms and mb to Mw.* J. Seismology 10:225. DOI 10.1007/s10950-006-9012-4.
- Shi, Y., Bolt, B. (1982). *The standard error of the magnitude-frequency b value.* BSSA 72:1677.
- Velasco, A. et al. (2008). *Global ubiquity of dynamic earthquake triggering.* Nat. Geosci. 1:375.
- Vorobieva, I. et al. (2022). *Benchmarking… catalog merging.* Front. Earth Sci. DOI 10.3389/feart.2022.820277.
- Wiemer, S., Wyss, M. (2000). *Minimum magnitude of completeness (GFT).* BSSA 90:859.
- Woessner, J., Wiemer, S. (2005). *Assessing the quality of earthquake catalogs (Mc).* BSSA 95:684.
- Zaliapin, I., Ben-Zion, Y. (2013). *Earthquake clusters… nearest-neighbor.* JGR 118:2847.
- Zúñiga, R. et al. (2019). *A unified Mexican earthquake catalog.* Sci. Data 6. DOI 10.1038/s41597-019-0234-z.
- **pyCSEP** — Savran, W. et al. (2022). SCECcode/pycsep, docs.cseptesting.org.

*Compilado con verificación en vivo de fuentes y código propio testeado (2026).*
