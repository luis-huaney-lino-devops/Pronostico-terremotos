# EDA SEIS-PERU — catálogo del Perú

- Eventos: **26,459** | 1960–2026 | Mw 4.0–8.4

## Magnitud de completitud (Mc), 2000+ (17,151 eventos)
- MAXC+0.2 = **4.70**
- GFT (Wiemer-Wyss) = **4.50** (R=91.7%, nivel 90%)
- Mc adoptado para b-value = **4.50**

## Gutenberg-Richter b-value (Perú, 2000+)
- **b = 1.457 ± 0.012** | a = 10.71 | n = 14,271

### Diagnóstico — b-value SOLO con Mw directa (1,045 eventos)
- Mc(Mw)=5.20 | **b = 0.937 ± 0.046**
- Si b(Mw) << b(global 1.46), la conversión de magnitudes inflaba el b global (heterogeneidad de escalas).

## b-value según esquema de conversión de magnitud (2000+, Mc fijo=4.5)
Mismo Mc para todos → aísla el efecto de la relación de conversión.
| Esquema | b ± σ | n |
|---|---|---:|
| scordilis | 0.994 ± 0.009 | 6,492 |
| digiacomo | 0.751 ± 0.004 | 6,858 |
| peru | 1.093 ± 0.012 | 6,076 |
| igp_national | 1.034 ± 0.017 | 3,572 |

## b-value por macro-región (2000+, Mc por GFT regional)
| Región | n | Mc | b ± σ |
|---|---:|---:|---|
| Perú-Norte | 4,383 | 4.50 | 1.580 ± 0.025 |
| Perú-Centro | 4,307 | 4.50 | 1.489 ± 0.021 |
| Perú-Sur | 5,033 | 4.50 | 1.370 ± 0.018 |

## Mc por década (MAXC+0.2) — muestra la densificación de la red
| Década | n | Mc |
|---|---:|---:|
| 1960s | 997 | 5.10 |
| 1970s | 1,226 | 5.10 |
| 1980s | 2,191 | 5.10 |
| 1990s | 4,894 | 4.70 |
| 2000s | 4,886 | 4.70 |
| 2010s | 8,576 | 4.70 |
| 2020s | 3,689 | 4.20 |

## Figuras
Generadas en `reports/figures/` (6 PNG).

## Hallazgos e interpretación
- **Mc realista (~4.5)** en la era moderna, coherente con la literatura del Perú; la FMD se aplana por debajo (incompletitud).
- **b del catálogo completo alto (1.46)** pero es un ARTEFACTO: sobre el subconjunto de magnitudes TIPADAS con Scordilis (estándar nacional) b≈0.99, y solo con Mw directa b≈0.94. El **b real del Perú es ~0.9–1.0**. Lo inflan (a) las magnitudes del IGP SIN tipo y (b) la elección de conversión (la tabla de esquemas mueve b entre 0.75 y 1.09). Lección: fijar Mc, preferir Mw directa y documentar el esquema.
- **Gradiente norte→sur real** (b: Norte>Centro>Sur). b más bajo en el sur ⇒ relativamente más grandes ⇒ coherente con el gap sísmico del sur y el terremoto de Arequipa 2001 (M8.4).
- **Mc(t) NO es estacionaria**: baja de ~5.1 (1960-80s) a ~4.2 (2020s) por densificación de la red ⇒ hay que usar Mc por ventana temporal.

### Próximo refinamiento
- Recalcular b sobre una fuente **homogénea** (ISC-GEM Mw) para eliminar el sesgo de conversión; relaciones ML→Mw regionales del IGP; Mc(t,x) mapeado.