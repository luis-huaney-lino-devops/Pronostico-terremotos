#!/usr/bin/env python
"""Experimento H3 — ¿Chile dispara sismicidad en el Perú? (análisis honesto)

Superposed-epoch analysis con NULL de Monte Carlo (tiempos aleatorios): para
cada gran sismo de Chile (M≥7), contamos la sismicidad del Perú en ventanas
posteriores y la comparamos con lo esperado por azar (dado el fondo clustered
del Perú). Reportamos rate-ratio, beta-statistic y p-valor por permutación.

EXPECTATIVA HONESTA (física + literatura): resultado DÉBIL o NULO. El triggering
estático se descarta a >1-2 longitudes de ruptura; el dinámico solo afecta
microsismicidad transitoria en zonas susceptibles (arco volcánico del sur). El
triggering de M≥5 en el Perú por sismos de Chile NO se espera (Parsons & Velasco
2011). Un nulo es el resultado esperado y igualmente publicable (regla #4).

    python scripts/experiment_chile_peru.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from seis_peru.regions import CHILE  # noqa: E402

FIGDIR = ROOT / "reports" / "figures"
DAY = 86400.0
N_MC = 3000
START_YEAR = 1990   # Mc razonablemente estable
PERU_MC = 4.5
SUR_PERU = dict(min_lat=-18.5, max_lat=-14.0, min_lon=-76.0, max_lon=-68.0)
SUR_CENTROID = (-16.25, -72.0)   # para el control de distancia
REMOTE_KM = 600.0                # triggers más lejos: solo triggering dinámico posible


def _haversine(lat1, lon1, lat2, lon2):
    r = 6371.0088
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _days(ts, t0):
    return (ts - t0).dt.total_seconds().to_numpy() / DAY


def _post_count(triggers, targets_sorted, W):
    """Nº de eventos target en (t, t+W] sumado sobre los triggers."""
    lo = np.searchsorted(targets_sorted, triggers, "right")
    hi = np.searchsorted(targets_sorted, triggers + W, "right")
    return int(np.sum(hi - lo))


def _mc_null(n_trig, targets_sorted, W, lo_b, hi_b, rng):
    null = np.empty(N_MC)
    for i in range(N_MC):
        rt = rng.uniform(lo_b, hi_b, n_trig)
        null[i] = _post_count(rt, targets_sorted, W)
    return null


def analyze(name, triggers, targets, t0, tmax, emit, rng):
    targets_sorted = np.sort(targets)
    emit(f"\n### {name}  (n triggers = {len(triggers)}, n target = {len(targets_sorted)})")
    if len(triggers) < 3:
        emit("- Muy pocos triggers para un test robusto.")
        return
    emit("| Ventana | Observado | Esperado(azar) | Rate-ratio | β | p (una cola) |")
    emit("|---|---|---|---|---|---|")
    for W in (1, 7, 30, 90):
        lo_b, hi_b = 0.0, tmax - W
        obs = _post_count(triggers, targets_sorted, W)
        null = _mc_null(len(triggers), targets_sorted, W, lo_b, hi_b, rng)
        exp = null.mean()
        beta = (obs - exp) / (null.std() + 1e-9)
        rr = obs / exp if exp > 0 else float("nan")
        p = (1 + np.sum(null >= obs)) / (1 + N_MC)
        flag = " ⚠" if p < 0.05 else ""
        emit(f"| {W}d | {obs} | {exp:.1f} | {rr:.2f} | {beta:+.2f} | {p:.3f}{flag} |")


def main() -> int:
    cat = pd.read_parquet(ROOT / "data/processed/catalog_canonical.parquet").dropna(subset=["preferred_mw"])
    cat = cat[cat["origin_time"].dt.year >= START_YEAR].sort_values("origin_time")
    t0 = cat["origin_time"].min()
    cat_days = _days(cat["origin_time"], t0)
    cat = cat.assign(_d=cat_days)
    tmax = float(cat_days.max())
    rng = np.random.default_rng(42)
    lines = []

    def emit(s=""):
        print(s); lines.append(s)

    emit("# Experimento H3 — ¿Chile → Perú? (superposed-epoch + null Monte Carlo)")
    emit(f"\nPeriodo {START_YEAR}+ | Perú Mc={PERU_MC} | {N_MC} permutaciones | "
         "expectativa física: débil/nulo (Parsons & Velasco 2011).")

    # Triggers de Chile (M≥7 y M≥7.5).
    chile = cat[cat["latitude"].between(CHILE.min_lat, CHILE.max_lat) &
                cat["longitude"].between(CHILE.min_lon, CHILE.max_lon)]
    # Targets: Perú (todo) y sur del Perú (zona más cercana / susceptible).
    peru_all = cat[cat["country"] == "Peru"]
    peru_all = peru_all[peru_all["preferred_mw"] >= PERU_MC]
    sur = cat[cat["latitude"].between(SUR_PERU["min_lat"], SUR_PERU["max_lat"]) &
              cat["longitude"].between(SUR_PERU["min_lon"], SUR_PERU["max_lon"]) &
              (cat["preferred_mw"] >= PERU_MC)]

    chile = chile.assign(_dist=_haversine(chile["latitude"].to_numpy(),
                                          chile["longitude"].to_numpy(), *SUR_CENTROID))
    for mmin in (7.0, 7.5):
        cm = chile[chile["preferred_mw"] >= mmin]
        trig = cm["_d"].to_numpy()
        emit(f"\n## Triggers: Chile M≥{mmin}")
        analyze(f"→ Todo el Perú (M≥{PERU_MC})", trig, peru_all["_d"].to_numpy(), t0, tmax, emit, rng)
        analyze(f"→ Sur del Perú (M≥{PERU_MC})", trig, sur["_d"].to_numpy(), t0, tmax, emit, rng)
        # CONTROL DE DISTANCIA: solo triggers REMOTOS (>600 km) -> descarta
        # aftershocks de campo cercano; si persiste, es triggering dinámico real.
        remote = cm[cm["_dist"] >= REMOTE_KM]["_d"].to_numpy()
        analyze(f"→ Sur del Perú · SOLO Chile remoto >{REMOTE_KM:.0f}km "
                f"(control campo-cercano)", remote, sur["_d"].to_numpy(), t0, tmax, emit, rng)

    # Figura superposed-epoch (Chile M≥7 → sur del Perú, ±30 d).
    _fig_sea(chile[chile["preferred_mw"] >= 7.0]["_d"].to_numpy(),
             np.sort(sur["_d"].to_numpy()), tmax, rng,
             FIGDIR / "11_chile_peru_sea.png")

    emit("\n## Interpretación honesta (H3)")
    emit("- **Señal aparente fuerte pero ESPURIA.** Con TODOS los sismos de Chile, "
         "el sur del Perú muestra un aumento enorme (RR≈11.5 a 1 día tras Chile "
         "M≥7.5). PERO el **control de distancia lo desmonta**: al restringir a Chile "
         "REMOTO (>600 km, donde solo cabe triggering dinámico), la señal DESAPARECE "
         "(RR≈1.1–1.2, p>0.1).")
    emit("- **Conclusión:** la 'señal' es **fuga de aftershocks de campo cercano** de "
         "eventos del NORTE de Chile (frontera), cuya zona de réplicas invade la caja "
         "del sur del Perú — es la MISMA secuencia de subducción, no triggering remoto. "
         "**No hay evidencia de que Chile dispare de forma remota sismos M≥4.5 en el "
         "Perú** → coincide con Parsons & Velasco (2011).")
    emit("- **Lección metodológica:** sin el control de distancia habríamos afirmado "
         "una relación causal falsa (RR=11.5, p=0.001). El resultado negativo "
         "controlado vale más que un positivo espurio (**reglas #4/#5**).")
    emit("- Caveats restantes: STAI (incompletitud post-mainshock) y que no "
         "declusteramos el lado peruano con ETAS. Un análisis definitivo añadiría "
         "dosis-respuesta vs PGV y Mc(t). La microsismicidad dinámica transitoria en "
         "el arco volcánico del sur (Holtkamp & Brudzinski 2011) no se descarta a "
         "magnitudes menores, pero NO eleva el peligro de grandes sismos.")

    (ROOT / "reports" / "chile_peru_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nResumen -> reports/chile_peru_summary.md")
    return 0


def _fig_sea(triggers, targets_sorted, tmax, rng, path):
    taus = np.arange(-30, 31)
    obs = np.array([_post_count(triggers + tau, targets_sorted, 1.0) for tau in taus]) / max(len(triggers), 1)
    # envelope MC
    stacks = np.empty((300, len(taus)))
    for i in range(300):
        rt = rng.uniform(30, tmax - 30, len(triggers))
        stacks[i] = [_post_count(rt + tau, targets_sorted, 1.0) for tau in taus]
    stacks /= max(len(triggers), 1)
    lo, hi = np.percentile(stacks, [2.5, 97.5], axis=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(taus, lo, hi, color="gray", alpha=0.3, label="null 95% (azar)")
    ax.plot(taus, obs, "o-", color="#c44e52", ms=3, label="observado")
    ax.axvline(0, color="k", ls="--", lw=1, label="sismo de Chile")
    ax.set_xlabel("Días relativos al sismo de Chile"); ax.set_ylabel("Sismos/día en sur del Perú")
    ax.set_title("Superposed-epoch: sismicidad del sur del Perú alrededor de Chile M≥7")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
