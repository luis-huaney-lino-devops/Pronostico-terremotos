# ---
# SEIS-PERU · Notebook 01 — EDA + Gutenberg-Richter + Mc + b-value
# Archivo con celdas `# %%` (VSCode: "Run Cell" / Jupyter interactive).
# Reproduce el análisis de scripts/run_eda.py de forma interactiva.
# ---

# %% [markdown]
# # EDA del catálogo sísmico del Perú
# Distribución frecuencia-magnitud, magnitud de completitud (Mc) y b-value.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parents[0] if (Path.cwd().name == "notebooks") else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seis_peru.analysis.completeness import fmd, mc_gft, mc_maxc
from seis_peru.analysis.gutenberg_richter import b_value_aki, gr_fit
from seis_peru.analysis.regions_peru import PERU_MACRO

df = pd.read_parquet(ROOT / "data/processed/catalog_canonical_peru.parquet")
df = df.dropna(subset=["preferred_mw"])
df["year"] = df["origin_time"].dt.year
print(f"{len(df):,} eventos | {df.year.min()}–{df.year.max()}")
df.head()

# %% [markdown]
# ## 1. Distribución frecuencia-magnitud (FMD) y Mc

# %%
modern = df[df.year >= 2000]["preferred_mw"].to_numpy()
mc = mc_gft(modern, 0.1).mc
bv = b_value_aki(modern, mc, 0.1)
print(f"Mc (GFT) = {mc:.2f} | b = {bv.b:.3f} ± {bv.sigma_b:.3f} | n = {bv.n:,}")

centers, inc, cum = fmd(modern, 0.1)
plt.figure(figsize=(7, 5))
plt.scatter(centers, cum, s=18, label="N(≥M)")
plt.scatter(centers, inc, s=12, c="gray", marker="s", label="incremental")
mr = centers[centers >= mc - 0.05]
plt.plot(mr, 10 ** (bv.a - bv.b * mr), "r-", lw=2, label=f"GR b={bv.b:.2f}")
plt.axvline(mc, color="g", ls="--", label=f"Mc={mc:.2f}")
plt.yscale("log"); plt.xlabel("Mw"); plt.ylabel("Nº eventos")
plt.legend(); plt.grid(alpha=0.3); plt.title("FMD Perú 2000+"); plt.show()

# %% [markdown]
# ## 2. Diagnóstico: ¿el b está inflado por la mezcla de magnitudes?
# Comparamos con el b calculado SOLO con magnitud-momento (Mw) directa.

# %%
mw_only = df[(df.year >= 2000) &
             df.preferred_mag_type.fillna("").str.lower().str.startswith("mw")]["preferred_mw"].to_numpy()
mc_mw = mc_maxc(mw_only, 0.1)
bv_mw = b_value_aki(mw_only, mc_mw, 0.1)
print(f"Mixto:    b = {bv.b:.3f}  (Mc {mc:.2f})")
print(f"Mw solo:  b = {bv_mw.b:.3f}  (Mc {mc_mw:.2f}, n={bv_mw.n})")

# %% [markdown]
# ## 3. b-value por macro-región (norte / centro / sur)

# %%
for name, box in PERU_MACRO.items():
    m = df[(df.year >= 2000) &
           df.latitude.between(box.min_lat, box.max_lat) &
           df.longitude.between(box.min_lon, box.max_lon)]["preferred_mw"].to_numpy()
    if len(m) < 200:
        continue
    mc_r = mc_gft(m, 0.1).mc
    b = b_value_aki(m, mc_r, 0.1)
    print(f"{name:14s} n={b.n:5d}  Mc={mc_r:.2f}  b={b.b:.3f} ± {b.sigma_b:.3f}")

# %% [markdown]
# ## 4. Mc por década — la red se densifica con el tiempo

# %%
for dec in range(1960, 2030, 10):
    m = df[(df.year >= dec) & (df.year < dec + 10)]["preferred_mw"].to_numpy()
    if len(m) < 100:
        continue
    print(f"{dec}s  n={len(m):5d}  Mc={mc_maxc(m, 0.1):.2f}")
