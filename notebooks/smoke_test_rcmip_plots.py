# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Joint FaIRv2 + CICEROSCMPY2 RCMIP smoke test - plots
#
# Loads the CSVs written by `scripts/smoke_test_rcmip.py` and plots
# both models on the same axes for visual inspection.
#
# - Rows: ssp126 / ssp245 / ssp370 / ssp585
# - Cols: Surface Air Temperature Change / Atmospheric Concentrations|CO2 /
#         Effective Radiative Forcing
# - Lines: each climate model's per-member ensemble with median +
#          5-95 % envelope shaded.
#
# Visual sanity checks:
#
# 1. Both models should give monotonically warming GSAT for ssp245/370/585,
#    peak-then-stabilise for ssp126.
# 2. 2100 GSAT envelopes should overlap or be close: CICERO typically a
#    bit warmer than FaIR on ssp245 in AR6-era calibrations.
# 3. CO2 should rise steadily under ssp370/585 and dip under ssp126.
# 4. ERF should track CO2 + non-CO2 in the expected order ssp126 <
#    ssp245 < ssp370 < ssp585 across 2100.
#
# Re-run `python scripts/smoke_test_rcmip.py` if you need to refresh the
# CSVs (e.g. after changing the calibration bundle or member count).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = REPO_ROOT / "out" / "smoke_test_rcmip"

SCENARIOS = ("ssp126", "ssp245", "ssp370", "ssp585")
VARIABLES = (
    "Surface Air Temperature Change",
    "Atmospheric Concentrations|CO2",
    "Effective Radiative Forcing",
)
MODEL_COLOURS = {"CICERO-SCM-PY2.1.0": "C0", "FaIRv2.2.4": "C1"}


# %%
def _load(path: Path) -> pd.DataFrame:
    """Load one model's smoke-test CSV; long-form with a year index."""
    df = pd.read_csv(path, header=0, index_col=list(range(8)))
    df.columns = df.columns.astype(int)
    return df


csvs = sorted(DATA_DIR.glob("*.csv"))
print(f"Loaded CSVs from {DATA_DIR}:")
for p in csvs:
    print(f"  {p.name}")
frames = {p.stem: _load(p) for p in csvs}
for name, df in frames.items():
    print(f"\n{name}: shape {df.shape}")
    print(f"  meta levels: {df.index.names}")
    print(f"  variables: {sorted(set(df.index.get_level_values('variable')))}")
    print(f"  scenarios: {sorted(set(df.index.get_level_values('scenario')))}")


# %% [markdown]
# ## 4 (scenario) x 3 (variable) grid

# %%
def _envelope(df: pd.DataFrame):
    """Return (median, p05, p95) across the ensemble rows."""
    return (
        df.median(axis=0),
        df.quantile(0.05, axis=0),
        df.quantile(0.95, axis=0),
    )


fig, axes = plt.subplots(
    nrows=len(SCENARIOS),
    ncols=len(VARIABLES),
    figsize=(13, 11),
    sharex=True,
)

for row, scenario in enumerate(SCENARIOS):
    for col, variable in enumerate(VARIABLES):
        ax = axes[row, col]
        for model_name, df in frames.items():
            sel = df.xs(scenario, level="scenario").xs(variable, level="variable")
            if sel.empty:
                continue
            median, p05, p95 = _envelope(sel)
            colour = MODEL_COLOURS.get(model_name, "k")
            years = median.index
            # Keep only post-1850 years for visual cleanness; the
            # pre-1850 historical is heavily clipped to bundle baselines.
            mask = years >= 1850
            ax.plot(years[mask], median[mask], color=colour, lw=1.5, label=model_name)
            ax.fill_between(
                years[mask], p05[mask], p95[mask], color=colour, alpha=0.18
            )
        if row == 0:
            ax.set_title(variable, fontsize=10)
        if col == 0:
            ax.set_ylabel(scenario, fontsize=11)
        if row == len(SCENARIOS) - 1:
            ax.set_xlabel("Year")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1850, 2100)

# Single legend at top.
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(frames), frameon=False)
fig.suptitle(
    f"FaIRv2 + CICEROSCMPY2 smoke test  -  {len(frames)} models x "
    f"{len(SCENARIOS)} SSPs x median + 5-95% envelope",
    y=0.98,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))

fig_path = DATA_DIR / "ssp_envelopes.png"
fig.savefig(fig_path, dpi=120, bbox_inches="tight")
print(f"Saved {fig_path}")


# %% [markdown]
# ## 2100 GSAT distributions per model per scenario

# %%
fig2, ax = plt.subplots(figsize=(9, 5))

variable = "Surface Air Temperature Change"
year_target = 2100
positions = np.arange(len(SCENARIOS))
offsets = np.linspace(-0.18, 0.18, len(frames))

for offset, (model_name, df) in zip(offsets, frames.items()):
    colour = MODEL_COLOURS.get(model_name, "k")
    series_per_scenario = []
    for scenario in SCENARIOS:
        sel = df.xs(scenario, level="scenario").xs(variable, level="variable")
        if sel.empty:
            series_per_scenario.append(np.array([]))
            continue
        series_per_scenario.append(sel[year_target].values)
    # Strip plot: one dot per ensemble member.
    for i, vals in enumerate(series_per_scenario):
        ax.scatter(
            np.full(vals.size, positions[i] + offset),
            vals,
            color=colour,
            s=22,
            alpha=0.55,
            edgecolor="none",
            label=model_name if i == 0 else None,
        )
        if vals.size:
            ax.plot(
                [positions[i] + offset - 0.05, positions[i] + offset + 0.05],
                [np.median(vals)] * 2,
                color=colour,
                lw=2.0,
            )

ax.set_xticks(positions)
ax.set_xticklabels(SCENARIOS)
ax.set_ylabel(f"{variable} at {year_target} (K)")
ax.set_title(
    f"{year_target} {variable} per scenario - per-member dots, horizontal "
    "bar = median"
)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="upper left", frameon=False)
fig2.tight_layout()

fig2_path = DATA_DIR / "ssp_gsat_2100_scatter.png"
fig2.savefig(fig2_path, dpi=120, bbox_inches="tight")
print(f"Saved {fig2_path}")

# %% [markdown]
# ## Sanity-check summary table

# %%
rows = []
for model_name, df in frames.items():
    for scenario in SCENARIOS:
        sel = (
            df.xs(scenario, level="scenario")
            .xs("Surface Air Temperature Change", level="variable")
        )
        if sel.empty or 2100 not in sel.columns:
            continue
        vals_2100 = sel[2100].values
        rows.append(
            {
                "model": model_name,
                "scenario": scenario,
                "GSAT_2100_median": np.median(vals_2100),
                "GSAT_2100_p05": np.quantile(vals_2100, 0.05),
                "GSAT_2100_p95": np.quantile(vals_2100, 0.95),
                "n_members": vals_2100.size,
            }
        )

summary = pd.DataFrame(rows).round(2)
print(summary.to_string(index=False))
