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
# # RCMIP runner output - visual QA
#
# Loads every `{scenario}_rcmip_{model}.csv` under `out/rcmip/`
# (written by `scripts/run_rcmip.py`) and produces a categorised
# grid of (scenario x variable) panels, with each model's per-member
# ensemble shown as a median + 5-95 % envelope.
#
# Scenarios are grouped by class for legibility:
#
# - **SSPs** (`ssp*`): the four protocol SSPs (and 119/434/460/534-over
#   if present). Both CICEROSCMPY2 (concentration-driven) and FaIRv2
#   (emissions-driven, when fixture data exists) on the same axes.
# - **esm-allGHG SSPs** (`esm-allGHG-ssp*`): emissions-driven SSP variants.
# - **scen7 family**: `scen7-*` and `esm-scen7-*` variants.
# - **Idealised**: `1pctCO2*`, `abrupt-*`, `esm-flat*`, `esm-bell-*`, etc.
# - **Historical**: `historical*`, `hist-*`, `piControl`.
# - **methanemip**: `methanemip-*`.
#
# Each scenario has its own row; output variables in the columns.
#
# Re-run `python scripts/run_rcmip.py` to regenerate the CSVs.

# %%
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = REPO_ROOT / "out" / "rcmip"

VARIABLES = (
    "Surface Air Temperature Change",
    "Atmospheric Concentrations|CO2",
    "Effective Radiative Forcing",
)
MODEL_COLOURS = {"cicero": "C0", "fair": "C1"}


def _model_from_filename(stem: str) -> str:
    m = re.match(r".*_rcmip_(cicero|fair)$", stem)
    return m.group(1) if m else "unknown"


def _scenario_from_filename(stem: str) -> str:
    return re.sub(r"_rcmip_(cicero|fair)$", "", stem)


# %%
all_csvs = sorted(DATA_DIR.glob("*_rcmip_*.csv"))
print(f"Loaded {len(all_csvs)} CSVs from {DATA_DIR}:")
for p in all_csvs[:10]:
    print(f"  {p.name}")
if len(all_csvs) > 10:
    print(f"  ... and {len(all_csvs) - 10} more")


def _load(path: Path) -> pd.DataFrame:
    """Load one CSV; index = scmdata meta multiindex, columns = years (int)."""
    df = pd.read_csv(path, header=0, index_col=list(range(8)))
    df.columns = df.columns.astype(int)
    return df


loaded: dict[tuple[str, str], pd.DataFrame] = {}
for p in all_csvs:
    scen = _scenario_from_filename(p.stem)
    model = _model_from_filename(p.stem)
    loaded[(scen, model)] = _load(p)

scenarios_seen = sorted({k[0] for k in loaded})
models_seen = sorted({k[1] for k in loaded})
print(f"\n{len(scenarios_seen)} scenario(s) across {len(models_seen)} model(s): {models_seen}")


# %% [markdown]
# ## Categorise scenarios

# %%
def _category(scen: str) -> str:
    s = scen.lower()
    if s.startswith("esm-allghg-ssp"):
        return "esm-allGHG-SSP"
    if s.startswith("ssp"):
        return "SSP"
    if "scen7" in s:
        return "scen7"
    if s.startswith(("historical", "hist-", "esm-hist", "picontrol", "esm-picontrol")):
        return "historical"
    if s.startswith("methanemip"):
        return "methanemip"
    if s.startswith(("1pctco2", "abrupt-", "esm-flat", "esm-bell", "esm-pi-", "esm-1pct")):
        return "idealised"
    return "other"


by_cat: dict[str, list[str]] = {}
for scen in scenarios_seen:
    by_cat.setdefault(_category(scen), []).append(scen)

print("Scenario categories:")
for cat, scens in sorted(by_cat.items()):
    print(f"  {cat:<20} ({len(scens):>2}): {scens}")


# %% [markdown]
# ## Plot: one figure per category, scenario x variable grid

# %%
def _envelope(df: pd.DataFrame):
    """Return (median, p05, p95) across the ensemble rows."""
    return (
        df.median(axis=0),
        df.quantile(0.05, axis=0),
        df.quantile(0.95, axis=0),
    )


def plot_category(category: str, scens: list[str]) -> Path | None:
    if not scens:
        return None
    nrows = len(scens)
    ncols = len(VARIABLES)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(13, 2.2 * max(nrows, 1) + 1.0),
        sharex=True,
        squeeze=False,
    )
    for row, scen in enumerate(scens):
        for col, var in enumerate(VARIABLES):
            ax = axes[row, col]
            any_data = False
            for model in models_seen:
                df = loaded.get((scen, model))
                if df is None:
                    continue
                sel = df.xs(var, level="variable")
                if sel.empty:
                    continue
                median, p05, p95 = _envelope(sel)
                colour = MODEL_COLOURS.get(model, "k")
                years = median.index
                mask = years >= 1850
                ax.plot(
                    years[mask], median[mask], color=colour, lw=1.3,
                    label=model,
                )
                ax.fill_between(
                    years[mask], p05[mask], p95[mask],
                    color=colour, alpha=0.18,
                )
                any_data = True
            if not any_data:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
            if row == 0:
                ax.set_title(var, fontsize=9)
            if col == 0:
                ax.set_ylabel(scen, fontsize=9)
            if row == nrows - 1:
                ax.set_xlabel("Year")
            ax.grid(True, alpha=0.3)

    # Single legend at top
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
        if handles:
            break
    if handles:
        fig.legend(
            handles, labels, loc="upper center",
            ncol=len(labels), frameon=False,
        )
    fig.suptitle(
        f"RCMIP runner — {category} ({len(scens)} scenarios, median + 5-95 %)",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97 - 0.02 * (nrows > 2)))
    out_path = DATA_DIR / f"plot_{category.replace('-', '_').lower()}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"Saved {out_path.name}")
    return out_path


saved = []
for cat in sorted(by_cat):
    p = plot_category(cat, by_cat[cat])
    if p:
        saved.append(p)
print()
print(f"Saved {len(saved)} per-category figures under {DATA_DIR}/")


# %% [markdown]
# ## 2100 GSAT summary across all scenarios

# %%
rows = []
for (scen, model), df in loaded.items():
    sel = df.xs("Surface Air Temperature Change", level="variable")
    if sel.empty or 2100 not in sel.columns:
        continue
    vals = sel[2100].values
    rows.append({
        "scenario": scen, "model": model,
        "category": _category(scen),
        "GSAT_2100_median": float(np.median(vals)),
        "GSAT_2100_p05": float(np.quantile(vals, 0.05)),
        "GSAT_2100_p95": float(np.quantile(vals, 0.95)),
        "n_members": vals.size,
    })

if rows:
    summary = (
        pd.DataFrame(rows)
        .sort_values(["category", "scenario", "model"])
        .round(2)
    )
    print(summary.to_string(index=False))
    summary.to_csv(DATA_DIR / "summary_gsat_2100.csv", index=False)
    print(f"\nWrote summary to {DATA_DIR}/summary_gsat_2100.csv")
else:
    print("No scenarios reach 2100 in the output.")
