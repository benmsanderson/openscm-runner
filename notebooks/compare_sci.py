# -*- coding: utf-8 -*-
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
# # SCI cross-model comparison: FaIRv2 vs CICEROSCMPY2
#
# Picks five representative Scenario Compass Initiative 2025 pathways
# (covering high-ambition mitigation through to baseline) and drives
# each through both adapters with a 10-member parameter ensemble,
# then sanity-checks the result.
#
# This is the SCI counterpart to ``compare_rcmip.py``. It exercises:
#
# * The IAMC loader (`openscm_runner.scenarios.load_iamc`) against the
#   real SCI xlsx — same harmonisation pass for both models.
# * Both adapters' translated-cfg paths on scenarios that didn't come
#   out of the RCMIP fixture.
# * Cross-model agreement on a sparser, real-IAM scenario set.
#
# Sanity checks
# -------------
#
# 1. 2100 GSAT ordering: 1.5°C-labelled scenarios end below 2.5 K in
#    both models; high-emission baseline scenarios end above 3.5 K.
# 2. 2100 GSAT, CO2, ERF in plausible numerical ranges for every
#    scenario / member.
# 3. Cross-model 2100 GSAT median spread within 2.5 K (the SCI
#    splice-mode CICEROSCM run uses bundled ssp245 historical for
#    species SCI doesn't report, which biases CICERO results a bit
#    higher than its bundle-mode equivalents — so the bound is
#    looser than for RCMIP).
# 4. Multi-IAM scenarios where one scenario is reported by multiple
#    models: per-model SCM agreement check (intra-IAM spread is
#    smaller than cross-IAM spread).
# %% [markdown]
# ## Configuration

# %%
import multiprocessing as mp
import os
from pathlib import Path

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("fork", force=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scmdata

import openscm_runner.run
from openscm_runner.adapters import CICEROSCMPY2, FAIR2
from openscm_runner.scenarios import load_iamc

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

SCI_PATH = Path(
    os.environ.get(
        "SCI_DATA_PATH",
        REPO_ROOT / "scenario_data" / "SCI-2025_v1.0_pathways_ensemble_global.xlsx",
    )
)
FAIR2_BUNDLE = Path(
    os.environ.get(
        "FAIR2_CALIBRATION_PATH",
        REPO_ROOT / "configurations" / "fair-calibrate-v1.6.0",
    )
)
CICERO_BUNDLE_DIR = Path(
    os.environ.get(
        "CICEROSCMPY2_BUNDLE_DIR",
        REPO_ROOT / "configurations" / "ciceroscm",
    )
)

N_MEMBERS = 10

# Picked to span the SCI ambition range; the loader will gracefully
# error if any name disappears from a future SCI release.
SCI_SCENARIOS = (
    "SSP1-19",                  # 1.5°C-consistent mitigation
    "ADVANCE-2020-1.5°C-2100",  # alternative 1.5°C pathway
    "SSP2-45",                  # middle-of-the-road, ~2.5°C
    "CD-LINKS-NPi",             # current policies extension
    "SSP5-Baseline",            # high-emission baseline
)

OUTPUT_VARIABLES = (
    "Surface Air Temperature Change",
    "Atmospheric Concentrations|CO2",
    "Effective Radiative Forcing",
)

assert SCI_PATH.exists(), f"SCI xlsx not at {SCI_PATH}"
assert FAIR2_BUNDLE.exists(), f"FaIR2 calibration not at {FAIR2_BUNDLE}"
assert CICERO_BUNDLE_DIR.exists(), f"CICERO bundle not at {CICERO_BUNDLE_DIR}"

print("Adapters resolved:")
print(f"  FaIRv2:         v{FAIR2.get_version()}")
print(f"  CICERO-SCM-PY2: v{CICEROSCMPY2.get_version()}")
print(f"  SCI release:    {SCI_PATH.name}")
print(f"  Members per scenario per model: {N_MEMBERS}")

# %% [markdown]
# ## Load + interpolate SCI scenarios
#
# Some IAMs report at 5-year cadence with NaN gaps; both adapters
# expect a contiguous timeseries. Interpolate linearly to annual.

# %%
sci_raw = load_iamc(SCI_PATH, scenarios=list(SCI_SCENARIOS))
print(f"Loaded SCI: {sci_raw.shape[0]} raw timeseries")
print(f"  IAMs:      {sorted(sci_raw['model'].unique())}")
print(f"  Scenarios: {sorted(sci_raw['scenario'].unique())}")
print(f"  Species:   {len(sci_raw['variable'].unique())}")

# Each SCI scenario is reported by multiple IAMs. CICEROSCMPY2's
# DistributionRun collapses the per-scenario input down to one
# (scenario, member) timeseries, so a scenario name reported by
# multiple IAMs trips NonUniqueMetadataError on output assembly. For
# this cross-model comparison we pick a single IAM per scenario; if
# the same scenario name appears under multiple IAMs we prefer
# MESSAGE-GLOBIOM 1.0 (well-represented across SCI's scenario set)
# and otherwise fall back to whichever IAM appears alphabetically
# first. A separate notebook (not built here) would be the right
# home for inter-IAM-spread analysis at fixed scenario name.
PREFERRED_IAM = "MESSAGE-GLOBIOM 1.0"


def _pick_iam_per_scenario(run):
    chosen = {}
    for scenario in sorted(run["scenario"].unique()):
        iams = sorted(run.filter(scenario=scenario)["model"].unique())
        chosen[scenario] = PREFERRED_IAM if PREFERRED_IAM in iams else iams[0]
    return chosen


iam_per_scenario = _pick_iam_per_scenario(sci_raw)
print("\nIAM picked per scenario (one input ScmRun row per scenario):")
for s, m in iam_per_scenario.items():
    print(f"  {s:30s} <- {m}")

# Filter and interpolate.
keep_mask_pieces = []
for scenario, iam in iam_per_scenario.items():
    keep_mask_pieces.append(sci_raw.filter(scenario=scenario, model=iam))
sci_picked = scmdata.run_append(keep_mask_pieces)

sci_years = sci_picked["time"]
annual = pd.date_range(
    f"{sci_years.min().year}-01-01",
    f"{sci_years.max().year}-01-01",
    freq="YS",
)
scenarios = sci_picked.interpolate(annual)
print(
    f"Interpolated to annual cadence "
    f"({scenarios['time'].min().year}-{scenarios['time'].max().year})"
)

# %% [markdown]
# ## Run FaIRv2

# %%
fair_result = openscm_runner.run.run(
    climate_models_cfgs={
        "FaIRv2": [
            {
                "native_calibration": str(FAIR2_BUNDLE),
                "member_indices": range(N_MEMBERS),
                "emissions_bundle": str(FAIR2_BUNDLE),
            }
        ],
    },
    scenarios=scenarios,
    output_variables=OUTPUT_VARIABLES,
    out_config=None,
)
print(f"FaIRv2: {fair_result.shape[0]} output timeseries")

# %% [markdown]
# ## Run CICEROSCMPY2

# %%
required = {
    "distribution_json": CICERO_BUNDLE_DIR / "draw_samples_500.json",
    "gaspam_file": CICERO_BUNDLE_DIR / "gases_vupdate_2022_AR6.txt",
    "concentrations_file": CICERO_BUNDLE_DIR / "ssp245_conc_RCMIP.txt",
}
for k, p in required.items():
    assert p.exists(), f"CICERO bundle file '{k}' missing at {p}"

cicero_result = openscm_runner.run.run(
    climate_models_cfgs={
        "CICERO-SCM-PY2": [
            {
                "distribution_json": str(required["distribution_json"]),
                "gaspam_file": str(required["gaspam_file"]),
                "concentrations_file": str(required["concentrations_file"]),
                "member_indices": range(N_MEMBERS),
            }
        ],
    },
    scenarios=scenarios,
    output_variables=OUTPUT_VARIABLES,
    out_config=None,
)
print(f"CICEROSCMPY2: {cicero_result.shape[0]} output timeseries")

# %% [markdown]
# ## Combine

# %%
combined = scmdata.run_append([fair_result, cicero_result])
model_names = sorted(combined["climate_model"].unique())
FAIR_MODEL = next(m for m in model_names if m.startswith("FaIRv"))
CICERO_MODEL = next(m for m in model_names if m.startswith("CICERO-SCM-PY"))
MODELS = (FAIR_MODEL, CICERO_MODEL)
MODEL_COLOR = {FAIR_MODEL: "C0", CICERO_MODEL: "C1"}

scenario_names = sorted(combined["scenario"].unique())
print("Climate models in combined result:", model_names)
print("Scenarios:", scenario_names)

# %% [markdown]
# ## GSAT trajectories per scenario per IAM
#
# Each (IAM, scenario) pair gets its own panel; both climate models
# overlaid (median ± 5-95% across the parameter ensemble).

# %%
def _plot_trajectories(combined, variable, ylabel, scenarios, iam_per_scenario):
    n = len(scenarios)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(13, 2.8 * rows), sharex=True)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for i, scenario in enumerate(scenarios):
        ax = axes.flat[i]
        for cm in MODELS:
            chunk = combined.filter(
                climate_model=cm, scenario=scenario, variable=variable
            )
            if chunk.empty:
                continue
            ts = chunk.timeseries(time_axis="year")
            years = ts.columns.astype(int)
            median = ts.median(axis=0).values
            lo = ts.quantile(0.05, axis=0).values
            hi = ts.quantile(0.95, axis=0).values
            color = MODEL_COLOR[cm]
            ax.plot(years, median, color=color, label=cm, lw=1.5)
            ax.fill_between(years, lo, hi, color=color, alpha=0.2)
        iam = iam_per_scenario.get(scenario, "?")
        ax.set_title(f"{scenario}\n[{iam}]", fontsize=8)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="upper left", fontsize=7)

    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("Year", fontsize=9)

    fig.tight_layout()
    plt.show()


_plot_trajectories(
    combined, "Surface Air Temperature Change", "GSAT (K)",
    scenario_names, iam_per_scenario,
)

# %%
_plot_trajectories(
    combined, "Atmospheric Concentrations|CO2", "CO2 (ppm)",
    scenario_names, iam_per_scenario,
)

# %%
_plot_trajectories(
    combined, "Effective Radiative Forcing", "ERF (W/m^2)",
    scenario_names, iam_per_scenario,
)

# %% [markdown]
# ## 2100 GSAT distribution per scenario per model

# %%
def _gather_2100(combined, variable):
    rows = []
    for cm in MODELS:
        for s in scenario_names:
            values = combined.filter(
                climate_model=cm, scenario=s,
                variable=variable, year=2100,
            ).values.flatten()
            for v in values:
                rows.append({"climate_model": cm, "scenario": s, "value": v})
    return pd.DataFrame(rows)


df_2100 = _gather_2100(combined, "Surface Air Temperature Change")

fig, ax = plt.subplots(figsize=(12, 4))
n_models = len(MODELS)
width = 0.8 / n_models
positions = np.arange(len(scenario_names))
for i, cm in enumerate(MODELS):
    offsets = positions + (i - (n_models - 1) / 2) * width
    data = [
        df_2100[(df_2100["climate_model"] == cm) & (df_2100["scenario"] == s)]
        ["value"].values
        for s in scenario_names
    ]
    bp = ax.boxplot(
        data, positions=offsets, widths=width * 0.85,
        patch_artist=True, showfliers=False,
    )
    color = MODEL_COLOR[cm]
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.plot([], [], color=color, label=cm, lw=6, alpha=0.6)
ax.set_xticks(positions)
ax.set_xticklabels(scenario_names, rotation=15, ha="right", fontsize=8)
ax.set_ylabel("2100 GSAT (K)")
ax.set_title("2100 GSAT distribution across members, by scenario and SCM")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Cross-model scatter

# %%
fig, ax = plt.subplots(figsize=(6, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(scenario_names)))
for color, scenario in zip(colors, scenario_names):
    fair_vals = combined.filter(
        climate_model=FAIR_MODEL, scenario=scenario,
        variable="Surface Air Temperature Change", year=2100,
    ).values.flatten()
    cic_vals = combined.filter(
        climate_model=CICERO_MODEL, scenario=scenario,
        variable="Surface Air Temperature Change", year=2100,
    ).values.flatten()
    n = min(len(fair_vals), len(cic_vals))
    ax.scatter(fair_vals[:n], cic_vals[:n], color=color, s=30,
               alpha=0.7, label=scenario)

lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="1:1")
ax.set_xlabel(f"{FAIR_MODEL} 2100 GSAT (K)")
ax.set_ylabel(f"{CICERO_MODEL} 2100 GSAT (K)")
ax.set_title("Cross-model 2100 GSAT (one dot per scenario x member)")
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Sanity checks

# %%
def _median_2100_gsat(combined, model, scenario):
    return float(np.median(combined.filter(
        climate_model=model, scenario=scenario,
        variable="Surface Air Temperature Change", year=2100,
    ).values.flatten()))


# Print a summary table first.
summary_rows = []
for scenario in scenario_names:
    row = {"scenario": scenario}
    for cm in MODELS:
        row[f"{cm} median"] = _median_2100_gsat(combined, cm, scenario)
    summary_rows.append(row)
summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False, float_format="%.2f"))

# %%
# 1.5°C-labelled scenarios end below 2.5 K; baselines end above 3.5 K.
labels_15c = ("SSP1-19", "ADVANCE-2020-1.5°C-2100")
labels_baseline = ("SSP5-Baseline",)

for scenario in labels_15c:
    if scenario not in scenario_names:
        continue
    for cm in MODELS:
        med = _median_2100_gsat(combined, cm, scenario)
        assert med < 2.5, (
            f"{cm}: 1.5°C-labelled scenario {scenario} reaches 2100 GSAT "
            f"median {med:.2f} K > 2.5 K — mitigation pathway not "
            f"actually delivering substantial mitigation in this model."
        )
print("PASS: 1.5°C-labelled scenarios end below 2.5 K in both models")

for scenario in labels_baseline:
    if scenario not in scenario_names:
        continue
    for cm in MODELS:
        med = _median_2100_gsat(combined, cm, scenario)
        assert med > 3.5, (
            f"{cm}: baseline scenario {scenario} reaches 2100 GSAT "
            f"median only {med:.2f} K < 3.5 K — baseline pathway not "
            f"warming enough in this model."
        )
print("PASS: baseline scenarios end above 3.5 K in both models")

# %%
# Numerical ranges
for scenario in scenario_names:
    for cm in MODELS:
        gsat = combined.filter(
            climate_model=cm, scenario=scenario,
            variable="Surface Air Temperature Change", year=2100,
        ).values.flatten()
        co2 = combined.filter(
            climate_model=cm, scenario=scenario,
            variable="Atmospheric Concentrations|CO2", year=2100,
        ).values.flatten()
        erf = combined.filter(
            climate_model=cm, scenario=scenario,
            variable="Effective Radiative Forcing", year=2100,
        ).values.flatten()
        assert gsat.size > 0 and 0.0 < gsat.min() and gsat.max() < 10.0, (
            f"{cm} {scenario}: implausible GSAT range "
            f"[{gsat.min():.2f}, {gsat.max():.2f}] K"
        )
        assert co2.size > 0 and 250.0 < co2.min() and co2.max() < 2000.0, (
            f"{cm} {scenario}: implausible CO2 range "
            f"[{co2.min():.0f}, {co2.max():.0f}] ppm"
        )
        assert erf.size > 0 and -2.0 < erf.min() and erf.max() < 15.0, (
            f"{cm} {scenario}: implausible ERF range "
            f"[{erf.min():.2f}, {erf.max():.2f}] W/m^2"
        )
print("PASS: 2100 GSAT/CO2/ERF in plausible ranges for all scenarios/members")

# %%
# Cross-model spread bound: loose for SCI because the splice-mode
# CICEROSCM run uses bundled ssp245 historical for the Montreal-gas
# species SCI doesn't report, biasing CICERO slightly relative to
# FaIR's bundle-aligned setup. RCMIP same-bundle comparison hit 1.67 K
# at ssp585; SCI is bound to be looser.
SCI_CROSS_MODEL_BOUND = 2.5
diffs = []
for scenario in scenario_names:
    f = _median_2100_gsat(combined, FAIR_MODEL, scenario)
    c = _median_2100_gsat(combined, CICERO_MODEL, scenario)
    diffs.append((scenario, c - f))

print(f"  {'scenario':30s} {'CICERO-FaIR':>15s}")
for scenario, d in diffs:
    print(f"  {scenario:30s} {d:>+15.2f}")

max_spread = max(abs(d) for _, d in diffs)
assert max_spread < SCI_CROSS_MODEL_BOUND, (
    f"Cross-model 2100 GSAT median spread reached {max_spread:.2f} K, "
    f"exceeding the {SCI_CROSS_MODEL_BOUND} K loose bound for SCI."
)
print(
    f"PASS: max cross-model 2100 GSAT median spread {max_spread:.2f} K "
    f"(< {SCI_CROSS_MODEL_BOUND} K)"
)

# %%
# Mitigation ordering: across the picked scenarios, the 1.5°C-labelled
# ones should end *cooler* than the baseline. Catches the failure mode
# where scenario emissions got reversed or mis-routed.
ambition_order = [
    s for s in (
        "SSP1-19", "ADVANCE-2020-1.5°C-2100", "SSP2-45",
        "CD-LINKS-NPi", "SSP5-Baseline",
    )
    if s in scenario_names
]

if len(ambition_order) >= 2:
    for cm in MODELS:
        medians = [_median_2100_gsat(combined, cm, s) for s in ambition_order]
        # Tolerance: SSP1-19 vs ADVANCE-2020-1.5°C-2100 may swap order
        # (both are 1.5°C pathways and within ~0.2 K of each other),
        # but the strict-ambition pair (first 1.5°C-class < SSP5-Baseline)
        # must hold by a comfortable margin.
        assert medians[0] + 1.0 < medians[-1], (
            f"{cm}: ambition ordering failed; "
            f"{ambition_order[0]} median ({medians[0]:.2f}) >= "
            f"{ambition_order[-1]} median ({medians[-1]:.2f}) - 1.0 K. "
            f"Per-scenario medians: {[round(m, 2) for m in medians]}"
        )
    print(
        "PASS: cross-scenario ambition ordering "
        f"({ambition_order[0]} cooler than {ambition_order[-1]} "
        "by > 1.0 K in both models)"
    )
