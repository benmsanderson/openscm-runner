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
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # RCMIP3 protocol: cross-model cross-mode evaluation
#
# Reads the netCDF tree produced by ``scripts/run_rcmip3.py`` (one
# file per `(mode, model, scenario)` tuple under ``out/rcmip3/``) and
# produces four figure families demonstrating the multi-SCM AR7
# automation:
#
# 1. **Historical validation** — modelled 1850–2024 trajectories vs
#    the IGCC2024 / GCB2024 / AR6 constraint targets from Table 3 of
#    the RCMIP3 paper.
# 2. **Scenario projections** — 8 CMIP6 SSPs and 7 CMIP7 ``scen7-*``
#    markers side by side; emissions-driven and concentration-driven
#    overlaid so the carbon-cycle slack is visible.
# 3. **Idealised diagnostics** — ``esm-flat*`` constant-emissions
#    families showing TCRE convergence and the Zero Emissions
#    Commitment (ZEC) plateau after emissions stop.
# 4. **ED vs CD per scenario** — for the SSPs, paired emissions- and
#    concentration-driven runs of each model, so the
#    emissions-driven biases (in either model's carbon cycle) are
#    quantified against the prescribed-concentration "truth".
#
# Re-running the notebook does not invoke any SCM; it just reads the
# cached chunk files. Regenerate the cache via::
#
#     scripts/run_rcmip3.py --members 10 --scenario-set all --mode both
#
# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scmdata

import openscm_runner  # noqa: F401  (applies scmdata pandas-3 patches)
from openscm_runner.scenarios import CONSTRAINT_TARGETS

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
# DATA_ROOT: pick the most recent sweep that exists. Falls back to the
# original out/rcmip3 (Phase B baseline) if no rearch sweeps are present.
_SWEEP_CANDIDATES = (
    "rcmip3_issue45fair_full",  # All five fixes (loader mixed-mode +
                                # adapter consumption + LUC/VOLC/solar
                                # stripping + idealised PI-flat
                                # emissions on both adapters)
    "rcmip3_issue45_full",      # All except FaIR idealised fix
    "rcmip3_bpartial",          # B-partial only
    "rcmip3_step5ab",           # PR #12 baseline (post-rearch + bundle-name strip)
    "rcmip3",                   # Phase B baseline (pre-rearch)
)
DATA_ROOT = next(
    (REPO_ROOT / "out" / s for s in _SWEEP_CANDIDATES
     if (REPO_ROOT / "out" / s / "emissions" / "CICERO-SCM-PY2").is_dir()
     and any((REPO_ROOT / "out" / s / "emissions" / "CICERO-SCM-PY2").iterdir())),
    REPO_ROOT / "out" / "rcmip3",
)
print(f"DATA_ROOT = {DATA_ROOT.relative_to(REPO_ROOT)}")
FIGURE_DIR = REPO_ROOT / "notebooks" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Marit Sandstad's native CICERO-SCM RCMIP3 reference submissions.
# 500-member ensembles per scenario, one CSV per scenario, no header.
MARIT_DIR = REPO_ROOT / "configurations" / "ciceroscm" / "marittmp"
MARIT_VARIABLE = "Surface Air Ocean Blended Temperature Change"

MODELS = ("FaIRv2", "CICERO-SCM-PY2")
MODES = ("emissions", "concentrations")

# Colour palette: each (model, mode) gets its own line so eight bands
# can co-exist without confusion. Model picks the hue, mode picks the
# saturation/style.
MODEL_HUE = {"FaIRv2": "tab:blue", "CICERO-SCM-PY2": "tab:red"}
MODE_LINESTYLE = {"emissions": "-", "concentrations": "--"}
MODE_ALPHA = {"emissions": 1.0, "concentrations": 0.7}

# Scenario groups for the panel grids.
SSPS = (
    "ssp119", "ssp126", "ssp245", "ssp370",
    "ssp434", "ssp460", "ssp534-over", "ssp585",
)
SCEN7 = ("scen7-VL", "scen7-LN", "scen7-L", "scen7-ML",
         "scen7-M", "scen7-H", "scen7-HL")
FLAT_BASES = ("esm-flat7.5", "esm-flat10", "esm-flat20")


def _save(fig, name: str) -> Path:
    """Save figure under notebooks/figures/ with a consistent prefix."""
    path = FIGURE_DIR / f"compare_rcmip3_{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  wrote {path.relative_to(REPO_ROOT)}")
    return path


# %% [markdown]
# ## Load the output tree
#
# Lazy, file-by-file: each chunk is the full output of one
# ``(mode, model, scenario)`` run. We do *not* concatenate everything
# into one ScmRun (that would lose mode info anyway since both modes
# share the same scmdata metadata schema).

# %%
def _chunk_path(mode: str, model: str, scenario: str) -> Path:
    """Mirror NetCDFChunkWriter's filename convention."""
    import re
    safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)) or "unnamed"
    return DATA_ROOT / mode / safe(model) / f"{safe(scenario)}.nc"


def load_chunk(mode: str, model: str, scenario: str) -> scmdata.ScmRun | None:
    """Read one chunk, or return None if it isn't on disk."""
    path = _chunk_path(mode, model, scenario)
    if not path.exists():
        return None
    return scmdata.ScmRun.from_nc(path)


def available_scenarios(mode: str, model: str) -> list[str]:
    """List scenarios with a chunk on disk for one (mode, model)."""
    import re
    safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)) or "unnamed"
    model_dir = DATA_ROOT / mode / safe(model)
    if not model_dir.is_dir():
        return []
    return sorted(p.stem for p in model_dir.glob("*.nc"))


print("Output tree contents:")
for mode in MODES:
    for model in MODELS:
        scens = available_scenarios(mode, model)
        print(f"  {mode}/{model}: {len(scens)} scenarios")
        if scens:
            print(f"    {', '.join(scens[:6])}"
                  + (f", … (+{len(scens) - 6} more)" if len(scens) > 6 else ""))


# %% [markdown]
# ## Plotting helpers
#
# All figures use the same ensemble-band convention: median line plus
# a 5-95% percentile band. Models distinguished by hue, modes by
# linestyle.

# %%
def _band(ax, run: scmdata.ScmRun, variable: str, *,
          color: str, linestyle: str, alpha: float,
          label: str, year_range: tuple[int, int] | None = None) -> None:
    """Plot a 5-95% percentile band + median line for one variable.

    ``year_range`` (optional) clips the timeseries before plotting so
    the y-axis auto-scale doesn't include trailing years that fall
    outside the visible x-range.
    """
    sub = run.filter(variable=variable)
    if sub.empty:
        return
    ts = sub.timeseries(time_axis="year")
    years = ts.columns.astype(int).values
    if year_range is not None:
        mask = (years >= year_range[0]) & (years <= year_range[1])
        years = years[mask]
        ts = ts.iloc[:, mask]
    median = ts.median(axis=0).values
    q05 = ts.quantile(0.05, axis=0).values
    q95 = ts.quantile(0.95, axis=0).values
    ax.fill_between(years, q05, q95, color=color, alpha=0.15 * alpha)
    ax.plot(years, median, color=color, lw=1.6, linestyle=linestyle,
            alpha=alpha, label=label)


def _overlay_constraint(ax, key: str) -> None:
    """Shade the constraint period and overlay the IGCC/GCB target band."""
    ct = CONSTRAINT_TARGETS[key]
    yspan = ct.constraint_period
    ax.axvspan(yspan[0], yspan[1], color="grey", alpha=0.08)
    # Horizontal target band only really makes sense for scalar
    # quantities; for OHC the constraint is a single-year delta.
    ax.axhspan(ct.lower, ct.upper, color="black", alpha=0.08)
    ax.axhline(ct.central, color="black", lw=1.0, linestyle=":",
               label=f"{ct.source} target")


# %% [markdown]
# ## Figure 1 — Historical validation against IGCC2024 / GCB2024
#
# Reads the ``historical`` scenario from each (mode, model) and
# overlays the four scalar constraint targets from Table 3 of the
# RCMIP3 paper:
#
# | Target            | Source    | Constraint period | Value           |
# |-------------------|-----------|-------------------|-----------------|
# | GMST anomaly      | IGCC2024  | 2014-2023         | 1.19 [1.0, 1.4] K |
# | OHC change        | IGCC2024  | 1971 → 2020       | 435 [316, 561] ZJ |
# | CO2 concentration | NOAA GML  | 2014-2023         | 408.65 [407, 410] ppm |
# | Aerosol ERF       | AR6       | 2005-2014         | -1.3 [-2.0, -0.6] W/m² |

# %%
def plot_historical_panels():
    # 3 panels in a 1x3 row: GMST, CO2, OHC. Aerosol ERF dropped for
    # now since CICEROSCM's _OUTPUT_VARIABLES only exposes the bulk
    # ERF; the per-component breakdown (Anthropogenic|Aerosol) would
    # need an upstream / adapter extension.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panel_specs = [
        ("GMST_anomaly", "Surface Air Temperature Change",
         "GMST anomaly (K)", (-0.5, 2.0), axes[0]),
        ("CO2_concentration", "Atmospheric Concentrations|CO2",
         "Atmospheric CO2 (ppm)", (260, 450), axes[1]),
        ("OHC_change", "Heat Content|Ocean",
         "Ocean Heat Content (ZJ)", (-50, 800), axes[2]),
    ]
    x_range = (1850, 2024)
    for ct_key, variable, ylabel, ylim, ax in panel_specs:
        for mode in MODES:
            for model in MODELS:
                run = load_chunk(mode, model, "historical")
                if run is None:
                    continue
                _band(ax, run, variable,
                      color=MODEL_HUE[model],
                      linestyle=MODE_LINESTYLE[mode],
                      alpha=MODE_ALPHA[mode],
                      label=f"{model} ({mode})",
                      year_range=x_range)
        _overlay_constraint(ax, ct_key)
        ax.set_xlim(*x_range)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Year")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Historical validation against RCMIP3 Table 3 constraints "
        "(IGCC2024, GCB2024)",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "historical_validation")
    return fig


fig_hist = plot_historical_panels()
plt.show()


# %% [markdown]
# ## Figure 2 — Scenario projections (CMIP6 SSPs + CMIP7 scen7)
#
# Three rows: GSAT, atmospheric CO2 concentration, total ERF.
# Eight + seven cols: one column per scenario. Both adapters in
# both modes, so every cell holds up to four bands.
#
# Where a model is concentration-driven, atmospheric CO2 collapses
# to a single line (the prescribed trajectory); that's a useful
# visual: the spread is whatever the model's *forcing* response
# produces given identical input.

# %%
def _scenario_grid(scenarios: Iterable[str], title: str, tag: str):
    scenarios = list(scenarios)
    if not scenarios:
        print(f"  (skipping {tag}: no scenarios)")
        return None
    nrows, ncols = 3, len(scenarios)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.8 * ncols, 2.6 * nrows),
        sharex=True, sharey="row",
    )
    if ncols == 1:
        axes = np.array(axes).reshape(nrows, 1)
    row_specs = [
        ("Surface Air Temperature Change", "GSAT (K)"),
        ("Atmospheric Concentrations|CO2", "CO2 (ppm)"),
        ("Effective Radiative Forcing", "ERF (W/m²)"),
    ]
    x_range = (1980, 2100)
    for col_idx, scenario in enumerate(scenarios):
        for row_idx, (variable, ylabel) in enumerate(row_specs):
            ax = axes[row_idx, col_idx]
            for mode in MODES:
                for model in MODELS:
                    run = load_chunk(mode, model, scenario)
                    if run is None:
                        continue
                    _band(ax, run, variable,
                          color=MODEL_HUE[model],
                          linestyle=MODE_LINESTYLE[mode],
                          alpha=MODE_ALPHA[mode],
                          label=f"{model} ({mode})",
                          year_range=x_range)
            ax.grid(alpha=0.3)
            ax.set_xlim(*x_range)
            if row_idx == 0:
                ax.set_title(scenario, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == nrows - 1:
                ax.set_xlabel("Year")
    # Single legend for the whole grid.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(title, y=1.06)
    fig.tight_layout()
    _save(fig, f"scenarios_{tag}")
    return fig


fig_ssps = _scenario_grid(SSPS, "CMIP6 SSPs — emissions vs concentration driven", "ssps")
plt.show()


# %%
fig_scen7 = _scenario_grid(
    SCEN7, "CMIP7 scen7-* markers — emissions vs concentration driven", "scen7",
)
plt.show()


# %% [markdown]
# ## Figure 3 — Idealised diagnostics (esm-flat* family)
#
# These are emissions-driven by construction (constant CO2 emissions
# trajectory; concentrations are a model diagnostic, not an input).
#
# Two views:
#
# 1. **TCRE convergence**: each flat-emissions scenario should
#    produce a near-linear cumulative-CO2 vs GSAT relationship over
#    the constant-emissions phase. Slope = TCRE (K per 1000 GtC).
# 2. **ZEC plateau**: after the constant-emissions phase ends (at
#    year 100 in the protocol), GSAT should plateau or slowly drop.
#    The shape of the post-stop trajectory is the model's ZEC.

# %%
def plot_flat_family():
    # GSAT vs cumulative emissions for the three flat-* base cases.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: TCRE convergence on the plain flat-* (no -zec suffix)
    # scenarios. X-axis is the input cumulative emissions (the protocol's
    # constant trajectory: 7.5, 10, 20 PgC/yr starting at 1850), NOT a
    # model diagnostic — Net Flux to Atmosphere is the residual after
    # ocean+land uptake and has a sign convention that doesn't track
    # cumulative anthropogenic emissions.
    ax_tcre = axes[0]
    flat_emission_rate = {  # PgC/yr, constant from 1850 onwards
        "esm-flat7.5": 7.5, "esm-flat10": 10.0, "esm-flat20": 20.0,
    }
    for base in FLAT_BASES:
        rate = flat_emission_rate[base]
        for model in MODELS:
            run = load_chunk("emissions", model, base)
            if run is None:
                continue
            ts_t = run.filter(
                variable="Surface Air Temperature Change",
            ).timeseries(time_axis="year")
            if ts_t.empty:
                continue
            years = ts_t.columns.astype(int).values
            # Cumulative input emissions: rate * (year - 1850),
            # clipped to zero before 1850.
            cum = np.clip(years - 1850, 0, None) * rate
            gsat = ts_t.median(axis=0).values
            mask = cum >= 50.0
            ax_tcre.plot(
                cum[mask] / 1000.0, gsat[mask],
                color=MODEL_HUE[model],
                linestyle=("-" if base == "esm-flat10" else
                           ":" if base == "esm-flat7.5" else "--"),
                lw=1.5, label=f"{model} {base}",
            )
    ax_tcre.set_xlabel("Cumulative input emissions (1000 PgC = TtC)")
    ax_tcre.set_ylabel("GSAT (K)")
    ax_tcre.set_title("TCRE: cumulative input emissions vs GSAT (esm-flat*)")
    ax_tcre.legend(fontsize=7)
    ax_tcre.grid(alpha=0.3)

    # Right panel: GSAT timeseries showing the ZEC plateau.
    ax_zec = axes[1]
    for base in FLAT_BASES:
        scen = f"{base}-zec"
        for model in MODELS:
            run = load_chunk("emissions", model, scen)
            if run is None:
                continue
            _band(ax_zec, run, "Surface Air Temperature Change",
                  color=MODEL_HUE[model],
                  linestyle=("-" if base == "esm-flat10" else
                             ":" if base == "esm-flat7.5" else "--"),
                  alpha=1.0,
                  label=f"{model} {base}")
    ax_zec.axvline(1850 + 100, color="black", lw=0.8, linestyle="-",
                   alpha=0.5, label="emissions stop")
    ax_zec.set_xlim(1850, 2450)
    ax_zec.set_xlabel("Year")
    ax_zec.set_ylabel("GSAT (K)")
    ax_zec.set_title("Zero-emissions commitment after year 100")
    ax_zec.legend(fontsize=7)
    ax_zec.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "flat_family_diagnostics")
    return fig


fig_flat = plot_flat_family()
plt.show()


# %% [markdown]
# ## Figure 4 — ED vs CD per scenario
#
# For each model and each SSP, paired emissions-driven and
# concentration-driven runs. The CO2 panel makes the asymmetry most
# visible: CD runs collapse to a single concentration trajectory
# (5-95% band has zero width because every ensemble member sees the
# same prescribed CO2). GSAT spread under CD reflects only the
# climate sensitivity / aerosol-forcing axes of the calibration; ED
# spread additionally folds in carbon-cycle uncertainty.

# %%
def plot_ed_vs_cd():
    nrows, ncols = len(MODELS), 3
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(13, 3.2 * nrows),
        sharex=True,
    )
    row_specs = [
        ("Surface Air Temperature Change", "GSAT (K)"),
        ("Atmospheric Concentrations|CO2", "CO2 (ppm)"),
        ("Effective Radiative Forcing", "ERF (W/m²)"),
    ]
    # Only the SSPs for legibility; could add scen7 too if you want.
    selected_ssps = ("ssp126", "ssp245", "ssp585")
    palette = {"ssp126": "tab:green", "ssp245": "tab:olive", "ssp585": "tab:red"}
    x_range = (2000, 2100)
    for row_idx, model in enumerate(MODELS):
        for col_idx, (variable, ylabel) in enumerate(row_specs):
            ax = axes[row_idx, col_idx]
            for scen in selected_ssps:
                for mode in MODES:
                    run = load_chunk(mode, model, scen)
                    if run is None:
                        continue
                    _band(ax, run, variable,
                          color=palette[scen],
                          linestyle=MODE_LINESTYLE[mode],
                          alpha=MODE_ALPHA[mode],
                          label=f"{scen} ({mode})",
                          year_range=x_range)
            ax.grid(alpha=0.3)
            ax.set_xlim(*x_range)
            if row_idx == 0:
                ax.set_title(variable.split("|")[-1], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{model}\n{ylabel}")
            if row_idx == nrows - 1:
                ax.set_xlabel("Year")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=6, fontsize=7, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(
        "Emissions-driven (solid) vs concentration-driven (dashed) — "
        "selected SSPs per model",
        y=1.05,
    )
    fig.tight_layout()
    _save(fig, "ed_vs_cd")
    return fig


fig_ed_cd = plot_ed_vs_cd()
plt.show()


# %% [markdown]
# ## Figure 5 — Validation against Marit Sandstad's native CICERO-SCM reference
#
# The 97 RCMIP3 scenarios that Marit submitted to gitlab.com/rcmip/rcmip-phase-3
# are our scorecard target — they're the only published native-CICEROSCM
# RCMIP3 reference. For each scenario, Marit ran 500 ensemble members
# from ``draw_samples_500.json`` against her native CICEROSCM pipeline;
# we run the matched 10 members (same run_ids, deterministic seeded)
# through ``openscm_runner.adapters.ciceroscm_py2_adapter``.
#
# The figures below show the per-year GSAT diff (ours − Marit, both
# medians over their respective ensembles) for representative scenario
# families. Horizontal bands mark the scorecard thresholds:
# **PASS** if max |diff| ≤ 0.05 K, **WARN** if ≤ 0.15 K, **FAIL** above.

# %%
def load_marit_reference(scenario: str):
    """Read Marit's 500-member reference, indexed by run_id.

    Returns a DataFrame whose index is the per-row run_id (from the
    CSV's 4th column) and whose columns are years 1750-2500. Mirrors
    ``scripts/validate_against_marit.py:_load_marit`` so the
    notebook's matched-member diff matches the scorecard.
    """
    import pandas as pd
    path = MARIT_DIR / f"{scenario}_rcmip_draw_samples_500.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None
    raw = pd.read_csv(path, header=None)
    n_year_cols = raw.shape[1] - 8
    years = list(range(1750, 1750 + n_year_cols))
    data = raw.iloc[:, 8:]
    data.columns = years
    is_var = raw.iloc[:, 6].values == MARIT_VARIABLE
    out = data[is_var].copy()
    out.index = raw.iloc[:, 3][is_var].values
    out.index.name = "run_id"
    return out


def _our_chunk_for_marit_compare(scenario: str):
    """Pick the right (mode, model) chunk per Marit's protocol convention."""
    s = scenario.lower()
    mode = (
        "emissions" if s.startswith("esm-")
        or s in ("hist-aer", "hist-co2", "hist-ghg")
        else "concentrations"
    )
    return load_chunk(mode, "CICERO-SCM-PY2", scenario)


def diff_against_marit(scenario: str):
    """Matched-member median diff (our - Marit) over the year intersection.

    Mirrors the scorecard's per-run_id matching: pull the matched
    subset of Marit's 500-member ensemble against our 10-member ensemble
    and take the median over the matched runs only. This is the same
    statistic the scorecard reports as ``max|med dif|``.
    """
    marit = load_marit_reference(scenario)
    if marit is None:
        return None
    our_run = _our_chunk_for_marit_compare(scenario)
    if our_run is None:
        return None
    sub = our_run.filter(variable=MARIT_VARIABLE)
    if sub.empty:
        return None
    our_ts = sub.timeseries(time_axis="year")
    our_ts.index = our_ts.index.get_level_values("run_id")
    matched = sorted(set(our_ts.index) & set(marit.index))
    if not matched:
        return None
    common_years = sorted(set(our_ts.columns) & set(marit.columns))
    if not common_years:
        return None
    m = marit.loc[matched, common_years].median(axis=0).values
    o = our_ts.loc[matched, common_years].median(axis=0).values
    return np.array(common_years), o - m


def plot_marit_diff_panel(scenarios, title, tag, *, palette=None, x_range=(1850, 2500)):
    """One axes; each scenario gets one line of (ours - Marit) over time."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    skipped = []
    for i, scen in enumerate(scenarios):
        result = diff_against_marit(scen)
        if result is None:
            skipped.append(scen)
            continue
        years, diff = result
        color = palette[scen] if palette and scen in palette else f"C{i % 10}"
        ax.plot(years, diff, lw=1.2, alpha=0.9, label=scen, color=color)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhspan(-0.05, 0.05, color="green", alpha=0.08, label="PASS band (+/-50 mK)")
    for thr, label in ((0.15, "FAIL threshold (+/-150 mK)"), (-0.15, None)):
        ax.axhline(thr, color="red", lw=0.7, linestyle="--", alpha=0.5,
                   label=label)
    ax.set_xlim(*x_range)
    ax.set_xlabel("Year")
    ax.set_ylabel("GSAT diff: ours - Marit's reference (K)")
    # Annotate the data source so figures from different sweeps are
    # distinguishable when they live side-by-side on disk.
    ax.set_title(f"{title}\n[source: {DATA_ROOT.relative_to(REPO_ROOT)}]", fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(alpha=0.3)
    if skipped:
        print(f"  skipped (missing data): {skipped}")
    fig.tight_layout()
    _save(fig, f"marit_diff_{tag}")
    return fig


# %% [markdown]
# ### 5a — esm-ssp* family (the rearch's biggest unlock)
#
# Pre-rearch (Phase B): every ``esm-ssp*`` scenario FAILed at 4-9 K
# max diff, driven by the CICEROSCM bundle resolver falling back to
# ``historical_em_*`` instead of finding ``ssp245_em_*`` (the bundle
# uses bare names). After the rearch's bundle-name stripping (step 5b)
# and B-partial's proper mixed-mode loader, these collapse to the
# ``esm-allGHG-`` siblings — matching Marit's bit-identical pair
# convention.

# %%
fig_marit_essp = plot_marit_diff_panel(
    ["esm-ssp119", "esm-ssp126", "esm-ssp245", "esm-ssp370",
     "esm-ssp434", "esm-ssp460", "esm-ssp534-over", "esm-ssp585"],
    "Marit diff — ED CO2-only SSPs (esm-ssp*)",
    "esm_ssps",
)
plt.show()


# %% [markdown]
# ### 5b — scen7-*C family (CD variants)
#
# Pre-rearch: 1-6 K max diffs, driven by the bundle resolver looking
# for ``scen7-HC_conc_*`` (doesn't exist) instead of ``scen7-H_conc_*``.
# Step 5b's name stripping pulls these into WARN at ~130 mK.

# %%
fig_marit_scen7c = plot_marit_diff_panel(
    ["scen7-HC", "scen7-HLC", "scen7-LC", "scen7-LNC",
     "scen7-MC", "scen7-MLC", "scen7-VLC"],
    "Marit diff — CD scen7-*C variants",
    "scen7_cd",
)
plt.show()


# %% [markdown]
# ### 5c — Historical and piControl
#
# Validation that real-world / control runs match. Historical,
# hist-* attribution runs should all be within ±50 mK of Marit.
#
# The three piControl variants show a positive bias peaking ~+0.4 K
# around 1980 in the diff. That bias is in **Marit's** reference,
# not ours: our piControl trajectory is bit-exact zero throughout
# (perfect PI control after the step-4b idealised-suppression set
# plus the post-PR-#12 ``esm-piControl_em_`` fallback for emissions),
# while Marit's pipeline leaks historical aerosol-precursor
# emissions into her piControl run, producing a cooling that peaks
# around the 1980 SO2 maximum and recovers as aerosols decline.
# The diff plot therefore traces the inverse of Marit's drift, not
# ours — the +0.4 K peak at 1980 is the historical SO2 aerosol
# forcing leaking into a run that should be pure pre-industrial.

# %%
fig_marit_hist = plot_marit_diff_panel(
    ["historical", "historical-cmip6", "hist-aer", "hist-CO2", "hist-GHG",
     "piControl", "esm-piControl", "esm-allGHG-piControl",
     "esm-hist", "esm-allGHG-hist"],
    "Marit diff — historical, attribution, and piControl variants",
    "historical_and_picontrol",
)
plt.show()


# %% [markdown]
# ### 5d — Idealised CD experiments (residual FAILs)
#
# 1pctCO2 and abrupt-* runs still show 0.5-1.0 K diffs (FAIL). The
# sign pattern (less warming under high CO2, less cooling under 0.5x)
# is consistent with a different ECS/TCR distribution vs Marit's
# pipeline. Not addressed by PR #12; documented as future work.

# %%
fig_marit_ideal = plot_marit_diff_panel(
    ["1pctCO2", "1pctCO2-4xext", "1pctCO2-cdr",
     "abrupt-0p5xCO2", "abrupt-2xCO2", "abrupt-4xCO2"],
    "Marit diff — CD idealised (residual ECS/TCR discrepancy)",
    "cd_idealised",
)
plt.show()


# %% [markdown]
# ## Summary
#
# All five figure families read from the same on-disk cache produced
# by ``scripts/run_rcmip3.py --members 10 --scenario-set all --mode both``.
# Re-running the cells regenerates figures without re-running models.
#
# The headline artefacts saved to ``notebooks/figures/`` for the
# RCMIP3 community / IPCC AR7 audience:
#
# - ``compare_rcmip3_historical_validation.png``
# - ``compare_rcmip3_scenarios_ssps.png``
# - ``compare_rcmip3_scenarios_scen7.png``
# - ``compare_rcmip3_flat_family_diagnostics.png``
# - ``compare_rcmip3_ed_vs_cd.png``
# - ``compare_rcmip3_marit_diff_esm_ssps.png``  (NEW: validation)
# - ``compare_rcmip3_marit_diff_scen7_cd.png``  (NEW: validation)
# - ``compare_rcmip3_marit_diff_historical_and_picontrol.png``  (NEW)
# - ``compare_rcmip3_marit_diff_cd_idealised.png``  (NEW)
