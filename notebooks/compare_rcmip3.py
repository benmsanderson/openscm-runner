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
DATA_ROOT = REPO_ROOT / "out" / "rcmip3"
FIGURE_DIR = REPO_ROOT / "notebooks" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

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
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    # Per-panel y-limits keep the constraint band readable when the
    # underlying scenario projects well past the historical period
    # (matplotlib auto-scale would otherwise stretch the axis to the
    # 2500 endpoint that the data carries but we don't plot).
    panel_specs = [
        ("GMST_anomaly", "Surface Air Temperature Change",
         "GMST anomaly (K)", (-0.5, 2.0), axes[0, 0]),
        ("CO2_concentration", "Atmospheric Concentrations|CO2",
         "Atmospheric CO2 (ppm)", (260, 450), axes[0, 1]),
        ("OHC_change", "Heat Content|Ocean",
         "Ocean Heat Content (ZJ)", (-50, 800), axes[1, 0]),
        ("aerosol_ERF",
         "Effective Radiative Forcing|Anthropogenic|Aerosol",
         "Aerosol ERF (W/m²)", (-2.2, 0.2), axes[1, 1]),
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
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    axes[-1, 0].set_xlabel("Year")
    axes[-1, 1].set_xlabel("Year")
    fig.suptitle(
        "Historical validation against RCMIP3 Table 3 constraints "
        "(IGCC2024, GCB2024, AR6)",
        y=1.0,
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
    # scenarios — these have constant emissions for the full 300 years
    # so the cumulative-vs-GSAT trajectory keeps growing rather than
    # plateauing.
    ax_tcre = axes[0]
    for base in FLAT_BASES:
        for model in MODELS:
            run = load_chunk("emissions", model, base)
            if run is None:
                continue
            ts_t = run.filter(
                variable="Surface Air Temperature Change",
            ).timeseries(time_axis="year")
            ts_e = run.filter(
                variable="Net Flux to Atmosphere|CO2",
            ).timeseries(time_axis="year")
            if ts_e.empty:
                continue
            # Cumulative net flux to atmosphere ≈ cumulative emissions
            # for the flat-* family (no negative natural fluxes baked in).
            cum = ts_e.cumsum(axis=1).median(axis=0).values
            gsat = ts_t.median(axis=0).values
            # Drop the pre-experiment spin-up rows where cumulative
            # is effectively zero (these compress the visible range).
            mask = cum > 50.0
            ax_tcre.plot(
                cum[mask] / 1000.0, gsat[mask],
                color=MODEL_HUE[model],
                linestyle=("-" if base == "esm-flat10" else
                           ":" if base == "esm-flat7.5" else "--"),
                lw=1.5, label=f"{model} {base}",
            )
    ax_tcre.set_xlabel("Cumulative net flux to atmosphere (kgCO2 × 10⁻³ ≈ PgC)")
    ax_tcre.set_ylabel("GSAT (K)")
    ax_tcre.set_title("TCRE: cumulative emissions vs GSAT (esm-flat*)")
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
# ## Summary
#
# All four figure families read from the same on-disk cache produced
# by ``scripts/run_rcmip3.py --members 10 --scenario-set all --mode both``.
# Re-running the cells regenerates figures without re-running models.
#
# The four artefacts saved to ``notebooks/figures/`` are the
# headline outputs to share with the RCMIP3 community:
#
# - ``compare_rcmip3_historical_validation.png``
# - ``compare_rcmip3_scenarios_ssps.png``
# - ``compare_rcmip3_scenarios_scen7.png``
# - ``compare_rcmip3_flat_family_diagnostics.png``
# - ``compare_rcmip3_ed_vs_cd.png``
