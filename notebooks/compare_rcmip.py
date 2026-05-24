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
# # RCMIP single-scenario cross-model comparison: FaIRv2 vs CICEROSCMPY2
#
# Takes one RCMIP scenario (default ``ssp119`` — the first scenario in
# the fixture; override via the ``SCENARIO`` env var) and runs it
# through both adapters with a **10-member subset** of each model's
# calibration posterior. The subset is small enough that the whole
# notebook runs in under a minute on a laptop, which is the right
# default for plot-style iteration and quick cross-model sanity
# checks.
#
# Set ``N_MEMBERS=full`` to drive the entire calibration posterior
# (841 FaIR / 500 CICEROSCM) — that takes ~12 minutes wallclock and
# is the right depth for uncertainty characterisation (5-95% bands,
# distribution shape, tail behaviour). Any other integer value picks
# that many members.
#
# CICEROSCM bundle vs splice
# --------------------------
#
# ``CICERO_MODE=bundle`` (default) points the adapter at
# ``$CICEROSCMPY2_BUNDLE_DIR/rcmip-march2026/`` — the Marit-RCMIP
# input bundle the ``draw_samples_500`` calibration was actually fit
# against. ``CICERO_MODE=splice`` uses the legacy v1.1.x splice path,
# which carries a documented ~0.3-0.5 K present-day warm bias
# (see PR #8 description for the diagnosis).
#
# Sanity checks performed
# -----------------------
#
# 1. 2100 GSAT, CO2 concentration, and ERF fall in plausible numerical
#    ranges across the ensemble.
# 2. For mitigation scenarios (``ssp119``, ``ssp126``, ``ssp534-over``):
#    the GSAT median peaks before run end.
# 3. For high-emission scenarios (``ssp370``, ``ssp585``): GSAT
#    decadal-mean is monotone over the projection period.
# 4. Cross-model 2100 GSAT median spread is reported but not asserted:
#    the two bundles have different ECS distributions, and the gap is
#    information, not a regression to alarm on.
# %% [markdown]
# ## Configuration

# %%
import multiprocessing as mp
import os
from pathlib import Path

# macOS defaults to 'spawn', which re-imports the entrypoint module in
# every worker. That works fine in a notebook kernel but breaks when
# the .py representation is run as a script (the re-import attempts to
# spawn its own pool, recursing). Force 'fork' so both invocation paths
# behave the same.
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("fork", force=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scmdata

import openscm_runner.run
from openscm_runner.adapters import CICEROSCMPY2, FAIR2

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

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
RCMIP_FIXTURE = REPO_ROOT / "tests" / "test-data" / "rcmip_scen_ssp_world_emissions.csv"

# Where to cache results and write figures. The cache lets plot
# iterations skip the model re-runs (set REGEN_CACHE=1 to force a
# re-run, e.g. after pulling a new calibration bundle).
CACHE_DIR = REPO_ROOT / "notebooks" / ".cache"
FIGURE_DIR = REPO_ROOT / "notebooks" / "figures"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REGEN_CACHE = bool(int(os.environ.get("REGEN_CACHE", "0")))

# Scenario + ensemble size. Defaults: ssp119, 10 members per model
# (fast iteration). Set N_MEMBERS=full to run the entire calibration
# posterior (841 FaIR / 500 CICEROSCM, ~12 min wallclock).
SCENARIO = os.environ.get("SCENARIO", "ssp119")
_n_members_env = os.environ.get("N_MEMBERS", "10")
try:
    N_MEMBERS: int | None = int(_n_members_env)
    MEMBER_TAG = f"n{N_MEMBERS}"
except ValueError:
    N_MEMBERS = None
    MEMBER_TAG = "nfull"

# CICEROSCM input mode. Default is "bundle" — the Marit-RCMIP-aligned
# path the draw_samples_500 calibration was fit against, which
# reproduces the reference protocol bit-exactly. "splice" is the
# legacy v1.1.x path; it carries a ~0.3-0.5 K present-day warm bias
# because the bundled ssp245 historical doesn't match v2.x
# calibrations (see PR #8 for the diagnosis).
CICERO_MODE = os.environ.get("CICERO_MODE", "bundle")
assert CICERO_MODE in ("splice", "bundle"), (
    f"CICERO_MODE must be 'splice' or 'bundle' (got {CICERO_MODE!r})"
)
CICERO_BUNDLE_RCMIP = CICERO_BUNDLE_DIR / "rcmip-march2026"


def _cache_path(model_short, scenario, extra_tag=""):
    suffix = f"_{extra_tag}" if extra_tag else ""
    name = f"compare_rcmip_{model_short}_{scenario}_{MEMBER_TAG}{suffix}.pkl"
    return CACHE_DIR / name


def _cached_run(cache_path, run_fn, label):
    # Local-only cache: this notebook is both writer and reader, so
    # the security warnings about pickle deserialisation of untrusted
    # data do not apply (ruff's notebooks/* per-file-ignores handle
    # the lint).
    if cache_path.exists() and not REGEN_CACHE:
        print(f"  [cache hit] loading {label} from {cache_path.name}")
        return _load_from_cache(cache_path)
    print(f"  [cache miss] running {label}; will write {cache_path.name}")
    result = run_fn()
    result.timeseries().to_pickle(cache_path)
    return result


def _load_from_cache(cache_path):
    ts = pd.read_pickle(cache_path)
    return scmdata.ScmRun(ts.reset_index())


# Mitigation scenarios (peak-and-decline expected) vs high-emission
# scenarios (monotone warming expected). Used to pick the right
# shape sanity check for SCENARIO.
MITIGATION_SCENARIOS = {"ssp119", "ssp126", "ssp534-over"}
HIGH_EMISSION_SCENARIOS = {"ssp370", "ssp585"}

OUTPUT_VARIABLES = (
    "Surface Air Temperature Change",
    "Atmospheric Concentrations|CO2",
    "Effective Radiative Forcing",
)

assert FAIR2_BUNDLE.exists(), f"FaIR2 calibration not at {FAIR2_BUNDLE}"
assert CICERO_BUNDLE_DIR.exists(), f"CICERO bundle dir not at {CICERO_BUNDLE_DIR}"
assert RCMIP_FIXTURE.exists(), f"RCMIP fixture not at {RCMIP_FIXTURE}"

print("Adapters resolved:")
print(f"  FaIRv2:         v{FAIR2.get_version()}")
print(f"  CICERO-SCM-PY2: v{CICEROSCMPY2.get_version()}")
print(f"  Scenario:       {SCENARIO}")
print(f"  Members:        {MEMBER_TAG} ({N_MEMBERS or 'full posterior'} per model)")
print(f"  CICERO mode:    {CICERO_MODE}")

# %% [markdown]
# ## Load scenario

# %%
scenarios = scmdata.ScmRun(RCMIP_FIXTURE, lowercase_cols=True).filter(scenario=SCENARIO)
assert not scenarios.empty, (
    f"Scenario {SCENARIO!r} not in the RCMIP fixture. Set SCENARIO to one of: "
    f"{sorted(scmdata.ScmRun(RCMIP_FIXTURE, lowercase_cols=True)['scenario'].unique())}"
)
print(f"Loaded {scenarios.shape[0]} input timeseries for {SCENARIO}")

# %% [markdown]
# ## Run FaIRv2
#
# ``N_MEMBERS`` from the configuration above selects an ensemble
# subset; ``N_MEMBERS=full`` runs every row of the calibration's
# parameter posterior (841 members for ``fair-calibrate-v1.6.0``).

# %%
def _fair_cfg():
    cfg = {
        "native_calibration": str(FAIR2_BUNDLE),
        "emissions_bundle": str(FAIR2_BUNDLE),
    }
    if N_MEMBERS is not None:
        cfg["member_indices"] = range(N_MEMBERS)
    return cfg


def _run_fair():
    return openscm_runner.run.run(
        climate_models_cfgs={"FaIRv2": [_fair_cfg()]},
        scenarios=scenarios,
        output_variables=OUTPUT_VARIABLES,
        out_config=None,
    )


fair_result = _cached_run(
    _cache_path("fair2", SCENARIO), _run_fair, f"FaIRv2 ({MEMBER_TAG})"
)
n_fair = len(fair_result.filter(
    variable="Surface Air Temperature Change", year=2100,
).values.flatten())
print(f"FaIRv2: {n_fair} ensemble members run on {SCENARIO}")

# %% [markdown]
# ## Run CICEROSCMPY2
#
# ``CICERO_MODE=bundle`` (default) uses the Marit-RCMIP-aligned input
# bundle — the path the ``draw_samples_500`` calibration was fit
# against and the only path that produces present-day warming
# consistent with the reference protocol. ``CICERO_MODE=splice``
# uses the legacy v1.1.x path (carries a ~0.3-0.5 K warm bias).
#
# ``N_MEMBERS=full`` runs every cfg in ``draw_samples_500.json``
# (500 members).

# %%
def _cicero_bundle_cfg():
    distribution_json = CICERO_BUNDLE_DIR / "draw_samples_500.json"
    assert distribution_json.exists(), (
        f"CICERO distribution JSON missing at {distribution_json}"
    )
    assert CICERO_BUNDLE_RCMIP.is_dir(), (
        f"CICERO RCMIP bundle dir missing at {CICERO_BUNDLE_RCMIP} — "
        f"needed for CICERO_MODE=bundle. Set CICEROSCMPY2_BUNDLE_DIR "
        f"to a directory containing rcmip-march2026/, or set "
        f"CICERO_MODE=splice for the legacy path."
    )
    cfg = {
        "distribution_json": str(distribution_json),
        "cicero_bundle_dir": str(CICERO_BUNDLE_RCMIP),
    }
    if N_MEMBERS is not None:
        cfg["member_indices"] = range(N_MEMBERS)
    return cfg


def _cicero_splice_cfg():
    required = {
        "distribution_json": CICERO_BUNDLE_DIR / "draw_samples_500.json",
        "gaspam_file": CICERO_BUNDLE_DIR / "gases_vupdate_2022_AR6.txt",
        "concentrations_file": CICERO_BUNDLE_DIR / "ssp245_conc_RCMIP.txt",
    }
    for k, p in required.items():
        assert p.exists(), f"CICERO splice file {k!r} missing at {p}"
    cfg = {
        "distribution_json": str(required["distribution_json"]),
        "gaspam_file": str(required["gaspam_file"]),
        "concentrations_file": str(required["concentrations_file"]),
    }
    if N_MEMBERS is not None:
        cfg["member_indices"] = range(N_MEMBERS)
    return cfg


def _run_cicero():
    cfg = _cicero_bundle_cfg() if CICERO_MODE == "bundle" else _cicero_splice_cfg()
    return openscm_runner.run.run(
        climate_models_cfgs={"CICERO-SCM-PY2": [cfg]},
        scenarios=scenarios,
        output_variables=OUTPUT_VARIABLES,
        out_config=None,
    )


cicero_result = _cached_run(
    _cache_path("cicero_py2", SCENARIO, extra_tag=CICERO_MODE),
    _run_cicero,
    f"CICEROSCMPY2 ({CICERO_MODE} mode, {MEMBER_TAG})",
)
n_cicero = len(cicero_result.filter(
    variable="Surface Air Temperature Change", year=2100,
).values.flatten())
print(f"CICEROSCMPY2 ({CICERO_MODE}): {n_cicero} ensemble members run on {SCENARIO}")

# %% [markdown]
# ## Combine into one ScmRun

# %%
combined = scmdata.run_append([fair_result, cicero_result])

# Both adapters tag results with a version-suffixed name (e.g.
# 'FaIRv2.2.4', 'CICERO-SCM-PY2.1.0'). Detect the actual strings
# rather than hard-coding them so the notebook keeps working when
# the upstream package versions tick.
model_names = sorted(combined["climate_model"].unique())
FAIR_MODEL = next(m for m in model_names if m.startswith("FaIRv"))
CICERO_MODEL = next(m for m in model_names if m.startswith("CICERO-SCM-PY"))
MODELS = (FAIR_MODEL, CICERO_MODEL)
MODEL_COLOR = {FAIR_MODEL: "C0", CICERO_MODEL: "C1"}

print("Climate models:", model_names)
print(f"  {FAIR_MODEL}: {n_fair} members")
print(f"  {CICERO_MODEL}: {n_cicero} members")

# %% [markdown]
# ## Ensemble temperature timeseries
#
# A dedicated GSAT plot: every member as a thin translucent line
# (thinned to 100 per model so the figure stays readable at the full
# posterior), with the median + 5-95% band overlaid. Shows the full
# ensemble structure, not just the summary statistics.

# %%
FIG_TAG = f"cicero-{CICERO_MODE}_{MEMBER_TAG}"


def _figure_path(label):
    return FIGURE_DIR / f"compare_rcmip_{SCENARIO}_{FIG_TAG}_{label}.png"


def _spaghetti_temperature(combined, scenario, max_lines_per_model=100):
    fig, ax = plt.subplots(figsize=(11, 6))
    for model in MODELS:
        chunk = combined.filter(
            climate_model=model, variable="Surface Air Temperature Change",
        )
        if chunk.empty:
            continue
        ts = chunk.timeseries(time_axis="year")
        years = ts.columns.astype(int).values
        n = ts.shape[0]
        color = MODEL_COLOR[model]
        if n > max_lines_per_model:
            idx = np.round(
                np.linspace(0, n - 1, max_lines_per_model)
            ).astype(int)
            displayed = ts.iloc[idx]
        else:
            displayed = ts
        for _, row in displayed.iterrows():
            ax.plot(years, row.values, color=color, lw=0.4, alpha=0.18)
        median = ts.median(axis=0).values
        q05 = ts.quantile(0.05, axis=0).values
        q95 = ts.quantile(0.95, axis=0).values
        ax.plot(years, median, color=color, lw=2.2,
                label=f"{model} median (n={n})")
        ax.fill_between(years, q05, q95, color=color, alpha=0.0,
                        edgecolor=color, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Year")
    ax.set_ylabel("GSAT change (K)")
    ax.set_title(
        f"{scenario}: ensemble GSAT trajectories "
        f"({CICERO_MODE} CICEROSCM, thin: members, thick: median, "
        "dashed: 5-95%)"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = _figure_path("temperature_ensemble")
    fig.savefig(out, dpi=140)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")
    plt.show()


_spaghetti_temperature(combined, SCENARIO)

# %% [markdown]
# ## Trajectory bands — GSAT, CO2, ERF
#
# Three nested bands per model: full envelope (min-max, faintest),
# 5-95% (medium), 17-83% (darker), plus the median line. Reveals
# tail structure that a single 5-95% band hides.

# %%
def _band_plot(combined, variable, ylabel, ax):
    for model in MODELS:
        chunk = combined.filter(climate_model=model, variable=variable)
        if chunk.empty:
            continue
        ts = chunk.timeseries(time_axis="year")
        years = ts.columns.astype(int).values
        median = ts.median(axis=0).values
        envelope_lo = ts.min(axis=0).values
        envelope_hi = ts.max(axis=0).values
        q05 = ts.quantile(0.05, axis=0).values
        q95 = ts.quantile(0.95, axis=0).values
        q17 = ts.quantile(0.17, axis=0).values
        q83 = ts.quantile(0.83, axis=0).values
        color = MODEL_COLOR[model]
        ax.fill_between(years, envelope_lo, envelope_hi, color=color,
                        alpha=0.08, label=f"{model} min-max")
        ax.fill_between(years, q05, q95, color=color, alpha=0.18,
                        label=f"{model} 5-95%")
        ax.fill_between(years, q17, q83, color=color, alpha=0.30,
                        label=f"{model} 17-83%")
        ax.plot(years, median, color=color, lw=1.8, label=f"{model} median")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
for ax, (variable, label) in zip(axes, [
    ("Surface Air Temperature Change", "GSAT (K)"),
    ("Atmospheric Concentrations|CO2", "CO2 (ppm)"),
    ("Effective Radiative Forcing", "ERF (W/m^2)"),
]):
    _band_plot(combined, variable, label, ax)
axes[0].set_title(
    f"{SCENARIO}: ensemble trajectories ({CICERO_MODE} CICEROSCM, "
    "median, 17-83%, 5-95%, min-max)"
)
axes[-1].set_xlabel("Year")
axes[0].legend(fontsize=7, loc="upper left", ncol=2)
fig.tight_layout()
out = _figure_path("bands")
fig.savefig(out, dpi=140)
print(f"  wrote {out.relative_to(REPO_ROOT)}")
plt.show()

# %% [markdown]
# ## 2100 distribution — histograms

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (variable, label, unit) in zip(axes, [
    ("Surface Air Temperature Change", "2100 GSAT", "K"),
    ("Atmospheric Concentrations|CO2", "2100 CO2", "ppm"),
    ("Effective Radiative Forcing", "2100 ERF", "W/m^2"),
]):
    for model in MODELS:
        values = combined.filter(
            climate_model=model, variable=variable, year=2100,
        ).values.flatten()
        ax.hist(
            values, bins=40, alpha=0.5, color=MODEL_COLOR[model],
            label=f"{model} (n={len(values)})", density=True,
        )
        ax.axvline(np.median(values), color=MODEL_COLOR[model], lw=2,
                   linestyle="--", label=f"{model} median")
    ax.set_xlabel(f"{label} ({unit})")
    ax.set_ylabel("density")
    ax.set_title(f"{label}, {SCENARIO}, {CICERO_MODE} CICEROSCM")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
fig.tight_layout()
out = _figure_path("2100_histograms")
fig.savefig(out, dpi=140)
print(f"  wrote {out.relative_to(REPO_ROOT)}")
plt.show()

# %% [markdown]
# ## Cross-model scatter — paired GSAT/CO2/ERF per member
#
# Pairwise scatter requires matched member counts (we don't have a
# 1-to-1 mapping between FaIR posterior rows and CICEROSCM cfg
# rows), so we instead plot the marginal joint by sorting each
# model's outputs and pairing by rank. Roughly a QQ-style view.

# %%
def _rank_pair(values_a, values_b):
    a = np.sort(values_a)
    b = np.sort(values_b)
    n = min(len(a), len(b))
    # Sub-sample the larger to align lengths (keeps the rank
    # interpretation roughly correct).
    if len(a) > n:
        a = a[np.round(np.linspace(0, len(a) - 1, n)).astype(int)]
    if len(b) > n:
        b = b[np.round(np.linspace(0, len(b) - 1, n)).astype(int)]
    return a, b


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (variable, label, unit) in zip(axes, [
    ("Surface Air Temperature Change", "2100 GSAT", "K"),
    ("Atmospheric Concentrations|CO2", "2100 CO2", "ppm"),
    ("Effective Radiative Forcing", "2100 ERF", "W/m^2"),
]):
    fair_vals = combined.filter(
        climate_model=FAIR_MODEL, variable=variable, year=2100,
    ).values.flatten()
    cic_vals = combined.filter(
        climate_model=CICERO_MODEL, variable=variable, year=2100,
    ).values.flatten()
    fa, cb = _rank_pair(fair_vals, cic_vals)
    ax.scatter(fa, cb, s=12, alpha=0.5, color="C2")
    lo = min(fa.min(), cb.min())
    hi = max(fa.max(), cb.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="1:1")
    ax.set_xlabel(f"{FAIR_MODEL} {label} ({unit})")
    ax.set_ylabel(f"{CICERO_MODEL} {label} ({unit})")
    ax.set_title(
        f"{label}, {SCENARIO} ({CICERO_MODE} CICEROSCM; "
        "rank-aligned ensembles)"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.tight_layout()
out = _figure_path("2100_scatter")
fig.savefig(out, dpi=140)
print(f"  wrote {out.relative_to(REPO_ROOT)}")
plt.show()

# %% [markdown]
# ## Sanity checks

# %%
def _values_2100(combined, model, variable):
    return combined.filter(
        climate_model=model, variable=variable, year=2100,
    ).values.flatten()


# Numerical ranges
for model in MODELS:
    gsat = _values_2100(combined, model, "Surface Air Temperature Change")
    co2 = _values_2100(combined, model, "Atmospheric Concentrations|CO2")
    erf = _values_2100(combined, model, "Effective Radiative Forcing")
    assert gsat.size > 0 and 0.0 < gsat.min() and gsat.max() < 12.0, (
        f"{model} {SCENARIO}: implausible GSAT range "
        f"[{gsat.min():.2f}, {gsat.max():.2f}] K"
    )
    assert co2.size > 0 and 250.0 < co2.min() and co2.max() < 2500.0, (
        f"{model} {SCENARIO}: implausible CO2 range "
        f"[{co2.min():.0f}, {co2.max():.0f}] ppm"
    )
    assert erf.size > 0 and -2.5 < erf.min() and erf.max() < 18.0, (
        f"{model} {SCENARIO}: implausible ERF range "
        f"[{erf.min():.2f}, {erf.max():.2f}] W/m^2"
    )
print("PASS: 2100 GSAT/CO2/ERF in plausible ranges across full ensembles")

# %%
# Shape check depends on scenario class
PROJECTION_FIRST_YEAR = 2020
DECADAL_WINDOW = 10

if SCENARIO in MITIGATION_SCENARIOS:
    for model in MODELS:
        ts = combined.filter(
            climate_model=model, variable="Surface Air Temperature Change",
            year=range(PROJECTION_FIRST_YEAR, 2101),
        ).timeseries(time_axis="year")
        median = ts.median(axis=0)
        peak_year = int(median.idxmax())
        end_year = int(median.index.max())
        assert peak_year < end_year, (
            f"{model} {SCENARIO}: GSAT median peaks at {peak_year}, "
            f"which equals or follows the run end ({end_year}); expected "
            f"peak-and-decline for a mitigation pathway."
        )
    print(
        f"PASS: {SCENARIO} GSAT median peaks before run end for both "
        f"models (peak-and-decline)"
    )
elif SCENARIO in HIGH_EMISSION_SCENARIOS:
    for model in MODELS:
        ts = combined.filter(
            climate_model=model, variable="Surface Air Temperature Change",
            year=range(PROJECTION_FIRST_YEAR, 2101),
        ).timeseries(time_axis="year")
        median = ts.median(axis=0)
        smoothed = (
            median.rolling(DECADAL_WINDOW, min_periods=DECADAL_WINDOW)
            .mean()
            .dropna()
        )
        diffs = np.diff(smoothed.values)
        n_decreases = (diffs < 0).sum()
        assert n_decreases == 0, (
            f"{model} {SCENARIO}: decadal-mean GSAT decreases at "
            f"{n_decreases} step(s) over {PROJECTION_FIRST_YEAR}-2100; "
            f"expected monotone warming on a high-emission pathway."
        )
    print(
        f"PASS: {SCENARIO} GSAT decadal mean monotone over "
        f"{PROJECTION_FIRST_YEAR}-2100 for both models"
    )
else:
    print(
        f"(no shape check for scenario class {SCENARIO!r}; pick a "
        f"mitigation or high-emission scenario for that)"
    )

# %% [markdown]
# ## Summary table

# %%
def _summary(values, name):
    return {
        "metric": name,
        "n": len(values),
        "min": values.min(),
        "5%": np.quantile(values, 0.05),
        "17%": np.quantile(values, 0.17),
        "median": np.median(values),
        "83%": np.quantile(values, 0.83),
        "95%": np.quantile(values, 0.95),
        "max": values.max(),
    }


rows = []
for variable, label in [
    ("Surface Air Temperature Change", "2100 GSAT (K)"),
    ("Atmospheric Concentrations|CO2", "2100 CO2 (ppm)"),
    ("Effective Radiative Forcing", "2100 ERF (W/m^2)"),
]:
    for model in MODELS:
        vals = _values_2100(combined, model, variable)
        rows.append({"model": model, **_summary(vals, label)})
summary = pd.DataFrame(rows).set_index(["metric", "model"])
print(summary.to_string(float_format="%.2f"))

# %%
# Cross-model gap report (not asserted at this depth — see notebook
# docstring; the gap is genuine ECS-distribution information).
print(f"\nCross-model 2100 GSAT comparison on {SCENARIO}:")
fair_med = float(np.median(
    _values_2100(combined, FAIR_MODEL, "Surface Air Temperature Change")
))
cic_med = float(np.median(
    _values_2100(combined, CICERO_MODEL, "Surface Air Temperature Change")
))
print(f"  {FAIR_MODEL:25s} median: {fair_med:.2f} K")
print(f"  {CICERO_MODEL:25s} median: {cic_med:.2f} K")
print(f"  CICERO - FaIR:                       {cic_med - fair_med:+.2f} K")
