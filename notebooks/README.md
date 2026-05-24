# Cross-model validation notebooks

These notebooks live on `modernisation/integration` only — they exercise
the FaIRv2 (PR #7), CICEROSCMPY2 (PR #8), IAMC loader (PR #9), and
scmdata pandas-3 shim (PR #11) together, so they cannot run against
any single feature branch.

Validation artifacts, not part of the package's regular test suite:
each cell exercises a real adapter end-to-end against a real
calibration bundle, prints plots, and asserts a small set of scientific
plausibility bounds. Run them by hand after a rebuild of the
integration branch to catch cross-model regressions before external
review pulls on the individual PRs.

## Notebooks

- `compare_rcmip.py` — **Single-scenario, full-ensemble characterisation**
  (default ssp119; override via `SCENARIO`). Runs the full calibration
  posterior of each model (841 FaIR / 500 CICEROSCM members by default;
  bump down via `N_MEMBERS=10` for fast iteration). Caches each model's
  output to `.cache/` so plot iterations don't re-run the models;
  saves figures to `figures/`. Plots: nested-band trajectories
  (median, 17-83%, 5-95%, min-max), per-variable 2100 histograms,
  rank-aligned cross-model scatter, focused temperature ensemble
  (members + bands). Sanity-checks plausible numerical ranges +
  scenario-class-appropriate shape (mitigation: peak-and-decline;
  high-emission: monotone decadal mean).

  `CICERO_MODE=splice|bundle` switches the CICEROSCMPY2 cfg between
  the legacy splice path (carries a +0.3-0.5 K present-day warm bias —
  see fork issue #10 / project memory) and the Marit-RCMIP-aligned
  bundle path (bit-exact match to the reference protocol). Default is
  splice for backward compatibility with the rebuild output, but new
  validation runs should pass `CICERO_MODE=bundle`.

- `compare_sci.py` — **Scenario Compass Initiative subset comparison**.
  Five representative SCI 2025 scenarios (SSP1-19, ADVANCE-2020-1.5°C,
  SSP2-45, CD-LINKS-NPi, SSP5-Baseline) × 10 members × both adapters,
  via the IAMC loader. One IAM per scenario (CICEROSCMPY2's
  `DistributionRun` collapses scenarios by name).

## Running

```sh
# From repo root, with the integration branch checked out and the
# venv activated:
export FAIR2_CALIBRATION_PATH=$PWD/configurations/fair-calibrate-v1.6.0
export CICEROSCMPY2_BUNDLE_DIR=$PWD/configurations/ciceroscm
export SCI_DATA_PATH=$PWD/scenario_data/SCI-2025_v1.0_pathways_ensemble_global.xlsx

# RCMIP, 10-member iteration mode, bundle CICEROSCM (recommended):
MPLBACKEND=Agg N_MEMBERS=10 CICERO_MODE=bundle \
    python notebooks/compare_rcmip.py

# Full posterior, ssp245, bundle CICEROSCM (~12 min wallclock):
MPLBACKEND=Agg N_MEMBERS=full SCENARIO=ssp245 CICERO_MODE=bundle \
    python notebooks/compare_rcmip.py

# SCI subset:
MPLBACKEND=Agg python notebooks/compare_sci.py

# As Jupyter notebooks (paired ipynb generated on the fly):
jupytext --to ipynb --execute notebooks/compare_rcmip.py
```

The `.cache/` directory holds per-(model, scenario, member-count,
mode) pickle of each model's output. Set `REGEN_CACHE=1` to force a
re-run, e.g. after pulling a new calibration bundle.

## Findings recorded from these notebooks

- **CICEROSCMPY2 splice mode carries a present-day warm bias.**
  At 2024 the splice-mode median is 1.85 K vs bundle-mode 1.50 K on
  ssp119 / 10 members. Confirmed by reproducing Marit's reference
  protocol (`cscm-calibrate@ben_rcmip_sandbox:scripts/run_full_rcmip_protocol.py`)
  directly — bundle-mode adapter matches the reference bit-exactly.
  Fixed in PR #8: bundle mode is now the default and splice mode
  emits a `LOGGER.warning`.
- **FaIR's 2100 GSAT on ssp119** centres on 1.47 K with 841-member
  posterior, close to the AR6 SPM Table SPM.1 best estimate of 1.4 K.
- **CICEROSCM's central tendency** in the `draw_samples_500` bundle
  sits slightly above AR6 (~1.5 K present-day) — not a translation
  bug; the calibration's central tendency is genuinely a touch high.

## Caveats

- The `mp.set_start_method("fork")` at the top of each notebook is
  needed for CICEROSCM's parallel pool when running on macOS via the
  plain-`python` path; the kernel path doesn't need it but it is a
  no-op on Linux and inside an already-spawned ipykernel.
- `figures/` PNGs are tracked so they're inspectable from GitHub; the
  `.cache/` pickles are ignored (regeneratable, large).
