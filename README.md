# OpenSCM-Runner

<!---
Can use start-after and end-before directives in docs, see
https://myst-parser.readthedocs.io/en/latest/syntax/organising_content.html#inserting-other-documents-directly-into-the-current-document
-->

<!--- sec-begin-description -->

OpenSCM-Runner provides a unified API for running emissions scenarios with different simple climate models.

[![CI](https://github.com/openscm/openscm-runner/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/openscm/openscm-runner/actions/workflows/ci.yaml)
[![Coverage](https://codecov.io/gh/openscm/openscm-runner/branch/main/graph/badge.svg)](https://codecov.io/gh/openscm/openscm-runner)
[![Docs](https://readthedocs.org/projects/openscm-runner/badge/?version=latest)](https://openscm-runner.readthedocs.io)

**PyPI :**
[![PyPI](https://img.shields.io/pypi/v/openscm-runner.svg)](https://pypi.org/project/openscm-runner/)
[![PyPI: Supported Python versions](https://img.shields.io/pypi/pyversions/openscm-runner.svg)](https://pypi.org/project/openscm-runner/)
[![PyPI install](https://github.com/openscm/openscm-runner/actions/workflows/install.yaml/badge.svg?branch=main)](https://github.com/openscm/openscm-runner/actions/workflows/install.yaml)

**Other info :**
[![License](https://img.shields.io/github/license/openscm/openscm-runner.svg)](https://github.com/openscm/openscm-runner/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/openscm/openscm-runner.svg)](https://github.com/openscm/openscm-runner/commits/main)
[![Contributors](https://img.shields.io/github/contributors/openscm/openscm-runner.svg)](https://github.com/openscm/openscm-runner/graphs/contributors)

<!--- sec-end-description -->

Full documentation can be found at:
[openscm-runner.readthedocs.io](https://openscm-runner.readthedocs.io/en/latest/).
We recommend reading the docs there because the internal documentation links
don't render correctly on GitHub's viewer.

## AR7 modernisation fork

This branch (`benmsanderson/openscm-runner`) hosts in-flight AR7-cycle work
that has not yet been upstreamed. The `modernisation/integration` branch is
the all-in-one working tree: it carries every in-flight feature branch
merged together so the demo notebooks and CLI can be exercised end-to-end
locally. Individual feature branches stay independent for upstream PR
review. See `ARCHITECTURE_NOTES.md` for the design rationale and
[scripts/rebuild_integration_branch.sh](scripts/rebuild_integration_branch.sh)
for how `integration` is reconstructed from the feature branches.

Modernisation deltas relative to upstream `openscm/openscm-runner`:

- **FaIRv2 adapter** (`FAIR2`), parallel to the existing FaIR 1.6 adapter.
  Takes a native FaIR calibration bundle (CSV ensemble + species config) and
  supports both emissions-driven and concentration-driven runs.
- **CICERO-SCM v2.x adapter** (`CICERO-SCM-PY2`), parallel to the existing
  v1.1.x adapter. Wraps the native `DistributionRun` parallel API and
  supports the Marit RCMIP-aligned bundle plus a splice fallback. Reads
  protocol metadata columns (mode, natural-forcing, land-use-forcing)
  from the loader rather than pattern-matching on scenario names.
- **Streaming netCDF output** (`NetCDFChunkWriter`) for ensembles too large
  to hold in memory, wired into `openscm_runner.run.run` via a new
  `output_writer` kwarg.
- **IAMC and RCMIP3 scenario loaders** for both emissions
  (`load_rcmip3_emissions`) and concentrations
  (`load_rcmip3_concentrations`), covering the 94-scenario RCMIP3 registry
  (8 CMIP6 SSPs, 7 CMIP7 `scen7-*` markers + 7 conc variants, 15 `esm-flat*`
  idealised, 6 historical + attribution, 9 control + pulse + bell + 1pct
  branch, 8 `methanemip` / CH4 sensitivity variants, plus CO2-only ED
  variants of the SSPs and scen7s). Tags each row with protocol metadata
  (`protocol_mode`, `protocol_natural_forcing`, `protocol_land_use_forcing`)
  that adapters consume directly. Includes a checked-in IAMC translation
  of the CICEROSCM RCMIP-march2026 bundle so scenarios whose emissions
  live only in that bundle (e.g. the CH4-swap scen7 variants) are loadable
  without bundle access. Plus the RCMIP3 Table 3 constraint targets
  (`CONSTRAINT_TARGETS`).
- **Unified RCMIP3 protocol runner** at `scripts/run_rcmip3.py`, dispatching
  multiple SCMs across the full registry in both emissions-driven and
  concentration-driven mode. Per-(mode, model, scenario) netCDF output to
  `out/rcmip3/{mode}/{model}/{scenario}.nc`.
- **Validation harness** (`scripts/validate_against_marit.py`) that
  compares the runner's CICEROSCMPY2 output against Marit's 500-member
  native RCMIP3 submission per scenario. Output is a PASS/WARN/FAIL
  scorecard with mean/max/RMSE diff; current state and per-failure
  follow-up plan documented in
  [PHASE_B_SCORECARD.md](PHASE_B_SCORECARD.md).
- **pandas 3.0 / xarray 2025+ compatibility shim** for `scmdata` 0.18,
  applied at package import time. Four patches (groupby StringDtype,
  xarray Series positional indexing, ScmRun.convert_unit read-only
  values, netcdf `_read_nc` use_cftime). Goes away when `scmdata` ships
  its own fix.
- **Parallel model dispatch and SLURM-aware worker counts** in `run.run`.

## Status (as of 2026-05-29)

| Area | What works | What's pending |
|---|---|---|
| Both modern adapters end-to-end | FaIRv2 + CICEROSCMPY2 in ED and CD across the 94 registered scenarios; single-CLI dispatch via `scripts/run_rcmip3.py` | FaIRv2 can't yet consume the bundle-only scenarios (esm-1pct-brch-*PgC, scen7 CH4 swaps); CICEROSCMPY2 handles them via bundle file lookup |
| Constraint validation | Historical, OHC change, CO2 concentration, GMST, aerosol ERF, carbon fluxes against IGCC2024 / GCB2024 / AR6 (Table 3 of the RCMIP3 protocol paper) | Aerosol ERF panel populates only when `Effective Radiative Forcing\|Anthropogenic\|Aerosol` is in the requested output set; needs to be added to the runner default |
| RCMIP3 reference parity | esm-flat10-zec matched-member diff against Marit's 500-member reference: median ≈ 0, max ≈ 30 mK; pi-controls and historical PASS at scorecard threshold | 60 FAIL / 20 WARN / 10 PASS across the full registry sweep; per-category PR plan in `PHASE_B_SCORECARD.md` |
| Idealised diagnostics | TCRE convergence and ZEC ≈ 0 (FaIRv2) / slightly-negative (CICEROSCMPY2) for `esm-flat*` family, matching Marit's reference | Same path-dependence diagnostics for `esm-bell*`, `esm-pi-*`, `esm-1pct-brch-*` not yet validated |
| Documentation | This README, `ARCHITECTURE_NOTES.md` (design rationale), `PHASE_B_SCORECARD.md` (validation state + open work) | API docstrings could be tighter; no integration tests for the full sweep yet |

See `PHASE_B_SCORECARD.md` for the categorised list of open work; the
four PR-sized fix categories there cover the bulk of the remaining
FAILs. The fork's feature-branch PRs (listed in
`scripts/rebuild_integration_branch.sh`) are the units of upstream
review; `modernisation/integration` is the local all-in-one tree
where new work lands before being backported to its target branch.

## Installation

<!--- sec-begin-installation -->

OpenSCM-Runner can be installed with conda or pip:

```bash
pip install openscm-runner
conda install -c conda-forge openscm-runner
```

Note: the modernisation deltas above are not yet on PyPI. To use them, install
from this fork's `modernisation/integration` branch:

```bash
pip install "git+https://github.com/benmsanderson/openscm-runner.git@modernisation/integration"
```

### Optional extras

Each climate model adapter is gated behind its own pyproject extra. The
core install does not pull in any of them. Choose one or more of:

```bash
# To add notebook dependencies
pip install openscm-runner[notebooks]

# MAGICC (Fortran binary called via pymagicc)
pip install openscm-runner[magicc]

# FaIR 1.6 adapter (original)
pip install openscm-runner[fair]

# FaIR 2.x adapter (modernisation fork, AR7-era)
pip install openscm-runner[fair2]

# CICERO-SCM v1.1.x Python adapter (original)
pip install openscm-runner[ciceroscmpy]

# CICERO-SCM v2.x Python adapter (modernisation fork)
pip install openscm-runner[ciceroscmpy2]

# All three legacy models in one go (FaIR 1.6 + MAGICC + CICEROSCM v1.1.x)
pip install openscm-runner[models]

# Streaming netCDF output (NetCDFChunkWriter) and notebook chunk-readback.
# Already pulled in by [notebooks]; install standalone if you want the
# CLI runner to write per-scenario .nc files without the notebook deps.
pip install openscm-runner[netcdf]

# CICERO-SCM's Fortran binary requires no additional dependencies to be
# installed; it ships with the adapter package.
```

**Mutual exclusion: pick the major version that matches the adapter.**
`[fair]` and `[fair2]` install incompatible major versions of the same
`fair` PyPI package, and `[ciceroscmpy]` and `[ciceroscmpy2]` likewise
share the `ciceroscm` PyPI package. The pyproject extras leave both
options open (`fair >=1.6,<3`, `ciceroscm >=1.1,<3`); the adapter
`_compat` shims raise a clear `ImportError` at runtime if you select an
adapter whose major version is not the one you installed. Hard-pin
yourself if you need a particular minor:

```bash
pip install "openscm-runner[fair2]" "fair>=2.2,<3"
pip install "openscm-runner[ciceroscmpy2]" "ciceroscm>=2.1,<3"
```

```bash
# If you are installing with conda, we recommend
# installing the extras by hand because there is no stable
# solution yet (issue here: https://github.com/conda/conda/issues/7502)
```

<!--- sec-end-installation -->

## Quick start: RCMIP3 protocol demo

The modernisation fork ships a three-step workflow that runs FaIRv2 and
CICERO-SCM-PY2 across the full RCMIP3 protocol scenario set and renders
five comparison figures.

```bash
# 1. Install both modern adapters plus the notebook + netCDF extras
pip install -e ".[fair2,ciceroscmpy2,notebooks]"
# (the [notebooks] extra already pulls in netcdf4, which the runner
# writes and the notebook reads back; install [netcdf] standalone if
# you skip the notebooks extra)

# 2. Fetch the FaIRv2 calibration bundle from Zenodo
scripts/download_fair2_calibration.py
export FAIR2_CALIBRATION_PATH=$PWD/configurations/fair-calibrate-v1.6.0

# CICERO-SCM bundle (Marit RCMIP-aligned setup) lives under
# configurations/ciceroscm/rcmip-march2026/. Point at it via:
export CICEROSCMPY2_BUNDLE_DIR=$PWD/configurations/ciceroscm

# 3. Run the protocol: SSPs + scen7-* + esm-flat*, both modes, 10 members
scripts/run_rcmip3.py --members 10 --scenario-set all --mode both

# 4. Render figures from the cached netCDF tree (no model re-runs)
jupyter execute notebooks/compare_rcmip3.py
```

The runner streams per-(mode, model, scenario) output to
`out/rcmip3/{mode}/{model}/{scenario}.nc`. The notebook reads that tree
back lazily so figure styling iterates in seconds, decoupled from the
~30 minute sweep that produced the data.

See [notebooks/compare_rcmip3.py](notebooks/compare_rcmip3.py) for the
figure code (jupytext-paired with the `.ipynb`), and Table 3 of the
RCMIP3 protocol paper (Romero-Prieto et al. 2025) for the constraint
targets the historical-validation panel overlays.

## Programmatic API

The original `openscm_runner.run.run` entry point is unchanged for
existing callers. The new capabilities are additive:

```python
import openscm_runner.run
from openscm_runner.output import NetCDFChunkWriter
from openscm_runner.scenarios import (
    load_iamc, load_rcmip3_emissions, CONSTRAINT_TARGETS,
)

# Stream results to disk instead of returning one big ScmRun:
result = openscm_runner.run.run(
    climate_models_cfgs={"FaIRv2": [...]},
    scenarios=load_rcmip3_emissions(["ssp245", "esm-flat10-zec"]),
    output_variables=("Surface Air Temperature Change", ...),
    output_writer=NetCDFChunkWriter("out/myrun"),
)
# result is a RunResult; iter_chunks() lazily yields one ScmRun per
# (climate_model, scenario) file written.
```

## For developers

<!--- sec-begin-installation-dev -->

For development, we rely on [poetry](https://python-poetry.org) for all our
dependency management. To get started, you will need to make sure that poetry
is installed
([instructions here](https://python-poetry.org/docs/#installing-with-the-official-installer),
we found that pipx and pip worked better to install on a Mac).

For all of work, we use our `Makefile`.
You can read the instructions out and run the commands by hand if you wish,
but we generally discourage this because it can be error prone.
In order to create your environment, run `make virtual-environment`.

If there are any issues, the messages from the `Makefile` should guide you
through. If not, please raise an issue in the [issue tracker][issue_tracker].

For the rest of our developer docs, please see [](development-reference).
On the modernisation fork, also see
[ARCHITECTURE_NOTES.md](ARCHITECTURE_NOTES.md) for the design rationale
behind the new adapters and the integration-branch workflow.

<!--- sec-end-installation-dev -->

[issue_tracker]: https://github.com/openscm/openscm-runner/issues
