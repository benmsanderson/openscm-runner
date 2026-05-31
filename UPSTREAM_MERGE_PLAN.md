# Upstream merge plan

How the modernisation fork at `benmsanderson/openscm-runner` lands at
upstream `openscm/openscm-runner`. Written for review by Zeb and anyone else looking
at the fork before the PRs open.

## Background

Ben's fork currently carries an AR7-cycle modernisation pass of openscm-runner:
two new SCM adapters (FaIRv2 and CICERO-SCM 2.1), an RCMIP3 protocol
loader and runner, streaming netCDF output, a pandas-3.0 /
xarray-2025+ compatibility shim against scmdata, and a 94-scenario
RCMIP3 registry with cross-model validation against Marit's reference.

The upstream contribution is deliberately narrow. openscm-runner is a
thin, explicit wrapper around a small number of SCMs; protocol-specific
loaders, scenario registries, native bundle handling, mixed driver-modes
and output databases sit outside that scope and live in an application
layer above the wrapper.

The split:

- **Upstream** — Python-version bump, parallel dispatch ergonomics,
  the two new adapters consuming the existing per-cfg dict API, an
  explicit `RunMode` enum, and four parameterised smoke tests.
- **Application layer** — RCMIP3 loader + runner, the 94-scenario
  registry, Marit's bundle expansion, mixed driver-mode (CO2-only
  ED + non-CO2 CD), SCI translation, idealised-scenario natural-forcing
  suppression, chunked-output writing on top of
  [`pandas_openscm.db`](https://github.com/openscm/pandas-openscm/tree/main/src/pandas_openscm/db),
  cross-model comparison notebooks, and the ~200 fork-only unit
  tests. Lives on this fork (current home) or a future
  IPCC/CICERO-hosted repo.

See [`ARCHITECTURE_NOTES.md`](ARCHITECTURE_NOTES.md) for the design
rationale and [`PHASE_B_SCORECARD.md`](PHASE_B_SCORECARD.md) for the
CICEROSCMPY2 validation status against Marit's RCMIP3 reference. Both
stay on the fork.

## Two PRs against `openscm/openscm-runner@main`

### PR A: pandas-3 / xarray-2025+ fixes against `openscm/scmdata`

Status: **open as
[scmdata#321](https://github.com/openscm/scmdata/pull/321)**, awaiting
review. Four fixes to the affected scmdata source files (no shim in
openscm-runner). Once a scmdata release is cut, openscm-runner pins
to it and the in-tree shim is deleted as part of PR B.

| Patch | scmdata file | Change |
|---|---|---|
| 1 | `scmdata/groupby.py` (`RunGroupBy.__init__`) | Replace `np.issubdtype(col.dtype, np.number)` with `pd.api.types.is_numeric_dtype(col)` so StringDtype meta columns don't trip the numeric-column detection |
| 2 | `scmdata/_xarray.py` (`_many_to_one`) | Replace `.max()[0]` with `.max().iloc[0]` (pandas 3 removed positional indexing on label-indexed Series) |
| 3 | `scmdata/run.py` (`ScmRun.convert_unit`, `_binary_op`, `_unary_op`) | Replace `self._df.values[:] = ...` with `self._df.iloc[:, :] = ...` since `DataFrame.values` is read-only in pandas 3; binary/unary ops additionally wrap the RHS in `np.asarray(..., dtype=float)` to preserve the prior bool-to-float silent cast for comparison ops |
| 4 | `scmdata/netcdf.py` (`_read_nc`) | Route through `xr.coders.CFDatetimeCoder(use_cftime=True)` instead of the bare `use_cftime` kwarg, with a fallback for xarray older than 2024.09; silences the FutureWarning xarray 2025+ emits on every `ScmRun.from_nc` call |

Regression tests for each patch land in scmdata's own test suite. The
shim and its tests in this repo
(`src/openscm_runner/_scmdata_patches.py`,
`tests/unit/test_scmdata_patches.py`, the import-time call in
`src/openscm_runner/__init__.py`) are deleted in PR B; the runner's
pyproject pins `scmdata>=<fixed-release>`.

**Sequencing**: PR A lands and gets tagged; PR B opens against
`openscm/openscm-runner@main` once that pin is available.

### PR B: adapter additions and runtime ergonomics

Source code from these feature branches merges in, trimmed to the
upstream API surface:

| Feature branch | Brings (upstream-relevant only) |
|---|---|
| `modernisation/python-3.12` | Python version bump prerequisite |
| `modernisation/worker-counts` | SLURM-aware worker counts in adapter parallel dispatch |
| `modernisation/parallel-models` | Parallel model dispatch in `run.run` (verify whether already folded into worker-counts) |
| `modernisation/fair2-adapter` (trimmed) | FaIRv2 adapter (`FAIR2`) consuming the existing per-cfg dict API. Strip: native calibration bundle loading, default-calibration Zenodo fetch, idealised forcing suppression, protocol-metadata consumption — all move to the application layer |
| `modernisation/ciceroscmpy2-adapter` (trimmed) | CICERO-SCM 2.1 adapter (`CICERO-SCM-PY2`) consuming the existing per-cfg dict API. Strip: `DistributionRun` bundle wrapper, RCMIP-aligned bundle handling, idealised PI-flat aerosols, protocol-metadata consumption — all move to the application layer |
| _(new)_ | Top-level `RunMode` enum (see "Mode enum" below) wired into `run.run`; each adapter declares which modes it supports |
| _(new, optional)_ | `check_variables_are_as_expected` helper that raises on unknown emissions variable names |

Not in PR B:

- `modernisation/netcdf-writer` — replaced by `pandas_openscm.db` on
  the application-layer side.
- `modernisation/iamc-loader` — RCMIP-style and SCI-style xlsx
  loading both belong above the wrapper.
- `modernisation/rcmip3-inputs` — RCMIP3 protocol loaders, the
  94-scenario registry, CO2-only ED mixed-mode via
  `secondary_source`, CICERO bundle IAMC translation script.
- `modernisation/rcmip3-runner` — `scripts/run_rcmip3.py`.
- `modernisation/cross-model-notebooks` — `notebooks/compare_*` demos.

The architecture-notes branch (this `UPSTREAM_MERGE_PLAN.md`,
`ARCHITECTURE_NOTES.md`, `PHASE_B_SCORECARD.md`, fork-specific README
sections) stays on the fork. Upstream's README scope statement is
updated to name the new adapters and the supported modes explicitly,
rather than removing the existing "emissions-driven only" language
silently.

## Mode enum

The wrapper grows a top-level `RunMode` enum, applied at the
`run.run` call site, applies to the whole call (no per-cfg or
per-scenario variation):

```python
from enum import Enum

class RunMode(str, Enum):
    EMISSIONS_DRIVEN = "emissions_driven"
    CONCENTRATION_DRIVEN = "concentration_driven"
```

Each adapter declares which modes it supports and raises
`NotImplementedError` for the others. The wrapper picks input
columns from `scenarios` accordingly (`Emissions|*` for ED,
`Atmospheric Concentrations|*` for CD). No dynamic mixing: the
fork's `secondary_source` mixed-mode disappears from the wrapper
entirely and lives application-side.

Further enum values (e.g. `ERF_DRIVEN`, named idealised protocols)
get added only when a concrete adapter-side wiring need surfaces.

## Test scope

2-5 member ensembles only, parametrised by adapter and ensemble cfg.
Single new file `tests/integration/test_modern_adapters.py`:

```python
@pytest.mark.parametrize(
    "scm, ensemble_cfg",
    [
        pytest.param("FaIRv2", FAIR2_ENSEMBLE_A, id="fair2-ens-a"),
        pytest.param("FaIRv2", FAIR2_ENSEMBLE_B, id="fair2-ens-b"),
        pytest.param("CICERO-SCM-PY2", CICERO_ENSEMBLE_A, id="cicero-ens-a"),
        pytest.param("CICERO-SCM-PY2", CICERO_ENSEMBLE_B, id="cicero-ens-b"),
    ],
)
def test_emissions_driven(scm, ensemble_cfg):
    scenarios = _load_test_scenarios()
    result = openscm_runner.run.run(
        climate_models_cfgs={scm: ensemble_cfg},
        scenarios=scenarios,
        mode=RunMode.EMISSIONS_DRIVEN,
        output_variables=(
            "Surface Air Temperature Change",
            "Effective Radiative Forcing",
        ),
    )
    # variables present, 2100 GSAT in plausible band, no NaNs


@pytest.mark.parametrize(...)
def test_concentration_driven(scm, ensemble_cfg):
    # mirror of the above with mode=RunMode.CONCENTRATION_DRIVEN
```

`ensemble_cfg` is a small list of 2-5 dicts of model parameters —
the same per-cfg API the existing FaIR 1.6 / MAGICC adapters use.
Test scenarios are loaded from a tiny in-repo CSV (one SSP, one
idealised protocol) so the tests don't depend on any external
bundle.

Files we drop from the upstream PR (kept on the fork's feature
branches for our own iteration):

- `tests/unit/adapters/test_fair2*.py` (5 files, ~61 tests)
- `tests/unit/adapters/test_ciceroscmpy2*.py` (3 files, ~38 tests)
- `tests/unit/test_iamc_loader.py` (24 tests)
- `tests/unit/test_rcmip3.py` (40 tests)
- `tests/unit/test_run_rcmip3_script.py` (14 tests)
- `tests/unit/test_output.py` NetCDFChunkWriter cases (~7 tests)
- `tests/unit/test_scmdata_patches.py` (6 tests) — patches move to scmdata in PR A; shim and tests deleted entirely
- `src/openscm_runner/_scmdata_patches.py` and its import-time call in `src/openscm_runner/__init__.py` (the shim itself) — deleted in PR B once scmdata releases the fix
- `PHASE_B_SCORECARD.md`, `ARCHITECTURE_NOTES.md`, this file
- `notebooks/figures/*` (gitignored; PNGs don't ship)
- Fork-specific README sections

## CI coverage

With native bundle loading stripped, the adapters no longer need
calibration bundles to import. The four parameterised tests use a
small in-repo CSV of scenarios and 2-5 member synthetic ensembles,
so upstream CI runs them without any external fetch.

Marit's `rcmip-march2026` bundle and FaIRv2's Zenodo calibration
record both stay on the application-layer side as the inputs to
bundle expansion.

## FaIR 1.6 / MAGICC compatibility

FaIR 1.6 and MAGICC numerical results must stay stable across this
merge — only plumbing tweaks are acceptable. Verification before
PR B opens: run the existing FaIR 1.6 + MAGICC test suites against
our modified source on the modern stack and confirm they pass with
unchanged answers.

If `tests/integration/test_fair.py` or `tests/unit/test_fair1x_utils.py`
fail under the modern stack:

1. **First try**: install whatever `fair` resolves to from the `[fair]`
   extra. The fork's `_scmdf_to_emissions.py:137` already uses
   `int(row[row].index[0]) + 1` (numpy 2.x-safe).
2. **If CI fails**: pin `fair` to OMS-NetZero/FAIR's `v1.6.2-gcages`
   branch in `pyproject.toml`:
   ```toml
   fair = { git = "https://github.com/OMS-NetZero/FAIR.git", branch = "v1.6.2-gcages", optional = true }
   ```
3. **If that still fails**: add the import-time monkey patch
   (`_get_fair_col_unit_context_fixed` using
   `int(row[row].index.values.squeeze()) + 1`) in
   `src/openscm_runner/adapters/fair_adapter/_compat.py`.

Push access to OMS-NetZero/FAIR can be requested if direct commits to
the v1.6.2-gcages branch are needed.

## Application layer (on fork, perhaps move to WGI repo in next phase)

The bulk of the modernisation lives in an application layer outside
openscm-runner — currently on this fork, possibly migrating to a
future IPCC/CICERO-hosted repo. What's in it:

- **RCMIP3 protocol**: `load_rcmip3_emissions`,
  `load_rcmip3_concentrations`, the 94-scenario registry, the
  CO2-only ED mixed-mode (CO2 emissions + non-CO2 concentrations
  pre-baked into a single scenario before the wrapper sees it),
  idealised-scenario natural-forcing suppression (sunvolc, natemis,
  LUC, emstart, PI-flat aerosols).
- **Bundle expansion**: turn Marit's `rcmip-march2026` and FaIRv2's
  Zenodo calibration into N per-member cfg dicts, then hand the
  list to `openscm_runner.run.run` through the standard API. The
  bundle path never enters the wrapper.
- **Runner**: `scripts/run_rcmip3.py` orchestrates per-(mode, model,
  scenario) batches.
- **Chunked output**: `pandas_openscm.db` replaces the in-fork
  `NetCDFChunkWriter`.
- **IAMC loaders**: both RCMIP-style and SCI-style xlsx; SCI-side
  translation from Scenario Compass names into openscm-runner names
  uses [`gcages.databases.emissions_variables`](https://github.com/openscm/gcages/blob/main/src/gcages/databases/emissions_variables.py)
  as the canonical source.
- **Cross-model notebooks**: `notebooks/compare_*` demos.
- **Validation**: `scripts/validate_against_marit.py`,
  `scripts/compare_flat10_zec_marit.py`, the
  `PHASE_B_SCORECARD.md`.
- **Fork-only unit tests**: the ~200 we wrote during the
  modernisation.

Whether to split this into a separate package or keep it as the
`modernisation/*` branches on the fork is a follow-up decision once
PR B is in.

## Resolved

- **Sequencing**: 2-shot. PR A (scmdata) lands and gets tagged
  first; PR B opens against `openscm/openscm-runner@main` once
  that pin is available, with the in-tree shim removed.
- **Scope split**: native bundle loading, default calibrations,
  mixed driver-mode, RCMIP3 loader/runner, SCI translation,
  chunked-output writer, and comparison notebooks all live in the
  application layer rather than in PR B.
- **Mode**: explicit `RunMode` enum at the `run.run` call site, no
  dynamic mixing, no per-cfg toggle.
- **NetCDFChunkWriter**: application layer uses `pandas_openscm.db`.
- **SCI translation**: application layer routes through `gcages`.
- **climate-assessment**: not a target; gcages is the canonical
  AR7 ScenarioMIP workflow tool.
- **Marit's tests, scorecard, validation scripts**: stay on the
  fork.

## Open questions

1. **Concentration-driven mode**: ship `RunMode.CONCENTRATION_DRIVEN`
   in PR B alongside `EMISSIONS_DRIVEN`, or hold it back to a
   follow-up PR? CD on the fork is fine if upstream wants to start
   ED-only.
2. **`check_variables_are_as_expected`**: source the canonical name
   list from `gcages.databases.emissions_variables`, or bake the
   same list into openscm-runner so the runner doesn't take a new
   import dependency?

## Verification

Once PR B is assembled on a local `upstream-merge` branch off
`openscm/openscm-runner@main`:

```bash
# All existing upstream unit tests (including FaIR 1.6, MAGICC) pass
# with our source changes; numerical answers stable.
pytest tests/ -q --ignore=tests/integration

# The new parameterised integration tests pass without any external
# bundle fetch.
pytest tests/integration/test_modern_adapters.py -v
```

Push, open PR, watch CI. Apply the FaIR 1.6 fallback per the
section above if needed.

For PR A standalone (against `openscm/scmdata`):

```bash
# In the scmdata checkout, after applying the four fixes
pytest tests/ -v   # existing scmdata tests stay green + new patch tests pass
```

For the application layer (fork-side), the existing RCMIP3 runner
and notebooks are the verification harness; nothing about the
narrowed PR B changes those workflows.
