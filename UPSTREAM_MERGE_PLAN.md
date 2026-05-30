# Upstream merge plan

How the modernisation fork at `benmsanderson/openscm-runner` will land at
upstream `openscm/openscm-runner`. Written for review by Zeb (upstream
historical lead) and Marit (CICERO-SCM 2.x lead, owner of the
calibration bundle the CICEROSCMPY2 adapter is wired to), and anyone
else looking at the fork before the PRs open.

## Background

The fork carries an AR7-cycle modernisation pass of openscm-runner:
two new SCM adapters (FaIRv2 and CICERO-SCM 2.1), an RCMIP3 protocol
loader and runner, streaming netCDF output, and a pandas-3.0 /
xarray-2025+ compatibility shim against scmdata. All in-flight feature
branches are listed in
[`scripts/rebuild_integration_branch.sh`](scripts/rebuild_integration_branch.sh);
`modernisation/integration` is the all-in-one tree where new work lands
before being backported to its target branch (see
[`ARCHITECTURE_NOTES.md`](ARCHITECTURE_NOTES.md) for the design rationale
and [`PHASE_B_SCORECARD.md`](PHASE_B_SCORECARD.md) for the current
state of the CICEROSCMPY2 validation against Marit's RCMIP3 reference).

Zeb's response to our outreach (paraphrased):

> Let's jam this through. Add tests for the key use cases you care about
> but don't add any other tests, I'll just delete them anyway. Probably
> ends up as four parameterised tests: FaIRv2 ED, FaIRv2 CD, CICERO 2.1
> ED, CICERO 2.1 CD. Go crazy on scmdata as you need. If existing FaIR
> 1.6 tests break under modern stacks, switch to
> OMS-NetZero/FAIR@v1.6.2-gcages.

This plan is shaped around that signal: one PR for the scmdata
compatibility shim, one for everything else, minimal tests, no
backport ceremony.

## Two PRs against `openscm/openscm-runner@main`

### PR A: pandas-3 / xarray-2025+ fixes against `openscm/scmdata`

Four fixes to the affected scmdata source files (not a shim in
openscm-runner). Zeb maintains both repos and our last scmdata PRs
merged quickly, so the proper fix at the source goes in the same FOD
window. Once a scmdata release is cut, openscm-runner just pins to
the fixed version and the in-tree shim is deleted as part of PR B.

| Patch | scmdata file | Change |
|---|---|---|
| 1 | `scmdata/groupby.py` (`RunGroupBy.__init__`) | Replace `np.issubdtype(col.dtype, np.number)` with `pd.api.types.is_numeric_dtype(col)` so StringDtype meta columns don't trip the numeric-column detection |
| 2 | `scmdata/_xarray.py` (`_many_to_one`) | Replace `.max()[0]` with `.max().iloc[0]` (pandas 3 removed positional indexing on label-indexed Series) |
| 3 | `scmdata/run.py` (`ScmRun.convert_unit`) | Replace `group._df.values[:] = ...` with `group._df.iloc[:, :] = ...` since `DataFrame.values` is read-only in pandas 3 |
| 4 | `scmdata/netcdf.py` (`_read_nc`) | Replace `xr.load_dataset(fname, use_cftime=True)` with `xr.load_dataset(fname, decode_times=CFDatetimeCoder(use_cftime=True))` to silence the FutureWarning xarray 2025+ emits on every `ScmRun.from_nc` call |

Tests for each patch land in scmdata's own test suite alongside the
fixes (4-6 tests covering the broken cases). The shim and its tests
currently in this repo (`src/openscm_runner/_scmdata_patches.py`,
`tests/unit/test_scmdata_patches.py`, the import-time call in
`src/openscm_runner/__init__.py`) are deleted in PR B; the runner's
pyproject pins `scmdata>=<fixed-release>`.

**Sequencing**: open PR A first; Zeb merges + cuts a scmdata release;
open PR B against `openscm/openscm-runner@main` with the new scmdata
pin and the shim removed.

### PR B: everything else

Source code from these feature branches merges in, in this order, with
their test additions dropped (see "Test scope" below):

| Feature branch | Brings |
|---|---|
| `modernisation/python-3.12` | Python version bump prerequisite |
| `modernisation/worker-counts` | SLURM-aware worker counts in adapter parallel dispatch |
| `modernisation/parallel-models` | Parallel model dispatch in `run.run` (may already be folded into worker-counts; verify) |
| `modernisation/netcdf-writer` | `output.py` with NetCDFChunkWriter; `[netcdf]` extra; netcdf4 added to `[notebooks]` |
| `modernisation/iamc-loader` | `openscm_runner.scenarios.load_iamc` for RCMIP-style and SCI-style xlsx |
| `modernisation/fair2-adapter` | FaIRv2 adapter (`FAIR2`), native calibration bundle, ED + CD, idealised forcing suppression, protocol-metadata consumption |
| `modernisation/ciceroscmpy2-adapter` | CICERO-SCM 2.1 adapter (`CICERO-SCM-PY2`), DistributionRun wrapper, RCMIP-aligned bundle handling, idealised PI-flat aerosols, protocol-metadata consumption |
| `modernisation/rcmip3-inputs` | `load_rcmip3_emissions`, `load_rcmip3_concentrations`, 94-scenario registry, CO2-only ED mixed-mode (`secondary_source`), CICERO bundle IAMC translation script + translated CSV |
| `modernisation/rcmip3-runner` | `scripts/run_rcmip3.py` CLI runner |
| `modernisation/cross-model-notebooks` | `notebooks/compare_*` demos (drop figures, keep `.py`) |

The architecture-notes branch (this `MERGE_PLAN.md`,
`ARCHITECTURE_NOTES.md`, `PHASE_B_SCORECARD.md`, fork-specific README
sections) stays on the fork; upstream gets the original upstream
README content restored.

## Test scope

We have about 200 unit tests we wrote during the modernisation. None
of them ship upstream per Zeb's ask. Just the four parameterised
integration tests below, in a single new file
`tests/integration/test_modern_adapters.py`:

```python
@pytest.mark.parametrize("adapter,conc_driven", [
    ("FaIRv2",         False),  # emissions-driven
    ("FaIRv2",         True),   # concentration-driven
    ("CICERO-SCM-PY2", False),  # emissions-driven
    ("CICERO-SCM-PY2", True),   # concentration-driven
])
def test_adapter_ssp245_smoke(adapter, conc_driven):
    scenarios = load_rcmip3_emissions(["ssp245"])
    result = openscm_runner.run.run(
        climate_models_cfgs={adapter: [_build_cfg(adapter, conc_driven)]},
        scenarios=scenarios,
        output_variables=(
            "Surface Air Temperature Change",
            "Atmospheric Concentrations|CO2",
            "Effective Radiative Forcing",
        ),
    )
    # variables present, 2100 GSAT in plausible band, no NaNs
```

CI runs both bundles for real (see "CI coverage" below); the skip
guards only fire on contributor machines where the env vars aren't
set. Optional fifth case: both adapters in one `run.run` call.

Files we drop from the upstream PR (kept on the fork's feature branches
for our own iteration):

- `tests/unit/adapters/test_fair2*.py` (5 files, ~61 tests)
- `tests/unit/adapters/test_ciceroscmpy2*.py` (3 files, ~38 tests)
- `tests/unit/test_iamc_loader.py` (24 tests)
- `tests/unit/test_rcmip3.py` (40 tests)
- `tests/unit/test_run_rcmip3_script.py` (14 tests)
- `tests/unit/test_output.py` NetCDFChunkWriter cases (~7 tests)
- `tests/unit/test_scmdata_patches.py` (6 tests) — patches move to scmdata in PR A; the shim and its tests are deleted entirely
- `src/openscm_runner/_scmdata_patches.py` and its import-time call in `src/openscm_runner/__init__.py` (the shim itself) — deleted in PR B once scmdata releases the fix
- `PHASE_B_SCORECARD.md`, `ARCHITECTURE_NOTES.md`, this file
- `notebooks/figures/*` (gitignored; PNGs don't ship)
- Fork-specific README sections

## FaIR 1.6 compatibility fallback

The upstream FaIR 1.6 adapter tests may or may not pass against
current `fair`. The plan:

1. **First try**: install whatever `fair` resolves to from the `[fair]`
   extra and run the upstream tests. The fork's
   `_scmdf_to_emissions.py:137` already uses `int(row[row].index[0]) + 1`
   (numpy 2.x-safe), which is the same intent as the slicing change
   Zeb suggested.
2. **If CI fails**: pin `fair` to OMS-NetZero/FAIR's `v1.6.2-gcages`
   branch in `pyproject.toml`:
   ```toml
   fair = { git = "https://github.com/OMS-NetZero/FAIR.git", branch = "v1.6.2-gcages", optional = true }
   ```
3. **If that still fails**: add Zeb's monkey patch
   (`_get_fair_col_unit_context_fixed` using
   `int(row[row].index.values.squeeze()) + 1`) as an import-time
   patch in `src/openscm_runner/adapters/fair_adapter/_compat.py`.

We'd ask Zeb for push access to OMS-NetZero/FAIR if we need to commit
fixes to the v1.6.2-gcages branch directly.

## CI coverage

The four parameterised tests need bundles to do anything useful. The
existing upstream convention is to skip when env vars are absent, which
silently hides coverage gaps. We'd rather have CI exercise the new
adapters end-to-end:

- **FaIRv2**: the calibration CSVs (~2 MB) are already published on
  Zenodo (record `18828694`). `scripts/download_fair2_calibration.py`
  is in the fork; we wire it into the CI workflow and set
  `FAIR2_CALIBRATION_PATH` from the cached path.
- **CICERO-SCM-PY2**: the `rcmip-march2026` bundle Marit produced is
  what these tests need. It's not on Zenodo yet. **Ask for Marit**:
  publish it (or a CI-sized 10-member subset) under a citable DOI so
  the CI workflow can fetch + cache it the same way we fetch the FaIRv2
  bundle. Once that's available, CI sets `CICEROSCMPY2_BUNDLE_DIR` from
  the cache path and the four tests all run for real.

The CI workflow change goes in PR B alongside the test file; both
bundle downloads are cached by hash of the upstream record so the cost
is one fetch per cache miss.

## What we're not bringing upstream

Out of scope for these PRs, kept on the fork for our own use:

- `scripts/validate_against_marit.py`, `scripts/compare_flat10_zec_marit.py`
  (debugging utilities for matching our CICEROSCMPY2 output against
  Marit's 500-member reference). Useful tools but not runtime; we'd
  open them in a follow-up if anyone asks.
- `scripts/translate_cicero_bundle_to_iamc.py` and the translated CSV
  — actually let's keep this; the loader depends on the CSV at runtime
  for the CH4-swap scen7 scenarios.
- `notebooks/compare_rcmip3.py` figures (the `.py` notebook ships but
  the regenerated PNGs are gitignored and don't go in the PR).
- The Phase B scorecard, this merge plan, and the fork-specific README
  sections — all stay on the fork.
- The SCI integration story (Zeb pointed us at `gcages` for that; we'd
  pick it up in a follow-up once these PRs are in).

## Resolved (decisions, not questions)

- **Sequencing**: 2-shot. PR A (scmdata) lands and gets tagged first;
  PR B opens against `openscm/openscm-runner@main` once that pin is
  available, with the in-tree shim removed.
- **Marit-specific tests, scorecard, validation scripts**: stay on the
  fork. `PHASE_B_SCORECARD.md`, `validate_against_marit.py`,
  `compare_flat10_zec_marit.py` and the ~99 fork-only unit tests for
  the new adapters all live on `modernisation/integration` and its
  feature branches; upstream only sees source + the four parameterised
  integration tests.

## CICERO-SCM needs


1. Can we publish the `rcmip-march2026` CICERO-SCM calibration bundle
   to Zenodo under a citable DOI? A CI-sized subset (10 members) is
   fine if the full ensemble is awkward to host. Without it the
   upstream CI either skips the CICERO tests entirely or relies on a
   path we ship in-tree, neither of which is great.
2. Same question for any future bundle revisions: ideally each version
   we calibrate against gets its own DOI so the runner can pin to a
   known artefact rather than a mutable directory.

## Verification

Once PR B is assembled on a local `upstream-merge` branch off
`openscm/openscm-runner@main`:

```bash
# All existing upstream unit tests pass with our source changes
pytest tests/ -q --ignore=tests/integration

# The four new integration tests pass with bundles present
FAIR2_CALIBRATION_PATH=$PWD/configurations/fair-calibrate-v1.6.0 \
CICEROSCMPY2_BUNDLE_DIR=$PWD/configurations/ciceroscm \
  pytest tests/integration/test_modern_adapters.py -v

# Smoke-test the unified runner end-to-end
scripts/run_rcmip3.py --members 2 --scenarios ssp245 --mode both
```

Then push, open PR, watch CI. Apply the FaIR 1.6 fallback per the
section above if needed.

For PR A standalone (against `openscm/scmdata`):

```bash
# In the scmdata checkout, after applying the four fixes
pytest tests/ -v   # existing scmdata tests stay green + new patch tests pass
```
