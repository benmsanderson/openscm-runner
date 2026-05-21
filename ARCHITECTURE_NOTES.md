# openscm-runner architecture notes (AR7 modernisation)

Working notes ahead of the modernisation work (FaIR 2.x adapter, CICERO-SCM Python adapter
refresh, native calibration passthrough, concentration-driven runs). Captures how the
current code is organised so we know what to change and what to leave alone.

## Public surface

The library exposes one user-facing entry point, `openscm_runner.run.run`
([src/openscm_runner/run.py:32](src/openscm_runner/run.py#L32)). It takes:

- `climate_models_cfgs: dict[str, list[dict]]` (one config dict per ensemble member per model),
- `scenarios: scmdata.ScmRun | pyam.IamDataFrame` (IAMC-style hierarchical variable names),
- `output_variables: Iterable[str]`,
- `out_config: dict[str, tuple[str, ...]] | None` (which input-config fields to keep on the
  output `ScmRun` as metadata).

It loops over models, calls `get_adapter(name).run(...)`, asserts the per-model results share
the same meta columns, and `scmdata.run_append`s them. There is no central translation layer:
the orchestrator hands the raw `ScmRun` to each adapter and trusts the adapter to interpret it.

## Adapter pattern

`_Adapter` ([src/openscm_runner/adapters/base.py:9](src/openscm_runner/adapters/base.py#L9))
is a small ABC with:

- `model_name` class attribute (case-insensitive lookup key),
- `_init_model(*args, **kwargs)` called from `__init__` (typically ImportErrors if the model
  isn't installed),
- `_run(scenarios, cfgs, output_variables, output_config)` returning an `ScmRun` with
  conventional meta columns (`climate_model`, `model`, `scenario`, `region`, `variable`,
  `unit`, `run_id`, plus any `output_config` keys),
- a `get_version()` classmethod by convention (used in tests and in the `climate_model`
  metadata).

Registration is a module-level list in
[src/openscm_runner/adapters/__init__.py:11](src/openscm_runner/adapters/__init__.py#L11);
`register_adapter_class` allows third-party registration. Adding a new adapter is purely
additive: subclass `_Adapter`, add to the list (or call the register function), provide a
`_compat.py` shim, declare an extra in `pyproject.toml`.

## Existing adapters at a glance

| Adapter | Model dep | Parallel unit | Notes |
| --- | --- | --- | --- |
| `FAIR` (fair_adapter) | `fair<2` (FaIR 1.6) | one process per (scenario, cfg) | Hardcoded scientific defaults in `_make_full_cfgs` ([fair_adapter.py:101](src/openscm_runner/adapters/fair_adapter/fair_adapter.py#L101)). Emissions translated to a `(nt, 40)` array via `_scmdf_to_emissions.py`, with historical and Montreal-gas fill-in from a bundled RCMIP CSV. Output cherry-picks fixed species columns from FaIR's 7-tuple return. `output_config` not supported. `startyear` kwarg already exists; `endyear` derived from scenario. |
| `MAGICC7` | `pymagicc<3` (calls a Fortran binary) | one process per cfg, shared MAGICC instance per worker | Uses `pymagicc.definitions.convert_magicc7_to_openscm_variables` plus a tiny local `_VARIABLE_MAP` for HFC4310mee and surface temperature. Writes a `SCEN7` file per scenario; passes per-cfg kwargs into `pymagicc.MAGICC7.run(**cfg)`. `output_config` IS implemented (mirrors cfg into output meta, with warnings on mismatch). |
| `CICEROSCM` (Fortran) | bundled binary in `utils_templates/run_dir/scm_vCH4fb` | one process per scenario, cfgs iterated serially inside the worker | Writes `<scen>_em.txt` and a parameter file by template substitution. Result reading from text files via `read_results.py`. `output_config` not supported. |
| `CICEROSCMPY` | `ciceroscm==1.1.1` (pinned, Jan 2024) | one process per scenario, cfgs serial | Constructs one `cscmpy.CICEROSCM(...)` per scenario with hardcoded gaspam / natemis / concentration files (TODO comments flag this), then loops calling the **private** `self.cscm._run(...)` with `pamset_udm` and `pamset_emiconc`. Does not pass `pamset_carbon`. Does not use `ConfigDistro` / `DistributionRun`. Variable mapping shares `openscm_to_cscm_dict` with the Fortran adapter. The pinned version pre-dates the v2.0.0 rewrite by almost two years; see CICERO-SCM v2.1.0 notes below. |

Two CICERO adapters share `src/openscm_runner/adapters/utils/cicero_utils/`:
`COMMONSFILEWRITER` (emissions translation via `cicero_comp_dict` plus a `gases_v1RCMIP.txt`
unit table) and `_run_ciceroscm_parallel.py` (per-scenario `ProcessPoolExecutor` wrapper).
Output forcing aggregation lives in `cicero_forcing_postprocessing_common.py`.

Worker counts and temp dirs are configurable via env / dotenv through
[src/openscm_runner/settings.py](src/openscm_runner/settings.py):
`FAIR_WORKER_NUMBER`, `MAGICC_WORKER_NUMBER`, `MAGICC_WORKER_ROOT_DIR`,
`MAGICC_EXECUTABLE_7`, `CICEROSCM_WORKER_NUMBER`, `CICEROSCM_WORKER_ROOT_DIR`.

## Variable name mapping

There is no central registry. Each adapter owns its own translation in both directions:

- **Input emissions** are accepted as IAMC names (`Emissions|CO2|MAGICC Fossil and Industrial`,
  `Emissions|CH4`, `Emissions|Sulfur`, etc.). MAGICC renames Sulfur->SOx, HFC4310mee->HFC4310,
  VOC->NMVOC inside its adapter. FaIR 1.6 has its own ordered species table
  `EMISSIONS_SPECIES_UNITS_CONTEXT` and silently ignores species not in it. CICERO uses
  `cicero_comp_dict` and a tabular `gases_v1RCMIP.txt` to drive unit conversion via
  `openscm_units`.
- **Output variables** use a hierarchical pipe-delimited convention
  (`Effective Radiative Forcing|Aerosols|Direct Effect|BC`,
  `Atmospheric Concentrations|CO2`). FaIR 1.6 produces them by hand-mapping FaIR's positional
  arrays; MAGICC delegates to `pymagicc.definitions`; CICERO uses `openscm_to_cscm_dict` in
  `cicero_forcing_postprocessing_common.py`.

`_AdapterTester._common_variables` in [src/openscm_runner/testing.py:85](src/openscm_runner/testing.py#L85)
is the closest thing to an output contract.

## Test suite

- `tests/unit/` covers `run.run` arg validation, the adapter registry, fair1x utilities,
  cicero utilities, settings, progress, installation.
- `tests/integration/` has one file per adapter plus `test_run_multimodel.py`.
- All adapter tests subclass `_AdapterTester` (in `src/openscm_runner/testing.py`) and
  implement `test_run` plus `test_variable_naming`.
- Numerical regression uses pytest-regressions `num_regression.check(output_dict, rtol=1e-5)`,
  so any scientific-default change will trip a regression and require the stored YAMLs to be
  regenerated deliberately. That is the right behaviour for us.
- `pytest.mark.magicc` and `pytest.mark.ciceroscm` skip when the model isn't available;
  fixtures live in [tests/conftest.py](tests/conftest.py) and use bundled SSP RCMIP CSVs in
  `tests/test-data/`.

## Concrete interface for a new FaIR 2.x adapter

Minimal additive shape (parallel to FaIR 1.6, not replacing it):

1. New package `src/openscm_runner/adapters/fair2_adapter/` with `_compat.py` importing
   `fair` (>=2.2), and `FAIR2` subclass of `_Adapter` with `model_name = "FaIRv2"` (unique
   versus the existing `"FaIR"`).
2. `_init_model` raises ImportError if FaIR 2.x is missing; optionally loads a default AR7
   calibration (zenodo.org/records/18828694) from an env-configurable path.
3. `_run` builds a single FaIR 2.x `FAIR` object per scenario (FaIR 2.x is natively
   xarray-batched, so configs become the `config` dimension rather than separate processes;
   parallelism falls to scenarios, not configs). It populates `Scenario` emissions from the
   `ScmRun`, runs, and returns an `ScmRun` with the same meta columns. `get_version` returns
   `fair.__version__`.
4. Register in `adapters/__init__.py`. Add a `fair2` extra in `pyproject.toml`.
5. Integration test under `tests/integration/test_fair2.py` subclassing `_AdapterTester`;
   reuse the existing RCMIP SSP fixtures plus, separately, a Scenario Compass sample.

Open scientific choices to flag and make configurable in the adapter (not hardcode): species
set / mapping defaults, default calibration ensemble, natural emissions and solar / volcanic
forcing extension protocols, harmonisation year. Tag each with a `# scientific choice:`
comment per [[feedback-working-style]].

## CICERO-SCM Python adapter: jumping to v2.1.0

The pinned `ciceroscm==1.1.1` (Jan 2024) is structurally far from the v2.1.0 target
(May 2026). v2.0.0 (Dec 2025) was a major rewrite, and the new adapter has to be written
against the v2.x API rather than incrementally bumped from 1.1.1.

Relevant v2.x changes that shape the adapter:

- **Modular submodels.** Thermal model and carbon cycle are now swappable: in addition to
  the standard versions, a simplified two-layer ocean and a simplified box-model carbon
  cycle are selectable. The adapter must expose the model-choice options through cfg.
- **New calibration pipeline.** The pre-v2 calibration code was removed; calibration is now
  a separate, richer pipeline. The native passthrough mode should accept the new pipeline's
  output directly. The legacy `pamset_udm` / `pamset_emiconc` / `pamset_carbon` JSON list of
  dicts plus `Index` remains the per-run config shape.
- **`ConfigDistro` / `DistributionRun` parallel API** (introduced in 1.5.0, retained in 2.x)
  replaces the per-cfg serial loop the current adapter does inside one scenario worker.
  Switching to it means the adapter delegates parallelism for cfgs to ciceroscm itself
  rather than running cfgs serially inside our own process pool. This will simplify
  `_run_ciceroscm_parallel.py` use and unlock real cfg-level parallelism.
- **RCMIP output protocol alignment** (v2.1.0) means the model output variable names should
  now align with the RCMIP convention, which should reduce or remove the need for
  `openscm_to_cscm_dict` translation on the output side. Worth checking how much of the
  existing output mapping is still required versus pass-through.
- **Dynamic preindustrial CO2 baseline** (v2.1.0): the 278.0 ppm hardcode is gone; baseline
  comes from the start-of-run value. Concentration-driven runs naturally fit this since the
  user supplies the concentration trajectory.
- **Aerosol pattern effects** (v2.1.0) need no adapter-side ceremony: they are switched on
  by a single entry `delta_lambda_aero` in `pamset_udm` (default `0.0`, bit-for-bit
  identical to without). Both shipped thermal models implement the capability, so users
  who pass it through the cfg get it for free. The one edge case is a startup `ValueError`
  if a non-zero value is paired with a third-party thermal model that doesn't implement
  the protocol; we don't need to handle that explicitly.
- **Regional aerosol species** (v2.0.0) are out of scope for v1 of the modernisation and
  do require adapter-side care (extra input data shape), so we just don't expose them.
- **Tropospheric O3 decoupled from fossil CO2 concentration** (v2.0.0) is a numerical
  behaviour change that will show up in the integration test regression baselines and
  needs a deliberate reseed when we switch versions.

Suggested approach: build the new `CICEROSCMPY2` (or `CSCMPY2`) adapter as a sibling to the
existing one rather than an in-place upgrade. Pin the new extra to `ciceroscm>=2.1.0,<3`,
keep the 1.1.1-pinned `ciceroscmpy` extra and adapter working for backwards-compat,
deprecate the old one only after the new one is shipped and used.

## Core changes for native calibration passthrough

The current `cfgs: list[dict]` represents one runner-translated ensemble member per dict.
Native passthrough means letting the user hand the adapter the model's own calibration object
(FaIR 2.x xarray netCDF, CICERO-SCM JSON list-of-dicts with `pamset_udm` / `pamset_emiconc` /
`pamset_carbon` / `Index`) without runner translation.

Recommended approach (additive, mirrors how `output_config` was added):

- Add an optional kwarg to `_Adapter._run` (e.g. `native_calibration=None`) and thread it
  through `run.run`. Adapters that don't support it should ignore it or raise
  `NotImplementedError` (the precedent set by `output_config` in
  [fair_adapter.py:54](src/openscm_runner/adapters/fair_adapter/fair_adapter.py#L54) and
  the CICERO adapters).
- For each new adapter (FaIR 2.x, CSCM-Py modernised), implement `native_calibration` as
  either a file path or an in-memory object; when supplied, bypass the per-cfg translation
  path and call the model with its native ensemble API
  (`fair.fill_from_csv` / xarray Dataset for FaIR 2.x; `DistributionRun(ConfigDistro(...))`
  for CSCM-Py).
- Keep the existing translated-cfg path working for backwards compatibility (existing
  downstream users like climate-assessment will continue to call it).

## Core changes for concentration-driven runs and flexible start dates

`run.run` is variable-agnostic, so the change is mostly inside adapters and in the input
schema convention.

- Allow `Atmospheric Concentrations|*` variables alongside `Emissions|*` in the input
  `ScmRun`. Each adapter inspects the input and routes appropriately (FaIR 2.x and CSCM
  natively support both drivers; the open-source MAGICC will too).
- Surface a driver-mode toggle per cfg (e.g. `driver_mode: {"CO2": "concentration", ...}`)
  so mixed runs (CO2 conc-driven, non-CO2 emissions-driven, classic for DECK and idealised
  experiments) are expressible. Default behaviour stays emissions-driven for back-compat.
- Generalise `startyear` / `endyear` handling. FaIR 1.6 already exposes `startyear`
  ([fair_adapter.py:149](src/openscm_runner/adapters/fair_adapter/fair_adapter.py#L149));
  CSCM-Py exposes `nystart` / `emstart`. Make both kwargs explicit on the public adapter
  interface so historical-splice flexibility is consistent.
- Update the README to remove the "emissions driven runs only" claim once
  concentration-driven is wired in either adapter.

## Cluster usability (precondition for AR7 uptake)

In the AR6 cycle most modelling groups ran their own SCM rather than going through
openscm-runner, partly because submitting a multi-SCM, multi-thousand-member ensemble on a
cluster through the wrapper was painful. For AR7 the wrapper has to be easier than calling
the model directly, otherwise we get the same outcome. None of the issues below are strict
blockers, but together they will keep cluster users away. They are small, surgical, and
should land before or alongside the adapter work, not after.

- **Parallel model dispatch in `run.run`.** Today
  [run.py:74](src/openscm_runner/run.py#L74) loops over `climate_models_cfgs.items()`
  serially, so on a 100-core allocation with four SCMs three are idle at any moment.
  Fix: top-level `ProcessPoolExecutor` across models (processes, not threads, for
  consistency with what every adapter already does internally and to avoid GIL surprises
  from adapters that don't reliably release it), with an optional sequential fallback for
  debugging. One small PR, unblocks multi-SCM ensembles.
- **Respect cluster CPU allocations.** Every adapter falls back to
  `multiprocessing.cpu_count()` when its `*_WORKER_NUMBER` env var is unset; on a SLURM
  node `cpu_count()` reports the whole physical node rather than your `--cpus-per-task`
  share, which oversubscribes when two jobs share a node. Fix: prefer
  `SLURM_CPUS_PER_TASK`, then `OMP_NUM_THREADS`, then `cpu_count()`. One helper in
  `settings.py`, called from each adapter's worker-count lookup.
- **Chunked / streaming output.** `run.run` accumulates the whole ensemble in memory via
  `scmdata.run_append` ([run.py:112](src/openscm_runner/run.py#L112)). Fine at SSP scale,
  OOMs at Scenario Compass scale (~1600 scenarios x O(1000) configs x N variables x ~351
  years). Fix: an optional `output_writer=...` kwarg that writes per-(scenario, model)
  chunks to disk as netCDF (climate-community-friendly, xarray-native, plays well with
  CF conventions and downstream tooling) instead of returning one in-memory `ScmRun`.
  Default behaviour stays the same so existing callers don't change.

Beyond these three, cross-node parallelism (dask, mpi4py) and resume-from-disk
checkpointing are bigger lifts and are out of scope for v1; users can stripe across nodes
via SLURM job arrays as long as the three above are in place. The CICERO-SCM per-scenario
cfg-serial loop ([ciceroscm_wrapper.py:91](src/openscm_runner/adapters/ciceroscm_adapter/ciceroscm_wrapper.py#L91))
falls out naturally when the v2.1.0 adapter delegates cfg-level parallelism to
`ConfigDistro` / `DistributionRun`.

## Backwards compatibility stance

Decided 2026-05-22 ahead of any code changes.

- **Soft backward compat.** Public API of `run.run` may gain modest kwargs (e.g.
  `parallel_models=...`, `output_writer=...`) provided existing positional / keyword calls
  keep working with their current behaviour. Defaults can be changed in a later release
  but not in the same PR that introduces them.
- **Existing adapters keep working by default.** Fortran CICERO-SCM and MAGICC7 (pymagicc)
  are non-negotiable: they stay. FaIR 1.6 stays if cheap, but is not critical: if keeping
  it working becomes a real maintenance drag (e.g. shared utility code starts to diverge
  awkwardly to support both 1.6 and 2.x), it can be dropped with a deprecation cycle.
- **Numerical regression baselines may move** where the underlying model changed (the
  CSCM v2.0.0 tropospheric O3 refactor is the most obvious case). Baseline regenerations
  should land in their own PR with a one-line explanation in the changelog.
- **Downstream user check.** IIASA's climate-assessment is actively developed for AR7 and
  is the de-facto compatibility target: non-breaking changes preferred, but they can adapt
  to modest signature additions. Worth a courtesy heads-up before any change that touches
  the public `run.run` signature, even an additive one.

In practice: pre-1.0 we add kwargs and keep existing adapters. At a 1.0 cut we revisit
defaults and FaIR 1.6 retention.

## Things to leave alone (per scope)

- scmdata / pyam scenario representation (input shape stays the same).
- Existing variable name mapping infrastructure (we add to it, do not refactor).
- FaIR 1.6, Fortran CICERO-SCM, MAGICC7 (pymagicc) adapters: keep functional, add tests
  guarding their existing numerical regression baselines.
- Hector and OSCAR adapters: out of scope for v1.
