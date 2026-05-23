"""
Adapter for FaIR 2.x (native-calibration mode, v1 scope).

This adapter is a sibling to :class:`openscm_runner.adapters.fair_adapter.FAIR`
(which wraps FaIR 1.6) and does not replace it. The 1.6 adapter is kept
working for downstream users that depend on its outputs; this one
targets the AR7-relevant calibrations of FaIR 2.x.

**Scope of this first version**

Native-calibration mode only. Each entry in the per-model ``cfgs`` list
must carry a ``native_calibration`` sidecar key pointing at a FaIR 2.x
calibration bundle (a directory of CSVs as published on Zenodo, e.g.
https://zenodo.org/records/18828694). The runner loads the bundle,
builds a single :class:`fair.FAIR` instance per cfg entry whose
``config`` dimension is the calibration's parameter posterior (optionally
subset by ``member_indices``), and dispatches to FaIR's native xarray
ensemble API.

Translated-cfg mode (the FaIR 1.6 adapter's pattern of hand-built
per-member dicts) is deliberately **not** implemented in this PR. It
will land in a follow-up; until then, cfg dicts without a
``native_calibration`` key raise :class:`NotImplementedError`.

Variable mapping is currently minimal: emissions inputs supported are
the four main GHGs (CO2 FFI, CO2 AFOLU, CH4, N2O), and outputs are
Surface Air Temperature Change, total Effective Radiative Forcing, and
the CO2 concentration. Unmapped emissions species are logged at WARNING
and ignored (FaIR's bundle-provided historical / default values are
used in their place). The full mapping is a follow-up; see the
architecture notes.

**Per-cfg sidecar keys**

- ``native_calibration`` (native mode, required): filesystem path to
  the bundle directory, or an in-memory
  :class:`NativeFairCalibration` instance.
- ``member_indices`` (native mode, optional, default = all members):
  sequence of zero-based ``int`` selecting which rows of the
  parameter posterior to use as the FaIR config dimension.
  ``range(N)`` gives "first N members"; an explicit list lets the
  user stratify or reproduce a specific subset.
- ``emissions_bundle`` (translated mode, optional): filesystem path
  to a calibration bundle (or a :class:`NativeFairCalibration`).
  When set, the bundle's historical emissions, species_configs and
  natural forcings are used as the baseline; only the cfg dict's
  climate parameters vary per member. All cfgs in one call must
  share the same value. When omitted, translated mode falls back to
  FaIR's ``fill_from_rcmip()`` (known to break against fair 2.2.4
  for some species).
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from scmdata import ScmRun, run_append

from ..base import _Adapter
from ._compat import HAS_FAIR2, fair2
from ._emissions_translator import build_emissions_df
from ._native_calibration import NativeFairCalibration
from ._output_extractor import extract_outputs

LOGGER = logging.getLogger(__name__)


class FAIR2(_Adapter):
    """
    Adapter for running FaIR 2.x with a native calibration bundle.

    Registered under ``model_name = "FaIRv2"`` (distinct from the
    existing ``"FaIR"`` which is the FaIR 1.6 adapter).
    """

    model_name = "FaIRv2"

    def _init_model(self, *args, **kwargs):
        if not HAS_FAIR2:
            raise ImportError(
                "FaIR 2.x is not installed. Run 'pip install \"fair>=2.2,<3\"' "
                "or 'pip install openscm-runner[fair2]'. Note this conflicts "
                "with the FaIR 1.6 adapter's 'fair<2' pin; only one major "
                "version of fair can be installed at a time."
            )

    def _run(self, scenarios, cfgs, output_variables, output_config):
        if output_config is not None:
            raise NotImplementedError(
                "`output_config` not implemented for FaIRv2"
            )

        # Mode detection: each cfg dict is either native (carries a
        # `native_calibration` sidecar key, expands to the calibration's
        # parameter posterior) or translated (FaIR 2.x parameter names
        # as dict keys, becomes a single config in FaIR's config dim).
        # Mixed cfgs are rejected for now; supporting them is a
        # straightforward follow-up but adds bookkeeping that is not
        # warranted by any current use case.
        native_flags = ["native_calibration" in cfg for cfg in cfgs]
        if all(native_flags):
            out = _run_native_cfgs(scenarios, cfgs, output_variables)
        elif not any(native_flags):
            out = _run_translated_cfgs(scenarios, cfgs, output_variables)
        else:
            raise NotImplementedError(
                "FaIRv2 cfgs must be all-native or all-translated; mixing "
                "native and translated cfgs in one call is not supported "
                "in v1. Run the two modes in separate run.run calls."
            )

        out["climate_model"] = f"FaIRv{self.get_version()}"
        return out

    @staticmethod
    def get_version():
        """Return the installed FaIR version string (e.g. ``"2.2.4"``)."""
        if not HAS_FAIR2:
            raise ImportError("FaIR 2.x is not installed")
        return fair2.__version__


def _resolve_calibration(value: Any) -> NativeFairCalibration:
    """
    Accept either a path-like or an already-loaded
    :class:`NativeFairCalibration` and return a calibration instance.
    """
    if isinstance(value, NativeFairCalibration):
        return value
    return NativeFairCalibration(value)


def _run_native_cfgs(scenarios, cfgs, output_variables) -> ScmRun:
    """
    Native-calibration dispatch: each cfg is a calibration choice
    that expands to its parameter posterior on FaIR's config dim.

    Each entry in ``cfgs`` becomes one FAIR run; results are
    concatenated. ``run_id`` is assigned sequentially across all
    members of all cfgs so the combined ScmRun has unique run_ids.
    """
    results = []
    run_id_offset = 0
    for cfg_index, cfg in enumerate(cfgs):
        calibration = _resolve_calibration(cfg["native_calibration"])
        member_indices = cfg.get("member_indices")
        members = calibration.select_members(member_indices)

        LOGGER.info(
            "Running FaIRv2 (cfg %d/%d, native) with %d ensemble "
            "members from %s",
            cfg_index + 1,
            len(cfgs),
            len(members),
            calibration.path,
        )

        scmrun_chunk = _run_one_calibration(
            scenarios=scenarios,
            calibration=calibration,
            members=members,
            output_variables=output_variables,
            run_id_offset=run_id_offset,
        )
        results.append(scmrun_chunk)
        run_id_offset += len(members)

    return run_append(results)


def _run_translated_cfgs(  # noqa: PLR0912, PLR0915
    scenarios, cfgs, output_variables
) -> ScmRun:
    """
    Translated-cfg dispatch: each cfg dict is one ensemble member
    whose keys are FaIR 2.x parameter names.

    All cfgs go into a single FAIR instance with ``config`` dim sized
    to ``len(cfgs)``, which is much faster than spawning one FAIR
    per cfg. Climate configs are populated from the cfg dicts.

    Two sub-modes, distinguished by the optional ``emissions_bundle``
    sidecar key:

    - **Bundle-backed (recommended).** Each cfg carries
      ``emissions_bundle`` pointing at a calibration bundle (all
      cfgs in one call must agree). Species configs, historical
      emissions splice (with user ScmRun on top) and natural
      forcings come from the bundle. This is the workhorse path
      for parameter-sweep studies on realistic emissions.

    - **RCMIP-defaults fallback.** No cfg carries
      ``emissions_bundle``. Species configs come from FaIR's
      shipped AR6 defaults and emissions are seeded from FaIR's
      ``fill_from_rcmip()``. Note: fair 2.2.4's RCMIP loader is
      known to break on HFC-4310mee (compound-convert key mismatch
      with the upstream RCMIP CSV); use the bundle-backed path
      until upstream fixes it.

    Supported cfg keys are anything FaIR 2.x exposes on
    ``climate_configs`` or ``species_configs``, plus the
    ``emissions_bundle`` sidecar. Unknown keys are logged at
    WARNING and ignored. If after population some required
    climate_configs value is still NaN, FaIR's ``run()`` raises
    ``ValueError`` which we re-raise with a hint pointing back at
    native-calibration mode.
    """
    from fair.interface import fill
    from fair.io import read_properties

    scenario_run = ScmRun(scenarios.timeseries()) if scenarios is not None else None
    scenario_names = (
        sorted(set(scenario_run["scenario"]))
        if scenario_run is not None and not scenario_run.empty
        else ["historical"]
    )

    # All cfgs in one call share a single FAIR instance, so they must
    # also share the bundle (which sets the species list, time axis
    # of historical emissions, and natural forcings). Reject mixed
    # configurations rather than silently using only one.
    bundle_values = [cfg.get("emissions_bundle") for cfg in cfgs]
    unique_bundles = {id(b) if not isinstance(b, (str, type(None))) else b
                      for b in bundle_values}
    if len(unique_bundles) > 1:
        raise NotImplementedError(
            "All cfgs in a single FaIRv2 translated-cfg call must "
            "share the same `emissions_bundle` value (or all omit "
            "it). Got %d distinct values." % len(unique_bundles)
        )
    bundle_value = bundle_values[0]
    calibration = _resolve_calibration(bundle_value) if bundle_value else None

    # SCIENTIFIC CHOICE: same defaults as native mode (1750 start,
    # 1-year step). Made configurable in a follow-up.
    start_year = 1750
    end_year = (
        int(scenario_run.time_points.years().max())
        if scenario_run is not None and not scenario_run.empty
        else 2100
    )

    if calibration is not None:
        species, properties = read_properties(
            filename=calibration.file("species_configs")
        )
    else:
        species, properties = read_properties()

    config_labels = [f"config_{i}" for i in range(len(cfgs))]

    f = fair2.FAIR()
    f.define_time(start_year, end_year, 1)
    f.define_scenarios(scenario_names)
    f.define_configs(config_labels)
    f.define_species(species, properties)
    f.allocate()

    # allocate() leaves state arrays as NaN, which FaIR's run() then
    # propagates forward from t=0. Initialise to zero so the
    # integration has a defined starting point. See _run_one_calibration
    # for the equivalent in native mode.
    from fair.interface import initialise as _initialise

    _initialise(f.temperature, 0)
    _initialise(f.forcing, 0)
    _initialise(f.concentration, 0)
    _initialise(f.cumulative_emissions, 0)
    _initialise(f.airborne_emissions, 0)

    if calibration is not None:
        f.fill_species_configs(filename=calibration.file("species_configs"))
    else:
        f.fill_species_configs()

    LOGGER.info(
        "Running FaIRv2 (translated, %s) with %d ensemble members; "
        "climate_configs values come from cfg dicts",
        "bundle-backed" if calibration is not None else "RCMIP defaults",
        len(cfgs),
    )

    unknown_keys: set[str] = set()
    for cfg_idx, cfg in enumerate(cfgs):
        for key, value in cfg.items():
            if key == "emissions_bundle":
                continue  # sidecar, handled above
            if key in f.climate_configs:
                fill(f.climate_configs[key], value, config=config_labels[cfg_idx])
            elif key in f.species_configs:
                fill(f.species_configs[key], value, config=config_labels[cfg_idx])
            else:
                unknown_keys.add(key)

    if unknown_keys:
        LOGGER.warning(
            "FaIRv2 translated-cfg mode ignored unknown parameter "
            "names: %s. Valid names are FaIR 2.x climate_configs / "
            "species_configs keys.",
            sorted(unknown_keys),
        )

    if calibration is not None:
        bundle_emissions_csv = calibration.file("historical_emissions")
        if bundle_emissions_csv is not None or (
            scenario_run is not None and not scenario_run.empty
        ):
            emissions_df = build_emissions_df(
                scenario_run, bundle_emissions_csv, scenario_names
            )
            if not emissions_df.empty:
                f.fill_from_pandas(mode="emissions", df=emissions_df)
        _fill_natural_forcings(f, calibration)
    else:
        # No bundle: fall back to FaIR's RCMIP defaults. Known to
        # break on fair 2.2.4 for HFC-4310mee; recommend bundle path.
        # User emissions overrides via the ScmRun input are NOT merged
        # on top in this path (fill_from_pandas requires complete
        # coverage of every species, or it errors on the unit lookup).
        try:
            f.fill_from_rcmip()
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning(
                "FaIRv2 translated mode could not seed emissions from "
                "RCMIP defaults (%s). Pass `emissions_bundle` in each "
                "cfg to use a calibration bundle for emissions instead.",
                exc,
            )
        if scenario_run is not None and not scenario_run.empty:
            LOGGER.warning(
                "FaIRv2 translated mode without `emissions_bundle` "
                "ignores the user's `scenarios` input (RCMIP defaults "
                "are used). Pass `emissions_bundle` to splice user "
                "scenarios on top of bundle historicals."
            )

    try:
        f.run(progress=False, suppress_warnings=True)
    except ValueError as exc:
        if "NaN values" in str(exc):
            raise ValueError(
                "FaIR 2.x rejected the run because required "
                "climate_configs values are missing. Translated-cfg "
                "mode requires each cfg to fully specify the climate "
                "parameters FaIR needs (ocean_heat_capacity, "
                "ocean_heat_transfer, deep_ocean_efficacy, "
                "forcing_4co2 at minimum). For runs that should use "
                "a published calibration as the baseline, pass "
                "'native_calibration' in the cfg instead. Original "
                f"FaIR error: {exc}"
            ) from exc
        raise

    return extract_outputs(
        f,
        scenario_names,
        pd.DataFrame(index=range(len(cfgs))),  # one row per ensemble member
        output_variables,
        run_id_offset=0,
        properties_df=getattr(f, "properties_df", None),
    )


def _run_one_calibration(  # noqa: PLR0913
    scenarios,
    calibration: NativeFairCalibration,
    members: pd.DataFrame,
    output_variables,
    run_id_offset: int,
) -> ScmRun:
    """
    Run FaIR 2.x once with a single calibration choice and return the
    requested output variables as an :class:`scmdata.ScmRun`.

    Each row in ``members`` becomes one entry in FaIR's ``config``
    dimension. Output run_ids are ``run_id_offset + i`` so that
    concatenations across multiple calibration choices in the same
    ``run.run`` call stay unique.
    """
    from fair.io import read_properties

    scenario_run = ScmRun(scenarios.timeseries()) if scenarios is not None else None
    scenario_names = (
        sorted(set(scenario_run["scenario"]))
        if scenario_run is not None and not scenario_run.empty
        else ["historical"]
    )

    # Time axis spans the bundle's historical period through whatever the
    # scenario goes out to. Default to 2100 if no scenario provided.
    # SCIENTIFIC CHOICE: timestep=1 year, start=1750 (matches the bundle's
    # historical_emissions file). Made configurable in a follow-up.
    start_year = 1750
    end_year = (
        int(scenario_run.time_points.years().max())
        if scenario_run is not None and not scenario_run.empty
        else 2100
    )

    species, properties = read_properties(filename=calibration.file("species_configs"))

    # Config labels MUST match the row labels in the calibration CSV;
    # FaIR's override_defaults uses self.configs to index into the
    # parameter file via df_configs.loc[config, col], which is type-
    # sensitive. Pass the parameter DataFrame's index values through
    # with their native dtype (typically int seed labels for the
    # AR7-relevant fair-calibrate bundles).
    f = fair2.FAIR()
    f.define_time(start_year, end_year, 1)
    f.define_scenarios(scenario_names)
    f.define_configs(list(members.index))
    f.define_species(species, properties)
    f.allocate()

    # allocate() leaves temperature / forcing / concentration arrays
    # as NaN. FaIR's run() integrates forward from year 0 (1750) using
    # values from the previous timestep, so the NaN at t=0 propagates
    # to every subsequent timestep and the whole simulation comes out
    # NaN. Initialise the state variables to zero before populating
    # inputs and running.
    from fair.interface import initialise

    initialise(f.temperature, 0)
    initialise(f.forcing, 0)
    initialise(f.concentration, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    # Populate species configs and override defaults from the calibration
    # bundle. The override step writes the per-member parameter posterior
    # into FaIR's config dimension.
    f.fill_species_configs(filename=calibration.file("species_configs"))
    f.override_defaults(calibration.file("parameters"))

    # Splice the bundle's historical emissions with the user's scenario
    # data and let FaIR ingest the combined frame. Bundle historical
    # provides the baseline for every user scenario; user values
    # overwrite per-year where supplied. Species the user does not
    # provide (and species not in OPENSCM_TO_FAIR2_SPECIES) stay at
    # bundle-historical values for the historical period; FaIR's
    # interpolator handles missing future years by leaving NaN, which
    # the model treats as zero forcing for those species.
    bundle_emissions_csv = calibration.file("historical_emissions")
    if bundle_emissions_csv is None:
        LOGGER.warning(
            "Calibration bundle at %s does not contain %s. The user's "
            "scenario data will be passed straight through to FaIR; "
            "FaIR's interpolator will leave NaN for years the user did "
            "not cover.",
            calibration.path,
            calibration.FILES["historical_emissions"],
        )

    if bundle_emissions_csv is not None or (
        scenario_run is not None and not scenario_run.empty
    ):
        emissions_df = build_emissions_df(
            scenario_run, bundle_emissions_csv, scenario_names
        )
        if not emissions_df.empty:
            f.fill_from_pandas(mode="emissions", df=emissions_df)

    # Natural (solar / volcanic) forcings live in separate CSVs in the
    # bundle and use a forcing-mode input rather than emissions.
    _fill_natural_forcings(f, calibration)

    f.run(progress=False, suppress_warnings=True)

    return extract_outputs(
        f,
        scenario_names,
        members,
        output_variables,
        run_id_offset,
        properties_df=getattr(f, "properties_df", None),
    )


# SCIENTIFIC CHOICE: which column of the multi-scenario land_use /
# irrigation forcing CSVs to use. The bundle ships these forcings for
# several CMIP7 climate-target categories (VL, LN, L, ML, M, H, HL);
# we pick "M" (medium) as a sensible default. Will be made configurable
# (probably via a per-cfg key like `land_use_forcing_scenario`) when a
# real use case demands a specific mapping from user scenario names
# (e.g. ssp245) to these categories.
_DEFAULT_LAND_USE_SCENARIO = "M"


def _fill_natural_forcings(f, calibration: NativeFairCalibration) -> None:
    """
    Populate FaIR's forcing arrays for the bundle's forcing-input species.

    Solar, Volcanic, Land use, and Irrigation are forcing-input species
    in the v1.6.0 calibration bundle (the species_configs CSV overrides
    Land use and Irrigation from FaIR's default ``calculated`` mode to
    ``forcing`` mode). They bypass ``fill_from_pandas``'s emissions
    path, so we read each bundle CSV, reindex onto FaIR's timebounds,
    broadcast across scenario / config, and write into ``f.forcing``
    via ``fair.interface.fill``.

    The Solar and Volcanic CSVs are simple year + value; the
    Land use and Irrigation CSVs ship several CMIP7-target columns
    (VL, LN, L, ML, M, H, HL) and we pick ``_DEFAULT_LAND_USE_SCENARIO``
    (currently ``"M"``) with a warning.

    Missing CSVs leave the arrays at their default (zero) baseline.
    """
    import numpy as np

    from fair.interface import fill

    n_t = len(f.timebounds)
    n_scen = len(f.scenarios)
    n_cfg = len(f.configs)

    def _write(species_name, series):
        # Reindex onto FaIR's timebounds and fill missing as zero.
        series = series.reindex(f.timebounds).fillna(0.0)
        broadcasted = np.broadcast_to(
            series.values[:, None, None], (n_t, n_scen, n_cfg)
        )
        fill(f.forcing, broadcasted, specie=species_name)

    # Single-column long-form CSVs (Solar, Volcanic).
    single_col = {
        "Solar": ("solar_forcing", "solar_erf"),
        "Volcanic": ("volcanic_forcing", "volcanic_erf"),
    }
    for species_name, (bundle_key, value_col_prefix) in single_col.items():
        csv_path = calibration.file(bundle_key)
        if csv_path is None:
            LOGGER.warning(
                "Calibration bundle is missing %s; FaIR will use the "
                "default zero baseline for %s forcing.",
                calibration.FILES[bundle_key],
                species_name,
            )
            continue
        df = pd.read_csv(csv_path)
        year_col = next((c for c in df.columns if c.lower() == "year"), None)
        value_col = next(
            (c for c in df.columns if c.lower().startswith(value_col_prefix)),
            None,
        )
        if year_col is None or value_col is None:
            LOGGER.warning(
                "Unexpected layout for %s; expected year + %s_* columns, "
                "got %s. %s forcing left at default.",
                csv_path,
                value_col_prefix,
                list(df.columns),
                species_name,
            )
            continue
        _write(species_name, df.set_index(year_col)[value_col])

    # Multi-column per-scenario CSVs (Land use, Irrigation). First
    # column is the year index (unnamed); remaining columns are
    # CMIP7-target categories.
    multi_col = {
        "Land use": "land_use_forcing",
        "Irrigation": "irrigation_forcing",
    }
    for species_name, bundle_key in multi_col.items():
        csv_path = calibration.file(bundle_key)
        if csv_path is None:
            LOGGER.warning(
                "Calibration bundle is missing %s; FaIR will use the "
                "default zero baseline for %s forcing.",
                calibration.FILES[bundle_key],
                species_name,
            )
            continue
        df = pd.read_csv(csv_path, index_col=0)
        choice = _DEFAULT_LAND_USE_SCENARIO
        if choice not in df.columns:
            LOGGER.warning(
                "Bundle %s does not contain column %r; falling back to "
                "the first available column %r. %s forcing will reflect "
                "that choice. Override coming in a follow-up.",
                csv_path,
                choice,
                df.columns[0],
                species_name,
            )
            choice = df.columns[0]
        else:
            LOGGER.info(
                "Using %s column %r from %s for %s forcing (scientific "
                "choice; configurable in a follow-up).",
                csv_path.name,
                choice,
                bundle_key,
                species_name,
            )
        _write(species_name, df[choice])
