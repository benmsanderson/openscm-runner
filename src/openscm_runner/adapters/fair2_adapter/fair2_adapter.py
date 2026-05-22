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

- ``native_calibration`` (required): filesystem path to the bundle
  directory, or an in-memory :class:`NativeFairCalibration` instance.
- ``member_indices`` (optional, default = all members): sequence of
  zero-based ``int`` selecting which rows of the parameter posterior
  to use as the FaIR config dimension. ``range(N)`` gives "first N
  members"; an explicit list lets the user stratify or reproduce a
  specific subset.
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


def _run_translated_cfgs(scenarios, cfgs, output_variables) -> ScmRun:
    """
    Translated-cfg dispatch: each cfg dict is one ensemble member
    whose keys are FaIR 2.x parameter names.

    All cfgs go into a single FAIR instance with ``config`` dim sized
    to ``len(cfgs)``, which is much faster than spawning one FAIR
    per cfg. Species configs come from FaIR's shipped AR6 defaults
    (``fill_species_configs()`` without arg). Climate configs are
    populated from the cfg dicts.

    Supported cfg keys are anything FaIR 2.x exposes on
    ``climate_configs`` or ``species_configs``; unknown keys are
    logged at WARNING and ignored. If after population some required
    climate_configs value is still NaN, FaIR's ``run()`` raises a
    clear ``ValueError`` which we re-raise with a hint pointing back
    at native-calibration mode for users who want defaults from a
    published calibration.
    """
    from fair.interface import fill
    from fair.io import read_properties

    scenario_run = ScmRun(scenarios.timeseries()) if scenarios is not None else None
    scenario_names = (
        sorted(set(scenario_run["scenario"]))
        if scenario_run is not None and not scenario_run.empty
        else ["historical"]
    )

    # SCIENTIFIC CHOICE: same defaults as native mode (1750 start,
    # 1-year step). Made configurable in a follow-up.
    start_year = 1750
    end_year = (
        int(scenario_run.time_points.years().max())
        if scenario_run is not None and not scenario_run.empty
        else 2100
    )

    species, properties = read_properties()  # FaIR's shipped AR6 defaults

    config_labels = [f"config_{i}" for i in range(len(cfgs))]

    f = fair2.FAIR()
    f.define_time(start_year, end_year, 1)
    f.define_scenarios(scenario_names)
    f.define_configs(config_labels)
    f.define_species(species, properties)
    f.allocate()

    # Species configs from FaIR's shipped AR6 defaults; cfg-dict
    # overrides are applied on top so user values win.
    f.fill_species_configs()

    LOGGER.info(
        "Running FaIRv2 (translated) with %d ensemble members; "
        "climate_configs values come from cfg dicts",
        len(cfgs),
    )

    unknown_keys: set[str] = set()
    for cfg_idx, cfg in enumerate(cfgs):
        for key, value in cfg.items():
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

    if scenario_run is not None and not scenario_run.empty:
        emissions_df = build_emissions_df(
            scenario_run, None, scenario_names
        )
        if not emissions_df.empty:
            f.fill_from_pandas(mode="emissions", df=emissions_df)

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

    f = fair2.FAIR()
    f.define_time(start_year, end_year, 1)
    f.define_scenarios(scenario_names)
    f.define_configs([f"config_{i}" for i in members.index])
    f.define_species(species, properties)
    f.allocate()

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

    f.run(progress=False, suppress_warnings=True)

    return extract_outputs(
        f,
        scenario_names,
        members,
        output_variables,
        run_id_offset,
        properties_df=getattr(f, "properties_df", None),
    )
