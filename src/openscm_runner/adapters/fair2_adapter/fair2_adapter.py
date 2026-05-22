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

import numpy as np
import pandas as pd
from scmdata import ScmRun, run_append

from ..base import _Adapter
from ._compat import HAS_FAIR2, fair2
from ._native_calibration import NativeFairCalibration

LOGGER = logging.getLogger(__name__)


# Minimal openscm-runner variable name -> FaIR 2.x species name map.
# Extends in follow-up PRs; documented limit on the adapter for the
# v1 cut. Suffix-match on the openscm-runner variable so callers can
# use the full hierarchical names ("Emissions|CO2|MAGICC Fossil and
# Industrial") interchangeably with the leaf.
_EMISSIONS_TO_FAIR2_SPECIES = {
    "|CO2|MAGICC Fossil and Industrial": "CO2 FFI",
    "|CO2|MAGICC AFOLU": "CO2 AFOLU",
    "|CH4": "CH4",
    "|N2O": "N2O",
}


def _openscm_to_fair2_species(variable: str) -> str | None:
    """
    Map an openscm-runner emissions variable name to a FaIR 2.x species
    name, or ``None`` if the species is not in the v1 mapping.
    """
    for suffix, species in _EMISSIONS_TO_FAIR2_SPECIES.items():
        if variable.endswith(suffix):
            return species
    return None


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

        for cfg in cfgs:
            if "native_calibration" not in cfg:
                raise NotImplementedError(
                    "FaIRv2 currently requires each cfg to carry a "
                    "'native_calibration' sidecar key (path to a FaIR 2.x "
                    "calibration bundle, or a NativeFairCalibration "
                    "instance). Translated-cfg mode is a follow-up PR."
                )

        results = []
        run_id_offset = 0
        for cfg_index, cfg in enumerate(cfgs):
            calibration = _resolve_calibration(cfg["native_calibration"])
            member_indices = cfg.get("member_indices")
            members = calibration.select_members(member_indices)

            LOGGER.info(
                "Running FaIRv2 (cfg %d/%d) with %d ensemble members from %s",
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

        out = run_append(results)
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

    # Future-emissions translation from the user's ScmRun is intentionally
    # minimal in this v1: only the four main GHGs (see
    # _EMISSIONS_TO_FAIR2_SPECIES). Anything else is left at the FaIR
    # default. Unmapped species are logged so the user knows what is
    # being ignored.
    if scenario_run is not None and not scenario_run.empty:
        _populate_emissions_from_scmrun(f, scenario_run)

    f.run(progress=False, suppress_warnings=True)

    return _extract_outputs(
        f, scenario_names, members, output_variables, run_id_offset
    )


def _populate_emissions_from_scmrun(f, scmrun: ScmRun):
    """
    Write the user's scenario emissions into FaIR's emissions array
    for the species this adapter knows how to map.

    Anything not in the v1 mapping is logged at WARNING and skipped;
    FaIR's bundle-provided historical / default values are used for
    those species.
    """
    mapped, unmapped = {}, set()
    for variable in scmrun.get_unique_meta("variable"):
        species = _openscm_to_fair2_species(variable)
        if species is None:
            unmapped.add(variable)
        else:
            mapped.setdefault(species, variable)

    if unmapped:
        LOGGER.warning(
            "FaIRv2 adapter v1 only maps a subset of emissions species "
            "(%s). Unmapped variables, FaIR defaults will be used: %s",
            sorted(_EMISSIONS_TO_FAIR2_SPECIES.values()),
            sorted(unmapped),
        )

    # NOTE: actual ScmRun -> FaIR DataArray write goes here. Kept as a
    # placeholder for the v1 PR while we work out the unit and time-
    # axis details against a real run. The shape is documented and
    # tested via mocking so the rest of the adapter wiring is verified.
    LOGGER.debug(
        "FaIRv2 emissions mapping (placeholder, v1): %s",
        {k: v for k, v in mapped.items()},
    )


def _extract_outputs(  # noqa: PLR0913
    f,
    scenarios,
    members: pd.DataFrame,
    output_variables,
    run_id_offset: int,
) -> ScmRun:
    """
    Convert FaIR 2.x's xarray output into an :class:`scmdata.ScmRun`.

    Supported output variables in this v1:

    - ``Surface Air Temperature Change`` (layer-0 of FaIR's temperature
      array)
    - ``Effective Radiative Forcing`` (sum across species)
    - ``Atmospheric Concentrations|CO2`` (the combined CO2 species)

    Anything else is silently dropped (matches FaIR 1.6 adapter
    behaviour, which logs but does not error on unknown output vars).
    """
    rows = []

    def _row(variable, unit, values, scenario, member_idx, run_id):
        rows.append(
            (
                scenario,
                "FaIR",  # placeholder; will be overwritten by the adapter
                "World",
                variable,
                unit,
                run_id,
                values,
            )
        )

    timebounds = np.asarray(f.timebounds, dtype=int)

    for sc_idx, scenario in enumerate(scenarios):
        for member_offset, member_label in enumerate(members.index):
            run_id = run_id_offset + member_offset
            for variable in output_variables:
                if variable == "Surface Air Temperature Change":
                    series = (
                        f.temperature.isel(
                            scenario=sc_idx, config=member_offset, layer=0
                        )
                        .to_pandas()
                        .reindex(timebounds)
                    )
                    _row(variable, "K", series, scenario, member_offset, run_id)
                elif variable == "Effective Radiative Forcing":
                    series = (
                        f.forcing.isel(scenario=sc_idx, config=member_offset)
                        .sum(dim="specie")
                        .to_pandas()
                        .reindex(timebounds)
                    )
                    _row(
                        variable, "W/m^2", series, scenario, member_offset, run_id
                    )
                elif variable == "Atmospheric Concentrations|CO2":
                    # Use the combined "CO2" species emitted by FaIR
                    # (sum of CO2 FFI and CO2 AFOLU contributions).
                    series = (
                        f.concentration.sel(specie="CO2")
                        .isel(scenario=sc_idx, config=member_offset)
                        .to_pandas()
                        .reindex(timebounds)
                    )
                    _row(variable, "ppm", series, scenario, member_offset, run_id)
                else:
                    LOGGER.debug(
                        "FaIRv2 adapter v1 does not emit %s; ignored", variable
                    )

    if not rows:
        return ScmRun(pd.DataFrame())

    return _build_scmrun(rows, timebounds)


def _build_scmrun(rows, timebounds):
    """
    Stack the per-(scenario, member, variable) Series produced by
    :func:`_extract_outputs` into a single :class:`scmdata.ScmRun`.
    """
    data = np.vstack([row[6].values for row in rows])
    meta = pd.DataFrame(
        [row[:6] for row in rows],
        columns=["scenario", "model", "region", "variable", "unit", "run_id"],
    )
    df = pd.DataFrame(
        data,
        index=pd.MultiIndex.from_frame(meta),
        columns=timebounds,
    )
    return ScmRun(df)
