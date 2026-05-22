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
