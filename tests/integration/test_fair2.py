"""
Integration tests for the FaIRv2 adapter against a real calibration
bundle.

Gated on the ``FAIR2_CALIBRATION_PATH`` environment variable: when set
to a directory containing the AR7-relevant Smith calibration (or any
compatible bundle in the same shape as
https://zenodo.org/records/18828694), the tests below run a small
FaIR 2.x ensemble end-to-end and check the output shape, climate_model
metadata, and that a sensible Surface Air Temperature Change comes out.

CI does not have the bundle; set ``FAIR2_CALIBRATION_PATH`` locally
before running.
"""
from __future__ import annotations

import os

import pytest

import openscm_runner.run
from openscm_runner.adapters import FAIR2
from openscm_runner.adapters.fair2_adapter._compat import HAS_FAIR2

pytestmark = [
    pytest.mark.skipif(not HAS_FAIR2, reason="FaIR 2.x not installed"),
    pytest.mark.skipif(
        "FAIR2_CALIBRATION_PATH" not in os.environ,
        reason="FAIR2_CALIBRATION_PATH not set; integration test skipped",
    ),
]


@pytest.fixture()
def calibration_path():
    return os.environ["FAIR2_CALIBRATION_PATH"]


def test_fair2_runs_small_native_ensemble(calibration_path, test_scenarios):
    """
    Run the FaIRv2 adapter with the smallest possible ensemble (the
    first three members of the calibration's posterior) and verify
    the output is shaped as expected.
    """
    res = openscm_runner.run.run(
        climate_models_cfgs={
            "FaIRv2": [
                {
                    "native_calibration": calibration_path,
                    "member_indices": range(3),
                }
            ],
        },
        scenarios=test_scenarios.filter(scenario="ssp245"),
        output_variables=("Surface Air Temperature Change",),
        out_config=None,
    )

    assert (
        res.get_unique_meta("climate_model", no_duplicates=True)
        == f"FaIRv{FAIR2.get_version()}"
    )
    assert set(res.get_unique_meta("variable")) == {"Surface Air Temperature Change"}
    assert set(res["run_id"].tolist()) == {0, 1, 2}

    # Sanity check: 2100 GSAT for SSP2-4.5 should sit roughly in the
    # observed AR6 envelope of ~1.5-3.5 K. This is a coarse bound;
    # narrower envelopes belong in the calibration-specific regression
    # tests, not in the adapter integration test.
    temp_2100 = res.filter(
        variable="Surface Air Temperature Change", year=2100
    ).values
    assert temp_2100.shape[0] == 3  # one row per ensemble member
    assert (temp_2100 > 0.5).all()
    assert (temp_2100 < 6.0).all()


def test_fair2_adapter_matches_native_call(calibration_path, test_scenarios):
    """
    Parity test: adapter output must equal a direct ``fair.FAIR()`` call
    to machine precision for the same calibration, scenario, and
    ensemble members.

    Regression baselines catch *unintended changes* from a recorded
    output but assume the baseline was correct to begin with. This
    test catches the other direction: silent divergence between the
    adapter's wiring (output extractor, openscm-runner run dispatch,
    scenario / run_id metadata) and what a user would get by calling
    FaIR 2.x directly with the same inputs. Tolerance is ``rtol=0``
    (bit-identical); any drift is a real bug.
    """
    import numpy as np
    import numpy.testing as npt
    import scmdata
    from fair import FAIR
    from fair.interface import initialise
    from fair.io import read_properties

    from openscm_runner.adapters.fair2_adapter._emissions_translator import (
        build_emissions_df,
    )
    from openscm_runner.adapters.fair2_adapter._native_calibration import (
        NativeFairCalibration,
    )
    from openscm_runner.adapters.fair2_adapter.fair2_adapter import (
        _fill_natural_forcings,
    )

    scenarios = test_scenarios.filter(scenario="ssp245")
    member_indices = [0, 1, 2]

    # Adapter path.
    adapter_res = openscm_runner.run.run(
        climate_models_cfgs={
            "FaIRv2": [
                {
                    "native_calibration": calibration_path,
                    "member_indices": member_indices,
                }
            ],
        },
        scenarios=scenarios,
        output_variables=("Surface Air Temperature Change",),
        out_config=None,
    )

    # Native path: replicate _run_one_calibration's setup manually
    # using only public fair / NativeFairCalibration entry points.
    # Any change to the adapter that breaks parity here is a wiring
    # bug worth surfacing.
    calibration = NativeFairCalibration(calibration_path)
    members = calibration.select_members(member_indices)

    scenario_run = scmdata.ScmRun(scenarios.timeseries())
    scenario_names = sorted(set(scenario_run["scenario"]))
    end_year = int(scenario_run.time_points.years().max())

    species, properties = read_properties(
        filename=calibration.file("species_configs")
    )

    f = FAIR()
    f.define_time(1750, end_year, 1)
    f.define_scenarios(scenario_names)
    f.define_configs(list(members.index))
    f.define_species(species, properties)
    f.allocate()

    initialise(f.temperature, 0)
    initialise(f.forcing, 0)
    initialise(f.concentration, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    f.fill_species_configs(filename=calibration.file("species_configs"))
    f.override_defaults(calibration.file("parameters"))

    emissions_df = build_emissions_df(
        scenario_run, calibration.file("historical_emissions"), scenario_names
    )
    f.fill_from_pandas(mode="emissions", df=emissions_df)
    _fill_natural_forcings(f, calibration)

    f.run(progress=False, suppress_warnings=True)

    # Compare per-member 1850-end_year GSAT timeseries. Adapter rows
    # come back keyed by (scenario, variable, run_id); native is a
    # (timebound, scenario, config, layer) xarray. Pick layer=0
    # (surface), scenario=0 (single-ssp run), and iterate configs.
    native_temp = (
        f.temperature.isel(scenario=0, layer=0).to_pandas().to_numpy()
    )  # shape (timebound, config)

    for member_offset in range(len(member_indices)):
        adapter_series = (
            adapter_res.filter(
                variable="Surface Air Temperature Change",
                scenario="ssp245",
                run_id=member_offset,
            )
            .values.squeeze()
        )
        # Both adapter and native span 1750..end_year on timebounds.
        # Adapter reindexes onto timebounds in the extractor; native is
        # the raw f.temperature array.
        native_series = native_temp[:, member_offset]
        assert adapter_series.shape == native_series.shape, (
            f"member {member_offset}: shape mismatch adapter "
            f"{adapter_series.shape} vs native {native_series.shape}"
        )
        # Mask NaN year-0 timebound from both sides (f.temperature is
        # initialised to 0 at t=0 but the adapter reindexes onto
        # timebounds, so behaviour at the very first / last bound can
        # differ harmlessly). Compare every other year.
        mask = ~(np.isnan(adapter_series) | np.isnan(native_series))
        npt.assert_array_equal(
            adapter_series[mask],
            native_series[mask],
            err_msg=(
                f"adapter and native FaIR 2.x diverged for member "
                f"{member_offset} of ssp245 (calibration "
                f"{calibration_path}). Any non-zero diff is a wiring "
                "bug in the FaIRv2 adapter."
            ),
        )
