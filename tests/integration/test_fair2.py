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
