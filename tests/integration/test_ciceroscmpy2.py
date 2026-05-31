"""
Integration tests for the CICEROSCMPY2 adapter against a real
parameter distribution.

Gated on the ``CICEROSCMPY2_CALIBRATION_PATH`` environment variable:
when set to a directory containing the distribution JSON, gaspam file
and historical concentrations file (in the shape of
``configurations/ciceroscm/``), the tests below run a small ensemble
end-to-end and verify the adapter's output shape and parity with a
direct ``DistributionRun`` call.

CI does not have the bundle; set ``CICEROSCMPY2_CALIBRATION_PATH``
locally before running.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import openscm_runner.run
from openscm_runner.adapters import CICEROSCMPY2
from openscm_runner.adapters.ciceroscm_py2_adapter._compat import (
    HAS_CICEROSCM_PY2,
)

pytestmark = [
    pytest.mark.skipif(
        not HAS_CICEROSCM_PY2, reason="ciceroscm not installed"
    ),
    pytest.mark.skipif(
        "CICEROSCMPY2_CALIBRATION_PATH" not in os.environ,
        reason="CICEROSCMPY2_CALIBRATION_PATH not set; integration test skipped",
    ),
]


@pytest.fixture()
def bundle_dir() -> Path:
    return Path(os.environ["CICEROSCMPY2_CALIBRATION_PATH"])


@pytest.fixture()
def cfg(bundle_dir: Path) -> dict:
    return {
        "distribution_json": str(bundle_dir / "draw_samples_500.json"),
        "gaspam_file": str(bundle_dir / "gases_vupdate_2022_AR6.txt"),
        "concentrations_file": str(bundle_dir / "ssp245_conc_RCMIP.txt"),
        "member_indices": [0, 1, 2],
    }


def test_ciceroscmpy2_runs_small_native_ensemble(cfg, test_scenarios):
    """
    Run the CICEROSCMPY2 adapter with the smallest possible ensemble
    (first three members of the parameter distribution) and verify the
    output is shaped as expected.
    """
    res = openscm_runner.run.run(
        climate_models_cfgs={"CICERO-SCM-PY2": [cfg]},
        scenarios=test_scenarios.filter(scenario="ssp245"),
        output_variables=("Surface Air Temperature Change",),
        out_config=None,
    )

    assert res.get_unique_meta(
        "climate_model", no_duplicates=True
    ) == f"CICERO-SCM-PY{CICEROSCMPY2.get_version()}"
    assert set(res.get_unique_meta("variable")) == {
        "Surface Air Temperature Change"
    }

    # Sanity: 2100 GSAT for SSP2-4.5 sits roughly in the observed
    # AR6 envelope of ~1.5-3.5 K (coarse bound).
    temp_2100 = res.filter(
        variable="Surface Air Temperature Change", year=2100
    ).values
    assert (temp_2100 > 0.5).all()
    assert (temp_2100 < 6.0).all()


def test_ciceroscmpy2_adapter_matches_native_call(cfg, test_scenarios):
    """
    Parity test: adapter output must equal a direct
    ``DistributionRun.run_over_distribution`` call to machine precision
    for the same distribution, scenario and ensemble members.

    The CICEROSCMPY2 adapter is a thin shim that builds a scendata
    dict from the input ScmRun and dispatches to upstream's
    ``DistributionRun``; this test verifies that our scendata
    construction (gaspam / concentrations paths, emissions-DataFrame
    splice via the reused v1.1.x SCENARIODATAGETTER, nystart / nyend /
    emstart defaults) produces the same numbers as a direct upstream
    call with equivalent inputs.

    Regression baselines would catch *changes* from a frozen output
    but not silent divergence between adapter and upstream on day one.
    """
    import numpy as np
    import numpy.testing as npt
    from ciceroscm.parallel.distributionrun import DistributionRun

    from openscm_runner.adapters.ciceroscm_py2_adapter.ciceroscmpy2_adapter import (
        _BUNDLED_HISTORICAL_DIR,
    )
    from openscm_runner.adapters.ciceroscm_py_adapter.make_scenario_data import (
        SCENARIODATAGETTER,
    )

    scenarios = test_scenarios.filter(scenario="ssp245")
    member_indices = cfg["member_indices"]
    output_variables = ["Surface Air Temperature Change"]

    # Adapter path.
    adapter_res = openscm_runner.run.run(
        climate_models_cfgs={"CICERO-SCM-PY2": [cfg]},
        scenarios=scenarios,
        output_variables=tuple(output_variables),
        out_config=None,
    )

    # Native path: build scendata exactly the same way the adapter
    # does, then call DistributionRun.run_over_distribution directly.
    # If the adapter's scendata construction drifts from this recipe,
    # parity breaks - which is the wiring bug we want to surface.
    nystart = 1750
    nyend = int(scenarios.time_points.years().max())
    emstart = int(scenarios.time_points.years().min())

    sdatagetter = SCENARIODATAGETTER(_BUNDLED_HISTORICAL_DIR, nystart, nyend)
    timeseries = scenarios.timeseries(time_axis="year")
    scendata_list = []
    for (scen_name, _model), scen_df in timeseries.groupby(["scenario", "model"]):
        emissions_df = sdatagetter.get_scenario_data(scen_df, nystart)
        scendata_list.append(
            {
                "nystart": nystart,
                "nyend": nyend,
                "emstart": emstart,
                "sunvolc": 1,
                "scenname": scen_name,
                "gaspam_file": cfg["gaspam_file"],
                "concentrations_file": cfg["concentrations_file"],
                "emissions_data": emissions_df,
            }
        )

    dist = DistributionRun(
        distro_config=None,
        json_file_name=cfg["distribution_json"],
    )
    dist.cfgs = [dist.cfgs[i] for i in member_indices]

    # max_workers=1 forces serial execution so reproducibility doesn't
    # depend on ProcessPoolExecutor scheduling order.
    native_df = dist.run_over_distribution(
        scendata_list, output_variables, max_workers=1
    )

    # Compare per-member timeseries. The CICEROSCM v2.x wrapper sets
    # run_id from each cfg's "Index" field (e.g. the seed-derived
    # "_95_8597" for the v1.6.0 sample posterior); we filter on those
    # values, not on positional indices.
    from scmdata import ScmRun

    native_run = ScmRun(native_df).filter(
        variable="Surface Air Temperature Change",
        scenario="ssp245",
    )
    for cfg_dict in dist.cfgs:
        run_id = cfg_dict["Index"]
        adapter_vals = (
            adapter_res.filter(
                variable="Surface Air Temperature Change",
                scenario="ssp245",
                run_id=run_id,
            )
            .values.squeeze()
        )
        native_vals = (
            native_run.filter(run_id=run_id).values.squeeze()
        )

        assert adapter_vals.shape == native_vals.shape, (
            f"member {run_id}: shape mismatch adapter "
            f"{adapter_vals.shape} vs native {native_vals.shape}"
        )
        mask = ~(np.isnan(adapter_vals) | np.isnan(native_vals))
        npt.assert_array_equal(
            adapter_vals[mask],
            native_vals[mask],
            err_msg=(
                f"adapter and native CICERO-SCM v2.1.0 diverged for "
                f"member {run_id} of ssp245 (distribution "
                f"{cfg['distribution_json']}). Any non-zero diff is a "
                "wiring bug in the CICEROSCMPY2 adapter."
            ),
        )
