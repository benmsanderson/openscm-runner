"""
End-to-end demonstration of the FaIRv2 adapter.

Runs the FaIRv2 adapter against a real FaIR 2.x calibration bundle
(e.g. the AR7-relevant Smith calibration on Zenodo,
https://zenodo.org/records/18828694), with a small SSP-style
emissions scenario, and prints a summary of the 2100 GSAT response
across the first few ensemble members.

Intended as a quick smoke-test for the adapter, and as a starting
point for sharing the wrapper's value with the community while the
PRs are in review.

Usage:

    export FAIR2_CALIBRATION_PATH=/path/to/extracted/zenodo/bundle
    python scripts/demo_fair2.py

The bundle must contain at minimum:
- calibrated_constrained_parameters.csv
- species_configs_properties.csv
- historical_emissions_1750-2023_cmip7.csv

Optional knobs (env vars):
- FAIR2_DEMO_MEMBERS=10       how many ensemble members to use
- FAIR2_DEMO_SCENARIO=ssp245  which built-in fixture scenario to load
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import scmdata

import openscm_runner.run
from openscm_runner.adapters import FAIR2


def main() -> int:
    bundle = os.environ.get("FAIR2_CALIBRATION_PATH")
    if not bundle:
        print(
            "ERROR: set FAIR2_CALIBRATION_PATH to the directory containing "
            "the FaIR 2.x calibration bundle (e.g. an extract of "
            "https://zenodo.org/records/18828694)",
            file=sys.stderr,
        )
        return 1

    n_members = int(os.environ.get("FAIR2_DEMO_MEMBERS", "10"))
    scenario_name = os.environ.get("FAIR2_DEMO_SCENARIO", "ssp245")

    test_data = (
        Path(__file__).parent.parent
        / "tests"
        / "test-data"
        / "rcmip_scen_ssp_world_emissions.csv"
    )
    if not test_data.exists():
        print(f"ERROR: scenario fixture not found at {test_data}", file=sys.stderr)
        return 1

    scenarios = scmdata.ScmRun(test_data, lowercase_cols=True).filter(
        scenario=scenario_name
    )
    if scenarios.empty:
        print(
            f"ERROR: scenario {scenario_name!r} not in the fixture; "
            "try ssp126 / ssp245 / ssp370 / ssp585",
            file=sys.stderr,
        )
        return 1

    print(
        f"Running FaIRv{FAIR2.get_version()} with {n_members} members of "
        f"the calibration at {bundle}, scenario {scenario_name}"
    )

    result = openscm_runner.run.run(
        climate_models_cfgs={
            "FaIRv2": [
                {
                    "native_calibration": bundle,
                    "member_indices": range(n_members),
                }
            ],
        },
        scenarios=scenarios,
        output_variables=(
            "Surface Air Temperature Change",
            "Effective Radiative Forcing",
            "Atmospheric Concentrations|CO2",
        ),
        out_config=None,
    )

    gsat_2100 = result.filter(
        variable="Surface Air Temperature Change",
        year=2100,
        scenario=scenario_name,
    ).values

    print()
    print(f"{scenario_name} GSAT in 2100 ({n_members}-member subset of bundle):")
    print(f"  median: {np.median(gsat_2100):.2f} K")
    print(
        f"  5-95% range: {np.quantile(gsat_2100, 0.05):.2f} - "
        f"{np.quantile(gsat_2100, 0.95):.2f} K"
    )
    print(f"  min / max:  {np.min(gsat_2100):.2f} - {np.max(gsat_2100):.2f} K")

    co2_2100 = result.filter(
        variable="Atmospheric Concentrations|CO2",
        year=2100,
        scenario=scenario_name,
    ).values
    print(
        f"  2100 CO2 concentration: median {np.median(co2_2100):.1f} ppm "
        f"(range {np.min(co2_2100):.1f} - {np.max(co2_2100):.1f})"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
