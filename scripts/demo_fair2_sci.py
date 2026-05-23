"""
End-to-end demonstration: Scenario Compass Initiative -> FaIRv2.

Loads the SCI 2025 emissions ensemble (xlsx) via
``openscm_runner.scenarios.load_iamc``, picks one (model, scenario)
pair, interpolates to annual cadence (some IAMs report at 10-year
steps with NaN gaps), and runs it through the FaIRv2 adapter in
native-calibration mode.

This is the SCI counterpart to ``scripts/demo_fair2.py`` (which runs
the FaIR 1.6 test fixture) and exercises the loader added in PR #9
end-to-end against a real climate model.

Usage:

    # One-time setup (FaIRv2 extra + openpyxl for xlsx reading):
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[fair2]" openpyxl

    # Fetch the FaIR 2.x calibration bundle (~2 MB of CSVs):
    python scripts/download_fair2_calibration.py
    export FAIR2_CALIBRATION_PATH=$PWD/configurations/fair-calibrate-v1.6.0

    # SCI data: download SCI-2025_v1.0_pathways_ensemble_global.xlsx
    # from https://doi.org/10.5281/zenodo.18598251 and put it under
    # scenario_data/ (or set SCI_DATA_PATH to its location).
    export SCI_DATA_PATH=$PWD/scenario_data/SCI-2025_v1.0_pathways_ensemble_global.xlsx

    python scripts/demo_fair2_sci.py

Optional knobs (env vars):

- SCI_MODEL=MESSAGE-GLOBIOM 1.0  IAM to read from the ensemble
- SCI_SCENARIO=SSP1-19           scenario name
- FAIR2_DEMO_MEMBERS=10          how many calibration members to use
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import openscm_runner.run
from openscm_runner.adapters import FAIR2
from openscm_runner.scenarios import load_iamc


def main() -> int:
    bundle = os.environ.get("FAIR2_CALIBRATION_PATH")
    if not bundle:
        print(
            "ERROR: set FAIR2_CALIBRATION_PATH to the directory containing "
            "the FaIR 2.x calibration bundle (see scripts/download_fair2_calibration.py)",
            file=sys.stderr,
        )
        return 1

    sci_path = os.environ.get(
        "SCI_DATA_PATH",
        str(
            Path(__file__).parent.parent
            / "scenario_data"
            / "SCI-2025_v1.0_pathways_ensemble_global.xlsx"
        ),
    )
    if not Path(sci_path).exists():
        print(
            f"ERROR: SCI ensemble not found at {sci_path}. Download from "
            "https://doi.org/10.5281/zenodo.18598251 and place under "
            "scenario_data/, or set SCI_DATA_PATH.",
            file=sys.stderr,
        )
        return 1

    model = os.environ.get("SCI_MODEL", "MESSAGE-GLOBIOM 1.0")
    scenario_name = os.environ.get("SCI_SCENARIO", "SSP1-19")
    n_members = int(os.environ.get("FAIR2_DEMO_MEMBERS", "10"))

    print(
        f"Loading SCI scenario {scenario_name!r} from {sci_path} "
        f"(this can take a minute for the 118 MB xlsx)..."
    )
    scenarios = load_iamc(sci_path, scenarios=[scenario_name])
    scenarios = scenarios.filter(model=model)
    if scenarios.empty:
        available = sorted(
            load_iamc(sci_path, scenarios=[scenario_name])["model"].unique()
        )
        print(
            f"ERROR: model {model!r} did not report scenario {scenario_name!r}. "
            f"Available IAMs for this scenario: {available}",
            file=sys.stderr,
        )
        return 1

    print(
        f"  Loaded {scenarios.shape[0]} timeseries across "
        f"{len(scenarios['variable'].unique())} species"
    )

    scenarios = _interpolate_to_annual(scenarios)
    print(
        f"  Interpolated to annual cadence "
        f"({scenarios['time'].min().year}-{scenarios['time'].max().year})"
    )

    print(
        f"Running FaIRv{FAIR2.get_version()} with {n_members} members of "
        f"the calibration at {bundle}"
    )

    result = openscm_runner.run.run(
        climate_models_cfgs={
            "FaIRv2": [
                {
                    "native_calibration": bundle,
                    "member_indices": range(n_members),
                    "emissions_bundle": bundle,
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

    print()
    print(f"=== {model} | {scenario_name} (SCI 2025, {n_members} members) ===")
    _print_summary(result, "Surface Air Temperature Change", 2100, "K")
    _print_summary(result, "Atmospheric Concentrations|CO2", 2100, "ppm")
    _print_summary(result, "Effective Radiative Forcing", 2100, "W/m^2")

    return 0


def _interpolate_to_annual(scenarios):
    """
    Drop sparse cadences (some IAMs in SCI report decadally with NaN
    in mid-decade years) onto an annual grid spanning the data.

    The FaIRv2 adapter's pandas fill expects a contiguous timeseries;
    NaNs break the splice. scmdata's ``interpolate`` does a simple
    linear interpolation in time which is fine for the smooth
    socioeconomic trajectories SCI provides.
    """
    times = scenarios["time"]
    start, end = times.min().year, times.max().year
    annual = pd.date_range(f"{start}-01-01", f"{end}-01-01", freq="YS")
    return scenarios.interpolate(annual)


def _print_summary(result, variable, year, unit):
    values = result.filter(variable=variable, year=year).values
    if values.size == 0:
        print(f"  {variable}: (no rows)")
        return
    print(
        f"  {variable} @ {year}:  "
        f"median {np.median(values):.2f} {unit}  "
        f"(5-95% {np.quantile(values, 0.05):.2f} - "
        f"{np.quantile(values, 0.95):.2f}  "
        f"min/max {np.min(values):.2f} / {np.max(values):.2f})"
    )


if __name__ == "__main__":
    sys.exit(main())
