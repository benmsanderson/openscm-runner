"""
End-to-end demonstration of the FaIRv2 adapter.

Runs the FaIRv2 adapter against a real FaIR 2.x calibration bundle
(e.g. the AR7-relevant Smith calibration on Zenodo,
https://zenodo.org/records/18828694), and prints two demos:

1. **Native-calibration mode.** Runs an SSP-style emissions scenario
   across the first few members of the bundle's parameter posterior
   and prints a 2100 GSAT summary.
2. **Translated-cfg mode.** Sweeps the 4xCO2 effective radiative
   forcing parameter (``forcing_4co2``) across a few values, using
   the bundle's first posterior member as the baseline for the rest
   of the climate parameters. Shows how the wrapper exposes FaIR
   2.x's parameter space directly for sensitivity studies.

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

- FAIR2_DEMO_MEMBERS=10       how many ensemble members to use (native demo)
- FAIR2_DEMO_SCENARIO=ssp245  which built-in fixture scenario to load
- FAIR2_DEMO_SKIP_TRANSLATED=1  skip the translated-cfg demo
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import scmdata

import openscm_runner.run
from openscm_runner.adapters import FAIR2
from openscm_runner.adapters.fair2_adapter._compat import fair2
from openscm_runner.adapters.fair2_adapter._native_calibration import (
    NativeFairCalibration,
)


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

    if os.environ.get("FAIR2_DEMO_SKIP_TRANSLATED"):
        return 0

    return _translated_demo(bundle, scenarios, scenario_name)


def _translated_demo(bundle_path: str, scenarios, scenario_name: str) -> int:
    """
    Sweep ``forcing_4co2`` in translated-cfg mode and print 2100 GSAT.

    Uses the bundle's first posterior member as the baseline for all
    climate_configs other than ``forcing_4co2`` so we are not
    hardcoding physics values; only the swept parameter varies.
    """
    print()
    print("=== Translated-cfg mode demo: forcing_4co2 sensitivity sweep ===")

    try:
        baseline = _load_baseline_climate_configs(bundle_path)
    except Exception as exc:
        print(
            f"Could not load baseline climate_configs from {bundle_path}: "
            f"{exc}. Set FAIR2_DEMO_SKIP_TRANSLATED=1 to suppress this "
            "section.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Loaded {len(baseline)} climate_configs keys from the bundle's "
        f"first posterior member: {sorted(baseline)}"
    )

    forcing_values = (6.0, 7.0, 8.0, 9.0)
    cfgs = [{**baseline, "forcing_4co2": fv} for fv in forcing_values]

    result = openscm_runner.run.run(
        climate_models_cfgs={"FaIRv2": cfgs},
        scenarios=scenarios,
        output_variables=("Surface Air Temperature Change",),
        out_config=None,
    )

    print(
        f"2100 GSAT response to forcing_4co2 sweep "
        f"(scenario {scenario_name}, all other climate_configs held at the "
        "bundle's first-member values):"
    )
    for run_id, fv in enumerate(forcing_values):
        gsat = result.filter(
            variable="Surface Air Temperature Change",
            year=2100,
            scenario=scenario_name,
            run_id=run_id,
        ).values
        print(f"  forcing_4co2 = {fv:.1f} W/m^2  ->  2100 GSAT = {float(gsat):.2f} K")

    return 0


def _load_baseline_climate_configs(bundle_path: str) -> dict:
    """
    Build a translated-cfg dict from the bundle's first posterior member.

    Reads the parameter posterior, discovers which columns belong to
    FaIR 2.x's ``climate_configs`` by spinning up a bare FAIR instance,
    and re-assembles ``name[index]`` columns into lists. Returns a
    dict suitable for use as a translated-mode cfg base.
    """
    import warnings

    warnings.filterwarnings("ignore")

    from fair.io import read_properties

    bundle = NativeFairCalibration(bundle_path)
    row = bundle.parameters.iloc[0]

    f = fair2.FAIR()
    f.define_time(1750, 1751, 1)
    f.define_scenarios(["d"])
    f.define_configs(["c"])
    species, properties = read_properties()
    f.define_species(species, properties)
    f.allocate()
    climate_keys = set(f.climate_configs.keys())

    flat: dict = {}
    indexed: dict = {}
    for col, value in row.items():
        if "[" in col:
            name, idx_str = col.split("[", 1)
            name = name.strip()
            try:
                idx = int(idx_str.rstrip("]"))
            except ValueError:
                continue
            if name in climate_keys:
                indexed.setdefault(name, {})[idx] = value
        else:
            name = col.strip()
            if name in climate_keys:
                flat[name] = value

    out = dict(flat)
    for name, idx_to_val in indexed.items():
        out[name] = [idx_to_val[i] for i in sorted(idx_to_val)]
    return out


if __name__ == "__main__":
    sys.exit(main())
