"""
RCMIP protocol run: all available SSPs through the FaIRv2 adapter.

Runs every scenario in ``scripts/rcmip_scen_ssp_world_emissions.csv``
(ssp119, ssp126, ssp245, ssp370, ssp370-lowNTCF-aerchemmip,
ssp370-lowNTCF-gidden, ssp434, ssp460, ssp534-over, ssp585) against
``--members`` posterior members of a FaIR 2.x calibration bundle and
prints a per-scenario 2100 GSAT / CO2 / ERF summary.

Results are kept in memory: a 20-member sweep is well under 1 MB
total and a 1000-member sweep is still only ~50 MB.

Usage
-----

    # One-time setup:
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[fair2]"
    python scripts/download_fair2_calibration.py

    # Smoke run (20 members, default):
    export FAIR2_CALIBRATION_PATH=$PWD/configurations/fair-calibrate-v1.6.0
    python scripts/run_rcmip_fair2.py

    # Larger ensemble:
    python scripts/run_rcmip_fair2.py --members 100

Use ``--scenarios ssp245 ssp585`` to subset; omit for all scenarios in
the fixture.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import scmdata

import openscm_runner.run
from openscm_runner.adapters import FAIR2


_SUMMARY_VARS = (
    ("Surface Air Temperature Change", 2100, "K"),
    ("Atmospheric Concentrations|CO2", 2100, "ppm"),
    ("Effective Radiative Forcing", 2100, "W/m^2"),
)


def main(argv=None) -> int:
    """Run the RCMIP protocol and print a summary; return process exit code."""
    args = _parse_args(argv)

    bundle = os.environ.get("FAIR2_CALIBRATION_PATH")
    if not bundle:
        print(
            "ERROR: set FAIR2_CALIBRATION_PATH to the directory containing "
            "the FaIR 2.x calibration bundle "
            "(see scripts/download_fair2_calibration.py).",
            file=sys.stderr,
        )
        return 1

    fixture = (
        Path(__file__).parent.parent
        / "tests"
        / "test-data"
        / "rcmip_scen_ssp_world_emissions.csv"
    )
    if not fixture.exists():
        print(f"ERROR: scenario fixture not found at {fixture}", file=sys.stderr)
        return 1

    scenarios = scmdata.ScmRun(fixture, lowercase_cols=True)
    if args.scenarios:
        scenarios = scenarios.filter(scenario=args.scenarios)
        if scenarios.empty:
            available = sorted(
                scmdata.ScmRun(fixture, lowercase_cols=True)["scenario"].unique()
            )
            print(
                f"ERROR: none of the requested scenarios {args.scenarios!r} "
                f"are in the fixture. Available: {available}",
                file=sys.stderr,
            )
            return 1

    scenario_names = sorted(scenarios["scenario"].unique())

    print(
        f"FaIRv{FAIR2.get_version()} RCMIP protocol run\n"
        f"  bundle:    {bundle}\n"
        f"  members:   {args.members}\n"
        f"  scenarios: {len(scenario_names)} ({', '.join(scenario_names)})\n"
    )

    t0 = time.time()
    result = openscm_runner.run.run(
        climate_models_cfgs={
            "FaIRv2": [
                {
                    "native_calibration": bundle,
                    "member_indices": range(args.members),
                    "emissions_bundle": bundle,
                }
            ],
        },
        scenarios=scenarios,
        output_variables=tuple(v for v, _, _ in _SUMMARY_VARS),
        out_config=None,
    )
    elapsed = time.time() - t0

    print(
        f"\nFinished {len(scenario_names)} scenarios x {args.members} members "
        f"in {elapsed:.1f}s "
        f"({elapsed / len(scenario_names):.2f}s per scenario)"
    )
    _print_summary(result, scenario_names)
    return 0


def _parse_args(argv):
    """Build the CLI argument parser and parse ``argv``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "--members", type=int, default=20,
        help="Number of calibration posterior members to use (default: 20).",
    )
    p.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Subset of scenarios to run (default: all in the RCMIP fixture).",
    )
    return p.parse_args(argv)


def _print_summary(result, scenario_names):
    """Print a per-scenario 2100 quantile summary across members."""
    print("\n=== 2100 summary (median / 5-95% / min-max across members) ===")
    for scenario in scenario_names:
        chunk = result.filter(scenario=scenario)
        if chunk.empty:
            print(f"\n  {scenario}: (no rows)")
            continue
        print(f"\n  {scenario}:")
        for variable, year, unit in _SUMMARY_VARS:
            values = chunk.filter(variable=variable, year=year).values
            if values.size == 0:
                print(f"    {variable}: (no rows)")
                continue
            print(
                f"    {variable} @ {year}:  "
                f"median {np.median(values):.2f} {unit}  "
                f"(5-95% {np.quantile(values, 0.05):.2f} - "
                f"{np.quantile(values, 0.95):.2f}  "
                f"min/max {np.min(values):.2f} / {np.max(values):.2f})"
            )


if __name__ == "__main__":
    sys.exit(main())
