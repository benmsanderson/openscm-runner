"""
RCMIP protocol run: all available SSPs through the CICEROSCMPY2 adapter.

Mirror of ``scripts/run_rcmip_fair2.py`` for the CICEROSCMPY2 (native
DistributionRun) adapter. Runs every scenario in
``tests/test-data/rcmip_scen_ssp_world_emissions.csv`` (ssp119,
ssp126, ssp245, ssp370 and variants, ssp434, ssp460, ssp534-over,
ssp585) against ``--members`` posterior members of a CICERO-SCM v2.x
parameter distribution and prints a per-scenario 2100 GSAT / CO2 / ERF
summary.

Results are kept in memory; ``ScmRun.to_nc`` currently breaks on
pandas 3.0 (see ``openscm_runner._scmdata_patches`` in PR #11 for
the in-tree shim that fixes that — once PR #11 is merged to main and
the integration branch picks it up, the writer path can be wired in).

Usage
-----

    # One-time setup:
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[ciceroscmpy2]"

    # The bundle directory must contain:
    #   - draw_samples_500.json          (posterior; one cfg dict per member)
    #   - gases_vupdate_2022_AR6.txt     (gas definitions)
    #   - ssp245_conc_RCMIP.txt          (historical concentrations)
    export CICEROSCMPY2_BUNDLE_DIR=$PWD/configurations/ciceroscm

    # Smoke run (20 members, default):
    python scripts/run_rcmip_ciceroscm_py2.py

    # Larger ensemble:
    python scripts/run_rcmip_ciceroscm_py2.py --members 100

Use ``--scenarios ssp245 ssp585`` to subset.

A ``--cicero-bundle-dir`` override switches the adapter to bundle
mode (the Marit RCMIP-aligned path); when given, ``gaspam_file`` and
``concentrations_file`` are resolved per-scenario from inside the
bundle directory instead of from the top-level bundle files.
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
from openscm_runner.adapters import CICEROSCMPY2


_SUMMARY_VARS = (
    ("Surface Air Temperature Change", 2100, "K"),
    ("Atmospheric Concentrations|CO2", 2100, "ppm"),
    ("Effective Radiative Forcing", 2100, "W/m^2"),
)


def main(argv=None) -> int:
    """Run the RCMIP protocol and print a summary; return process exit code."""
    args = _parse_args(argv)

    bundle_dir = Path(
        os.environ.get(
            "CICEROSCMPY2_BUNDLE_DIR",
            Path(__file__).parent.parent / "configurations" / "ciceroscm",
        )
    )

    required = {
        "distribution_json": bundle_dir / "draw_samples_500.json",
        "gaspam_file": bundle_dir / "gases_vupdate_2022_AR6.txt",
        "concentrations_file": bundle_dir / "ssp245_conc_RCMIP.txt",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        print(
            "ERROR: missing CICEROSCMPY2 bundle files:\n  "
            + "\n  ".join(missing),
            file=sys.stderr,
        )
        print(
            "\nSet CICEROSCMPY2_BUNDLE_DIR to a directory containing "
            "draw_samples_500.json, gases_vupdate_2022_AR6.txt, and "
            "ssp245_conc_RCMIP.txt.",
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

    cfg: dict = {
        "distribution_json": str(required["distribution_json"]),
        "member_indices": range(args.members),
    }
    if args.cicero_bundle_dir is not None:
        cfg["cicero_bundle_dir"] = str(args.cicero_bundle_dir)
        mode = f"bundle ({args.cicero_bundle_dir})"
    else:
        cfg["gaspam_file"] = str(required["gaspam_file"])
        cfg["concentrations_file"] = str(required["concentrations_file"])
        mode = "splice (ssp245 historical + user emissions)"

    print(
        f"CICERO-SCM-PY{CICEROSCMPY2.get_version()} RCMIP protocol run\n"
        f"  distribution: {required['distribution_json']}\n"
        f"  mode:         {mode}\n"
        f"  members:      {args.members}\n"
        f"  scenarios:    {len(scenario_names)} ({', '.join(scenario_names)})\n"
    )

    t0 = time.time()
    result = openscm_runner.run.run(
        climate_models_cfgs={"CICERO-SCM-PY2": [cfg]},
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
        help="Number of distribution posterior members to use (default: 20).",
    )
    p.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Subset of scenarios to run (default: all in the RCMIP fixture).",
    )
    p.add_argument(
        "--cicero-bundle-dir", type=Path, default=None,
        help=(
            "Switch to bundle mode (Marit RCMIP-aligned setup). gaspam "
            "and concentrations files are resolved per-scenario from "
            "inside this directory."
        ),
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
