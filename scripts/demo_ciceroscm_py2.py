"""
End-to-end demonstration of the CICEROSCMPY2 adapter.

Runs the CICEROSCMPY2 adapter against a CICERO-SCM v2.1.0 parameter
distribution and prints a 2100 GSAT summary across the ensemble. The
distribution + supporting input files live under
``configurations/ciceroscm/`` by convention:

- ``draw_samples_500.json`` -- the parameter posterior (one
  ``{pamset_udm, pamset_emiconc, ...}`` dict per ensemble member).
- ``gases_vupdate_2022_AR6.txt`` -- gas definitions (molecular
  weights, lifetimes, radiative properties).
- ``ssp245_conc_RCMIP.txt`` -- historical concentrations covering
  ``nystart`` to ``emstart``.

Usage:

    # One-time setup: minimal install with the CICEROSCMPY2 extra.
    # ciceroscm 2.x requires Python >= 3.10.
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[ciceroscmpy2]"

    # Run the demo (defaults to ssp245 / first 10 members):
    python scripts/demo_ciceroscm_py2.py

Optional knobs (env vars):

- ``CICEROSCMPY2_BUNDLE_DIR``   directory with the JSON + gaspam +
                                 concentrations files (default
                                 ``configurations/ciceroscm/``).
- ``CICEROSCMPY2_DEMO_MEMBERS`` how many ensemble members to use
                                 (default ``10``; pass ``500`` to run
                                 the full posterior).
- ``CICEROSCMPY2_DEMO_SCENARIO`` which built-in fixture scenario to
                                  load (default ``ssp245``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import scmdata

import openscm_runner.run
from openscm_runner.adapters import CICEROSCMPY2


def main() -> int:
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
            "\nPoint CICEROSCMPY2_BUNDLE_DIR at a directory containing "
            "draw_samples_500.json, gases_vupdate_2022_AR6.txt, and "
            "ssp245_conc_RCMIP.txt, or copy them into "
            "configurations/ciceroscm/.",
            file=sys.stderr,
        )
        return 1

    n_members = int(os.environ.get("CICEROSCMPY2_DEMO_MEMBERS", "10"))
    scenario_name = os.environ.get("CICEROSCMPY2_DEMO_SCENARIO", "ssp245")

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
        f"Running CICERO-SCM-PY{CICEROSCMPY2.get_version()} with {n_members} "
        f"members from {required['distribution_json']}, scenario {scenario_name}"
    )

    result = openscm_runner.run.run(
        climate_models_cfgs={
            "CICERO-SCM-PY2": [
                {
                    "distribution_json": str(required["distribution_json"]),
                    "gaspam_file": str(required["gaspam_file"]),
                    "concentrations_file": str(required["concentrations_file"]),
                    "member_indices": range(n_members),
                }
            ],
        },
        scenarios=scenarios,
        output_variables=("Surface Air Temperature Change",),
        out_config=None,
    )

    gsat_2100 = result.filter(
        variable="Surface Air Temperature Change",
        year=2100,
        scenario=scenario_name,
    ).values

    print()
    print(
        f"{scenario_name} GSAT in 2100 ({n_members}-member subset of "
        "draw_samples_500):"
    )
    print(f"  median: {np.median(gsat_2100):.2f} K")
    print(
        f"  5-95% range: {np.quantile(gsat_2100, 0.05):.2f} - "
        f"{np.quantile(gsat_2100, 0.95):.2f} K"
    )
    print(f"  min / max:  {np.min(gsat_2100):.2f} - {np.max(gsat_2100):.2f} K")

    return 0


if __name__ == "__main__":
    sys.exit(main())
