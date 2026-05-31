"""
End-to-end demonstration of the CICEROSCMPY2 adapter.

Runs the CICEROSCMPY2 adapter against a CICERO-SCM v2.1.0 parameter
distribution in **bundle mode** (the path that bit-exactly reproduces
Marit's reference RCMIP protocol) and prints a 2100 GSAT summary
across the ensemble.

Bundle mode is the recommended path. The previously-shipped splice
mode uses a v1.1.x-era ssp245 historical that does not match v2.x
calibrations and produces a ~0.3-0.5 K present-day warm bias —
see the adapter module docstring for the diagnosis.

Layout expected under ``configurations/ciceroscm/``:

- ``draw_samples_500.json`` -- the parameter posterior (one
  ``{pamset_udm, pamset_emiconc, ...}`` dict per ensemble member).
- ``rcmip-march2026/`` -- Marit-RCMIP-aligned bundle directory
  containing per-scenario ``{scen}_em_*``, ``{scen}_conc_*``,
  ``solar_RCMIP_*``, ``VOLC_RCMIP_*``, ``LUCalbedo_RCMIP_*``,
  ``natemis_CH4_*``, ``natemis_N2O_*`` files plus the gaspam.

Usage:

    # One-time setup: minimal install with the CICEROSCMPY2 extra.
    # ciceroscm 2.x requires Python >= 3.10.
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[ciceroscmpy2]"

    # Run the demo (defaults to ssp245 / first 10 members):
    python scripts/demo_ciceroscm_py2.py

Optional knobs (env vars):

- ``CICEROSCMPY2_BUNDLE_DIR``       directory holding
                                     ``draw_samples_500.json`` (default
                                     ``configurations/ciceroscm/``).
- ``CICEROSCMPY2_RCMIP_BUNDLE_DIR`` Marit-RCMIP-aligned bundle
                                     directory (default
                                     ``$CICEROSCMPY2_BUNDLE_DIR/rcmip-march2026``).
- ``CICEROSCMPY2_DEMO_MEMBERS``     how many ensemble members to use
                                     (default ``10``; pass ``500`` to
                                     run the full posterior).
- ``CICEROSCMPY2_DEMO_SCENARIO``    which built-in fixture scenario to
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
    """Run the CICEROSCMPY2 bundle-mode demo; return process exit code."""
    bundle_dir = Path(
        os.environ.get(
            "CICEROSCMPY2_BUNDLE_DIR",
            Path(__file__).parent.parent / "configurations" / "ciceroscm",
        )
    )
    rcmip_bundle_dir = Path(
        os.environ.get(
            "CICEROSCMPY2_RCMIP_BUNDLE_DIR",
            bundle_dir / "rcmip-march2026",
        )
    )

    distribution_json = bundle_dir / "draw_samples_500.json"
    if not distribution_json.exists():
        print(
            f"ERROR: CICEROSCMPY2 distribution JSON not at {distribution_json}",
            file=sys.stderr,
        )
        return 1
    if not rcmip_bundle_dir.is_dir():
        print(
            f"ERROR: CICEROSCMPY2 RCMIP bundle dir not at {rcmip_bundle_dir}",
            file=sys.stderr,
        )
        print(
            "\nThe bundle is too large to ship in-tree; obtain it from the "
            "CICERO RCMIP setup (Marit's cscm-calibrate working directory) "
            "and point CICEROSCMPY2_RCMIP_BUNDLE_DIR at it.",
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
        f"Running CICERO-SCM-PY{CICEROSCMPY2.get_version()} bundle mode "
        f"with {n_members} members from {distribution_json}, "
        f"scenario {scenario_name} (bundle: {rcmip_bundle_dir})"
    )

    result = openscm_runner.run.run(
        climate_models_cfgs={
            "CICERO-SCM-PY2": [
                {
                    "distribution_json": str(distribution_json),
                    "cicero_bundle_dir": str(rcmip_bundle_dir),
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
