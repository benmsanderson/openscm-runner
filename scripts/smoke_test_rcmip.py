"""
Joint smoke test: FaIRv2 + CICEROSCMPY2 over 4 SSPs x 20 members.

Drives both adapters with the same `openscm-runner` ScmRun input
(``tests/test-data/rcmip_scen_ssp_world_emissions.csv``, the existing
SSP fixture), 20 ensemble members per model, and saves the combined
result as a single netCDF chunk per (model, scenario) under
``out/smoke_test_rcmip/``.

Output variables: Surface Air Temperature Change, Atmospheric
Concentrations|CO2, Effective Radiative Forcing.

Intended as a visual sanity check: plot the resulting trajectories
side-by-side and confirm the two models broadly agree on
ssp126/245/370/585 GSAT, CO2, and ERF responses. Run once, then
iterate on plots via ``notebooks/smoke_test_rcmip_plots.py`` (loads
the saved netCDFs).

Usage:

    # Defaults: bundles under configurations/, 20 members each, 4 SSPs.
    python scripts/smoke_test_rcmip.py

Env vars:

- ``FAIR2_CALIBRATION_PATH``     Override FaIR bundle directory.
                                  Default: configurations/fair-calibrate-v1.6.0/.
- ``CICEROSCMPY2_BUNDLE_DIR``    Override CICERO bundle directory.
                                  Default: configurations/ciceroscm/.
- ``SMOKE_TEST_RCMIP_OUT``       Output directory.
                                  Default: out/smoke_test_rcmip/.
- ``SMOKE_TEST_RCMIP_MEMBERS``   Members per model (default 20).
- ``SMOKE_TEST_RCMIP_SCENARIOS`` Comma-separated SSP names
                                  (default: ssp126,ssp245,ssp370,ssp585).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import scmdata

import openscm_runner.run

REPO_ROOT = Path(__file__).parent.parent

DEFAULT_FAIR_BUNDLE = REPO_ROOT / "configurations" / "fair-calibrate-v1.6.0"
DEFAULT_CICERO_BUNDLE = REPO_ROOT / "configurations" / "ciceroscm"
DEFAULT_OUT = REPO_ROOT / "out" / "smoke_test_rcmip"
DEFAULT_SCENARIOS = ("ssp126", "ssp245", "ssp370", "ssp585")
DEFAULT_MEMBERS = 20

OUTPUT_VARIABLES = (
    "Surface Air Temperature Change",
    "Atmospheric Concentrations|CO2",
    "Effective Radiative Forcing",
)


def main() -> int:
    fair_bundle = Path(
        os.environ.get("FAIR2_CALIBRATION_PATH", DEFAULT_FAIR_BUNDLE)
    )
    cicero_bundle = Path(
        os.environ.get("CICEROSCMPY2_BUNDLE_DIR", DEFAULT_CICERO_BUNDLE)
    )
    out_dir = Path(os.environ.get("SMOKE_TEST_RCMIP_OUT", DEFAULT_OUT))
    n_members = int(os.environ.get("SMOKE_TEST_RCMIP_MEMBERS", DEFAULT_MEMBERS))
    scenario_names = tuple(
        os.environ.get(
            "SMOKE_TEST_RCMIP_SCENARIOS", ",".join(DEFAULT_SCENARIOS)
        ).split(",")
    )

    # CICERO bundle mode: a Marit-aligned directory at
    # configurations/ciceroscm/rcmip-march2026/ contains per-scenario
    # ssp{N}_em / ssp{N}_conc files plus the 2024 WMO gaspam, March-
    # 2026 natural CH4/N2O and RCMIP3 sun/volc/LUC forcings. We point
    # the adapter at the directory; it picks the right files per
    # scenario. The legacy splice mode (gaspam_file + concentrations_file
    # on top of the v1.1.x ssp245_em_RCMIP.txt splice) over-stated
    # historical ERF and would give 1.89 K GSAT @ 2024 vs observed
    # ~1.2 K; bundle mode reproduces present-day cleanly.
    cicero_rcmip_bundle = cicero_bundle / "rcmip-march2026"
    cicero_required = {
        "distribution_json": cicero_bundle / "draw_samples_500.json",
        "cicero_bundle_dir": cicero_rcmip_bundle,
    }
    fair_required = {
        "calibrated_constrained_parameters.csv": fair_bundle
        / "calibrated_constrained_parameters.csv",
        "species_configs_properties.csv": fair_bundle
        / "species_configs_properties.csv",
        "historical_emissions": fair_bundle
        / "historical_emissions_1750-2023_cmip7.csv",
    }
    missing = [
        str(p)
        for p in {**cicero_required, **fair_required}.values()
        if not p.exists()
    ]
    if missing:
        print(
            "ERROR: missing required bundle files / directories:\n  "
            + "\n  ".join(missing),
            file=sys.stderr,
        )
        print(
            "\nFaIR: see scripts/download_fair2_calibration.py to fetch "
            "the v1.6.0 bundle.\n"
            "CICERO: configurations/ciceroscm/rcmip-march2026/ should "
            "contain Marit's RCMIP input bundle "
            "(cscm-calibrate data/calibration_data_RCMIP/ with "
            "temp_indata/ symlinks resolved).",
            file=sys.stderr,
        )
        return 1

    fixture = REPO_ROOT / "tests" / "test-data" / "rcmip_scen_ssp_world_emissions.csv"
    if not fixture.exists():
        print(f"ERROR: scenario fixture not found at {fixture}", file=sys.stderr)
        return 1

    scenarios = scmdata.ScmRun(fixture, lowercase_cols=True).filter(
        scenario=list(scenario_names)
    )
    found = sorted(set(scenarios["scenario"]))
    missing_scen = sorted(set(scenario_names) - set(found))
    if missing_scen:
        print(
            f"WARNING: requested scenarios {missing_scen} not in fixture; "
            f"continuing with {found}",
            file=sys.stderr,
        )
    if scenarios.empty:
        print("ERROR: no requested scenarios present in fixture", file=sys.stderr)
        return 1

    print(
        f"Running FaIRv2 + CICEROSCMPY2 over {found} ({n_members} members "
        f"each), output -> {out_dir}/"
    )

    cfgs = {
        "FaIRv2": [
            {
                "native_calibration": str(fair_bundle),
                "member_indices": range(n_members),
            }
        ],
        "CICERO-SCM-PY2": [
            {
                "distribution_json": str(cicero_required["distribution_json"]),
                "cicero_bundle_dir": str(cicero_required["cicero_bundle_dir"]),
                "member_indices": range(n_members),
            }
        ],
    }

    # NOTE: bypassing NetCDFChunkWriter because its per-(model,
    # scenario) groupby trips the scmdata 0.18.0 + numpy 2.x StringDtype
    # bug (our tracking issue #6; upstream scmdata PR #319). Pulling
    # the combined ScmRun into memory and writing one netCDF per model
    # via ScmRun.to_nc avoids the groupby code path.
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = openscm_runner.run.run(
        climate_models_cfgs=cfgs,
        scenarios=scenarios,
        output_variables=OUTPUT_VARIABLES,
        out_config=None,
        parallel_models=True,
    )
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f} s.")

    # CSV per model. ScmRun.to_nc complains about non-unique extras
    # because CICERO's wrapper sets `model` from scenname, varying
    # per row; CSV via .timeseries().to_csv() sidesteps that entirely
    # and is plenty fast at this size (~4 SSPs x 20 members x 86 yr
    # x 3 vars per model).
    written = []
    for climate_model in sorted(set(result["climate_model"])):
        chunk = result.filter(climate_model=climate_model)
        out_path = out_dir / f"{climate_model.replace('/', '_')}.csv"
        chunk.timeseries(time_axis="year").to_csv(out_path)
        written.append(out_path)
        print(
            f"  {out_path.name}: {len(chunk['run_id'].unique())} run_ids x "
            f"{len(set(chunk['scenario']))} scenarios x "
            f"{len(set(chunk['variable']))} variables"
        )
    print()
    print(f"Wrote {len(written)} CSV files under {out_dir}/")
    print("Next: plot via notebooks/smoke_test_rcmip_plots.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
