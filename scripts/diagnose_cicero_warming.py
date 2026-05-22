"""
Diagnose the CICERO present-day warming bias.

Runs the draw_samples_500 posterior through Marit's RCMIP-aligned
input setup (directly, bypassing the CICEROSCMPY2 adapter and our
v1.1.x SCENARIODATAGETTER splice).

Our smoke test gives CICERO 1.89 K at 2024 for ssp245 (observed ~1.2 K)
and an ERF of 3.0 W/m^2 at 2014 (AR6 IGCC says ~2.0 W/m^2 in 2014).
That bias is present in FaIR too (ERF 2.81 W/m^2 at 2014), pointing
at our input pipeline rather than either model's posterior.

This script reproduces the setup from cscm-calibrate's
``set_up_calibration_configs_and_run.define_scendata_for_scm``:
2024 WMO gaspam, March-2026 natural CH4/N2O, RCMIP3 historical
solar/volcanic/LUC, full historical_em / historical_conc files with
emstart=1850 and conc_run=False. If the posterior then reproduces
observed ~1.2 K at 2024, the bias is confirmed to be input alignment.

Usage:

    # Drop Marit's input files under configurations/ciceroscm/rcmip-march2026/:
    #   gases_vupdate_2024_WMO_added_new.txt                        (have)
    #   natemis_CH4_ode_method_from_March2026_vupdate_2024_WMO_added_new.txt  (have)
    #   natemis_N2O_ode_method_from_March2026_vupdate_2024_WMO_added_new.txt  (have)
    #   historical_em_gases_vupdate_2024_WMO_added_new.txt
    #   historical_conc_gases_vupdate_2024_WMO_added_new.txt
    #   solar_RCMIP_historical_RCMIP3.txt
    #   VOLC_RCMIP_historical_RCMIP3.txt
    #   LUCalbedo_RCMIP_historical_RCMIP3.txt
    python scripts/diagnose_cicero_warming.py

Env vars:

- ``DIAGNOSE_BUNDLE_DIR``  Override input directory (default
                            configurations/ciceroscm/rcmip-march2026/)
- ``DIAGNOSE_MEMBERS``     Members to run (default 20)
- ``DIAGNOSE_NYEND``       Final year (default 2024)
- ``DIAGNOSE_DISTRO_JSON`` Override path to draw_samples_500.json
                            (default configurations/ciceroscm/draw_samples_500.json)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
DEFAULT_DIR = REPO / "configurations" / "ciceroscm" / "rcmip-march2026"
DEFAULT_DISTRO = REPO / "configurations" / "ciceroscm" / "draw_samples_500.json"

REQUIRED_FILES = (
    "gases_vupdate_2024_WMO_added_new.txt",
    "natemis_CH4_ode_method_from_March2026_vupdate_2024_WMO_added_new.txt",
    "natemis_N2O_ode_method_from_March2026_vupdate_2024_WMO_added_new.txt",
    "historical_em_gases_vupdate_2024_WMO_added_new.txt",
    "historical_conc_gases_vupdate_2024_WMO_added_new.txt",
    "solar_RCMIP_historical_RCMIP3.txt",
    "VOLC_RCMIP_historical_RCMIP3.txt",
    "LUCalbedo_RCMIP_historical_RCMIP3.txt",
)


def _build_marit_scendata(
    test_data_dir: Path,
    *,
    nyend: int,
    nystart: int = 1750,
    emstart: int = 1850,
):
    """
    Mirror cscm-calibrate's ``define_scendata_for_scm``.

    Inlined here rather than imported because cscm-calibrate isn't a
    PyPI package; we only need this one function for the diagnostic.
    """
    import pandas as pd
    from ciceroscm import input_handler

    def _load(path: Path, case_type: str):
        if case_type in ("CH4", "N2O"):
            # ``read_natural_emissions`` requires exact (endyear - startyear
            # + 1) rows in the file. The March-2026 natural-emissions
            # files cover 1750-2022 (273 rows); pass that explicitly
            # rather than the default endyear=2500.
            with open(path) as fh:
                n_rows = sum(1 for _ in fh)
            file_endyear = nystart + n_rows - 1
            df = input_handler.read_natural_emissions(
                str(path), case_type, startyear=nystart, endyear=file_endyear
            )
            return df.loc[: min(nyend, file_endyear)]
        if case_type == "emis":
            ih = input_handler.InputHandler(
                {"nyend": nyend, "nystart": nystart, "emstart": emstart}
            )
            df = ih.read_emissions(str(path))
            df.rename(columns={"CO2": "CO2_FF", "CO2.1": "CO2_AFOLU"}, inplace=True)
            return df
        if case_type == "conc":
            return input_handler.read_inputfile(str(path), cut_years=False)
        if case_type == "gaspam":
            return input_handler.read_components(str(path))
        raise ValueError(case_type)

    gaspam = _load(
        test_data_dir / "gases_vupdate_2024_WMO_added_new.txt", "gaspam"
    )
    df_nat_ch4 = _load(
        test_data_dir / REQUIRED_FILES[1], "CH4"
    )
    df_nat_n2o = _load(
        test_data_dir / REQUIRED_FILES[2], "N2O"
    )
    df_conc = _load(
        test_data_dir / "historical_conc_gases_vupdate_2024_WMO_added_new.txt",
        "conc",
    )
    df_emis = _load(
        test_data_dir / "historical_em_gases_vupdate_2024_WMO_added_new.txt",
        "emis",
    )

    rf_solar_data = pd.read_csv(
        test_data_dir / "solar_RCMIP_historical_RCMIP3.txt",
        header=None,
        sep=r"\s+",
    )
    rf_volc_data = pd.read_csv(
        test_data_dir / "VOLC_RCMIP_historical_RCMIP3.txt",
        header=None,
        sep=r"\s+",
    )
    rf_luc_data = pd.read_csv(
        test_data_dir / "LUCalbedo_RCMIP_historical_RCMIP3.txt",
        header=None,
        sep=r"\s+",
    )

    return [
        {
            "gaspam_data": gaspam,
            "emstart": emstart,
            "conc_run": False,
            "nystart": nystart,
            "nyend": nyend,
            "concentrations_data": df_conc,
            "emissions_data": df_emis,
            "nat_ch4_data": df_nat_ch4,
            "nat_n2o_data": df_nat_n2o,
            "idtm": 24,
            "scenname": "historical-marit-aligned",
            "sunvolc": 1,
            "rf_solar": rf_solar_data,
            "rf_volc": rf_volc_data,
            "rf_luc": rf_luc_data,
        }
    ]


def main() -> int:
    """Run the Marit-aligned diagnostic and print present-day GSAT / ERF / CO2."""
    bundle_dir = Path(os.environ.get("DIAGNOSE_BUNDLE_DIR", DEFAULT_DIR))
    n_members = int(os.environ.get("DIAGNOSE_MEMBERS", "20"))
    nyend = int(os.environ.get("DIAGNOSE_NYEND", "2024"))
    distro_json = Path(os.environ.get("DIAGNOSE_DISTRO_JSON", DEFAULT_DISTRO))

    missing = [f for f in REQUIRED_FILES if not (bundle_dir / f).exists()]
    if missing:
        print(
            f"ERROR: {len(missing)} required files missing from {bundle_dir}:",
            file=sys.stderr,
        )
        for f in missing:
            print(f"  {f}", file=sys.stderr)
        print(
            "\nDrop them in from Marit's temp_indata/ (the cscm-calibrate "
            "data/calibration_data_RCMIP/ entries for these names are git "
            "symlinks pointing at her cluster).",
            file=sys.stderr,
        )
        return 1
    if not distro_json.exists():
        print(f"ERROR: distribution JSON not found at {distro_json}", file=sys.stderr)
        return 1

    print(f"Loading Marit-aligned scendata from {bundle_dir}")
    print(f"  members:  {n_members}")
    print(f"  nyend:    {nyend}")
    print(f"  distro:   {distro_json.name}")
    print()

    scendata = _build_marit_scendata(bundle_dir, nyend=nyend)

    from ciceroscm.parallel.distributionrun import DistributionRun

    dist = DistributionRun(distro_config=None, json_file_name=str(distro_json))
    dist.cfgs = dist.cfgs[:n_members]

    import time

    t0 = time.time()
    result = dist.run_over_distribution(
        scendata,
        output_vars=[
            "Surface Air Temperature Change",
            "Effective Radiative Forcing",
            "Atmospheric Concentrations|CO2",
        ],
        max_workers=min(n_members, 8),
    )
    elapsed = time.time() - t0
    print(f"Ran {n_members} members in {elapsed:.1f} s.")

    import numpy as np
    import scmdata

    run = scmdata.ScmRun(result)
    target_years = [yr for yr in (1980, 2000, 2014, 2019, 2024) if yr <= nyend]

    for var, label, reference in [
        ("Surface Air Temperature Change", "GSAT", {
            2014: ("~1.1 K", "AR6 2010-2019 mean"),
            2019: ("~1.2 K", "AR6 IGCC 2019"),
            2024: ("~1.2-1.3 K", "AR6 IGCC 2024 estimate"),
        }),
        ("Effective Radiative Forcing", "ERF", {
            2019: ("~2.7 W/m^2", "AR6 ERF total 2019"),
            2024: ("~2.8 W/m^2", "AR6 IGCC 2024"),
        }),
        ("Atmospheric Concentrations|CO2", "CO2", {
            2014: ("~398 ppm", "NOAA"),
            2019: ("~410 ppm", "NOAA"),
            2024: ("~422 ppm", "NOAA"),
        }),
    ]:
        sel = run.filter(variable=var)
        if sel.empty:
            continue
        ts = sel.timeseries(time_axis="year")
        ts.columns = ts.columns.astype(int)
        print(f"\n{label}:")
        for yr in target_years:
            if yr not in ts.columns:
                continue
            vals = ts[yr].values
            ref = reference.get(yr, ("", ""))
            ref_str = f"  (obs/AR6 {ref[0]}, {ref[1]})" if ref[0] else ""
            print(
                f"  {yr}: median {np.median(vals):>6.2f}  "
                f"range {np.min(vals):>5.2f} - {np.max(vals):>5.2f}{ref_str}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
