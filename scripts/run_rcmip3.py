"""
RCMIP3 protocol runner: dispatch one or more SCMs across the RCMIP3
experiment set in emissions-driven and/or concentration-driven mode,
streaming per-(mode, model, scenario) output to a netCDF tree on disk.

Replaces the per-model demos ``scripts/run_rcmip_fair2.py`` and
``scripts/run_rcmip_ciceroscm_py2.py``. Scenarios are resolved through
:func:`openscm_runner.scenarios.load_rcmip3_emissions` so the same CLI
covers the CMIP6 SSPs, the CMIP7 ScenarioMIP ``scen7-*`` set, and the
``esm-flat*`` idealised family.

Modes
-----
Both FaIRv2 and CICEROSCMPY2 can run either emissions-driven or
concentration-driven on every SSP / scen7 / historical scenario.
``--mode both`` (default) dispatches each pair through both modes so
the notebook can compare carbon-cycle behaviour against prescribed
concentrations. The idealised ``esm-flat*`` family is emissions-driven
by construction and is silently skipped from any concentration-driven
sub-run.

Example
-------

    # One-time setup
    pip install -e ".[fair2,ciceroscmpy2]"
    scripts/download_fair2_calibration.py
    # plus a CICERO-SCM bundle at $CICEROSCMPY2_BUNDLE_DIR

    export FAIR2_CALIBRATION_PATH=$PWD/configurations/fair-calibrate-v1.6.0
    export CICEROSCMPY2_BUNDLE_DIR=$PWD/configurations/ciceroscm

    # 10-member sweep across SSPs + scen7 + flat-*, both modes:
    scripts/run_rcmip3.py --members 10 --scenario-set all

    # Smaller / faster iteration:
    scripts/run_rcmip3.py --members 5 --scenarios ssp245 ssp585 esm-flat10-zec

Output layout
-------------
``--output-dir`` (default ``out/rcmip3/``) ends up containing one
file per (mode, model, scenario) tuple::

    out/rcmip3/
        emissions/
            FaIRv2/
                ssp119.nc
                ssp245.nc
                …
            CICERO-SCM-PY2/
                ssp119.nc
                …
        concentrations/
            FaIRv2/
                ssp119.nc
                …
            CICERO-SCM-PY2/
                ssp119.nc
                …

Read back with::

    from pathlib import Path
    import scmdata
    chunks = [scmdata.ScmRun.from_nc(p) for p in Path("out/rcmip3").rglob("*.nc")]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scmdata

import openscm_runner.run
from openscm_runner.output import NetCDFChunkWriter
from openscm_runner.scenarios import available_scenarios, load_rcmip3_emissions

LOGGER = logging.getLogger("rcmip3-runner")


# ---------------------------------------------------------------------------
# Output variables we drive the runner with. Both FaIRv2 and CICEROSCMPY2
# claim support for the full set; CICEROSCMPY2's _output_variables
# validator catches anything they don't, so feel free to extend this
# alongside the notebook panels.
# ---------------------------------------------------------------------------

_OUTPUT_VARIABLES: tuple[str, ...] = (
    # GSAT (2m over land + 2m over ocean): the variable historically
    # reported as the model's "temperature anomaly", used in the
    # plotting notebooks.
    "Surface Air Temperature Change",
    # Blended (2m over land + SST over ocean): what observational
    # records like HadCRUT/IGCC measure, and the variable Marit's
    # native RCMIP3 submissions report for validation against this
    # adapter. Both adapters compute this natively.
    "Surface Air Ocean Blended Temperature Change",
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Effective Radiative Forcing",
    "Heat Content|Ocean",
    "Heat Uptake",
    "Net Flux to Atmosphere|CO2",
)


# Per-scenario summary printed at the end of the run for quick sanity-checking.
_SUMMARY_AT_2100 = (
    ("Surface Air Temperature Change", "K"),
    ("Atmospheric Concentrations|CO2", "ppm"),
    ("Effective Radiative Forcing", "W/m^2"),
)


# Convenience scenario sets exposed via --scenario-set.
_SCENARIO_SETS: dict[str, tuple[str, ...]] = {
    "ssps": (
        "ssp119", "ssp126", "ssp245", "ssp370",
        "ssp434", "ssp460", "ssp534-over", "ssp585",
    ),
    "esm-ssps": tuple(
        f"esm-ssp{s}" for s in ("119", "126", "245", "370",
                                "434", "460", "534-over", "585")
    ),
    "esm-allghg-ssps": tuple(
        f"esm-allGHG-ssp{s}" for s in ("119", "126", "245", "370",
                                       "434", "460", "534-over", "585")
    ),
    "ssp-sensitivity": (
        "esm-allGHG-ssp370-lowCH4",
        "esm-allGHG-ssp370-lowNTCF",
        "esm-allGHG-ssp370-lowNTCF-HighCH4",
        "esm-allGHG-ssp534-over-highCH4",
        "esm-allGHG-ssp585-lowCH4",
    ),
    "scen7": (
        "scen7-VL", "scen7-LN", "scen7-L", "scen7-ML",
        "scen7-M", "scen7-H", "scen7-HL",
    ),
    "scen7-cd": (
        "scen7-HC", "scen7-HLC", "scen7-LC", "scen7-LNC",
        "scen7-MC", "scen7-MLC", "scen7-VLC",
    ),
    "esm-scen7": tuple(
        f"esm-scen7-{x}" for x in ("VL", "LN", "L", "ML", "M", "H", "HL")
    ),
    "esm-allghg-scen7": tuple(
        f"esm-allGHG-scen7-{x}"
        for x in ("VL", "LN", "L", "ML", "M", "H", "HL")
    ) + ("esm-allGHG-scen7-H-CH4L", "esm-allGHG-scen7-L-CH4H"),
    "flat": tuple(
        f"esm-flat{base}{suffix}"
        for base in ("7.5", "10", "20")
        for suffix in ("", "-zec", "-cdr", "-nz", "-rev")
    ),
    "bell": ("esm-bell-750PgC", "esm-bell-1000PgC", "esm-bell-2000PgC"),
    "pulse": ("esm-pi-CO2pulse", "esm-pi-cdr-pulse"),
    "1pct-brch": (
        "esm-1pct-brch-750PgC",
        "esm-1pct-brch-1000PgC",
        "esm-1pct-brch-2000PgC",
    ),
    "abrupt": ("abrupt-0p5xCO2", "abrupt-2xCO2", "abrupt-4xCO2"),
    "1pctco2": ("1pctCO2", "1pctCO2-4xext", "1pctCO2-cdr"),
    "historical": (
        "historical", "historical-cmip6",
        "esm-hist", "esm-hist-cmip6",
        "esm-allGHG-hist", "esm-allGHG-hist-cmip6",
        "hist-aer", "hist-CO2", "hist-GHG",
    ),
    "picontrol": ("piControl", "esm-piControl", "esm-allGHG-piControl"),
}
# Union of every individual set. Excludes the 4 shelved scenarios
# (1pctCO2-bgc/-rad need non-standard model config; methanemip-*
# need scenario-specific MethaneMIP inputs not yet wired).
_SCENARIO_SETS["all"] = tuple(
    sorted(set().union(*(v for k, v in _SCENARIO_SETS.items())))
)


# ---------------------------------------------------------------------------
# Adapter cfg builders
# ---------------------------------------------------------------------------

# Mode labels used both in the CLI surface and as the top-level output
# subdirectory. Keep stable: the notebook reads these by name.
MODE_EMISSIONS = "emissions"
MODE_CONCENTRATIONS = "concentrations"
VALID_MODES = (MODE_EMISSIONS, MODE_CONCENTRATIONS)


def _cicero_bundle_dir() -> Path:
    """Resolve the CICERO RCMIP-aligned bundle directory used by both adapters.

    FaIRv2 conc-driven mode reads ``{scen}_conc_*`` files from the same
    Marit RCMIP-aligned bundle that CICEROSCMPY2's bundle mode uses, so
    one env var (``CICEROSCMPY2_BUNDLE_DIR``) drives both.
    """
    return Path(
        os.environ.get(
            "CICEROSCMPY2_BUNDLE_DIR",
            Path(__file__).parent.parent / "configurations" / "ciceroscm",
        )
    )


def _build_cfg_fair2(members: int, mode: str) -> list[dict[str, Any]]:
    bundle = os.environ.get("FAIR2_CALIBRATION_PATH")
    if not bundle:
        raise SystemExit(
            "FaIRv2 selected but FAIR2_CALIBRATION_PATH is not set. "
            "See scripts/download_fair2_calibration.py."
        )
    cfg: dict[str, Any] = {
        "native_calibration": bundle,
        "member_indices": range(members),
        "emissions_bundle": bundle,
    }
    if mode == MODE_CONCENTRATIONS:
        rcmip_bundle = _cicero_bundle_dir() / "rcmip-march2026"
        if not rcmip_bundle.is_dir():
            raise SystemExit(
                "FaIRv2 conc-driven mode requires the CICERO RCMIP bundle at "
                f"{rcmip_bundle}. Install it under $CICEROSCMPY2_BUNDLE_DIR/"
                "rcmip-march2026/ or run with --mode emissions only."
            )
        cfg["fair2_conc_driven"] = True
        cfg["fair2_conc_bundle_dir"] = str(rcmip_bundle)
    return [cfg]


def _build_cfg_ciceroscmpy2(members: int, mode: str) -> list[dict[str, Any]]:
    """Bundle mode if rcmip-march2026/ is present, otherwise splice fallback.

    In concentration-driven mode the bundle is mandatory (splice mode
    has no clean conc-driven story); the adapter raises if the bundle
    is missing.
    """
    bundle_dir = _cicero_bundle_dir()
    distribution_json = bundle_dir / "draw_samples_500.json"
    if not distribution_json.exists():
        raise SystemExit(
            f"CICERO-SCM-PY2 selected but {distribution_json} is missing. "
            "Set CICEROSCMPY2_BUNDLE_DIR to a directory containing "
            "draw_samples_500.json."
        )

    cfg: dict[str, Any] = {
        "distribution_json": str(distribution_json),
        "member_indices": range(members),
    }

    rcmip_aligned = bundle_dir / "rcmip-march2026"
    if rcmip_aligned.is_dir():
        cfg["cicero_bundle_dir"] = str(rcmip_aligned)
    elif mode == MODE_CONCENTRATIONS:
        raise SystemExit(
            "CICERO-SCM-PY2 conc-driven mode requires the RCMIP bundle at "
            f"{rcmip_aligned}; splice fallback has no clean conc-driven path."
        )
    else:
        # Emissions-driven splice fallback. Adapter warns at runtime
        # about the present-day warm bias; echo it here too.
        LOGGER.warning(
            "CICERO-SCM bundle dir %s not found; falling back to splice "
            "mode (present-day ~0.3-0.5 K warm bias — install the "
            "Marit RCMIP-aligned bundle for bit-exact reproduction).",
            rcmip_aligned,
        )
        gaspam = bundle_dir / "gases_vupdate_2022_AR6.txt"
        concentrations = bundle_dir / "ssp245_conc_RCMIP.txt"
        missing = [p for p in (gaspam, concentrations) if not p.exists()]
        if missing:
            raise SystemExit(
                "CICERO-SCM-PY2 splice-mode fallback also lacks files: "
                + ", ".join(str(p) for p in missing)
            )
        cfg["gaspam_file"] = str(gaspam)
        cfg["concentrations_file"] = str(concentrations)

    if mode == MODE_CONCENTRATIONS:
        cfg["cicero_conc_run"] = True
    elif mode == MODE_EMISSIONS:
        # Override the adapter's name-based auto-detection so non-`esm-`
        # scenarios (ssp245, scen7-VL, …) still run emissions-driven
        # when the user explicitly asks for emissions mode.
        cfg["cicero_conc_run"] = False
    return [cfg]


# Map CLI model alias -> (canonical model name, cfg builder taking (members, mode)).
_MODEL_DISPATCH: dict[str, tuple[str, Any]] = {
    "fair2": ("FaIRv2", _build_cfg_fair2),
    "ciceroscmpy2": ("CICERO-SCM-PY2", _build_cfg_ciceroscmpy2),
}


def _scenarios_for_mode(scenarios: Sequence[str], mode: str) -> tuple[str, ...]:
    """Drop scenarios that are conceptually incompatible with ``mode``.

    The ``esm-flat*`` idealised family is emissions-driven by
    construction (the protocol prescribes constant CO2 emissions and
    reads out concentrations as a diagnostic); CD mode would require
    bundle concentration files derived from the same model run, which
    defeats the purpose of the experiment. Skip them with a warning.
    """
    if mode == MODE_EMISSIONS:
        return tuple(scenarios)
    filtered = []
    dropped = []
    for s in scenarios:
        if s.startswith("esm-flat"):
            dropped.append(s)
        else:
            filtered.append(s)
    if dropped:
        LOGGER.warning(
            "Skipping %d esm-flat* scenario(s) from --mode concentrations "
            "(they are emissions-driven by construction): %s",
            len(dropped), ", ".join(dropped),
        )
    return tuple(filtered)


# ---------------------------------------------------------------------------
# Per-model dispatch
# ---------------------------------------------------------------------------


def _run_one_model(
    model_alias: str,
    mode: str,
    scenarios: scmdata.ScmRun,
    members: int,
    writer: NetCDFChunkWriter,
) -> None:
    """Run one (adapter, mode) pair across the loaded scenarios; stream to disk."""
    canonical, build_cfgs = _MODEL_DISPATCH[model_alias]
    cfgs = build_cfgs(members, mode)

    scen_names = sorted(scenarios["scenario"].unique())
    LOGGER.info(
        "%s [%s]: dispatching %d scenarios x %d members",
        canonical, mode, len(scen_names), members,
    )

    t0 = time.time()
    result = openscm_runner.run.run(
        climate_models_cfgs={canonical: cfgs},
        scenarios=scenarios,
        output_variables=_OUTPUT_VARIABLES,
        out_config=None,
        output_writer=writer,
    )
    elapsed = time.time() - t0
    LOGGER.info(
        "%s [%s]: finished %d chunks in %.1fs (%.2fs / scenario)",
        canonical, mode, result.n_chunks, elapsed,
        elapsed / max(len(scen_names), 1),
    )

    _print_summary(canonical, mode, result, scen_names)


def _print_summary(
    canonical_model: str,
    mode: str,
    result,
    scenario_names: Sequence[str],
) -> None:
    """Print a per-scenario 2100 quantile summary across members."""
    print(f"\n=== {canonical_model} [{mode}]: 2100 summary "
          "(median / 5-95% across members) ===")
    for scenario in scenario_names:
        # The runner writes one file per (model, scenario), so we
        # read back just the file we need rather than concatenating.
        matches = [
            p for p in result.chunk_paths
            if p.parent.name == _safe_dir(canonical_model)
            and p.stem == _safe_dir(scenario)
        ]
        if not matches:
            print(f"\n  {scenario}: (no output)")
            continue
        run = scmdata.ScmRun.from_nc(matches[0])
        print(f"\n  {scenario}:")
        for variable, unit in _SUMMARY_AT_2100:
            sub = run.filter(variable=variable, year=2100)
            if sub.empty:
                print(f"    {variable} @ 2100: (no rows)")
                continue
            vals = sub.values.flatten()
            print(
                f"    {variable} @ 2100:  "
                f"median {np.median(vals):.2f} {unit}  "
                f"(5-95% {np.quantile(vals, 0.05):.2f} - "
                f"{np.quantile(vals, 0.95):.2f})"
            )


def _safe_dir(name: str) -> str:
    """Mirror NetCDFChunkWriter._safe_name without importing the private helper."""
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)) or "unnamed"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_scenarios(args: argparse.Namespace) -> tuple[str, ...]:
    if args.scenarios and args.scenario_set:
        raise SystemExit(
            "Pass either --scenarios <ids...> or --scenario-set <name>, not both."
        )
    if args.scenarios:
        return tuple(args.scenarios)
    return _SCENARIO_SETS[args.scenario_set or "ssps"]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "--models", nargs="+", default=["fair2", "ciceroscmpy2"],
        choices=sorted(_MODEL_DISPATCH),
        help="Which adapters to dispatch (default: both fair2 and ciceroscmpy2).",
    )
    p.add_argument(
        "--members", type=int, default=10,
        help="Ensemble members per model (default: 10).",
    )
    p.add_argument(
        "--mode", choices=("emissions", "concentrations", "both"),
        default="both",
        help=(
            "Whether to dispatch each (model, scenario) pair through the "
            "adapter's emissions-driven path, concentration-driven path, or "
            "both (default). Concentration-driven mode skips esm-flat* "
            "scenarios because they are emissions-driven by construction."
        ),
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--scenarios", nargs="+", default=None,
        help=(
            "Explicit list of RCMIP3 protocol scenario IDs to run "
            "(e.g. ssp245 esm-flat10-zec scen7-VL)."
        ),
    )
    group.add_argument(
        "--scenario-set", choices=sorted(_SCENARIO_SETS), default=None,
        help=(
            "Named bundle of scenarios. 'ssps' = 8 CMIP6 SSPs, 'scen7' = "
            "7 CMIP7 markers, 'flat' = 15 esm-flat* variants, 'historical' "
            "= historical only, 'all' = union of the four. Default: ssps."
        ),
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "out" / "rcmip3",
        help="Root of the netCDF output tree (default: out/rcmip3/).",
    )
    p.add_argument(
        "--cache-dir", type=Path,
        default=Path(__file__).parent.parent / "configurations" / "rcmip3",
        help=(
            "Where load_rcmip3_emissions caches the upstream emissions CSVs "
            "(default: configurations/rcmip3/)."
        ),
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Set log level to INFO (default: WARNING).",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the RCMIP3 protocol; return process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    requested = _resolve_scenarios(args)
    unknown = sorted(set(requested) - set(available_scenarios()))
    if unknown:
        print(
            f"ERROR: unknown RCMIP3 protocol scenarios: {unknown}\n"
            f"  Use one of: {available_scenarios()}",
            file=sys.stderr,
        )
        return 1

    modes = VALID_MODES if args.mode == "both" else (args.mode,)
    print(
        "RCMIP3 protocol runner\n"
        f"  models:       {', '.join(args.models)}\n"
        f"  modes:        {', '.join(modes)}\n"
        f"  members:      {args.members}\n"
        f"  scenarios:    {len(requested)} ({', '.join(requested)})\n"
        f"  output-dir:   {args.output_dir}\n"
        f"  cache-dir:    {args.cache_dir}\n"
    )

    scenarios_all = load_rcmip3_emissions(
        list(requested), cache_dir=args.cache_dir
    )

    for mode in modes:
        scen_subset = _scenarios_for_mode(requested, mode)
        if not scen_subset:
            LOGGER.warning("Mode %s has no eligible scenarios; skipping.", mode)
            continue
        scenarios_for_mode = scenarios_all.filter(scenario=list(scen_subset))
        mode_dir = args.output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        writer = NetCDFChunkWriter(mode_dir)
        for model_alias in args.models:
            _run_one_model(
                model_alias, mode, scenarios_for_mode, args.members, writer
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
