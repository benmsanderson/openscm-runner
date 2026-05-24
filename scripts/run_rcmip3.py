"""
RCMIP3 protocol runner: dispatch one or more SCMs across the RCMIP3
emissions-driven experiment set and stream per-(model, scenario) output
to a netCDF tree on disk.

Replaces the per-model demos ``scripts/run_rcmip_fair2.py`` and
``scripts/run_rcmip_ciceroscm_py2.py``. Scenarios are resolved through
:func:`openscm_runner.scenarios.load_rcmip3_emissions` so the same CLI
covers the CMIP6 SSPs, the CMIP7 ScenarioMIP ``scen7-*`` set, and the
``esm-flat*`` idealised family.

Example
-------

    # One-time setup
    pip install -e ".[fair2,ciceroscmpy2]"
    scripts/download_fair2_calibration.py
    # plus a CICERO-SCM bundle at $CICEROSCMPY2_BUNDLE_DIR

    export FAIR2_CALIBRATION_PATH=$PWD/configurations/fair-calibrate-v1.6.0
    export CICEROSCMPY2_BUNDLE_DIR=$PWD/configurations/ciceroscm

    # 10-member sweep across SSPs + scen7 + flat-*:
    scripts/run_rcmip3.py --members 10 --scenario-set all

    # Smaller / faster iteration:
    scripts/run_rcmip3.py --members 5 --scenarios ssp245 ssp585 esm-flat10-zec

Output layout
-------------
``--output-dir`` (default ``out/rcmip3/``) ends up containing one
file per (model, scenario) pair::

    out/rcmip3/
        FaIRv2/
            ssp119.nc
            ssp245.nc
            …
        CICERO-SCM-PY2/
            ssp119.nc
            …

Read back with::

    from openscm_runner.output import RunResult
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
    "Surface Air Temperature Change",
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
    "scen7": (
        "scen7-VL", "scen7-LN", "scen7-L", "scen7-ML",
        "scen7-M", "scen7-H", "scen7-HL",
    ),
    "flat": tuple(
        f"esm-flat{base}{suffix}"
        for base in ("7.5", "10", "20")
        for suffix in ("", "-zec", "-cdr", "-nz", "-rev")
    ),
    "historical": ("historical",),
}
_SCENARIO_SETS["all"] = (
    _SCENARIO_SETS["ssps"]
    + _SCENARIO_SETS["scen7"]
    + _SCENARIO_SETS["flat"]
    + _SCENARIO_SETS["historical"]
)


# ---------------------------------------------------------------------------
# Adapter cfg builders
# ---------------------------------------------------------------------------


def _build_cfg_fair2(members: int) -> list[dict[str, Any]]:
    bundle = os.environ.get("FAIR2_CALIBRATION_PATH")
    if not bundle:
        raise SystemExit(
            "FaIRv2 selected but FAIR2_CALIBRATION_PATH is not set. "
            "See scripts/download_fair2_calibration.py."
        )
    return [
        {
            "native_calibration": bundle,
            "member_indices": range(members),
            "emissions_bundle": bundle,
        }
    ]


def _build_cfg_ciceroscmpy2(members: int) -> list[dict[str, Any]]:
    """Bundle mode if rcmip-march2026/ is present, otherwise splice fallback."""
    bundle_dir = Path(
        os.environ.get(
            "CICEROSCMPY2_BUNDLE_DIR",
            Path(__file__).parent.parent / "configurations" / "ciceroscm",
        )
    )
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
        return [cfg]

    # Splice fallback. The adapter itself warns about the present-day
    # warm bias when this path runs; we still echo the warning here so
    # users running with --models ciceroscmpy2 see it in CLI output.
    LOGGER.warning(
        "CICERO-SCM bundle dir %s not found; falling back to splice mode "
        "(present-day ~0.3-0.5 K warm bias — install the Marit "
        "RCMIP-aligned bundle for bit-exact reference reproduction).",
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
    return [cfg]


# Map CLI model alias -> (canonical model name, cfg builder).
_MODEL_DISPATCH: dict[str, tuple[str, Any]] = {
    "fair2": ("FaIRv2", _build_cfg_fair2),
    "ciceroscmpy2": ("CICERO-SCM-PY2", _build_cfg_ciceroscmpy2),
}


# ---------------------------------------------------------------------------
# Per-model dispatch
# ---------------------------------------------------------------------------


def _run_one_model(
    model_alias: str,
    scenarios: scmdata.ScmRun,
    members: int,
    writer: NetCDFChunkWriter,
) -> None:
    """Run one adapter across all loaded scenarios; stream output to disk."""
    canonical, build_cfgs = _MODEL_DISPATCH[model_alias]
    cfgs = build_cfgs(members)

    scen_names = sorted(scenarios["scenario"].unique())
    LOGGER.info(
        "%s: dispatching %d scenarios x %d members",
        canonical, len(scen_names), members,
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
        "%s: finished %d chunks in %.1fs (%.2fs / scenario)",
        canonical, result.n_chunks, elapsed,
        elapsed / max(len(scen_names), 1),
    )

    _print_summary(canonical, result, scen_names)


def _print_summary(
    canonical_model: str,
    result,
    scenario_names: Sequence[str],
) -> None:
    """Print a per-scenario 2100 quantile summary across members."""
    print(f"\n=== {canonical_model}: 2100 summary "
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

    print(
        "RCMIP3 protocol runner\n"
        f"  models:       {', '.join(args.models)}\n"
        f"  members:      {args.members}\n"
        f"  scenarios:    {len(requested)} ({', '.join(requested)})\n"
        f"  output-dir:   {args.output_dir}\n"
        f"  cache-dir:    {args.cache_dir}\n"
    )

    scenarios = load_rcmip3_emissions(list(requested), cache_dir=args.cache_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = NetCDFChunkWriter(args.output_dir)

    for model_alias in args.models:
        _run_one_model(model_alias, scenarios, args.members, writer)

    return 0


if __name__ == "__main__":
    sys.exit(main())
