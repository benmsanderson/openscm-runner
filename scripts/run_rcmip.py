"""
Operational RCMIP3-protocol runner for openscm-runner.

Iterates the scenario list from the RCMIP3 protocol XLSX and drives
each scenario through the registered SCM adapters that can handle it:

- **CICEROSCMPY2** (bundle mode against Marit's
  ``rcmip-march2026/`` bundle) runs every scenario; the adapter
  auto-detects concentration- vs emissions-driven from the scenario
  name (``esm-*`` / ``methanemip-*`` -> emissions; everything else
  -> concentrations), matching ``run_full_rcmip_protocol.py``.
- **FaIRv2** runs only on scenarios present in openscm-runner's
  existing ``tests/test-data/rcmip_scen_ssp_world_emissions.csv``
  fixture (the four canonical SSPs ssp126 / ssp245 / ssp370 /
  ssp585). For other RCMIP scenarios FaIR is auto-skipped with a
  clear log; the CICERO-format-to-ScmRun translation needed to feed
  FaIR with arbitrary RCMIP scenarios is a follow-up.

One CSV per ``(scenario, model)`` is written under ``--output-dir``
(default ``out/rcmip/``) following Marit's convention
``{scenario}_rcmip_{model}.csv`` so the outputs drop straight into
``cscm-calibrate/scripts/reformat_csvs_for_rcmip.py`` if needed.

Usage:

    # Full protocol, default ensemble (20 members each):
    python scripts/run_rcmip.py

    # Subset by glob:
    python scripts/run_rcmip.py --scenarios "ssp*"
    python scripts/run_rcmip.py --scenarios "esm-allGHG-*,ssp245"

    # Smaller ensemble for development:
    python scripts/run_rcmip.py --members 5 --scenarios ssp245

    # Single model:
    python scripts/run_rcmip.py --models cicero

    # Skip scenarios whose output CSV already exists (resume support):
    python scripts/run_rcmip.py --skip-existing

    # List which scenarios would run without actually running:
    python scripts/run_rcmip.py --dry-run --scenarios "esm-flat*"

Env-var overrides:

- ``RCMIP_PROTOCOL_XLSX``       protocol XLSX path
- ``RCMIP_CICERO_BUNDLE_DIR``   CICERO RCMIP bundle directory
- ``RCMIP_CICERO_JSON``         CICERO posterior JSON
- ``RCMIP_FAIR_BUNDLE_DIR``     FaIR Zenodo bundle
- ``RCMIP_OUTPUT_DIR``          where CSVs go
- ``RCMIP_FIXTURE``             openscm-runner SSP emissions fixture
"""
from __future__ import annotations

import argparse
import fnmatch
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import scmdata

import openscm_runner.run

REPO = Path(__file__).parent.parent

DEFAULT_OUTPUT = REPO / "out" / "rcmip"
DEFAULT_PROTOCOL = (
    REPO
    / "configurations"
    / "ciceroscm"
    / "rcmip-march2026"
    / "rcmip_phase3_protocol_v1.1.6.xlsx"
)
DEFAULT_CICERO_BUNDLE = REPO / "configurations" / "ciceroscm" / "rcmip-march2026"
DEFAULT_CICERO_JSON = REPO / "configurations" / "ciceroscm" / "draw_samples_500.json"
DEFAULT_FAIR_BUNDLE = REPO / "configurations" / "fair-calibrate-v1.6.0"
DEFAULT_FIXTURE = REPO / "tests" / "test-data" / "rcmip_scen_ssp_world_emissions.csv"

OUTPUT_VARIABLES = (
    "Surface Air Temperature Change",
    "Atmospheric Concentrations|CO2",
    "Effective Radiative Forcing",
)

LOG = logging.getLogger("rcmip")


def _envpath(name: str, default: Path) -> Path:
    """Return ``Path(os.environ[name])`` if set, else ``default``."""
    val = os.environ.get(name)
    return Path(val) if val else default


def load_protocol(xlsx_path: Path) -> pd.DataFrame:
    """
    Load the RCMIP3 protocol XLSX scenario list.

    Returns a DataFrame with ``Scenario``, ``Type``, ``Duration of
    scenario``, ``Priority`` plus a normalised ``_type`` column.
    """
    df = pd.read_excel(
        xlsx_path,
        sheet_name="scenario_info",
        skiprows=[1, 2],
        usecols=[1, 2, 5, 6],
    )
    df["_type"] = (
        df["Type"]
        .astype(str)
        .str.lower()
        .str.replace(r"[\s\-_]+", "", regex=True)
    )
    return df


def select_scenarios(df: pd.DataFrame, patterns: list[str]) -> pd.DataFrame:
    """Filter scenario DataFrame to those matching any glob in ``patterns``."""
    if not patterns or patterns == ["all"]:
        return df
    all_names = df["Scenario"].astype(str).tolist()
    selected = set()
    for pat in patterns:
        for name in all_names:
            if fnmatch.fnmatch(name, pat):
                selected.add(name)
    return df[df["Scenario"].isin(selected)].copy()


def is_emissions_driven(scenario_name: str) -> bool:
    """
    Mirror :func:`_auto_conc_run` from the CICEROSCMPY2 adapter.

    ``esm-*`` and ``methanemip-*`` are emissions-driven; everything
    else (ssp*, scen7-*, 1pctCO2*, abrupt*, hist-*, piControl) is
    concentration-driven, matching Marit's protocol runner.
    """
    s = scenario_name.lower()
    return s.startswith(("esm-", "esm_", "methanemip"))


def _stub_scmrun(scenario_name: str, nyend: int) -> scmdata.ScmRun:
    """
    Build a one-row ScmRun naming ``scenario_name``.

    CICEROSCMPY2 bundle mode uses the input ScmRun only to enumerate
    scenario names + pick ``nyend``; the actual emissions come from
    the bundle's ``{scen}_em_…`` file (or the bundle scenario
    concentrations file for conc-driven runs). A stub ScmRun gives
    bundle mode the scenario label without having to wire arbitrary
    RCMIP scenarios through the openscm-runner input pipeline.
    """
    df = pd.DataFrame(
        [[0.0, 0.0]],
        index=pd.MultiIndex.from_tuples(
            [
                (
                    scenario_name,
                    "World",
                    "Mt CO2/yr",
                    "Emissions|CO2|MAGICC Fossil and Industrial",
                    "rcmip-runner-stub",
                )
            ],
            names=["scenario", "region", "unit", "variable", "model"],
        ),
        columns=[2015, nyend],
    )
    return scmdata.ScmRun(df)


def _bundle_lookup_name(scenario_name: str) -> str:
    """
    Translate a protocol scenario name to the bundle file-lookup name.

    Matches the stripping ``run_full_rcmip_protocol.py`` applies:
    drop the ``esm-`` and ``allGHG-`` prefixes (the bundle's CICERO-
    format emissions / concentration files are named after the
    scenario *content*, not the protocol's emissions-driven labelling),
    and drop a trailing ``C`` from scen7 variants (``scen7-HC`` ->
    ``scen7-H``: the trailing C indicates "concentration-driven" in
    the protocol but the bundle file name uses the bare variant).
    """
    stripped = scenario_name.split("esm-")[-1].split("allGHG-")[-1]
    if stripped.startswith("scen7") and stripped.endswith("C"):
        stripped = stripped[:-1]
    return stripped


def _run_cicero(
    scenario_name: str, duration: int, args: argparse.Namespace
) -> scmdata.ScmRun:
    nyend = 1750 + int(duration) - 1
    lookup_name = _bundle_lookup_name(scenario_name)
    # Pass an explicit conc_run override so the adapter's auto-detect
    # (which looks at the stub's scenario name = lookup_name) doesn't
    # flip the answer for cases like esm-scen7-H, where the stripped
    # lookup name is `scen7-H` (would auto-detect as conc-driven)
    # but the original protocol scenario is emissions-driven.
    cicero_conc_run = not is_emissions_driven(scenario_name)
    stub = _stub_scmrun(lookup_name, nyend)
    cfg = {
        "distribution_json": str(args.cicero_json),
        "cicero_bundle_dir": str(args.cicero_bundle),
        "member_indices": list(range(args.members)),
        "nyend": nyend,
        "cicero_conc_run": cicero_conc_run,
    }
    result = openscm_runner.run.run(
        climate_models_cfgs={"CICERO-SCM-PY2": [cfg]},
        scenarios=stub,
        output_variables=OUTPUT_VARIABLES,
        out_config=None,
    )
    # The adapter labels output rows with the stub's scenario value
    # (lookup_name). Restore the protocol scenario name so the CSV is
    # consistent with what the user requested.
    if lookup_name != scenario_name:
        result["scenario"] = scenario_name
    return result


def _run_fair(
    scenario_name: str, args: argparse.Namespace
) -> scmdata.ScmRun:
    if not args.fixture.exists():
        raise FileNotFoundError(f"FaIR fixture not found at {args.fixture}")
    fixture = scmdata.ScmRun(args.fixture, lowercase_cols=True)
    scen = fixture.filter(scenario=scenario_name)
    if scen.empty:
        raise NotImplementedError(
            f"FaIRv2: no input data for scenario {scenario_name!r} in "
            f"{args.fixture}. FaIR currently runs on scenarios present "
            "in openscm-runner's SSP fixture (ssp126/245/370/585). The "
            "CICERO-format-to-ScmRun translator needed for arbitrary "
            "RCMIP scenarios is a follow-up."
        )
    cfg = {
        "native_calibration": str(args.fair_bundle),
        "member_indices": list(range(args.members)),
    }
    return openscm_runner.run.run(
        climate_models_cfgs={"FaIRv2": [cfg]},
        scenarios=scen,
        output_variables=OUTPUT_VARIABLES,
        out_config=None,
    )


def _save_csv(result: scmdata.ScmRun, out_path: Path) -> None:
    result.timeseries(time_axis="year").to_csv(out_path)


def main() -> int:  # noqa: PLR0912, PLR0915
    """CLI entry point; returns 0 on success, 1-2 on errors / failures."""
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scenarios",
        default="all",
        help='Comma-separated scenario names or glob patterns (e.g. '
        '"ssp*", "esm-allGHG-*,ssp245", "all").',
    )
    p.add_argument(
        "--members", type=int, default=20, help="Ensemble members per model."
    )
    p.add_argument(
        "--models",
        default="all",
        choices=["all", "cicero", "fair", "both"],
        help='Which adapters to run. "all"/"both" = CICERO + FaIR where applicable.',
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_envpath("RCMIP_OUTPUT_DIR", DEFAULT_OUTPUT),
    )
    p.add_argument(
        "--protocol-xlsx",
        type=Path,
        default=_envpath("RCMIP_PROTOCOL_XLSX", DEFAULT_PROTOCOL),
    )
    p.add_argument(
        "--cicero-bundle",
        type=Path,
        default=_envpath("RCMIP_CICERO_BUNDLE_DIR", DEFAULT_CICERO_BUNDLE),
    )
    p.add_argument(
        "--cicero-json",
        type=Path,
        default=_envpath("RCMIP_CICERO_JSON", DEFAULT_CICERO_JSON),
    )
    p.add_argument(
        "--fair-bundle",
        type=Path,
        default=_envpath("RCMIP_FAIR_BUNDLE_DIR", DEFAULT_FAIR_BUNDLE),
    )
    p.add_argument(
        "--fixture",
        type=Path,
        default=_envpath("RCMIP_FIXTURE", DEFAULT_FIXTURE),
        help="openscm-runner SSP emissions fixture (FaIR input).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (scenario, model) pairs whose output CSV already exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the (scenario, model) pairs that would run and exit.",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    for path, label in [
        (args.protocol_xlsx, "protocol XLSX"),
        (args.cicero_bundle, "CICERO bundle dir"),
        (args.cicero_json, "CICERO distribution JSON"),
        (args.fair_bundle, "FaIR bundle dir"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}", file=sys.stderr)
            return 1

    df = load_protocol(args.protocol_xlsx)
    patterns = [s.strip() for s in args.scenarios.split(",")]
    df = select_scenarios(df, patterns)
    if df.empty:
        print(f"ERROR: no scenarios match {patterns}", file=sys.stderr)
        return 1

    # "Variable" duration scenarios in the protocol have no fixed
    # endpoint and need special handling we don't support yet.
    df_var = df[df["Duration of scenario"].astype(str) == "Variable"]
    if not df_var.empty:
        LOG.warning(
            "Skipping %d scenario(s) with Variable duration: %s",
            len(df_var),
            df_var["Scenario"].tolist(),
        )
        df = df[df["Duration of scenario"].astype(str) != "Variable"]

    models_arg = args.models
    models = (
        ["cicero", "fair"] if models_arg in ("all", "both") else [models_arg]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("=" * 70)
    LOG.info(
        "RCMIP runner: %d scenario(s), %d model(s), %d members each",
        len(df), len(models), args.members,
    )
    LOG.info("  Models:        %s", models)
    LOG.info("  Output:        %s", args.output_dir)
    LOG.info("  Protocol:      %s", args.protocol_xlsx.name)
    LOG.info("  CICERO bundle: %s", args.cicero_bundle.name)
    LOG.info("  CICERO JSON:   %s", args.cicero_json.name)
    LOG.info("  FaIR bundle:   %s", args.fair_bundle.name)
    LOG.info("=" * 70)

    if args.dry_run:
        for _, row in df.iterrows():
            scen = row["Scenario"]
            mode = "EMIS" if is_emissions_driven(scen) else "CONC"
            apply = []
            if "cicero" in models:
                apply.append("cicero")
            fair_applies = is_emissions_driven(scen) or scen.startswith("ssp")
            if "fair" in models and fair_applies:
                apply.append("fair?")
            LOG.info("  %-40s [%s] models=%s", scen, mode, apply)
        return 0

    successes: list[str] = []
    failures: list[tuple[str, str, str]] = []
    skipped: list[str] = []

    t_total = time.time()
    for _, row in df.iterrows():
        scen = row["Scenario"]
        duration = row["Duration of scenario"]
        emis_driven = is_emissions_driven(scen)

        for model in models:
            out_path = args.output_dir / f"{scen}_rcmip_{model}.csv"

            if args.skip_existing and out_path.exists():
                LOG.info("  SKIP %-40s %-7s (output exists)", scen, model)
                skipped.append(out_path.name)
                continue

            if model == "fair" and not emis_driven and not scen.startswith("ssp"):
                LOG.info(
                    "  SKIP %-40s %-7s (conc-driven; FaIR has no input)",
                    scen, model,
                )
                skipped.append(out_path.name)
                continue

            t0 = time.time()
            try:
                if model == "cicero":
                    result = _run_cicero(scen, int(duration), args)
                else:  # fair
                    result = _run_fair(scen, args)
                _save_csv(result, out_path)
                elapsed = time.time() - t0
                n_members_out = len(set(result["run_id"].tolist()))
                LOG.info(
                    "  OK   %-40s %-7s -> %s  (%.1f s, %d members)",
                    scen, model, out_path.name, elapsed, n_members_out,
                )
                successes.append(out_path.name)
            except NotImplementedError as exc:
                LOG.info("  SKIP %-40s %-7s (%s)", scen, model, exc)
                skipped.append(out_path.name)
            except Exception as exc:  # pylint: disable=broad-except
                LOG.warning("  FAIL %-40s %-7s: %s", scen, model, exc)
                failures.append((scen, model, str(exc)))

    LOG.info("=" * 70)
    LOG.info(
        "Done in %.1f s. %d succeeded, %d failed, %d skipped.",
        time.time() - t_total, len(successes), len(failures), len(skipped),
    )
    if failures:
        LOG.warning("Failures:")
        for scen, model, exc in failures:
            LOG.warning("  %s %s: %s", scen, model, exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
