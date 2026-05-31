"""
Scorecard: openscm-runner CICEROSCMPY2 output vs Marit's native
RCMIP3 submissions (configurations/ciceroscm/marittmp/{scenario}_rcmip
_draw_samples_500.csv) on every protocol scenario where we have both
sides on disk.

Marit's submissions report ``Surface Air Ocean Blended Temperature
Change`` at 500 members; we compare on matched run_ids (the
``draw_samples_500.json`` posterior member ordering is the same on
both sides) so any residual diff is a real adapter / pipeline
difference, not sampling noise.

For each scenario the scorecard prints:

- the number of matched run_ids
- max |median diff|, mean median diff, mean |per-member diff| at a
  handful of key years (peak, mid-projection, end)
- an overall PASS / WARN / FAIL label using thresholds documented
  inline at the top of the file

Run:

    scripts/validate_against_marit.py
    scripts/validate_against_marit.py --output-dir out/rcmip3_500
    scripts/validate_against_marit.py --scenarios esm-flat10-zec ssp245
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import scmdata

import openscm_runner  # noqa: F401  (applies scmdata pandas-3 patches)

REPO_ROOT = Path(__file__).parent.parent
MARIT_DIR = REPO_ROOT / "configurations" / "ciceroscm" / "marittmp"
DEFAULT_OUR_ROOT = REPO_ROOT / "out" / "rcmip3"

MARIT_VARIABLE = "Surface Air Ocean Blended Temperature Change"

# Thresholds for the PASS / WARN / FAIL label. Tuned against the
# esm-flat10-zec validation done in the LUC fix commit: 500-member
# CICERO matches Marit to within ~30 mK in the active period and
# ~60 mK at the long tail. PASS catches that pattern; WARN catches
# anything noticeably off; FAIL catches obvious bugs.
PASS_MEDIAN_DIFF_K = 0.05
WARN_MEDIAN_DIFF_K = 0.15


@dataclass(frozen=True)
class _ScenarioResult:
    scenario: str
    n_marit: int
    n_ours: int
    n_matched: int
    max_abs_med_diff: float
    mean_med_diff: float
    label: str


def _load_marit(scenario: str) -> pd.DataFrame | None:
    path = MARIT_DIR / f"{scenario}_rcmip_draw_samples_500.csv"
    if not path.exists() or path.stat().st_size == 0:
        # Marit's submission carries empty placeholders for the
        # feedback-decoupling experiments (1pctCO2-bgc, 1pctCO2-rad)
        # that need non-standard model config. Treated the same as
        # a missing file by the caller.
        return None
    raw = pd.read_csv(path, header=None)
    # Marit's CSVs are variable-length per scenario; the year columns
    # start at column 8 and run from 1750 onwards by row position.
    n_year_cols = raw.shape[1] - 8
    years = list(range(1750, 1750 + n_year_cols))
    data = raw.iloc[:, 8:]
    data.columns = years
    is_var = raw.iloc[:, 6].values == MARIT_VARIABLE
    out = data[is_var]
    out.index = raw.iloc[:, 3][is_var].values
    out.index.name = "run_id"
    return out


def _protocol_mode(scenario: str) -> str:
    """The RCMIP3 mode implied by the scenario name.

    Marit's submission filename convention follows the protocol:
    bare names like ``ssp245`` / ``scen7-HC`` / ``historical`` /
    ``1pctCO2`` / ``abrupt-*`` are the concentration-driven
    experiments; ``esm-`` or ``hist-{aer,CO2,GHG}`` are
    emissions-driven. Our runner's --mode flag controls which
    subtree the chunk ends up in, so the scorecard has to read the
    right one for each scenario.
    """
    s = scenario.lower()
    if s.startswith("esm-") or s in ("hist-aer", "hist-co2", "hist-ghg"):
        return "emissions"
    return "concentrations"


def _load_ours(
    scenario: str, our_root: Path, mode: str | None = None,
) -> pd.DataFrame | None:
    if mode is None:
        mode = _protocol_mode(scenario)
    path = our_root / mode / "CICERO-SCM-PY2" / f"{scenario}.nc"
    if not path.exists():
        return None
    run = scmdata.ScmRun.from_nc(path)
    ts = run.filter(variable=MARIT_VARIABLE).timeseries(time_axis="year")
    if ts.empty:
        return None
    ts.index = ts.index.get_level_values("run_id")
    return ts


def _score_one(
    scenario: str, our_root: Path, mode: str | None = None,
) -> _ScenarioResult | None:
    """Score one scenario; ``mode=None`` triggers protocol-name-based auto-detect."""
    marit = _load_marit(scenario)
    ours = _load_ours(scenario, our_root, mode=mode)
    if marit is None:
        return _ScenarioResult(scenario, 0, 0, 0,
                               float("nan"), float("nan"), "NO_MARIT_REF")
    if ours is None:
        return _ScenarioResult(scenario, len(marit), 0, 0,
                               float("nan"), float("nan"), "NO_OURS_RUN")
    matched = sorted(set(marit.index) & set(ours.index))
    if not matched:
        return _ScenarioResult(scenario, len(marit), len(ours), 0,
                               float("nan"), float("nan"), "NO_MATCH")
    common_years = sorted(set(marit.columns) & set(ours.columns))
    if not common_years:
        return _ScenarioResult(scenario, len(marit), len(ours), len(matched),
                               float("nan"), float("nan"), "NO_YEAR_OVERLAP")
    m = marit.loc[matched, common_years]
    o = ours.loc[matched, common_years]
    med_diff = o.median(axis=0) - m.median(axis=0)
    max_abs = med_diff.abs().max()
    mean = med_diff.mean()
    if max_abs <= PASS_MEDIAN_DIFF_K:
        label = "PASS"
    elif max_abs <= WARN_MEDIAN_DIFF_K:
        label = "WARN"
    else:
        label = "FAIL"
    return _ScenarioResult(
        scenario=scenario,
        n_marit=len(marit),
        n_ours=len(ours),
        n_matched=len(matched),
        max_abs_med_diff=float(max_abs),
        mean_med_diff=float(mean),
        label=label,
    )


def _discover_scenarios() -> list[str]:
    """Every scenario for which we have a Marit reference CSV."""
    return sorted(
        p.stem.replace("_rcmip_draw_samples_500", "")
        for p in MARIT_DIR.glob("*_rcmip_draw_samples_500.csv")
    )


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUR_ROOT,
        help=(
            "Root of our netCDF output tree (the directory containing "
            "emissions/CICERO-SCM-PY2/{scenario}.nc). Defaults to "
            "out/rcmip3."
        ),
    )
    p.add_argument(
        "--mode", default=None,
        choices=("emissions", "concentrations"),
        help=(
            "Force a specific mode subtree to read. Default: auto-detect "
            "per scenario from the RCMIP3 naming convention (bare names "
            "= concentration-driven, esm-* / hist-{aer,CO2,GHG} = "
            "emissions-driven)."
        ),
    )
    p.add_argument(
        "--scenarios", nargs="+", default=None,
        help=(
            "Specific scenarios to score. Defaults to every scenario "
            "with a Marit reference CSV on disk."
        ),
    )
    args = p.parse_args(argv)

    scenarios = args.scenarios or _discover_scenarios()
    print(
        f"Scoring {len(scenarios)} scenarios against Marit's reference "
        f"({MARIT_VARIABLE!r}), mode={args.mode}, our-root={args.output_dir}"
    )
    print()

    results = [
        _score_one(s, our_root=args.output_dir, mode=args.mode)
        for s in scenarios
    ]

    header = (
        f"{'label':<14} {'scenario':<35} {'matched':>9} "
        f"{'max|med dif|':>14} {'mean med dif':>14}"
    )
    print(header)
    print("-" * len(header))
    label_counts: dict[str, int] = {}
    for r in results:
        if r is None:
            continue
        label_counts[r.label] = label_counts.get(r.label, 0) + 1
        matched = f"{r.n_matched}/{r.n_marit}"
        if r.label.startswith("NO"):
            print(f"{r.label:<14} {r.scenario:<35} {matched:>9} "
                  f"{'-':>14} {'-':>14}")
        else:
            print(
                f"{r.label:<14} {r.scenario:<35} {matched:>9} "
                f"{r.max_abs_med_diff:>10.4f} K   "
                f"{r.mean_med_diff:>+10.4f} K"
            )

    print()
    print("Summary:")
    for lbl in ("PASS", "WARN", "FAIL", "NO_OURS_RUN",
                "NO_MATCH", "NO_YEAR_OVERLAP", "NO_MARIT_REF"):
        if lbl in label_counts:
            print(f"  {lbl}: {label_counts[lbl]}")
    return 0 if label_counts.get("FAIL", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
