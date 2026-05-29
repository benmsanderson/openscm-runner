"""
Compare CICEROSCMPY2 esm-flat10-zec output against Marit's native
500-member reference (configurations/ciceroscm/flat10_zec_marit.csv).

Run scripts/run_rcmip3.py first to produce out/rcmip3_500/emissions/
CICERO-SCM-PY2/esm-flat10-zec.nc (need --members 500 to match Marit's
ensemble size for a like-for-like median comparison).

Saves a two-panel figure to notebooks/figures/compare_flat10_zec_marit.png:
left panel = GSAT timeseries with median and 5-95% bands, right panel
= zoomed post-cessation period 1950-2200 to show the ZEC structure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scmdata

import openscm_runner  # noqa: F401  (applies scmdata pandas-3 patches)

REPO_ROOT = Path(__file__).parent.parent
MARIT_CSV = REPO_ROOT / "configurations" / "ciceroscm" / "flat10_zec_marit.csv"
OUR_NC = (
    REPO_ROOT / "out" / "rcmip3_500" / "emissions"
    / "CICERO-SCM-PY2" / "esm-flat10-zec.nc"
)
OUT_FIG = REPO_ROOT / "notebooks" / "figures" / "compare_flat10_zec_marit.png"


def _load_marit(path: Path) -> pd.DataFrame:
    """Return Marit's GSAT timeseries as members x years."""
    raw = pd.read_csv(path, header=None)
    years = list(range(1750, 2501))
    data = raw.iloc[:, 8:]
    data.columns = years
    is_gsat = raw.iloc[:, 6].values == "Surface Air Temperature Change"
    gsat = data[is_gsat]
    gsat.index = raw.iloc[:, 3][is_gsat].values
    gsat.index.name = "run_id"
    return gsat


def _load_ours(path: Path) -> pd.DataFrame:
    """Return our GSAT timeseries as members x years."""
    run = scmdata.ScmRun.from_nc(path)
    ts = run.filter(
        variable="Surface Air Temperature Change"
    ).timeseries(time_axis="year")
    # ScmRun gives a multi-index; flatten to run_id for the join.
    ts.index = ts.index.get_level_values("run_id")
    return ts


def _band(ax, df, color, label, year_range=None):
    """Plot median + 5-95% band + 17-83% band."""
    years = df.columns.values
    if year_range is not None:
        mask = (years >= year_range[0]) & (years <= year_range[1])
        years = years[mask]
        df = df.iloc[:, mask]
    median = df.median(axis=0).values
    q05 = df.quantile(0.05, axis=0).values
    q95 = df.quantile(0.95, axis=0).values
    q17 = df.quantile(0.17, axis=0).values
    q83 = df.quantile(0.83, axis=0).values
    ax.fill_between(years, q05, q95, color=color, alpha=0.12,
                    label=f"{label} 5-95%")
    ax.fill_between(years, q17, q83, color=color, alpha=0.20)
    ax.plot(years, median, color=color, lw=1.8, label=f"{label} median")


def main() -> int:
    if not MARIT_CSV.exists():
        print(f"ERROR: Marit reference not found at {MARIT_CSV}")
        return 1
    if not OUR_NC.exists():
        print(
            f"ERROR: our output not found at {OUR_NC}\n"
            "Run: scripts/run_rcmip3.py --models ciceroscmpy2 "
            "--members 500 --scenarios esm-flat10-zec --mode emissions "
            "--output-dir out/rcmip3_500"
        )
        return 1

    marit = _load_marit(MARIT_CSV)
    ours = _load_ours(OUR_NC)
    print(f"Marit: {len(marit)} members, years {marit.columns.min()}-{marit.columns.max()}")
    print(f"Ours:  {len(ours)} members, years {ours.columns.min()}-{ours.columns.max()}")

    # Median diff at key years for the text annotation.
    print()
    print(f"{'Year':<6} {'Marit median':<14} {'Ours median':<14} {'Diff (K)':<10}")
    for y in (1949, 1955, 2000, 2024, 2050, 2100, 2200, 2300, 2500):
        m = marit[y].median()
        o = ours[y].median()
        print(f"{y:<6} {m:>10.3f}{'':<6} {o:>10.3f}{'':<4} {o-m:>+8.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, x_range, title in (
        (axes[0], (1850, 2500), "Full timeseries 1850-2500"),
        (axes[1], (1950, 2200), "Post-cessation 1950-2200 (zoom)"),
    ):
        _band(ax, marit, "tab:gray", "Marit (500 members)", year_range=x_range)
        _band(ax, ours, "tab:red", "Ours (500 members)", year_range=x_range)
        ax.axvline(1950, color="black", lw=0.8, linestyle=":",
                   alpha=0.5, label="emissions stop (yr 100)")
        ax.set_xlim(*x_range)
        ax.set_xlabel("Year")
        ax.set_ylabel("Surface Air Temperature Change (K)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "CICEROSCMPY2 esm-flat10-zec: openscm-runner adapter vs Marit's "
        "native 500-member reference",
        y=1.0,
    )
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    print()
    print(f"Wrote {OUT_FIG.relative_to(REPO_ROOT)}")

    # Per-member comparison: first 10 members from Marit by row order,
    # matched to our first 10 by run_id. Plot Marit (solid) vs ours
    # (dashed) for each member individually so any per-member drift
    # is visible (a band match could hide compensating member-level
    # errors).
    first10_marit_runids = list(marit.index[:10])
    ours_runids = ours.index.tolist()
    matched = [rid for rid in first10_marit_runids if rid in ours_runids]
    print()
    print(f"Per-member overlay: {len(matched)}/10 first-10 Marit "
          f"run_ids matched in our output")

    cmap = plt.get_cmap("tab10")
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, x_range, title in (
        (axes2[0], (1850, 2500), "Full timeseries (first 10 members)"),
        (axes2[1], (1950, 2200), "Post-cessation zoom (first 10 members)"),
    ):
        for i, rid in enumerate(matched):
            color = cmap(i % 10)
            years_m = marit.columns.values
            years_o = ours.columns.values
            m_mask = (years_m >= x_range[0]) & (years_m <= x_range[1])
            o_mask = (years_o >= x_range[0]) & (years_o <= x_range[1])
            ax.plot(
                years_m[m_mask], marit.loc[rid].values[m_mask],
                color=color, lw=1.3, alpha=0.9,
                label=f"{rid}" if ax is axes2[0] else None,
            )
            ax.plot(
                years_o[o_mask], ours.loc[rid].values[o_mask],
                color=color, lw=1.3, alpha=0.9, linestyle="--",
            )
        ax.axvline(1950, color="black", lw=0.8, linestyle=":", alpha=0.5)
        ax.set_xlim(*x_range)
        ax.set_xlabel("Year")
        ax.set_ylabel("Surface Air Temperature Change (K)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes2[0].legend(fontsize=7, loc="upper right", ncol=2,
                    title="run_id (solid=Marit, dashed=ours)")
    fig2.suptitle(
        "Per-member trajectory check: first 10 matched run_ids "
        "from Marit's reference vs ours",
        y=1.0,
    )
    fig2.tight_layout()
    out2 = OUT_FIG.with_name("compare_flat10_zec_marit_per_member.png")
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"Wrote {out2.relative_to(REPO_ROOT)}")

    # Diff statistics on matched members.
    print()
    print(f"{'Year':<6} {'max |diff| (K)':<15} {'mean diff (K)':<15}")
    for y in (1955, 2000, 2100, 2300, 2500):
        diffs = (ours.loc[matched, y] - marit.loc[matched, y]).abs()
        meandiff = (ours.loc[matched, y] - marit.loc[matched, y]).mean()
        print(f"{y:<6} {diffs.max():>10.4f}{'':<5} {meandiff:>+10.4f}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
