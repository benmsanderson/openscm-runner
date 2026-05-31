#!/usr/bin/env python3
"""Translate CICERO RCMIP-aligned bundle ``_em_*`` files to IAMC long format.

The CICEROSCM ``rcmip-march2026`` bundle ships scenario-specific
``{scenario}_em_{gases_ep}.txt`` files. For most scenarios those files
have IAMC twins in the upstream archives (RCMIP3, scenariomip-paper),
which ``openscm_runner.scenarios.load_rcmip3_emissions`` already
serves. Two scenarios are an exception:

  - ``esm-allGHG-scen7-H-CH4L``
  - ``esm-allGHG-scen7-L-CH4H``

These are the CH4-swap experiments from section 2.2.4 of the RCMIP3
protocol: the same scen7-{H,L} all-GHG trajectory but with a different
CH4 sub-pathway. No published IAMC source carries them; the only copy
is in CICERO's bundle. This script reads the bundle file once and
emits a canonical IAMC long-format CSV that the loader can serve like
any other scenario, retiring the bundle-only stub path for these IDs.

Run from the repo root::

    python scripts/translate_cicero_bundle_to_iamc.py \\
        --bundle-dir configurations/ciceroscm/rcmip-march2026 \\
        --output src/openscm_runner/scenarios/data/bundle_translated/cicero_rcmip_march2026.csv

By default both CH4-swap scenarios are translated together into one
CSV. Re-run after the bundle is updated to refresh the file (it's a
committed artefact, not generated at install time).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from openscm_units import unit_registry as ureg

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from openscm_runner.adapters.utils.cicero_utils.make_scenario_common import (  # noqa: E402
    cicero_comp_dict,
)

DEFAULT_BUNDLE_DIR = REPO_ROOT / "configurations" / "ciceroscm" / "rcmip-march2026"
DEFAULT_GASES_EP = "gases_vupdate_2024_WMO_added_new.txt"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "src"
    / "openscm_runner"
    / "scenarios"
    / "data"
    / "bundle_translated"
    / "cicero_rcmip_march2026.csv"
)

# Scenarios this script can translate from bundle to IAMC. Each entry
# specifies the protocol id (used as Scenario in the output CSV) and
# the bundle stem (the part of the filename before ``_em_``).
SCENARIOS: dict[str, str] = {
    "esm-allGHG-scen7-H-CH4L": "esm-allGHG-scen7-H-CH4L",
    "esm-allGHG-scen7-L-CH4H": "esm-allGHG-scen7-L-CH4H",
}

# CICERO species -> (IAMC variable name, IAMC unit string).
#
# We hand-pick the IAMC unit to match what RCMIP3 / SCI ship for the
# same species, so the iamc_loader's harmonisation and the canonical
# allowlist accept this CSV unchanged. The conversion factor between
# CICERO's bundle unit and our chosen IAMC unit comes from
# openscm-units below.
_OUTPUT_UNITS: dict[str, str] = {
    # CO2: PgC -> Mt CO2/yr via the carbon context (PgC * 44/12 / 1e-3 = ...)
    # We emit MAGICC-style sector splits since the iamc loader's
    # canonical allowlist uses those.
    "CO2": ("Emissions|CO2|MAGICC Fossil and Industrial", "Mt CO2/yr"),
    "CO2_lu": ("Emissions|CO2|MAGICC AFOLU", "Mt CO2/yr"),
    "CH4": ("Emissions|CH4", "Mt CH4/yr"),
    "N2O": ("Emissions|N2O", "kt N2O/yr"),
    "SO2": ("Emissions|Sulfur", "Mt SO2/yr"),
    "NOx": ("Emissions|NOx", "Mt NO2/yr"),
    "CO": ("Emissions|CO", "Mt CO/yr"),
    "NMVOC": ("Emissions|VOC", "Mt VOC/yr"),
    "NH3": ("Emissions|NH3", "Mt NH3/yr"),
    "BC": ("Emissions|BC", "Mt BC/yr"),
    "OC": ("Emissions|OC", "Mt OC/yr"),
    "BMB_AEROS_BC": ("Emissions|BMB_AEROS_BC", "Mt BC/yr"),
    "BMB_AEROS_OC": ("Emissions|BMB_AEROS_OC", "Mt OC/yr"),
}
# F-gases and Montreal gases: openscm-units canonical is kt {species}/yr.
_F_GAS_AND_MONTREAL = (
    "CFC-11", "CFC-12", "CFC-113", "CFC-114", "CFC-115",
    "HCFC-22", "HCFC-141b", "HCFC-142b",
    "CCl4", "CH3CCl3", "CH3Br",
    "H-1211", "H-1301", "H-2402",
    "HFC125", "HFC134a", "HFC143a", "HFC227ea", "HFC23",
    "HFC245fa", "HFC32", "HFC4310mee",
    "CF4", "C2F6", "C6F14", "SF6",
)
for _gas in _F_GAS_AND_MONTREAL:
    iamc_name = cicero_comp_dict[_gas][0]
    iamc_var = f"Emissions|{iamc_name}"
    _OUTPUT_UNITS[_gas] = (iamc_var, f"kt {iamc_name}/yr")


def _cicero_unit_to_pint(cicero_unit: str, species: str) -> str:
    """Match :func:`ciceroscmpy2_adapter._cicero_unit_to_pint`.

    For halocarbons the pint registry uses the openscm-units short name
    (``Halon1211`` not ``H1211``), so we look the species name up in
    ``cicero_comp_dict`` when constructing the species-tagged unit.
    """
    if species == "N2O" and cicero_unit == "Tg_N":
        return "TgN2ON / yr"
    if "_" in cicero_unit:
        return cicero_unit.replace("_", "") + " / yr"
    # Species-tagged units (Gg / Tg / Mt + species name). Prefer the
    # openscm-units name from cicero_comp_dict when available so halons
    # map H-1211 -> Halon1211 etc. BMB_AEROS_* is a CICERO-internal
    # tag for biomass-burning aerosols; the unit follows the underlying
    # aerosol species (BC / OC) per the v1.1.x adapter's convention.
    iamc_name = cicero_comp_dict.get(species, [None])[0]
    if iamc_name is not None and not iamc_name.startswith("BMB_AEROS_"):
        comp_str = iamc_name
    else:
        comp_str = species.replace("-", "").replace("BMB_AEROS_", "")
    return f"{cicero_unit}{comp_str} / yr"


def _convert_series(
    values: pd.Series, cicero_unit: str, species: str, target_unit: str,
) -> pd.Series:
    """Convert a column from CICERO's per-year unit to the target IAMC unit."""
    cicero_pint = _cicero_unit_to_pint(cicero_unit, species)
    # NOx and NH3 need the openscm-units mass-N <-> mass-NO2/NH3 context.
    context = {"NOx": "NOx_conversions", "NH3": "NH3_conversions"}.get(species)
    if context is not None:
        with ureg.context(context):
            factor = (1.0 * ureg(cicero_pint)).to(target_unit).magnitude
    else:
        factor = (1.0 * ureg(cicero_pint)).to(target_unit).magnitude
    return values * factor


def _read_bundle_em(path: Path) -> tuple[list[str], list[str], pd.DataFrame]:
    """Read a CICERO bundle ``_em_*.txt`` file.

    Returns (components, units, df) where df is year-indexed with
    the two CO2 columns disambiguated as ``CO2_FF`` and ``CO2_lu``.
    """
    with path.open() as fh:
        components = [c.strip() for c in next(fh).rstrip("\n").split("\t")]
        units = [u.strip() for u in next(fh).rstrip("\n").split("\t")]

    df = pd.read_csv(
        path, delimiter="\t", index_col=0, skiprows=[1, 2, 3], dtype=float,
    )
    df.columns = [c.strip() for c in df.columns]
    df.index = df.index.astype(int)
    # The two CO2 columns are both labelled "CO2" in the file; pandas
    # disambiguates the second as "CO2.1" (or "CO2 .1" depending on
    # the strip pass).
    df.columns = [
        "CO2_FF" if c == "CO2"
        else "CO2_lu" if c in ("CO2.1", "CO2 .1")
        else c
        for c in df.columns
    ]
    # Drop the leading "Component" entry from each header list to align
    # with df's columns (which excludes the index column).
    return components[1:], units[1:], df


def _translate_one_scenario(
    bundle_path: Path, protocol_id: str,
) -> pd.DataFrame:
    components, units, df = _read_bundle_em(bundle_path)

    out_rows: list[dict] = []
    seen_co2: set[str] = set()
    for col_idx, cicero_species in enumerate(components):
        # Map the disambiguated CO2 col back to the cicero_comp_dict key.
        if cicero_species == "CO2":
            key = "CO2_lu" if "CO2" in seen_co2 else "CO2"
            df_col = "CO2_lu" if key == "CO2_lu" else "CO2_FF"
            seen_co2.add("CO2")
        else:
            key = cicero_species
            df_col = cicero_species

        if key not in _OUTPUT_UNITS:
            # Species we don't translate (e.g. H-1202 / Halon1212,
            # NF3, SO2F2, cC4F8, C3F8/C4F10/C5F12/C7F16/C8F18,
            # CH2Cl2 / CHCl3 / CH3Cl, HFC152a/236fa/365mfc, HCFC-123).
            # These are absent from cicero_comp_dict (the v1.1.x
            # adapter's species set) so the loader downstream wouldn't
            # consume them anyway.
            continue

        iamc_variable, target_unit = _OUTPUT_UNITS[key]
        try:
            converted = _convert_series(
                df[df_col], units[col_idx], cicero_species, target_unit,
            )
        except KeyError:
            print(
                f"  [skip] {protocol_id}: column {df_col!r} missing from "
                f"bundle file (header listed {cicero_species!r} at index "
                f"{col_idx})",
                file=sys.stderr,
            )
            continue
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"  [skip] {protocol_id}: {cicero_species} unit conversion "
                f"{units[col_idx]} -> {target_unit} failed: {exc}",
                file=sys.stderr,
            )
            continue

        row = {
            "Model": "CICERO-SCM-PY2 bundle (translated)",
            "Scenario": protocol_id,
            "Region": "World",
            "Variable": iamc_variable,
            "Unit": target_unit,
        }
        for year, value in converted.items():
            row[str(int(year))] = float(value)
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    p.add_argument("--gases-ep", default=DEFAULT_GASES_EP)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    args = p.parse_args(argv)

    if not args.bundle_dir.is_dir():
        p.error(f"--bundle-dir does not exist: {args.bundle_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for protocol_id in args.scenarios:
        stem = SCENARIOS.get(protocol_id, protocol_id)
        bundle_path = args.bundle_dir / f"{stem}_em_{args.gases_ep}"
        if not bundle_path.exists():
            p.error(f"Bundle file missing: {bundle_path}")
        print(f"Translating {protocol_id} from {bundle_path.name}")
        frames.append(_translate_one_scenario(bundle_path, protocol_id))

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.output, index=False)
    print(
        f"Wrote {len(combined)} rows ({combined['Scenario'].nunique()} "
        f"scenarios, {combined['Variable'].nunique()} variables) to "
        f"{args.output.relative_to(REPO_ROOT)}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
