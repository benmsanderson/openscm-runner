"""IAMC-format emissions scenario loader.

Reads a CSV or Excel file laid out in the standard IAMC long format
(``model, scenario, region, variable, unit, <year_1>, <year_2>, ...``)
and returns an :class:`scmdata.ScmRun` whose variable names match the
canonical openscm-runner convention used by the adapter layer.

Two source dialects are supported with the same code path:

* **RCMIP-style** CSV exports (parents include ``F-Gases|HFC|``,
  ``F-Gases|PFC|``, ``Montreal Gases|``).
* **Scenario Compass Initiative** xlsx releases (parents are shorter:
  ``HFC|``, ``PFC|``; CO2 uses ``Energy and Industrial Processes`` and
  ``AFOLU`` sector splits instead of MAGICC's naming).

The harmonisation is purely a relabel-and-filter; values and units are
passed through unchanged. Scenarios whose IAM did not report a given
species will simply have that species missing from the returned
ScmRun — the adapter's bundle merge / historical splice step is
responsible for filling such gaps.

Notes on a couple of deliberate choices:

* ``Emissions|CO2|AFOLU`` is kept; ``Emissions|CO2|AFOLU [NGHGI]`` is
  dropped. The plain variant is the physical land-atmosphere flux that
  the carbon cycle responds to; the ``[NGHGI]`` variant is an
  inventory-accounting construct and would mis-drive the SCM.
* SF6 is treated as a top-level species in all sources; we do not try
  to handle the (incorrect) ``F-Gases|PFC|SF6`` parent path that some
  exports use.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import scmdata

LOGGER = logging.getLogger(__name__)


# Parents stripped when present at the head of the variable body
# (i.e. after ``Emissions|``). Order matters: longest-prefix-first so
# ``F-Gases|HFC|`` matches before the bare ``F-Gases|``.
_PARENT_PATHS = (
    "F-Gases|HFC|",
    "F-Gases|PFC|",
    "F-Gases|CFC|",
    "F-Gases|",
    "Montreal Gases|",
    "HFC|",
    "PFC|",
    "CFC|",
)

# Per-species name fixups applied after parent stripping.
_SPECIES_RENAMES = {
    "HFC4310": "HFC4310mee",
    "HFC43-10": "HFC4310mee",
    "SOx": "Sulfur",
    "NMVOC": "VOC",
}

# Unit-string fixups applied to the ``unit`` column in lockstep with the
# species renames above. SCI in particular reports HFC4310mee as
# ``kt HFC43-10/yr``: the dash is not a valid pint symbol character so
# openscm-units rejects the string outright, and the downstream adapter
# converter then silently falls back to bundle defaults for that species.
# Rewriting to ``kt HFC4310mee/yr`` (the RCMIP convention) lets
# openscm-units parse the source unit; the downstream conversion to
# whatever the adapter bundle prefers (e.g. ``kt HFC4310/yr``) is then
# handled by openscm-units' molecule equivalence table.
#
# Order matters: HFC43-10 must be handled before HFC4310 (otherwise the
# bare-43-10 string never matches), and HFC4310 must only be rewritten
# when ``mee`` does not already follow.

# CO2 sector splits. Source-format sector names on the left, canonical
# MAGICC names on the right. ``AFOLU [NGHGI]`` is intentionally absent
# (see module docstring).
_CO2_SECTOR_RENAMES = {
    "Energy and Industrial Processes": "MAGICC Fossil and Industrial",
    "AFOLU": "MAGICC AFOLU",
}

# The set of canonical variable names openscm-runner adapters know how
# to drive. Anything outside this set is dropped during filtering.
# Emissions go through the per-species harmonisation in
# _canonicalise_variable; Atmospheric Concentrations follow the same
# parent-path stripping so any source file using
# ``Atmospheric Concentrations|F-Gases|HFC|HFC125`` ends up as the
# canonical ``Atmospheric Concentrations|HFC125``. Aerosols and most
# reactive species have no concentration entry by design (their
# protocol input is emissions-only).
_EMISSIONS_SPECIES = (
    "CO2|MAGICC Fossil and Industrial", "CO2|MAGICC AFOLU",
    "CH4", "N2O",
    "HFC125", "HFC134a", "HFC143a", "HFC227ea", "HFC23",
    "HFC245fa", "HFC32", "HFC4310mee",
    "CF4", "C2F6", "C6F14", "SF6",
    "BC", "OC", "Sulfur", "NOx", "NH3", "VOC", "CO",
)
_CONCENTRATION_SPECIES = (
    "CO2", "CH4", "N2O",
    "HFC125", "HFC134a", "HFC143a", "HFC227ea", "HFC23",
    "HFC245fa", "HFC32", "HFC4310mee",
    "HFC152a", "HFC236fa", "HFC365mfc",
    "CF4", "C2F6", "C3F8", "C4F10", "C5F12", "C6F14",
    "C7F16", "C8F18", "cC4F8", "SF6", "SO2F2", "NF3",
    "CFC11", "CFC12", "CFC113", "CFC114", "CFC115",
    "HCFC22", "HCFC141b", "HCFC142b",
    "CCl4", "CH3CCl3", "CH3Cl", "CH3Br", "CH2Cl2", "CHCl3",
    "Halon1202", "Halon1211", "Halon1301", "Halon2402",
)
CANONICAL_VARIABLES: frozenset[str] = frozenset(
    [f"Emissions|{s}" for s in _EMISSIONS_SPECIES]
    + [f"Atmospheric Concentrations|{s}" for s in _CONCENTRATION_SPECIES]
)

_REQUIRED_COLUMNS = frozenset({"model", "scenario", "region", "variable", "unit"})


def load_iamc(
    path: Union[str, Path],
    *,
    sheet: str = "data",
    scenarios: Optional[Sequence[str]] = None,
    region: str = "World",
    variables: Optional[Sequence[str]] = None,
) -> scmdata.ScmRun:
    """
    Load an IAMC-format scenario file and return a canonical ScmRun.

    Parameters
    ----------
    path
        Path to a ``.csv`` or ``.xlsx`` file in standard IAMC long format.
    sheet
        Worksheet name to read when ``path`` is an Excel file. Ignored
        for CSV. Defaults to ``"data"`` (the SCI convention).
    scenarios
        If given, restrict to scenarios in this collection. Otherwise
        all scenarios in the file are returned.
    region
        Region to keep. Defaults to ``"World"`` — adapters expect global
        timeseries.
    variables
        Override the canonical allowlist. Names should be supplied in
        post-harmonisation (canonical) form. If ``None`` (default) the
        module-level :data:`CANONICAL_VARIABLES` is used.

    Returns
    -------
    scmdata.ScmRun
        One timeseries per (model, scenario, variable) triple, with
        variable names harmonised to the canonical openscm-runner
        convention.

    Raises
    ------
    ValueError
        If the file extension is unrecognised, required IAMC columns
        are missing, or no rows survive filtering.
    """
    path = Path(path)
    df = _read_iamc(path, sheet=sheet)
    df = _harmonise(df)
    df = _filter(df, scenarios=scenarios, region=region, variables=variables)
    if df.empty:
        raise ValueError(
            f"No rows survived filtering of {path} (scenarios={scenarios!r}, "
            f"region={region!r}). Check the file contains the expected "
            f"region/scenario names and at least one of the canonical "
            f"variables after harmonisation."
        )
    return scmdata.ScmRun(df)


def _read_iamc(path: Path, *, sheet: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        raise ValueError(
            f"Unsupported file extension {suffix!r} for {path}; "
            f"expected .csv, .xlsx, or .xls."
        )

    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required IAMC columns: {sorted(missing)}. "
            f"Found columns: {sorted(df.columns)}"
        )
    return df


def _harmonise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["variable"] = out["variable"].map(_canonicalise_variable)
    out["unit"] = out["unit"].map(_canonicalise_unit)
    return out


def _canonicalise_unit(u: str) -> str:
    if not isinstance(u, str):
        return u
    if "HFC43-10" in u:
        u = u.replace("HFC43-10", "HFC4310mee")
    elif "HFC4310" in u and "HFC4310mee" not in u:
        u = u.replace("HFC4310", "HFC4310mee")
    if "SOx" in u:
        u = u.replace("SOx", "Sulfur")
    if "NMVOC" in u:
        u = u.replace("NMVOC", "VOC")
    return u


def _canonicalise_variable(name: str) -> str:
    if not isinstance(name, str):
        return name
    for prefix in ("Emissions|", "Atmospheric Concentrations|"):
        if name.startswith(prefix):
            return _canonicalise_with_prefix(name, prefix)
    return name


def _canonicalise_with_prefix(name: str, prefix: str) -> str:
    body = name[len(prefix):]

    # CO2 sector splits only apply to emissions; the concentrations
    # source publishes a single ``Atmospheric Concentrations|CO2`` row,
    # never split by sector, so the CO2| branch is emissions-only.
    if prefix == "Emissions|" and body.startswith("CO2|"):
        sector = body[len("CO2|"):]
        mapped = _CO2_SECTOR_RENAMES.get(sector)
        if mapped is not None:
            return f"Emissions|CO2|{mapped}"
        return name  # let the allowlist filter drop subcategory rows

    for parent in _PARENT_PATHS:
        if body.startswith(parent):
            body = body[len(parent):]
            break

    body = _SPECIES_RENAMES.get(body, body)
    return f"{prefix}{body}"


def _filter(
    df: pd.DataFrame,
    *,
    scenarios: Optional[Sequence[str]],
    region: str,
    variables: Optional[Sequence[str]],
) -> pd.DataFrame:
    out = df[df["region"] == region]
    if scenarios is not None:
        out = out[out["scenario"].isin(list(scenarios))]
    allowed = set(variables) if variables is not None else CANONICAL_VARIABLES
    out = out[out["variable"].isin(allowed)]
    return out
