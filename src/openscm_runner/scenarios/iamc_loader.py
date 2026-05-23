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

# CO2 sector splits. Source-format sector names on the left, canonical
# MAGICC names on the right. ``AFOLU [NGHGI]`` is intentionally absent
# (see module docstring).
_CO2_SECTOR_RENAMES = {
    "Energy and Industrial Processes": "MAGICC Fossil and Industrial",
    "AFOLU": "MAGICC AFOLU",
}

# The set of canonical variable names openscm-runner adapters know how
# to drive. Anything outside this set is dropped during filtering.
CANONICAL_VARIABLES: frozenset[str] = frozenset(
    {
        "Emissions|CO2|MAGICC Fossil and Industrial",
        "Emissions|CO2|MAGICC AFOLU",
        "Emissions|CH4",
        "Emissions|N2O",
        "Emissions|HFC125",
        "Emissions|HFC134a",
        "Emissions|HFC143a",
        "Emissions|HFC227ea",
        "Emissions|HFC23",
        "Emissions|HFC245fa",
        "Emissions|HFC32",
        "Emissions|HFC4310mee",
        "Emissions|CF4",
        "Emissions|C2F6",
        "Emissions|C6F14",
        "Emissions|SF6",
        "Emissions|BC",
        "Emissions|OC",
        "Emissions|Sulfur",
        "Emissions|NOx",
        "Emissions|NH3",
        "Emissions|VOC",
        "Emissions|CO",
    }
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
    df = _harmonise_variables(df)
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


def _harmonise_variables(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["variable"] = out["variable"].map(_canonicalise_variable)
    return out


def _canonicalise_variable(name: str) -> str:
    if not isinstance(name, str) or not name.startswith("Emissions|"):
        return name
    body = name[len("Emissions|") :]

    # CO2 sector splits handled separately; never strip parents from these.
    if body.startswith("CO2|"):
        sector = body[len("CO2|") :]
        mapped = _CO2_SECTOR_RENAMES.get(sector)
        if mapped is not None:
            return f"Emissions|CO2|{mapped}"
        return name  # let the allowlist filter drop subcategory rows

    for parent in _PARENT_PATHS:
        if body.startswith(parent):
            body = body[len(parent) :]
            break

    body = _SPECIES_RENAMES.get(body, body)
    return f"Emissions|{body}"


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
