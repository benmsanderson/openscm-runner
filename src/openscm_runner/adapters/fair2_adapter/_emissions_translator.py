"""
Convert ScmRun emissions into FaIR 2.x's expected DataFrame shape.

Pivots an openscm-runner :class:`scmdata.ScmRun` into the horizontal
DataFrame that :meth:`fair.FAIR.fill_from_pandas` expects, and splices
the user's scenario data on top of the calibration bundle's historical
emissions.

Shape expected by FaIR 2.x (lowercase column names; FaIR lowercases
them itself but we produce them lowercase up-front for clarity):

    scenario | variable | region | unit | 1750 | 1751 | ... | 2100
    ssp245   | CO2 FFI  | World  | ...  |  ... | ...  |     | ...

One row per (scenario, FaIR-species) combination. Time columns are
years (integers); FaIR interpolates to its own timepoints internally,
so the input grid does not have to align with FaIR's.

Splice rules:

- Bundle historical CSV is used as the baseline for every user scenario.
  The bundle's own "scenario" label (typically "historical") is
  relabelled to each user scenario in turn.
- For each (user_scenario, species) pair where the user provides data,
  the user's values overwrite the bundle's in the overlapping years
  (typically ~2015 onwards for IAMC scenarios). Years before the user's
  data start are taken from the bundle; years after are taken from the
  user.
- Species the user does not provide stay at bundle-historical values
  (and FaIR's defaults for any future years the bundle doesn't cover).
  Unmapped openscm-runner variables (anything not in the
  :data:`OPENSCM_TO_FAIR2_SPECIES` map) are logged at WARNING and
  ignored.

Unit conversion is left to FaIR 2.x (it has its own conversion tables
in :mod:`fair.io.fill_from`). We pass the openscm-runner unit string
through unchanged and trust FaIR to recognise it; FaIR raises a clear
``UnitParseError`` if it does not.

This module deliberately uses pandas directly rather than going through
scmdata.ScmRun helpers (which currently lean on the broken
``ScmRun.convert_unit`` -> groupby path on pandas 3.x; see openscm/scmdata#318
and benmsanderson/openscm-runner#6). Aligns with the long-term direction
of leaning less on scmdata.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)


# Suffix-match openscm-runner variable names to FaIR 2.x species names.
# Suffix-match lets callers use the full hierarchical names
# ("Emissions|CO2|MAGICC Fossil and Industrial") interchangeably with
# the leaf-only forms.
#
# v1 mapping covers the headline GHGs the AR7 priority workloads need.
# Extending this is the obvious follow-up: add the other species in
# fair.io.read_properties() (~64 in the AR6 default set).
OPENSCM_TO_FAIR2_SPECIES = {
    "|CO2|MAGICC Fossil and Industrial": "CO2 FFI",
    "|CO2|MAGICC AFOLU": "CO2 AFOLU",
    "|CH4": "CH4",
    "|N2O": "N2O",
}


def _openscm_to_fair2_species(variable: str) -> str | None:
    for suffix, species in OPENSCM_TO_FAIR2_SPECIES.items():
        if variable.endswith(suffix):
            return species
    return None


def _scmrun_to_fair2_rows(scmrun, scenario_names: Iterable[str]) -> pd.DataFrame:
    """
    Pivot the user's ScmRun into one row per (scenario, species).

    Unmapped variables are dropped with a warning. Years are integer
    column labels. ``unit`` is preserved verbatim (FaIR converts).
    """
    ts = scmrun.timeseries(time_axis="year")
    rows = []
    unmapped: set[str] = set()
    mapped_scenarios = set(scenario_names)

    # Use named index lookups so we do not depend on the order or
    # presence of meta columns (different callers carry different sets).
    index_names = list(ts.index.names)
    for index_values, series in ts.iterrows():
        meta = dict(zip(index_names, index_values))
        variable = meta.get("variable")
        species = _openscm_to_fair2_species(variable) if variable else None
        if species is None:
            if variable is not None:
                unmapped.add(variable)
            continue
        scenario = meta.get("scenario")
        if scenario not in mapped_scenarios:
            # Caller passed a subset of scenarios. Skip rows for scenarios
            # we are not running.
            continue
        years = [int(y) for y in series.index]
        row = {
            "scenario": scenario,
            "variable": species,
            "region": meta.get("region", "World"),
            "unit": meta.get("unit"),
        }
        for year, value in zip(years, series.values):
            row[year] = value
        rows.append(row)

    if unmapped:
        LOGGER.warning(
            "FaIRv2 adapter v1 emissions mapping covers %s. "
            "Unmapped variables (bundle defaults will be used instead): %s",
            sorted(OPENSCM_TO_FAIR2_SPECIES.values()),
            sorted(unmapped),
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _splice_bundle_with_user(
    bundle_df: pd.DataFrame,
    user_df: pd.DataFrame,
    scenario_names: Iterable[str],
) -> pd.DataFrame:
    """
    Take bundle historical as baseline and overwrite with user data.

    For each user scenario, the bundle's own scenario label is replaced
    by the user's scenario name, so each user scenario gets its own
    copy of the historical baseline. Where the user supplies values
    for the same (variable, year), the user's values take precedence.
    """
    # Normalise bundle column names; FaIR is case-insensitive on them.
    bundle_df = bundle_df.copy()
    bundle_df.columns = [
        c.lower() if isinstance(c, str) else c for c in bundle_df.columns
    ]
    bundle_year_cols = [c for c in bundle_df.columns if isinstance(c, (int, float))]
    bundle_year_cols += [
        c
        for c in bundle_df.columns
        if isinstance(c, str) and c.isdigit() and int(c) not in bundle_year_cols
    ]
    # FaIR's _check_csv expects year columns; if they came in as strings
    # (common when reading CSV without parse), coerce here so downstream
    # selection works on integer keys.
    rename_map = {
        c: int(c) for c in bundle_df.columns if isinstance(c, str) and c.isdigit()
    }
    if rename_map:
        bundle_df = bundle_df.rename(columns=rename_map)

    if not user_df.empty:
        user_df.columns = [
            c.lower() if isinstance(c, str) else c for c in user_df.columns
        ]

    spliced_rows = []
    for scenario in scenario_names:
        for _, bundle_row in bundle_df.iterrows():
            spliced = bundle_row.copy()
            spliced["scenario"] = scenario
            spliced_rows.append(spliced)

    if not spliced_rows:
        # No bundle to splice on top of. Caller has user data only;
        # pass it through unchanged so FaIR sees what the user supplied.
        return user_df.reset_index(drop=True)

    spliced_df = pd.DataFrame(spliced_rows).reset_index(drop=True)

    if user_df.empty:
        return spliced_df

    # Overlay user data on top, per (scenario, variable) row.
    user_year_cols = [c for c in user_df.columns if isinstance(c, int)]
    for _, user_row in user_df.iterrows():
        mask = (spliced_df["scenario"] == user_row["scenario"]) & (
            spliced_df["variable"] == user_row["variable"]
        )
        if not mask.any():
            # User provided a species the bundle didn't cover. Append
            # the row as-is so FaIR sees it (bundle baseline will be NaN
            # for the early years; FaIR's interpolator handles that with
            # bounds_error=False, leaving NaN, which the model may not
            # like; user can fill in their own historical if needed).
            spliced_df = pd.concat(
                [spliced_df, pd.DataFrame([user_row])], ignore_index=True
            )
            continue
        # Update the matching year columns
        for year in user_year_cols:
            value = user_row[year]
            if pd.notna(value):
                spliced_df.loc[mask, year] = value
        # Prefer the user's unit string (assumes the user knows what they
        # mean; FaIR will convert as needed).
        if "unit" in user_row and pd.notna(user_row["unit"]):
            spliced_df.loc[mask, "unit"] = user_row["unit"]

    return spliced_df


def build_emissions_df(
    scmrun,
    bundle_emissions_csv,
    scenario_names: Iterable[str],
) -> pd.DataFrame:
    """
    Build the FaIR 2.x-shaped emissions DataFrame for the given user
    scenarios, splicing the bundle's historical emissions with the
    user's scenario data.

    Parameters
    ----------
    scmrun : scmdata.ScmRun or None
        The user's emissions scenarios. When ``None`` or empty, the
        returned DataFrame is just the bundle's historical relabelled to
        each user scenario (useful for reproducing the bundle's own
        runs without further input).
    bundle_emissions_csv : pathlib.Path
        Path to the bundle's historical emissions CSV (typically
        ``historical_emissions_1750-2023_cmip7.csv``).
    scenario_names : iterable of str
        FaIR scenario labels to populate.

    Returns
    -------
    pandas.DataFrame
        Horizontal-form emissions DataFrame ready to pass to
        ``fair.FAIR.fill_from_pandas(mode="emissions", df=...)``.
    """
    bundle_df = (
        pd.read_csv(bundle_emissions_csv)
        if bundle_emissions_csv is not None
        else pd.DataFrame()
    )

    user_df = (
        _scmrun_to_fair2_rows(scmrun, scenario_names)
        if scmrun is not None
        else pd.DataFrame()
    )

    if bundle_df.empty and user_df.empty:
        return pd.DataFrame()

    return _splice_bundle_with_user(bundle_df, user_df, scenario_names)
