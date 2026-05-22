"""
Convert FaIR 2.x's xarray outputs into an openscm-runner ScmRun.

Variable name conventions are openscm-runner's (compatible with the
output of the FaIR 1.6 and MAGICC7 adapters where possible). FaIR 2.x
species names are hyphenated where the IAMC / openscm-runner
convention is not (e.g. FaIR "CFC-11" / openscm "CFC11"), so the
output mapping mirrors the inverse of
:data:`._emissions_translator.OPENSCM_TO_FAIR2_SPECIES` plus a few
special cases.

Supported output variables in this version:

- ``Surface Air Temperature Change`` (FaIR's surface layer)
- ``Surface Air Ocean Blended Temperature Change`` (scaled by
  ``GMST_TO_GSAT_SCALE``; documented scientific choice)
- ``Effective Radiative Forcing`` (FaIR's ``forcing_sum``)
- ``Effective Radiative Forcing|<species>`` for every FaIR 2.x species
  with a known openscm-runner name (see
  :data:`OUTPUT_LEAF_TO_FAIR2_SPECIES`)
- ``Effective Radiative Forcing|<category>`` aggregations
  (Anthropogenic, Greenhouse Gases, Kyoto Gases, F-Gases,
  CO2/CH4/N2O, Montreal Protocol Halogen Gases, Aerosols,
  Aerosols|Direct Effect, Aerosols|Indirect Effect, Ozone,
  CH4 Oxidation Stratospheric H2O, Contrails, Land-use Change,
  Black Carbon on Snow, Volcanic, Solar)
- ``Atmospheric Concentrations|<species>`` for every FaIR 2.x species
  with a known openscm-runner name
- ``Heat Content``, ``Heat Content|Ocean`` (FaIR's
  ``ocean_heat_content_change``, converted J -> ZJ)
- ``Heat Uptake``, ``Heat Uptake|Ocean``, ``Net Energy Imbalance``
  (FaIR's ``toa_imbalance``)
- ``Airborne Fraction`` (FaIR's ``airborne_fraction`` summed over
  CO2 FFI + CO2 AFOLU)

Unrecognised variables are logged at DEBUG and silently skipped, the
same forgiving pattern the FaIR 1.6 adapter follows.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scmdata import ScmRun

from ._emissions_translator import OPENSCM_TO_FAIR2_SPECIES

LOGGER = logging.getLogger(__name__)


# Scientific choice: factor used to convert surface temperature change
# (layer-0 of FaIR's temperature array, a model-internal "GSAT" proxy)
# to the GMST-blended convention reported in many observed datasets.
# 1/1.04 mirrors the value the FaIR 1.6 adapter has used (Cowtan et
# al. 2015 scaling, used in AR6 WGI Table 7.SM.1).
GMST_TO_GSAT_SCALE = 1.0 / 1.04

# Joules to zettajoules (10^21 J). FaIR 2.x exposes
# ocean_heat_content_change in J; openscm-runner uses ZJ.
J_TO_ZJ = 1e-21


# Output leaf names (the part after "Atmospheric Concentrations|" or
# "Effective Radiative Forcing|") -> FaIR 2.x species names.
#
# Derived from OPENSCM_TO_FAIR2_SPECIES by stripping the leading "|"
# and skipping the CO2 FFI/AFOLU split (output uses the calculated
# total "CO2"). Aliases collapse to one entry.
_BASE_OUTPUT_LEAVES = {
    suffix.lstrip("|"): fair_name
    for suffix, fair_name in OPENSCM_TO_FAIR2_SPECIES.items()
    if not suffix.startswith("|CO2|")
}
OUTPUT_LEAF_TO_FAIR2_SPECIES: dict[str, str] = {
    **_BASE_OUTPUT_LEAVES,
    # Output side uses the calculated total CO2, not the FFI/AFOLU split
    "CO2": "CO2",
}


# Per-species concentration units (FaIR 2.x's internal units, used as
# the output unit string). ppm for CO2, ppb for CH4 and N2O, ppt for
# everything else.
_CONCENTRATION_UNITS = {"CO2": "ppm", "CH4": "ppb", "N2O": "ppb"}


def _concentration_unit(species_name: str) -> str:
    return _CONCENTRATION_UNITS.get(species_name, "ppt")


# Forcing-aggregation recipes. Keyed by the openscm-runner output
# variable name; value is a callable that takes (forcing_da,
# properties_df, available_species) and returns a 1-D numpy array
# along the time axis (or None if the recipe can't be computed against
# the species set FaIR actually ran).
def _sum_forcing_over(
    forcing_da, sc_idx: int, member_offset: int, species_to_sum: list[str]
) -> np.ndarray:
    """Helper: sum FaIR's forcing array over a list of species names."""
    species_present = [s for s in species_to_sum if s in forcing_da["specie"].values]
    if not species_present:
        return None
    sliced = forcing_da.sel(specie=species_present).isel(
        scenario=sc_idx, config=member_offset
    )
    return sliced.sum(dim="specie").values


def _species_by_property(properties_df, predicate) -> list[str]:
    """Return the species names where ``predicate(row)`` is truthy."""
    return [
        name for name, row in properties_df.iterrows() if predicate(row)
    ]


def _build_forcing_aggregations(  # noqa: PLR0915
    forcing_da, sc_idx: int, member_offset: int, properties_df
) -> "dict[str, tuple[np.ndarray, str]]":
    """
    Build the value arrays for the supported forcing aggregations.

    Returns a dict keyed by openscm-runner variable name; values are
    ``(values, unit)`` tuples. Aggregations whose species list does
    not intersect FaIR's actual species are omitted, not zero-filled.
    """
    species_in_run = list(forcing_da["specie"].values)

    def sum_over(species_list: list[str]) -> np.ndarray:
        return _sum_forcing_over(
            forcing_da, sc_idx, member_offset, species_list
        )

    ghgs = _species_by_property(
        properties_df, lambda r: bool(r.get("greenhouse_gas", False))
    )
    f_gases = _species_by_property(
        properties_df, lambda r: r.get("type") == "f-gas"
    )
    montreal = _species_by_property(
        properties_df,
        lambda r: r.get("type") in ("cfc-11", "other halogen"),
    )

    # Kyoto basket = CO2 + CH4 + N2O + F-gases (the basket the Kyoto
    # Protocol covers; CFCs are governed by the Montreal Protocol).
    kyoto = [s for s in ("CO2", "CH4", "N2O") if s in species_in_run] + f_gases

    aero_indirect = [
        s
        for s in ("Aerosol-cloud interactions",)
        if s in species_in_run
    ]
    aero_direct = [
        s
        for s in ("Aerosol-radiation interactions",)
        if s in species_in_run
    ]

    out: dict[str, tuple[np.ndarray, str]] = {}
    forcing_unit = "W/m^2"

    def maybe(name: str, values):
        if values is None:
            return
        out[name] = (values, forcing_unit)

    # Anthropogenic = total forcing minus Solar minus Volcanic
    total = (
        forcing_da.isel(scenario=sc_idx, config=member_offset)
        .sum(dim="specie")
        .values
    )
    natural_species = [s for s in ("Solar", "Volcanic") if s in species_in_run]
    if natural_species:
        natural = sum_over(natural_species)
        anthro = total - natural if natural is not None else None
        maybe("Effective Radiative Forcing|Anthropogenic", anthro)
    else:
        maybe("Effective Radiative Forcing|Anthropogenic", total)

    maybe("Effective Radiative Forcing|Greenhouse Gases", sum_over(ghgs))
    maybe("Effective Radiative Forcing|F-Gases", sum_over(f_gases))
    maybe("Effective Radiative Forcing|Kyoto Gases", sum_over(kyoto))
    maybe(
        "Effective Radiative Forcing|CO2, CH4 and N2O",
        sum_over([s for s in ("CO2", "CH4", "N2O") if s in species_in_run]),
    )
    maybe(
        "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
        sum_over(montreal),
    )
    maybe(
        "Effective Radiative Forcing|Aerosols",
        sum_over(aero_direct + aero_indirect),
    )
    maybe(
        "Effective Radiative Forcing|Aerosols|Direct Effect", sum_over(aero_direct)
    )
    maybe(
        "Effective Radiative Forcing|Aerosols|Indirect Effect",
        sum_over(aero_indirect),
    )

    # Single-species pass-throughs that the FaIR 1.6 adapter exposes
    # under openscm-runner names that differ from the FaIR 2.x species
    # name. Map: openscm-runner name -> FaIR 2.x specie label.
    single_specie_aliases = {
        "Effective Radiative Forcing|Ozone": "Ozone",
        "Effective Radiative Forcing|CH4 Oxidation Stratospheric H2O": (
            "Stratospheric water vapour"
        ),
        "Effective Radiative Forcing|Contrails": "Contrails",
        "Effective Radiative Forcing|Land-use Change": "Land use",
        "Effective Radiative Forcing|Black Carbon on Snow": (
            "Light absorbing particles on snow and ice"
        ),
        "Effective Radiative Forcing|Volcanic": "Volcanic",
        "Effective Radiative Forcing|Solar": "Solar",
    }
    for openscm_name, fair_specie in single_specie_aliases.items():
        if fair_specie not in species_in_run:
            continue
        values = (
            forcing_da.sel(specie=fair_specie)
            .isel(scenario=sc_idx, config=member_offset)
            .values
        )
        maybe(openscm_name, values)

    return out


def _row(
    rows: list,
    scenario: str,
    variable: str,
    unit: str,
    run_id: int,
    values: np.ndarray,
):
    rows.append((scenario, "", "World", variable, unit, run_id, values))


def extract_outputs(  # noqa: PLR0913, PLR0912, PLR0915
    f,
    scenarios: list[str],
    members: pd.DataFrame,
    output_variables,
    run_id_offset: int,
    properties_df=None,
) -> ScmRun:
    """
    Convert FaIR 2.x's xarray output into an :class:`scmdata.ScmRun`.

    See module docstring for the supported variable set.
    """
    timebounds = np.asarray(f.timebounds, dtype=int)
    rows: list = []

    species_in_run = list(f.forcing["specie"].values)

    # Pre-compute aggregations once per (scenario, member) since the
    # recipes share intermediate sums.
    for sc_idx, scenario in enumerate(scenarios):
        for member_offset in range(len(members)):
            run_id = run_id_offset + member_offset

            aggregations = (
                _build_forcing_aggregations(
                    f.forcing, sc_idx, member_offset, properties_df
                )
                if properties_df is not None
                else {}
            )

            for variable in output_variables:
                # Scalars (no species suffix)
                if variable == "Surface Air Temperature Change":
                    values = (
                        f.temperature.isel(
                            scenario=sc_idx, config=member_offset, layer=0
                        )
                        .to_pandas()
                        .reindex(timebounds)
                    )
                    _row(rows, scenario, variable, "K", run_id, values)
                    continue
                if variable == "Surface Air Ocean Blended Temperature Change":
                    values = (
                        f.temperature.isel(
                            scenario=sc_idx, config=member_offset, layer=0
                        )
                        .to_pandas()
                        .reindex(timebounds)
                    )
                    _row(
                        rows,
                        scenario,
                        variable,
                        "K",
                        run_id,
                        values * GMST_TO_GSAT_SCALE,
                    )
                    continue
                if variable == "Effective Radiative Forcing":
                    values = (
                        f.forcing_sum.isel(scenario=sc_idx, config=member_offset)
                        .to_pandas()
                        .reindex(timebounds)
                    )
                    _row(rows, scenario, variable, "W/m^2", run_id, values)
                    continue
                if variable in ("Heat Content", "Heat Content|Ocean"):
                    values = (
                        f.ocean_heat_content_change.isel(
                            scenario=sc_idx, config=member_offset
                        )
                        .to_pandas()
                        .reindex(timebounds)
                    )
                    _row(
                        rows, scenario, variable, "ZJ", run_id, values * J_TO_ZJ
                    )
                    continue
                if variable in (
                    "Heat Uptake",
                    "Heat Uptake|Ocean",
                    "Net Energy Imbalance",
                ):
                    values = (
                        f.toa_imbalance.isel(scenario=sc_idx, config=member_offset)
                        .to_pandas()
                        .reindex(timebounds)
                    )
                    _row(rows, scenario, variable, "W/m^2", run_id, values)
                    continue
                if variable == "Airborne Fraction":
                    co2_components = [
                        s
                        for s in ("CO2 FFI", "CO2 AFOLU")
                        if s in species_in_run
                    ]
                    if not co2_components:
                        continue
                    values = (
                        f.airborne_fraction.sel(specie=co2_components)
                        .isel(scenario=sc_idx, config=member_offset)
                        .sum(dim="specie")
                        .to_pandas()
                        .reindex(timebounds)
                    )
                    _row(
                        rows,
                        scenario,
                        variable,
                        "dimensionless",
                        run_id,
                        values,
                    )
                    continue

                # Forcing aggregations
                if variable in aggregations:
                    values, unit = aggregations[variable]
                    series = pd.Series(values, index=timebounds)
                    _row(rows, scenario, variable, unit, run_id, series)
                    continue

                # Per-species patterns
                handled = False
                for prefix, extractor in (
                    ("Atmospheric Concentrations|", _atmos_conc),
                    ("Effective Radiative Forcing|", _erf_per_species),
                ):
                    if variable.startswith(prefix):
                        result = extractor(
                            f, variable[len(prefix) :], sc_idx, member_offset
                        )
                        if result is not None:
                            values, unit = result
                            series = pd.Series(values, index=timebounds)
                            _row(rows, scenario, variable, unit, run_id, series)
                            handled = True
                        break

                if not handled:
                    LOGGER.debug(
                        "FaIRv2 adapter does not emit %s; ignored", variable
                    )

    if not rows:
        return ScmRun(pd.DataFrame())
    return _build_scmrun(rows, timebounds)


def _atmos_conc(f, leaf: str, sc_idx: int, member_offset: int):
    species_name = OUTPUT_LEAF_TO_FAIR2_SPECIES.get(leaf)
    if species_name is None or species_name not in f.concentration["specie"].values:
        return None
    values = (
        f.concentration.sel(specie=species_name)
        .isel(scenario=sc_idx, config=member_offset)
        .values
    )
    return values, _concentration_unit(species_name)


def _erf_per_species(f, leaf: str, sc_idx: int, member_offset: int):
    species_name = OUTPUT_LEAF_TO_FAIR2_SPECIES.get(leaf)
    if species_name is None or species_name not in f.forcing["specie"].values:
        return None
    values = (
        f.forcing.sel(specie=species_name)
        .isel(scenario=sc_idx, config=member_offset)
        .values
    )
    return values, "W/m^2"


def _build_scmrun(rows, timebounds) -> ScmRun:
    """
    Stack per-(scenario, member, variable) Series into a single
    :class:`scmdata.ScmRun`.
    """
    data = np.vstack([np.asarray(row[6]) for row in rows])
    meta = pd.DataFrame(
        [row[:6] for row in rows],
        columns=["scenario", "model", "region", "variable", "unit", "run_id"],
    )
    df = pd.DataFrame(
        data, index=pd.MultiIndex.from_frame(meta), columns=timebounds
    )
    return ScmRun(df)
