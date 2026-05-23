"""
Unit tests for the FaIRv2 output extractor.

These tests fabricate a FAIR-shaped object: xarray DataArrays with the
right dim names and a couple of synthetic species, plus a
``properties_df`` mirroring the structure FaIR builds from
:func:`fair.io.read_properties`. Hand-rolled rather than running a
real FaIR ensemble because building a numerically valid FAIR run
without a calibration bundle requires filling in
``climate_configs`` (FAIR rejects NaN there). Mocking the xarray
shape covers exactly what the extractor consumes.

End-to-end runs against a real calibration bundle are covered by
``tests/integration/test_fair2.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openscm_runner.adapters.fair2_adapter._output_extractor import (
    GMST_TO_GSAT_SCALE,
    J_TO_ZJ,
    OUTPUT_LEAF_TO_FAIR2_SPECIES,
    _concentration_unit,
    extract_outputs,
)


# A small but realistic species set covering the categories the
# extractor's aggregations care about (GHGs, aerosols, natural).
_SPECIES = [
    "CO2 FFI",
    "CO2 AFOLU",
    "CO2",
    "CH4",
    "N2O",
    "CFC-11",
    "HFC-23",
    "Sulfur",
    "Aerosol-radiation interactions",
    "Aerosol-cloud interactions",
    "Ozone",
    "Solar",
    "Volcanic",
]


def _properties_df():
    """Mimic the columns and rows FaIR.properties_df exposes."""
    rows = []
    for s in _SPECIES:
        row = {"name": s}
        row["greenhouse_gas"] = s in (
            "CO2",
            "CH4",
            "N2O",
            "CFC-11",
            "HFC-23",
        )
        if s == "CFC-11":
            row["type"] = "cfc-11"
        elif s == "HFC-23":
            row["type"] = "f-gas"
        elif s == "Solar":
            row["type"] = "solar"
        elif s == "Volcanic":
            row["type"] = "volcanic"
        elif s == "Aerosol-radiation interactions":
            row["type"] = "ari"
        elif s == "Aerosol-cloud interactions":
            row["type"] = "aci"
        elif s == "Ozone":
            row["type"] = "ozone"
        else:
            row["type"] = "generic"
        rows.append(row)
    return pd.DataFrame(rows).set_index("name")


@dataclass
class _FakeFair:
    """Minimal FAIR-shaped object: just the xarrays the extractor reads."""

    timebounds: np.ndarray
    temperature: xr.DataArray
    forcing: xr.DataArray
    forcing_sum: xr.DataArray
    concentration: xr.DataArray
    ocean_heat_content_change: xr.DataArray
    toa_imbalance: xr.DataArray
    airborne_fraction: xr.DataArray
    properties_df: pd.DataFrame


@pytest.fixture()
def fake_fair():
    timebounds = np.arange(2000, 2011)  # 11 points
    n_t = len(timebounds)
    n_scen = 1
    n_cfg = 2  # two ensemble members
    scenarios = ["ssp245"]

    # Forcing per species: each species gets a constant value distinct
    # across species so the aggregations are testable.
    species_forcing_values = {
        "CO2 FFI": 0.0,
        "CO2 AFOLU": 0.0,
        "CO2": 2.0,
        "CH4": 0.5,
        "N2O": 0.2,
        "CFC-11": 0.1,
        "HFC-23": 0.01,
        "Sulfur": -0.5,
        "Aerosol-radiation interactions": -0.4,
        "Aerosol-cloud interactions": -0.6,
        "Ozone": 0.4,
        "Solar": 0.05,
        "Volcanic": -0.1,
    }
    forcing_data = np.zeros((n_t, n_scen, n_cfg, len(_SPECIES)))
    for i, s in enumerate(_SPECIES):
        forcing_data[:, :, :, i] = species_forcing_values[s]
    forcing = xr.DataArray(
        forcing_data,
        coords={
            "timebounds": timebounds,
            "scenario": scenarios,
            "config": [f"c{i}" for i in range(n_cfg)],
            "specie": _SPECIES,
        },
        dims=("timebounds", "scenario", "config", "specie"),
    )

    forcing_sum = forcing.sum(dim="specie")

    # Concentration: only the species openscm-runner asks about
    # (CO2, CH4, N2O, CFC-11, HFC-23). Others can be NaN since we
    # never request them.
    concentration_data = np.full(forcing_data.shape, np.nan)
    conc_levels = {"CO2": 400.0, "CH4": 1900.0, "N2O": 330.0, "CFC-11": 230.0, "HFC-23": 28.0}
    for s, level in conc_levels.items():
        i = _SPECIES.index(s)
        concentration_data[:, :, :, i] = level
    concentration = xr.DataArray(
        concentration_data,
        coords=forcing.coords,
        dims=forcing.dims,
    )

    # Temperature: 3 layers; surface is layer 0
    temperature_data = np.zeros((n_t, n_scen, n_cfg, 3))
    temperature_data[:, :, :, 0] = 1.0  # surface
    temperature_data[:, :, :, 1] = 0.5  # ocean shallow
    temperature_data[:, :, :, 2] = 0.1  # ocean deep
    temperature = xr.DataArray(
        temperature_data,
        coords={
            "timebounds": timebounds,
            "scenario": scenarios,
            "config": [f"c{i}" for i in range(n_cfg)],
            "layer": [0, 1, 2],
        },
        dims=("timebounds", "scenario", "config", "layer"),
    )

    # 1 ZJ = 1e21 J -> ohc set to 5e21 so converted output is 5 ZJ.
    ohc = xr.DataArray(
        np.full((n_t, n_scen, n_cfg), 5e21),
        coords=forcing_sum.coords,
        dims=forcing_sum.dims,
    )

    toa = xr.DataArray(
        np.full((n_t, n_scen, n_cfg), 0.7),
        coords=forcing_sum.coords,
        dims=forcing_sum.dims,
    )

    # Airborne fraction: only CO2 FFI / AFOLU matter for the
    # extractor's "Airborne Fraction" aggregation.
    af = np.zeros((n_t, n_scen, n_cfg, len(_SPECIES)))
    af[:, :, :, _SPECIES.index("CO2 FFI")] = 0.4
    af[:, :, :, _SPECIES.index("CO2 AFOLU")] = 0.05
    airborne_fraction = xr.DataArray(
        af, coords=forcing.coords, dims=forcing.dims
    )

    return _FakeFair(
        timebounds=timebounds,
        temperature=temperature,
        forcing=forcing,
        forcing_sum=forcing_sum,
        concentration=concentration,
        ocean_heat_content_change=ohc,
        toa_imbalance=toa,
        airborne_fraction=airborne_fraction,
        properties_df=_properties_df(),
    )


def _two_members():
    return pd.DataFrame(index=[0, 1])


def test_extract_surface_temperature(fake_fair):
    res = extract_outputs(
        fake_fair,
        scenarios=["ssp245"],
        members=_two_members(),
        output_variables=("Surface Air Temperature Change",),
        run_id_offset=0,
        properties_df=fake_fair.properties_df,
    )
    assert set(res["variable"]) == {"Surface Air Temperature Change"}
    assert set(res["unit"]) == {"K"}
    assert set(res["run_id"]) == {0, 1}
    # Layer 0 of our fake temperature is constant 1.0
    np.testing.assert_allclose(res.values, 1.0)


def test_extract_blended_temperature_applies_gmst_scale(fake_fair):
    res = extract_outputs(
        fake_fair,
        scenarios=["ssp245"],
        members=_two_members(),
        output_variables=("Surface Air Ocean Blended Temperature Change",),
        run_id_offset=0,
        properties_df=fake_fair.properties_df,
    )
    np.testing.assert_allclose(res.values, 1.0 * GMST_TO_GSAT_SCALE)


def test_extract_total_forcing(fake_fair):
    res = extract_outputs(
        fake_fair,
        scenarios=["ssp245"],
        members=_two_members(),
        output_variables=("Effective Radiative Forcing",),
        run_id_offset=0,
        properties_df=fake_fair.properties_df,
    )
    # Sum of constant per-species values defined in the fixture
    expected = (
        0.0 + 0.0 + 2.0 + 0.5 + 0.2 + 0.1 + 0.01 - 0.5 - 0.4 - 0.6 + 0.4
        + 0.05 - 0.1
    )
    np.testing.assert_allclose(res.values, expected)


def test_extract_per_species_concentration(fake_fair):
    res = extract_outputs(
        fake_fair,
        scenarios=["ssp245"],
        members=_two_members(),
        output_variables=(
            "Atmospheric Concentrations|CO2",
            "Atmospheric Concentrations|CH4",
            "Atmospheric Concentrations|CFC11",
            "Atmospheric Concentrations|HFC23",
        ),
        run_id_offset=0,
        properties_df=fake_fair.properties_df,
    )
    # Hyphenated FaIR names -> openscm-runner non-hyphenated leaves
    assert set(res["variable"]) == {
        "Atmospheric Concentrations|CO2",
        "Atmospheric Concentrations|CH4",
        "Atmospheric Concentrations|CFC11",
        "Atmospheric Concentrations|HFC23",
    }
    assert res.filter(variable="Atmospheric Concentrations|CO2").values[0, 0] == 400.0
    assert res.filter(variable="Atmospheric Concentrations|CH4").values[0, 0] == 1900.0
    # Units per species
    assert (
        res.filter(variable="Atmospheric Concentrations|CO2").get_unique_meta(
            "unit", True
        )
        == "ppm"
    )
    assert (
        res.filter(variable="Atmospheric Concentrations|CH4").get_unique_meta(
            "unit", True
        )
        == "ppb"
    )
    assert (
        res.filter(variable="Atmospheric Concentrations|HFC23").get_unique_meta(
            "unit", True
        )
        == "ppt"
    )


def test_extract_per_species_forcing(fake_fair):
    res = extract_outputs(
        fake_fair,
        scenarios=["ssp245"],
        members=_two_members(),
        output_variables=(
            "Effective Radiative Forcing|CO2",
            "Effective Radiative Forcing|CFC11",
        ),
        run_id_offset=0,
        properties_df=fake_fair.properties_df,
    )
    co2 = res.filter(variable="Effective Radiative Forcing|CO2").values
    cfc = res.filter(variable="Effective Radiative Forcing|CFC11").values
    np.testing.assert_allclose(co2, 2.0)
    np.testing.assert_allclose(cfc, 0.1)


def test_extract_forcing_aggregations(fake_fair):
    res = extract_outputs(
        fake_fair,
        scenarios=["ssp245"],
        members=_two_members(),
        output_variables=(
            "Effective Radiative Forcing|Greenhouse Gases",
            "Effective Radiative Forcing|F-Gases",
            "Effective Radiative Forcing|Kyoto Gases",
            "Effective Radiative Forcing|CO2, CH4 and N2O",
            "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
            "Effective Radiative Forcing|Aerosols",
            "Effective Radiative Forcing|Aerosols|Direct Effect",
            "Effective Radiative Forcing|Aerosols|Indirect Effect",
            "Effective Radiative Forcing|Ozone",
            "Effective Radiative Forcing|Solar",
            "Effective Radiative Forcing|Volcanic",
            "Effective Radiative Forcing|Anthropogenic",
        ),
        run_id_offset=0,
        properties_df=fake_fair.properties_df,
    )

    def val(var):
        return res.filter(variable=var).values[0, 0]

    # GHGs marked greenhouse_gas=True in the fixture: CO2, CH4, N2O, CFC-11, HFC-23
    np.testing.assert_allclose(
        val("Effective Radiative Forcing|Greenhouse Gases"),
        2.0 + 0.5 + 0.2 + 0.1 + 0.01,
    )
    # F-gases: HFC-23 only in the fixture
    np.testing.assert_allclose(
        val("Effective Radiative Forcing|F-Gases"), 0.01
    )
    # Kyoto basket = CO2 + CH4 + N2O + F-gases (CFC-11 is Montreal)
    np.testing.assert_allclose(
        val("Effective Radiative Forcing|Kyoto Gases"),
        2.0 + 0.5 + 0.2 + 0.01,
    )
    np.testing.assert_allclose(
        val("Effective Radiative Forcing|CO2, CH4 and N2O"),
        2.0 + 0.5 + 0.2,
    )
    # Montreal: CFC-11 (type="cfc-11" / "other halogen")
    np.testing.assert_allclose(
        val("Effective Radiative Forcing|Montreal Protocol Halogen Gases"),
        0.1,
    )
    # Aerosols = ari + aci
    np.testing.assert_allclose(
        val("Effective Radiative Forcing|Aerosols"), -0.4 + -0.6
    )
    np.testing.assert_allclose(
        val("Effective Radiative Forcing|Aerosols|Direct Effect"), -0.4
    )
    np.testing.assert_allclose(
        val("Effective Radiative Forcing|Aerosols|Indirect Effect"), -0.6
    )
    np.testing.assert_allclose(val("Effective Radiative Forcing|Ozone"), 0.4)
    np.testing.assert_allclose(val("Effective Radiative Forcing|Solar"), 0.05)
    np.testing.assert_allclose(val("Effective Radiative Forcing|Volcanic"), -0.1)
    # Anthropogenic = total - solar - volcanic
    total = (
        0.0 + 0.0 + 2.0 + 0.5 + 0.2 + 0.1 + 0.01 - 0.5 - 0.4 - 0.6 + 0.4
        + 0.05 - 0.1
    )
    np.testing.assert_allclose(
        val("Effective Radiative Forcing|Anthropogenic"),
        total - 0.05 - (-0.1),
    )


def test_extract_heat_content_converts_to_zj(fake_fair):
    res = extract_outputs(
        fake_fair,
        scenarios=["ssp245"],
        members=_two_members(),
        output_variables=("Heat Content",),
        run_id_offset=0,
        properties_df=fake_fair.properties_df,
    )
    # Fixture sets ocean_heat_content_change = 5e21 J; expected 5 ZJ
    assert res.get_unique_meta("unit", True) == "ZJ"
    np.testing.assert_allclose(res.values, 5e21 * J_TO_ZJ)


def test_extract_heat_uptake(fake_fair):
    from openscm_runner.adapters.fair2_adapter._output_extractor import (
        _TOA_W_PER_M2_TO_ZJ_PER_YR,
    )

    res = extract_outputs(
        fake_fair,
        scenarios=["ssp245"],
        members=_two_members(),
        output_variables=("Heat Uptake", "Net Energy Imbalance"),
        run_id_offset=0,
        properties_df=fake_fair.properties_df,
    )
    # Heat Uptake is reported in ZJ/yr (RCMIP convention) -
    # toa_imbalance W/m^2 * Earth surface * seconds/yr / 1e21.
    # Net Energy Imbalance keeps the raw W/m^2.
    uptake = res.filter(variable="Heat Uptake")
    nei = res.filter(variable="Net Energy Imbalance")
    np.testing.assert_allclose(
        uptake.values, 0.7 * _TOA_W_PER_M2_TO_ZJ_PER_YR
    )
    assert uptake.get_unique_meta("unit", True) == "ZJ/yr"
    np.testing.assert_allclose(nei.values, 0.7)
    assert nei.get_unique_meta("unit", True) == "W/m^2"


def test_extract_airborne_fraction_sums_co2_components(fake_fair):
    res = extract_outputs(
        fake_fair,
        scenarios=["ssp245"],
        members=_two_members(),
        output_variables=("Airborne Fraction",),
        run_id_offset=0,
        properties_df=fake_fair.properties_df,
    )
    # 0.4 (FFI) + 0.05 (AFOLU) = 0.45
    np.testing.assert_allclose(res.values, 0.45)
    assert res.get_unique_meta("unit", True) == "dimensionless"


def test_extract_unknown_variable_silently_skipped(fake_fair, caplog):
    import logging

    with caplog.at_level(
        logging.DEBUG,
        logger="openscm_runner.adapters.fair2_adapter._output_extractor",
    ):
        res = extract_outputs(
            fake_fair,
            scenarios=["ssp245"],
            members=_two_members(),
            output_variables=(
                "Surface Air Temperature Change",
                "Some Completely Made Up Variable",
            ),
            run_id_offset=0,
            properties_df=fake_fair.properties_df,
        )
    # Known variable still emitted
    assert set(res["variable"]) == {"Surface Air Temperature Change"}
    # Unknown one logged at DEBUG (forgiving, matches FaIR 1.6 adapter pattern)
    assert "Some Completely Made Up Variable" in caplog.text


def test_extract_run_id_offset_applied(fake_fair):
    res = extract_outputs(
        fake_fair,
        scenarios=["ssp245"],
        members=_two_members(),
        output_variables=("Surface Air Temperature Change",),
        run_id_offset=100,
        properties_df=fake_fair.properties_df,
    )
    assert set(res["run_id"]) == {100, 101}


def test_output_leaf_to_fair2_species_covers_no_co2_split():
    # The output map should NOT include CO2 FFI / AFOLU as openscm leaves;
    # those are emissions-side only (input).
    assert "CO2 FFI" not in OUTPUT_LEAF_TO_FAIR2_SPECIES
    assert "CO2 AFOLU" not in OUTPUT_LEAF_TO_FAIR2_SPECIES
    # But the calculated total CO2 IS available as an output species
    assert OUTPUT_LEAF_TO_FAIR2_SPECIES["CO2"] == "CO2"
    # Hyphenation differences round-trip correctly
    assert OUTPUT_LEAF_TO_FAIR2_SPECIES["CFC11"] == "CFC-11"
    assert OUTPUT_LEAF_TO_FAIR2_SPECIES["HFC4310mee"] == "HFC-4310mee"


def test_concentration_unit_defaults_to_ppt():
    assert _concentration_unit("CO2") == "ppm"
    assert _concentration_unit("CH4") == "ppb"
    assert _concentration_unit("N2O") == "ppb"
    assert _concentration_unit("CFC-11") == "ppt"
    assert _concentration_unit("HFC-23") == "ppt"


def test_extract_resolves_hierarchical_rcmip_paths(fake_fair):
    """
    RCMIP uses hierarchical variable paths like
    ``Atmospheric Concentrations|F-Gases|HFC|HFC23``. The extractor
    must strip to the leaf segment (``HFC23``) and look it up in
    ``OUTPUT_LEAF_TO_FAIR2_SPECIES`` the same way it would for the
    flat ``Atmospheric Concentrations|HFC23`` form.
    """
    res = extract_outputs(
        fake_fair,
        scenarios=["ssp245"],
        members=_two_members(),
        output_variables=(
            "Atmospheric Concentrations|F-Gases|HFC|HFC23",
            "Effective Radiative Forcing|Anthropogenic|F-Gases|HFC|HFC23",
            "Atmospheric Concentrations|Montreal Gases|CFC|CFC11",
        ),
        run_id_offset=0,
        properties_df=fake_fair.properties_df,
    )
    assert set(res["variable"]) == {
        "Atmospheric Concentrations|F-Gases|HFC|HFC23",
        "Effective Radiative Forcing|Anthropogenic|F-Gases|HFC|HFC23",
        "Atmospheric Concentrations|Montreal Gases|CFC|CFC11",
    }
    # Resolves to the same species data as the flat-form lookup
    deep = res.filter(
        variable="Atmospheric Concentrations|F-Gases|HFC|HFC23"
    ).values[0, 0]
    assert deep == 28.0  # HFC-23 conc level from fixture
    erf = res.filter(
        variable="Effective Radiative Forcing|Anthropogenic|F-Gases|HFC|HFC23"
    ).values[0, 0]
    assert erf == 0.01  # HFC-23 ERF level from fixture
