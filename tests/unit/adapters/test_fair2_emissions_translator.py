"""
Unit tests for the FaIRv2 emissions translator.

These verify the ScmRun -> FaIR 2.x DataFrame conversion and the
bundle / user splice logic without touching FaIR itself. The
translator is the load-bearing piece for getting user scenarios into
FaIR via the native-calibration path.
"""
from __future__ import annotations

import pandas as pd
import pytest
import scmdata

from openscm_runner.adapters.fair2_adapter._emissions_translator import (
    OPENSCM_TO_FAIR2_SPECIES,
    _openscm_to_fair2_species,
    build_emissions_df,
)


def _bundle_csv(tmp_path):
    """Tiny synthetic bundle historical CSV covering CO2 FFI + CH4, 1990-2010."""
    bundle = pd.DataFrame(
        [
            {
                "scenario": "historical",
                "variable": "CO2 FFI",
                "region": "World",
                "unit": "Gt CO2/yr",
                1990: 22.0,
                2000: 25.0,
                2010: 33.0,
            },
            {
                "scenario": "historical",
                "variable": "CH4",
                "region": "World",
                "unit": "Mt CH4/yr",
                1990: 310.0,
                2000: 320.0,
                2010: 340.0,
            },
        ]
    )
    path = tmp_path / "historical_emissions.csv"
    bundle.to_csv(path, index=False)
    return path


def _user_scmrun():
    """ScmRun spanning two scenarios + CO2 FFI for 2010-2020."""
    df = pd.DataFrame(
        [[35.0, 40.0], [38.0, 45.0]],
        index=pd.MultiIndex.from_tuples(
            [
                (
                    "iam",
                    "ssp126",
                    "World",
                    "GtCO2/yr",
                    "Emissions|CO2|MAGICC Fossil and Industrial",
                    0,
                ),
                (
                    "iam",
                    "ssp370",
                    "World",
                    "GtCO2/yr",
                    "Emissions|CO2|MAGICC Fossil and Industrial",
                    0,
                ),
            ],
            names=[
                "model",
                "scenario",
                "region",
                "unit",
                "variable",
                "run_id",
            ],
        ),
        columns=[2010, 2020],
    )
    return scmdata.ScmRun(df)


def test_species_mapping_matches_openscm_names():
    # GHGs
    assert (
        _openscm_to_fair2_species("Emissions|CO2|MAGICC Fossil and Industrial")
        == "CO2 FFI"
    )
    assert _openscm_to_fair2_species("Emissions|CO2|MAGICC AFOLU") == "CO2 AFOLU"
    assert _openscm_to_fair2_species("Emissions|CH4") == "CH4"
    assert _openscm_to_fair2_species("Emissions|N2O") == "N2O"
    # Aerosols / SLCFs (with MAGICC adapter aliases)
    assert _openscm_to_fair2_species("Emissions|Sulfur") == "Sulfur"
    assert _openscm_to_fair2_species("Emissions|SOx") == "Sulfur"
    assert _openscm_to_fair2_species("Emissions|VOC") == "VOC"
    assert _openscm_to_fair2_species("Emissions|NMVOC") == "VOC"
    # FaIR 2.x uses hyphenated names; the openscm-runner / MAGICC names
    # do not.
    assert _openscm_to_fair2_species("Emissions|CFC11") == "CFC-11"
    assert _openscm_to_fair2_species("Emissions|HFC4310mee") == "HFC-4310mee"
    assert _openscm_to_fair2_species("Emissions|Halon1211") == "Halon-1211"
    assert _openscm_to_fair2_species("Emissions|cC4F8") == "c-C4F8"
    # Genuinely unmapped: a non-emissions variable, and a made-up
    # species that does not exist in FaIR 2.x's AR6 default set.
    assert _openscm_to_fair2_species("Atmospheric Concentrations|CO2") is None
    assert _openscm_to_fair2_species("Emissions|MadeUpSpecies") is None


def test_build_emissions_df_bundle_only_relabels_scenarios(tmp_path):
    """No user data -> one bundle copy per requested scenario name."""
    bundle_path = _bundle_csv(tmp_path)
    df = build_emissions_df(
        scmrun=None,
        bundle_emissions_csv=bundle_path,
        scenario_names=["ssp126", "ssp370"],
    )
    # 2 species x 2 scenarios = 4 rows
    assert len(df) == 4
    assert sorted(df["scenario"].unique()) == ["ssp126", "ssp370"]
    assert sorted(df["variable"].unique()) == ["CH4", "CO2 FFI"]
    # CO2 FFI 2000 value preserved from bundle
    co2_ssp126 = df[
        (df["scenario"] == "ssp126") & (df["variable"] == "CO2 FFI")
    ].iloc[0]
    assert co2_ssp126[2000] == pytest.approx(25.0)


def test_build_emissions_df_user_overrides_bundle_in_overlap(tmp_path):
    """User values take precedence over bundle in overlapping years."""
    bundle_path = _bundle_csv(tmp_path)
    user = _user_scmrun()
    df = build_emissions_df(
        scmrun=user,
        bundle_emissions_csv=bundle_path,
        scenario_names=["ssp126", "ssp370"],
    )

    co2_ssp126 = df[
        (df["scenario"] == "ssp126") & (df["variable"] == "CO2 FFI")
    ].iloc[0]
    # Pre-user year stays bundle: 1990 = 22.0
    assert co2_ssp126[1990] == pytest.approx(22.0)
    # Overlap year overridden by user: 2010 = 35.0 (bundle was 33.0)
    assert co2_ssp126[2010] == pytest.approx(35.0)
    # Beyond-bundle year present from user: 2020 = 40.0
    assert co2_ssp126[2020] == pytest.approx(40.0)

    # CH4 (no user override) untouched
    ch4_ssp126 = df[
        (df["scenario"] == "ssp126") & (df["variable"] == "CH4")
    ].iloc[0]
    assert ch4_ssp126[2010] == pytest.approx(340.0)


def test_build_emissions_df_uses_user_unit(tmp_path):
    bundle_path = _bundle_csv(tmp_path)
    user = _user_scmrun()
    df = build_emissions_df(
        scmrun=user,
        bundle_emissions_csv=bundle_path,
        scenario_names=["ssp126", "ssp370"],
    )
    co2_ssp126 = df[
        (df["scenario"] == "ssp126") & (df["variable"] == "CO2 FFI")
    ].iloc[0]
    # User's "GtCO2/yr" overwrites bundle's "Gt CO2/yr" in the spliced row.
    assert co2_ssp126["unit"] == "GtCO2/yr"


def test_build_emissions_df_drops_scenarios_not_in_run(tmp_path, caplog):
    """User scenarios not in scenario_names are silently skipped."""
    bundle_path = _bundle_csv(tmp_path)
    user = _user_scmrun()
    df = build_emissions_df(
        scmrun=user,
        bundle_emissions_csv=bundle_path,
        scenario_names=["ssp126"],  # only one scenario asked for
    )
    assert set(df["scenario"].unique()) == {"ssp126"}


def test_build_emissions_df_warns_on_unmapped_variables(tmp_path, caplog):
    bundle_path = _bundle_csv(tmp_path)
    # A made-up species not in the FaIR 2.x AR6 default set; the
    # translator should warn and skip rather than crash.
    df_user = pd.DataFrame(
        [[10.0, 11.0]],
        index=pd.MultiIndex.from_tuples(
            [("iam", "ssp245", "World", "Mt/yr", "Emissions|MadeUpSpecies", 0)],
            names=["model", "scenario", "region", "unit", "variable", "run_id"],
        ),
        columns=[2010, 2020],
    )
    user = scmdata.ScmRun(df_user)

    import logging

    with caplog.at_level(
        logging.WARNING,
        logger="openscm_runner.adapters.fair2_adapter._emissions_translator",
    ):
        df = build_emissions_df(
            scmrun=user,
            bundle_emissions_csv=bundle_path,
            scenario_names=["ssp245"],
        )

    assert "Unmapped variables" in caplog.text
    assert "Emissions|MadeUpSpecies" in caplog.text
    # Bundle still produces rows for ssp245 (just nothing user-side overrides)
    assert set(df["scenario"].unique()) == {"ssp245"}


def test_build_emissions_df_user_only_no_bundle():
    """When no bundle file is provided, user data passes through as-is."""
    user = _user_scmrun()
    df = build_emissions_df(
        scmrun=user,
        bundle_emissions_csv=None,
        scenario_names=["ssp126", "ssp370"],
    )
    assert len(df) == 2
    assert sorted(df["variable"].unique()) == ["CO2 FFI"]
    assert sorted(df["scenario"].unique()) == ["ssp126", "ssp370"]


def test_build_emissions_df_empty_inputs_returns_empty():
    df = build_emissions_df(
        scmrun=None, bundle_emissions_csv=None, scenario_names=["ssp245"]
    )
    assert df.empty


def test_openscm_to_fair2_species_covers_documented_map():
    """The exported map and the lookup helper agree."""
    for suffix in OPENSCM_TO_FAIR2_SPECIES:
        # The suffix should match a variable that ends with it
        sample = f"Emissions{suffix}"
        assert _openscm_to_fair2_species(sample) is not None
