"""
Unit tests for the FaIRv2 concentrations translator.

These verify the CICERO-format ``{scen}_conc_{gases_ep}`` ->
FaIR-compatible DataFrame conversion that powers FaIRv2's
concentration-driven mode against Marit's RCMIP bundle.

The translator is small but load-bearing for the joint RCMIP
runner: CICERO and FaIR consume the SAME bundle conc files, so
the species name + unit mapping has to land exactly on FaIR's
species set.
"""
from __future__ import annotations

import pandas as pd
import pytest

from openscm_runner.adapters.fair2_adapter._concentrations_translator import (
    CICERO_TO_FAIR2_SPECIES,
    build_concentrations_df,
)


def _make_bundle_conc(
    tmp_path, scen_name="ssp245", gases_ep="gases_test.txt", years=(1750, 1751, 2024)
):
    """
    Write a minimal CICERO-format conc file with a handful of species.

    Layout matches Marit's RCMIP bundle: 4 header rows then year-
    indexed data, tab-delimited.
    """
    header_lines = [
        "Component \t CO2 \t CH4\tN2O\tCFC-11\tHFC125\tH-1211\tcC4F8",
        "Unit \t ppm\tppb\tppb\tppt\tppt\tppt\tppt",
        "Description \t historical \t historical \t historical \t historical \t historical \t historical \t historical",
        "Reference \t test \t test \t test \t test \t test \t test \t test",
    ]
    data_rows = []
    for i, yr in enumerate(years):
        row = [
            str(yr),
            f"{277 + i:.4f}",         # CO2
            f"{731.4 + i * 10:.4f}",  # CH4
            f"{273.9 + i:.4f}",       # N2O
            f"{i * 50.0:.4f}",        # CFC-11
            f"{i * 0.5:.4f}",         # HFC125
            f"{i * 0.1:.4f}",         # H-1211
            f"{i * 0.05:.4f}",        # cC4F8
        ]
        data_rows.append("\t".join(row))

    path = tmp_path / f"{scen_name}_conc_{gases_ep}"
    path.write_text("\n".join(header_lines + data_rows) + "\n")
    return path


def test_cicero_to_fair2_species_map_covers_known_renames():
    """
    Spot-check the renames we know matter for RCMIP bundles.
    """
    assert CICERO_TO_FAIR2_SPECIES["HFC125"] == "HFC-125"
    assert CICERO_TO_FAIR2_SPECIES["HFC4310mee"] == "HFC-4310mee"
    assert CICERO_TO_FAIR2_SPECIES["H-1211"] == "Halon-1211"
    assert CICERO_TO_FAIR2_SPECIES["H-2402"] == "Halon-2402"
    assert CICERO_TO_FAIR2_SPECIES["cC4F8"] == "c-C4F8"


def test_build_concentrations_df_renames_and_filters(tmp_path):
    """
    Species are translated through ``CICERO_TO_FAIR2_SPECIES`` and
    filtered to ``fair_species``; unmapped or out-of-set species
    drop out silently.
    """
    _make_bundle_conc(tmp_path)
    fair_species = {"CO2", "CH4", "N2O", "CFC-11", "HFC-125", "Halon-1211"}
    # `c-C4F8` is in our test file but excluded from fair_species ->
    # should be dropped from the output.
    df = build_concentrations_df(
        bundle_dir=str(tmp_path),
        gases_ep="gases_test.txt",
        scenario_names=["ssp245"],
        fair_species=fair_species,
        nystart=1750,
        nyend=2024,
    )
    variables = set(df["variable"].tolist())
    assert variables == fair_species
    # Verify ``cC4F8`` was dropped
    assert "c-C4F8" not in variables
    assert "cC4F8" not in variables


def test_build_concentrations_df_unit_passthrough(tmp_path):
    """
    Unit strings from the CICERO file's "Unit" header row are
    propagated unchanged to the output (FaIR's
    ``_concentration_unit_convert`` accepts ppm / ppb / ppt
    directly).
    """
    _make_bundle_conc(tmp_path)
    fair_species = {"CO2", "CH4", "CFC-11"}
    df = build_concentrations_df(
        bundle_dir=str(tmp_path),
        gases_ep="gases_test.txt",
        scenario_names=["ssp245"],
        fair_species=fair_species,
    )
    by_var = df.set_index("variable")["unit"].to_dict()
    assert by_var["CO2"] == "ppm"
    assert by_var["CH4"] == "ppb"
    assert by_var["CFC-11"] == "ppt"


def test_build_concentrations_df_values_match_file(tmp_path):
    """
    Numeric values round-trip through the parser unchanged.
    """
    years = (1750, 1751, 2024)
    _make_bundle_conc(tmp_path, years=years)
    fair_species = {"CO2"}
    df = build_concentrations_df(
        bundle_dir=str(tmp_path),
        gases_ep="gases_test.txt",
        scenario_names=["ssp245"],
        fair_species=fair_species,
        nystart=1750,
        nyend=2024,
    )
    co2_row = df[df["variable"] == "CO2"].iloc[0]
    # The fixture set CO2 = 277 + i for i = 0, 1, 2 across the three years.
    assert co2_row["1750"] == pytest.approx(277.0)
    assert co2_row["1751"] == pytest.approx(278.0)
    assert co2_row["2024"] == pytest.approx(279.0)


def test_build_concentrations_df_falls_back_to_historical(tmp_path):
    """
    Scenario-specific file missing -> use historical_conc_{gases_ep}
    instead (mirrors the CICEROSCMPY2 bundle fallback).
    """
    # Write only a historical_conc file, not a scenario-specific one.
    _make_bundle_conc(tmp_path, scen_name="historical")
    fair_species = {"CO2"}
    df = build_concentrations_df(
        bundle_dir=str(tmp_path),
        gases_ep="gases_test.txt",
        scenario_names=["my-novel-scenario"],
        fair_species=fair_species,
        nystart=1750,
        nyend=2024,
    )
    # Got data despite no my-novel-scenario_conc_… file existing.
    assert not df.empty
    assert df.iloc[0]["scenario"] == "my-novel-scenario"
    assert df.iloc[0]["variable"] == "CO2"


def test_build_concentrations_df_empty_when_no_files(tmp_path):
    """
    No bundle conc files at all -> empty DataFrame + warning.
    """
    fair_species = {"CO2"}
    df = build_concentrations_df(
        bundle_dir=str(tmp_path),
        gases_ep="gases_test.txt",
        scenario_names=["ssp245"],
        fair_species=fair_species,
    )
    assert df.empty
