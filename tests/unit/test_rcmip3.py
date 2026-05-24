"""Unit tests for the RCMIP3 scenario loader.

End-to-end behaviour against the real upstream CSVs lives in
``tests/integration/test_rcmip3_live_download.py`` (gated on a network
fixture); these unit tests exercise the registry, scenario filter, and
the source dispatch logic against synthetic in-memory CSVs so no
network is involved.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from openscm_runner.scenarios import (
    CONSTRAINT_TARGETS,
    available_scenarios,
    constraint_targets_dataframe,
    load_rcmip3_emissions,
)
from openscm_runner.scenarios.rcmip3 import (
    _SCEN7_MARKERS,
    _SOURCE_RCMIP3,
    _SOURCE_SCEN7,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_ssps_flat_family_and_scen7():
    scens = set(available_scenarios())
    # CMIP6 SSPs
    assert {
        "ssp119", "ssp126", "ssp245", "ssp370",
        "ssp434", "ssp460", "ssp534-over", "ssp585",
    } <= scens
    # Flat family - 3 base x 5 variants = 15
    for base in ("esm-flat7.5", "esm-flat10", "esm-flat20"):
        for suffix in ("", "-zec", "-cdr", "-nz", "-rev"):
            assert f"{base}{suffix}" in scens
    # scen7 markers
    assert set(_SCEN7_MARKERS) <= scens


def test_load_rcmip3_emissions_rejects_unknown_scenario():
    with pytest.raises(KeyError, match="Unknown RCMIP3 protocol scenarios"):
        load_rcmip3_emissions(["ssp245", "not-a-real-scenario"])


def test_load_rcmip3_emissions_rejects_empty_scenarios():
    with pytest.raises(ValueError, match="non-empty sequence"):
        load_rcmip3_emissions([])


def test_load_rcmip3_emissions_refuses_to_download_when_disabled(tmp_path):
    with pytest.raises(FileNotFoundError, match="not cached"):
        load_rcmip3_emissions(
            ["ssp245"], cache_dir=tmp_path, download_if_missing=False
        )


# ---------------------------------------------------------------------------
# Source dispatch and protocol-id renaming
# ---------------------------------------------------------------------------


def _write_rcmip3_like_csv(
    path: Path, rows: list[dict], year_cols: tuple[int, ...] = (2020, 2050, 2100)
) -> None:
    """Write a minimal CSV with the RCMIP3 v1.1.7 column shape."""
    base_cols = {
        "Model": "(unspecified)",
        "Scenario": "",
        "Region": "World",
        "Variable": "Emissions|CH4",
        "Unit": "Mt CH4/yr",
        "Mip_Era": "CMIP6",
        "Version": "1.1.7",
        "Activity_Id": "RCMIP",
        "Type": "Future",
        "Priority": 1,
    }
    full_rows = []
    for r in rows:
        row = {**base_cols, **r}
        for y in year_cols:
            row.setdefault(str(y), 1.0)
        full_rows.append(row)
    pd.DataFrame(full_rows).to_csv(path, index=False)


def _write_scen7_like_csv(
    path: Path, rows: list[dict], year_cols: tuple[int, ...] = (2020, 2050, 2100)
) -> None:
    """Write a minimal CSV with the scen7 column shape (extra workflow col)."""
    base_cols = {
        "model": "REMIND-MAgPIE 3.5-4.11",
        "scenario": "",
        "region": "World",
        "workflow": "for_scms",
        "variable": "Emissions|CH4",
        "unit": "Mt CH4/yr",
    }
    full_rows = []
    for r in rows:
        row = {**base_cols, **r}
        for y in year_cols:
            row.setdefault(str(y), 1.0)
        full_rows.append(row)
    pd.DataFrame(full_rows).to_csv(path, index=False)


def test_load_rcmip3_scenario_renames_back_to_protocol_id(tmp_path):
    # Pre-seed the cache so no download happens.
    _write_rcmip3_like_csv(
        tmp_path / _SOURCE_RCMIP3.cache_filename,
        rows=[
            {"Scenario": "ssp245", "Variable": "Emissions|CH4"},
            {"Scenario": "ssp370", "Variable": "Emissions|CH4"},
        ],
    )
    run = load_rcmip3_emissions(
        ["ssp245"], cache_dir=tmp_path, download_if_missing=False
    )
    # The cached scenario id is already the protocol id for the RCMIP3
    # source, so the rename is a no-op but exercises the codepath.
    assert sorted(run["scenario"].unique()) == ["ssp245"]


def test_load_rcmip3_scen7_renames_marker_iam_scenario_to_protocol_id(tmp_path):
    _write_scen7_like_csv(
        tmp_path / _SOURCE_SCEN7.cache_filename,
        rows=[
            # The protocol id ``scen7-VL`` resolves to this (scenario, model) pair.
            {
                "scenario": "SSP1 - Very Low Emissions",
                "model": "REMIND-MAgPIE 3.5-4.11",
                "variable": "Emissions|CH4",
            },
            # A distractor: same scenario name with a different IAM is
            # not the marker and must be filtered out.
            {
                "scenario": "SSP1 - Very Low Emissions",
                "model": "AIM 3.0",
                "variable": "Emissions|CH4",
            },
        ],
    )
    run = load_rcmip3_emissions(
        ["scen7-VL"], cache_dir=tmp_path, download_if_missing=False
    )
    assert sorted(run["scenario"].unique()) == ["scen7-VL"]
    assert sorted(run["model"].unique()) == ["REMIND-MAgPIE 3.5-4.11"]


def test_load_rcmip3_multiple_sources_concatenated(tmp_path):
    _write_rcmip3_like_csv(
        tmp_path / _SOURCE_RCMIP3.cache_filename,
        rows=[
            {"Scenario": "ssp245", "Variable": "Emissions|CH4"},
        ],
    )
    _write_scen7_like_csv(
        tmp_path / _SOURCE_SCEN7.cache_filename,
        rows=[
            {
                "scenario": "SSP3 - High Emissions",
                "model": "GCAM 8s",
                "variable": "Emissions|CH4",
            },
        ],
    )
    run = load_rcmip3_emissions(
        ["ssp245", "scen7-H"], cache_dir=tmp_path, download_if_missing=False
    )
    assert sorted(run["scenario"].unique()) == ["scen7-H", "ssp245"]


def test_load_rcmip3_harmonises_co2_sector_split(tmp_path):
    # Both sources use the new CMIP7-style sector name; the loader
    # should harmonise it via the same _CO2_SECTOR_RENAMES table that
    # the iamc loader already implements.
    _write_rcmip3_like_csv(
        tmp_path / _SOURCE_RCMIP3.cache_filename,
        rows=[
            {
                "Scenario": "ssp245",
                "Variable": "Emissions|CO2|Energy and Industrial Processes",
                "Unit": "Mt CO2/yr",
            },
            {
                "Scenario": "ssp245",
                "Variable": "Emissions|CO2|AFOLU",
                "Unit": "Mt CO2/yr",
            },
        ],
    )
    run = load_rcmip3_emissions(
        ["ssp245"], cache_dir=tmp_path, download_if_missing=False
    )
    assert set(run["variable"]) == {
        "Emissions|CO2|MAGICC Fossil and Industrial",
        "Emissions|CO2|MAGICC AFOLU",
    }


# ---------------------------------------------------------------------------
# Constraint targets table
# ---------------------------------------------------------------------------


def test_constraint_targets_table_covers_all_six_ar7_metrics():
    # Table 3 of the RCMIP3 paper has exactly these six rows.
    assert set(CONSTRAINT_TARGETS) == {
        "GMST_anomaly",
        "OHC_change",
        "CO2_concentration",
        "carbon_flux_to_oceans",
        "carbon_flux_to_land",
        "aerosol_ERF",
    }


def test_constraint_targets_dataframe_shape():
    df = constraint_targets_dataframe()
    assert len(df) == len(CONSTRAINT_TARGETS)
    expected = {
        "id", "variable", "unit", "source",
        "base_period_start", "base_period_end",
        "constraint_period_start", "constraint_period_end",
        "central", "lower", "upper",
    }
    assert expected <= set(df.columns)


def test_constraint_targets_central_within_uncertainty_bounds():
    # Guard against transcription errors from the paper table.
    for key, ct in CONSTRAINT_TARGETS.items():
        assert ct.lower <= ct.central <= ct.upper, (
            f"{key}: central {ct.central} outside [{ct.lower}, {ct.upper}]"
        )
