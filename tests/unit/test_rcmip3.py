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
    available_concentration_scenarios,
    available_scenarios,
    constraint_targets_dataframe,
    load_rcmip3_concentrations,
    load_rcmip3_emissions,
)
from openscm_runner.scenarios.rcmip3 import (
    _SCEN7_MARKERS,
    _SOURCE_RCMIP3,
    _SOURCE_RCMIP3_CONC,
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
# Protocol metadata (mode / natural_forcing / land_use_forcing)
# ---------------------------------------------------------------------------


_PROTOCOL_META_COLS = (
    "protocol_mode",
    "protocol_natural_forcing",
    "protocol_land_use_forcing",
)


def test_protocol_metadata_registry_covers_all_emissions_scenarios():
    # Step 3 promise: every registered emissions scenario carries the
    # three protocol metadata fields with one of the six valid profiles.
    from openscm_runner.scenarios.rcmip3 import _REGISTRY
    valid_modes = {"CD", "ED-CO2-only", "ED-all-GHG"}
    valid_natural = {"on", "off"}
    valid_lu = {"historical", "constant_zero"}
    for pid, spec in _REGISTRY.items():
        assert spec.protocol_mode in valid_modes, f"{pid}: {spec.protocol_mode}"
        assert spec.protocol_natural_forcing in valid_natural, \
            f"{pid}: {spec.protocol_natural_forcing}"
        assert spec.protocol_land_use_forcing in valid_lu, \
            f"{pid}: {spec.protocol_land_use_forcing}"


def test_protocol_metadata_spot_checks():
    # Representative members from each profile bucket. Guards against
    # accidental reclassification in future edits.
    from openscm_runner.scenarios.rcmip3 import _REGISTRY
    cases = {
        # CD real-world
        "ssp245": ("CD", "on", "historical"),
        "historical": ("CD", "on", "historical"),
        "hist-aer": ("CD", "on", "historical"),
        "scen7-H": ("CD", "on", "historical"),
        "scen7-HC": ("CD", "on", "historical"),
        # CD idealised
        "1pctCO2": ("CD", "off", "constant_zero"),
        "abrupt-4xCO2": ("CD", "off", "constant_zero"),
        "piControl": ("CD", "off", "constant_zero"),
        # ED CO2-only real-world
        "esm-ssp245": ("ED-CO2-only", "on", "historical"),
        "esm-scen7-H": ("ED-CO2-only", "on", "historical"),
        "esm-hist": ("ED-CO2-only", "on", "historical"),
        # ED CO2-only idealised
        "esm-flat10": ("ED-CO2-only", "off", "constant_zero"),
        "esm-bell-1000PgC": ("ED-CO2-only", "off", "constant_zero"),
        "esm-pi-CO2pulse": ("ED-CO2-only", "off", "constant_zero"),
        "esm-piControl": ("ED-CO2-only", "off", "constant_zero"),
        "esm-1pct-brch-1000PgC": ("ED-CO2-only", "off", "constant_zero"),
        # ED all-GHG real-world
        "esm-allGHG-ssp245": ("ED-all-GHG", "on", "historical"),
        "esm-allGHG-scen7-H": ("ED-all-GHG", "on", "historical"),
        "esm-allGHG-scen7-H-CH4L": ("ED-all-GHG", "on", "historical"),
        "esm-allGHG-ssp370-lowCH4": ("ED-all-GHG", "on", "historical"),
        "esm-allGHG-hist": ("ED-all-GHG", "on", "historical"),
        # ED all-GHG idealised (PI control of an ED-all-GHG run)
        "esm-allGHG-piControl": ("ED-all-GHG", "off", "constant_zero"),
    }
    for pid, (mode, natural, lu) in cases.items():
        spec = _REGISTRY[pid]
        actual = (spec.protocol_mode, spec.protocol_natural_forcing,
                  spec.protocol_land_use_forcing)
        expected = (mode, natural, lu)
        assert actual == expected, f"{pid}: expected {expected}, got {actual}"


def test_co2_only_property_derived_from_protocol_mode():
    # The co2_only flag the loader's mask reads must always agree with
    # protocol_mode == "ED-CO2-only"; this guards the property against
    # future drift.
    from openscm_runner.scenarios.rcmip3 import _REGISTRY
    for pid, spec in _REGISTRY.items():
        assert spec.co2_only == (spec.protocol_mode == "ED-CO2-only"), pid


def test_load_emissions_writes_protocol_meta_columns(tmp_path):
    _write_rcmip3_like_csv(
        tmp_path / _SOURCE_RCMIP3.cache_filename,
        rows=[
            {"Scenario": "ssp245",
             "Variable": "Emissions|CO2|Energy and Industrial Processes",
             "Unit": "Mt CO2/yr"},
        ],
    )
    run = load_rcmip3_emissions(
        ["ssp245", "esm-ssp245", "esm-allGHG-ssp245"],
        cache_dir=tmp_path, download_if_missing=False,
    )
    for col in _PROTOCOL_META_COLS:
        assert col in run.meta.columns, f"missing meta col {col!r}"
    by_scen = run.meta.set_index("scenario")[list(_PROTOCOL_META_COLS)]
    assert by_scen.loc["ssp245", "protocol_mode"] == "CD"
    assert by_scen.loc["esm-ssp245", "protocol_mode"] == "ED-CO2-only"
    assert by_scen.loc["esm-allGHG-ssp245", "protocol_mode"] == "ED-all-GHG"
    # All three SSP variants are real-world, so natural/LU are identical.
    for scen in ("ssp245", "esm-ssp245", "esm-allGHG-ssp245"):
        assert by_scen.loc[scen, "protocol_natural_forcing"] == "on"
        assert by_scen.loc[scen, "protocol_land_use_forcing"] == "historical"


def test_load_emissions_meta_survives_co2_only_mask(tmp_path):
    # Regression guard: the mask reconstructs a fresh ScmRun via
    # (data, index, columns), so any meta col that isn't explicitly
    # carried across would get dropped. esm-ssp245 goes through the
    # mask; meta cols must still be present afterwards.
    _write_rcmip3_like_csv(
        tmp_path / _SOURCE_RCMIP3.cache_filename,
        rows=[
            {"Scenario": "ssp245",
             "Variable": "Emissions|CO2|Energy and Industrial Processes",
             "Unit": "Mt CO2/yr"},
            {"Scenario": "ssp245",
             "Variable": "Emissions|CH4",
             "Unit": "Mt CH4/yr"},
        ],
    )
    run = load_rcmip3_emissions(
        ["esm-ssp245"], cache_dir=tmp_path, download_if_missing=False,
    )
    meta = run.meta.drop_duplicates(subset=["scenario"]).iloc[0]
    assert meta["protocol_mode"] == "ED-CO2-only"
    assert meta["protocol_natural_forcing"] == "on"
    assert meta["protocol_land_use_forcing"] == "historical"


def test_step5a_esm_hist_variants_source_map_to_historical_row(tmp_path):
    # esm-hist / esm-allGHG-hist source-map onto the bare `historical`
    # row in the RCMIP3 CSV; esm-hist gets CO2-only masking via the
    # protocol_mode flag, esm-allGHG-hist passes through unmasked.
    _write_rcmip3_like_csv(
        tmp_path / _SOURCE_RCMIP3.cache_filename,
        rows=[
            {"Scenario": "historical",
             "Variable": "Emissions|CO2|Energy and Industrial Processes",
             "Unit": "Mt CO2/yr", "2020": 35000.0},
            {"Scenario": "historical",
             "Variable": "Emissions|CH4",
             "Unit": "Mt CH4/yr", "2020": 300.0},
        ],
    )
    run = load_rcmip3_emissions(
        ["esm-hist", "esm-allGHG-hist"],
        cache_dir=tmp_path, download_if_missing=False,
    )
    ts = run.timeseries(time_axis="year").reset_index()
    ch4_by_scen = ts[ts["variable"] == "Emissions|CH4"].set_index("scenario")[2020]
    # esm-hist is ED-CO2-only -> CH4 zeroed by loader mask
    assert ch4_by_scen["esm-hist"] == 0.0
    # esm-allGHG-hist is ED-all-GHG -> CH4 passthrough
    assert ch4_by_scen["esm-allGHG-hist"] == 300.0


def test_step5c_ch4_swap_loads_from_translated_bundle():
    # Step 5c: the two CH4-swap scenarios come from
    # src/openscm_runner/scenarios/data/bundle_translated/cicero_rcmip_march2026.csv
    # (committed in repo, generated offline by
    # scripts/translate_cicero_bundle_to_iamc.py). The loader resolves
    # them without needing the CICERO bundle at runtime and without
    # touching the network.
    from openscm_runner.scenarios import load_rcmip3_emissions

    run = load_rcmip3_emissions(
        ["esm-allGHG-scen7-H-CH4L", "esm-allGHG-scen7-L-CH4H"],
        download_if_missing=False,
    )
    assert sorted(run["scenario"].unique()) == [
        "esm-allGHG-scen7-H-CH4L", "esm-allGHG-scen7-L-CH4H",
    ]
    # CH4-swap sanity: H scenario has lower CH4, L has higher.
    ts = run.timeseries(time_axis="year").reset_index()
    ch4_by_scen = ts[ts["variable"] == "Emissions|CH4"].set_index("scenario")[2050]
    assert ch4_by_scen["esm-allGHG-scen7-H-CH4L"] < ch4_by_scen[
        "esm-allGHG-scen7-L-CH4H"
    ], (
        f"H-CH4L ({ch4_by_scen['esm-allGHG-scen7-H-CH4L']}) should have "
        f"lower CH4 than L-CH4H ({ch4_by_scen['esm-allGHG-scen7-L-CH4H']})"
    )
    # Metadata: ED all-GHG real-world.
    for scen in ("esm-allGHG-scen7-H-CH4L", "esm-allGHG-scen7-L-CH4H"):
        sub = run.filter(scenario=scen)
        assert sub["protocol_mode"].unique().tolist() == ["ED-all-GHG"]
        assert sub["protocol_natural_forcing"].unique().tolist() == ["on"]


def test_step5a_esm_allghg_picontrol_source_maps_to_esm_picontrol(tmp_path):
    # esm-allGHG-piControl reads from the esm-piControl source row
    # (both are PI controls; emissions are ~zero either way, but this
    # gives the loader a real ScmRun to feed CICERO via hybrid mode
    # rather than a zero stub).
    _write_rcmip3_like_csv(
        tmp_path / _SOURCE_RCMIP3.cache_filename,
        rows=[
            {"Scenario": "esm-piControl",
             "Variable": "Emissions|CO2|Energy and Industrial Processes",
             "Unit": "Mt CO2/yr", "2020": 0.0},
        ],
    )
    run = load_rcmip3_emissions(
        ["esm-allGHG-piControl"],
        cache_dir=tmp_path, download_if_missing=False,
    )
    assert run["scenario"].unique().tolist() == ["esm-allGHG-piControl"]
    # protocol_mode tag distinguishes from the bare esm-piControl row.
    assert run["protocol_mode"].unique().tolist() == ["ED-all-GHG"]
    assert run["protocol_natural_forcing"].unique().tolist() == ["off"]


def test_load_bundle_only_stub_carries_protocol_meta(tmp_path):
    # Only scenarios still in _RCMIP3_BUNDLE_ONLY_ED / _CD should
    # short-circuit the download path. After step 5c, the only
    # bundle-only ED scenarios left are the esm-1pct-brch-* triplet.
    run = load_rcmip3_emissions(
        ["1pctCO2", "esm-1pct-brch-1000PgC"],
        cache_dir=tmp_path, download_if_missing=False,
    )
    by_scen = run.meta.set_index("scenario")[list(_PROTOCOL_META_COLS)]
    # 1pctCO2: CD idealised
    assert by_scen.loc["1pctCO2", "protocol_mode"] == "CD"
    assert by_scen.loc["1pctCO2", "protocol_natural_forcing"] == "off"
    assert by_scen.loc["1pctCO2", "protocol_land_use_forcing"] == "constant_zero"
    # esm-1pct-brch-1000PgC: ED CO2-only idealised
    assert by_scen.loc["esm-1pct-brch-1000PgC", "protocol_mode"] == "ED-CO2-only"
    assert by_scen.loc[
        "esm-1pct-brch-1000PgC", "protocol_natural_forcing"
    ] == "off"


def test_load_concentrations_writes_protocol_meta_columns(tmp_path):
    _write_rcmip3_conc_like_csv(
        tmp_path / _SOURCE_RCMIP3_CONC.cache_filename,
        rows=[
            {"Scenario": "ssp245",
             "Variable": "Atmospheric Concentrations|CO2", "Unit": "ppm"},
            {"Scenario": "1pctCO2",
             "Variable": "Atmospheric Concentrations|CO2", "Unit": "ppm"},
        ],
    )
    run = load_rcmip3_concentrations(
        ["ssp245", "1pctCO2"],
        cache_dir=tmp_path, download_if_missing=False,
    )
    by_scen = run.meta.set_index("scenario")[list(_PROTOCOL_META_COLS)]
    # Concentrations are always CD-mode regardless of lineage.
    assert by_scen.loc["ssp245", "protocol_mode"] == "CD"
    assert by_scen.loc["1pctCO2", "protocol_mode"] == "CD"
    # Natural / LU inherit from the idealised-vs-real classification.
    assert by_scen.loc["ssp245", "protocol_natural_forcing"] == "on"
    assert by_scen.loc["1pctCO2", "protocol_natural_forcing"] == "off"
    assert by_scen.loc["1pctCO2", "protocol_land_use_forcing"] == "constant_zero"


# ---------------------------------------------------------------------------
# Constraint targets table
# ---------------------------------------------------------------------------


def test_registry_includes_bundle_only_scenarios():
    """Bundle-only scenarios (no IAMC CSV source) are still registered.

    After step 2, the ``esm-ssp*`` / ``esm-allGHG-ssp*`` / ``esm-scen7-*``
    / ``esm-allGHG-scen7-*`` families read from the source CSVs via the
    co2_only flag. The remaining bundle-only set (ED) is the CH4-swap
    scen7 variants, esm-1pct-brch-*, the ED historicals and
    esm-allGHG-piControl. Bundle-only CD covers 1pctCO2 / abrupt-* /
    scen7-*C / piControl.
    """
    scens = set(available_scenarios())
    # esm-hist*, esm-allGHG-hist*, esm-allGHG-piControl are still in
    # the registry but no longer bundle-only after step 5a (they
    # source-map onto historical / esm-piControl IAMC rows).
    for s in ("esm-1pct-brch-1000PgC", "esm-hist", "esm-allGHG-piControl",
              "esm-allGHG-scen7-H-CH4L", "esm-allGHG-scen7-L-CH4H"):
        assert s in scens, f"{s} missing from registry"
    for s in ("1pctCO2", "abrupt-4xCO2", "scen7-HC", "piControl"):
        assert s in scens, f"{s} missing from registry"


def test_load_bundle_only_returns_stub_with_scenario_name(tmp_path):
    # esm-1pct-brch-* and 1pctCO2 are bundle-only (no IAMC source) and
    # short-circuit the download / load_iamc path entirely, so no cached
    # CSV is needed.
    run = load_rcmip3_emissions(
        ["esm-1pct-brch-1000PgC", "1pctCO2"],
        cache_dir=tmp_path, download_if_missing=False,
    )
    assert sorted(run["scenario"].unique()) == [
        "1pctCO2", "esm-1pct-brch-1000PgC",
    ]
    # Stub spans the full protocol window (1750-2500).
    years = run.time_points.years()
    assert years[0] == 1750
    assert years[-1] == 2500
    # Stub data is intentionally zeros (CICERO bundle resolution
    # overrides; FaIR cannot consume bundle-only stubs yet).
    assert (run.values == 0.0).all()


# ---------------------------------------------------------------------------
# CO2-only ED mask (esm-* non-allGHG variants of SSPs and scen7-*)
# ---------------------------------------------------------------------------


def test_registry_has_esm_ssp_co2_only_and_allghg_passthrough():
    # Step 2 promotes esm-ssp* / esm-allGHG-ssp* out of bundle-only:
    # both should now point at the bare ssp* source row in the RCMIP3
    # CSV, with the co2_only flag distinguishing them.
    from openscm_runner.scenarios.rcmip3 import _REGISTRY, _SOURCE_RCMIP3
    spec_co2 = _REGISTRY["esm-ssp245"]
    spec_all = _REGISTRY["esm-allGHG-ssp245"]
    assert spec_co2.source is _SOURCE_RCMIP3
    assert spec_co2.source_scenario == "ssp245"
    assert spec_co2.co2_only is True
    assert spec_all.source is _SOURCE_RCMIP3
    assert spec_all.source_scenario == "ssp245"
    assert spec_all.co2_only is False


def test_registry_has_esm_scen7_co2_only_and_allghg_passthrough():
    from openscm_runner.scenarios.rcmip3 import _REGISTRY, _SOURCE_SCEN7
    spec_co2 = _REGISTRY["esm-scen7-H"]
    spec_all = _REGISTRY["esm-allGHG-scen7-H"]
    assert spec_co2.source is _SOURCE_SCEN7
    assert spec_co2.source_scenario == "SSP3 - High Emissions"
    assert spec_co2.source_model == "GCAM 8s"
    assert spec_co2.co2_only is True
    assert spec_all.co2_only is False
    assert spec_all.source_scenario == spec_co2.source_scenario


def test_load_esm_ssp_zeros_non_co2_keeps_co2_sector_splits(tmp_path):
    # esm-ssp245 reads from the bare ssp245 source row; non-CO2 species
    # come back as zeros (the ED CO2-only protocol) while the two CO2
    # sector splits pass through unchanged.
    _write_rcmip3_like_csv(
        tmp_path / _SOURCE_RCMIP3.cache_filename,
        rows=[
            {"Scenario": "ssp245",
             "Variable": "Emissions|CO2|Energy and Industrial Processes",
             "Unit": "Mt CO2/yr", "2050": 35000.0},
            {"Scenario": "ssp245",
             "Variable": "Emissions|CO2|AFOLU",
             "Unit": "Mt CO2/yr", "2050": 4500.0},
            {"Scenario": "ssp245",
             "Variable": "Emissions|CH4",
             "Unit": "Mt CH4/yr", "2050": 300.0},
            {"Scenario": "ssp245",
             "Variable": "Emissions|N2O",
             "Unit": "kt N2O/yr", "2050": 8000.0},
            {"Scenario": "ssp245",
             "Variable": "Emissions|HFC|HFC125",
             "Unit": "kt HFC125/yr", "2050": 50.0},
        ],
    )
    run = load_rcmip3_emissions(
        ["esm-ssp245"], cache_dir=tmp_path, download_if_missing=False,
    )
    assert run["scenario"].unique().tolist() == ["esm-ssp245"]
    ts = run.timeseries(time_axis="year").reset_index()
    by_var = dict(zip(ts["variable"], ts[2050]))
    assert by_var["Emissions|CO2|MAGICC Fossil and Industrial"] == 35000.0
    assert by_var["Emissions|CO2|MAGICC AFOLU"] == 4500.0
    assert by_var["Emissions|CH4"] == 0.0
    assert by_var["Emissions|N2O"] == 0.0
    assert by_var["Emissions|HFC125"] == 0.0


def test_load_esm_allghg_ssp_passes_emissions_through_unmasked(tmp_path):
    # esm-allGHG-ssp245 reads from the same source row as ssp245 but
    # without the CO2-only mask.
    _write_rcmip3_like_csv(
        tmp_path / _SOURCE_RCMIP3.cache_filename,
        rows=[
            {"Scenario": "ssp245",
             "Variable": "Emissions|CO2|Energy and Industrial Processes",
             "Unit": "Mt CO2/yr", "2050": 35000.0},
            {"Scenario": "ssp245",
             "Variable": "Emissions|CH4",
             "Unit": "Mt CH4/yr", "2050": 300.0},
        ],
    )
    run = load_rcmip3_emissions(
        ["esm-allGHG-ssp245"], cache_dir=tmp_path, download_if_missing=False,
    )
    assert run["scenario"].unique().tolist() == ["esm-allGHG-ssp245"]
    ts = run.timeseries(time_axis="year").reset_index()
    by_var = dict(zip(ts["variable"], ts[2050]))
    assert by_var["Emissions|CO2|MAGICC Fossil and Industrial"] == 35000.0
    assert by_var["Emissions|CH4"] == 300.0


def test_load_co_request_of_three_protocol_variants_yields_distinct_rows(tmp_path):
    # ssp245 (CD-style passthrough), esm-ssp245 (CO2-only) and
    # esm-allGHG-ssp245 (all-GHG) all read the same source row.
    # The loader must materialise three distinct scenario timeseries
    # rather than collapsing them via the (source_scen, source_model)
    # rename map.
    _write_rcmip3_like_csv(
        tmp_path / _SOURCE_RCMIP3.cache_filename,
        rows=[
            {"Scenario": "ssp245",
             "Variable": "Emissions|CO2|Energy and Industrial Processes",
             "Unit": "Mt CO2/yr", "2050": 35000.0},
            {"Scenario": "ssp245",
             "Variable": "Emissions|CH4",
             "Unit": "Mt CH4/yr", "2050": 300.0},
        ],
    )
    run = load_rcmip3_emissions(
        ["ssp245", "esm-ssp245", "esm-allGHG-ssp245"],
        cache_dir=tmp_path, download_if_missing=False,
    )
    assert sorted(run["scenario"].unique()) == [
        "esm-allGHG-ssp245", "esm-ssp245", "ssp245",
    ]
    # Non-CO2 should be zeroed for esm-ssp245 only.
    ts = run.timeseries(time_axis="year").reset_index()
    ch4 = ts[ts["variable"] == "Emissions|CH4"].set_index("scenario")[2050]
    assert ch4["ssp245"] == 300.0
    assert ch4["esm-allGHG-ssp245"] == 300.0
    assert ch4["esm-ssp245"] == 0.0


def test_load_esm_scen7_uses_marker_iam_and_masks_non_co2(tmp_path):
    # The scen7 CSV ships one row per (scenario, IAM) pair; esm-scen7-H
    # must filter to the GCAM marker (not other IAMs that happen to
    # publish "SSP3 - High Emissions") and apply the CO2-only mask.
    _write_scen7_like_csv(
        tmp_path / _SOURCE_SCEN7.cache_filename,
        rows=[
            {"scenario": "SSP3 - High Emissions", "model": "GCAM 8s",
             "variable": "Emissions|CO2|Energy and Industrial Processes",
             "unit": "Mt CO2/yr", "2050": 50000.0},
            {"scenario": "SSP3 - High Emissions", "model": "GCAM 8s",
             "variable": "Emissions|CH4", "unit": "Mt CH4/yr", "2050": 400.0},
            # Distractor: same scenario name from a different IAM. Must
            # be excluded by the marker filter.
            {"scenario": "SSP3 - High Emissions", "model": "AIM 3.0",
             "variable": "Emissions|CH4", "unit": "Mt CH4/yr", "2050": 999.0},
        ],
    )
    run = load_rcmip3_emissions(
        ["esm-scen7-H"], cache_dir=tmp_path, download_if_missing=False,
    )
    assert run["scenario"].unique().tolist() == ["esm-scen7-H"]
    assert run["model"].unique().tolist() == ["GCAM 8s"]
    ts = run.timeseries(time_axis="year").reset_index()
    by_var = dict(zip(ts["variable"], ts[2050]))
    assert by_var["Emissions|CO2|MAGICC Fossil and Industrial"] == 50000.0
    assert by_var["Emissions|CH4"] == 0.0


# ---------------------------------------------------------------------------
# Concentrations loader
# ---------------------------------------------------------------------------


def _write_rcmip3_conc_like_csv(
    path: Path, rows: list[dict], year_cols: tuple[int, ...] = (2020, 2050, 2100)
) -> None:
    """Write a minimal CSV with the RCMIP3 v1.1.7 concentrations column shape.

    The concentrations CSV uses the same Title-case column layout and
    same IAMC parent-path conventions as the emissions CSV; only the
    Variable column differs (``Atmospheric Concentrations|*`` instead of
    ``Emissions|*``).
    """
    base_cols = {
        "Model": "(unspecified)",
        "Scenario": "",
        "Region": "World",
        "Variable": "Atmospheric Concentrations|CO2",
        "Unit": "ppm",
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
            row.setdefault(str(y), 280.0)
        full_rows.append(row)
    pd.DataFrame(full_rows).to_csv(path, index=False)


def test_available_concentration_scenarios_is_subset_of_all_scenarios():
    # Every concentration scenario must also be a known protocol
    # scenario in the emissions registry (its bundle-only ED variant
    # or the same CD id). Anything outside that set is a typo.
    conc = set(available_concentration_scenarios())
    all_scens = set(available_scenarios())
    assert conc <= all_scens, f"Unknown conc scenarios: {conc - all_scens}"


def test_available_concentration_scenarios_covers_expected_families():
    conc = set(available_concentration_scenarios())
    # All 8 CMIP6 SSPs (CD) ship concentrations.
    assert {
        "ssp119", "ssp126", "ssp245", "ssp370",
        "ssp434", "ssp460", "ssp534-over", "ssp585",
    } <= conc
    # CD-only idealised families.
    assert {"1pctCO2", "1pctCO2-4xext", "1pctCO2-cdr",
            "abrupt-0p5xCO2", "abrupt-2xCO2", "abrupt-4xCO2"} <= conc
    # Historical variants and piControl.
    assert {"historical", "historical-cmip6", "piControl"} <= conc


def test_available_concentration_scenarios_excludes_scen7_variants():
    # The CMIP7 scen7-*C variants are NOT in the RCMIP3 v1.1.7 archive
    # (they need bundle translation in step 5 of the rearch).
    conc = set(available_concentration_scenarios())
    assert "scen7-HC" not in conc
    assert "scen7-VLC" not in conc


def test_load_rcmip3_concentrations_rejects_unknown_scenario():
    with pytest.raises(KeyError, match="No RCMIP3 concentration source"):
        load_rcmip3_concentrations(["ssp245", "not-a-real-scenario"])


def test_load_rcmip3_concentrations_rejects_scen7_variants():
    # scen7-*C is a real protocol id (registered for emissions
    # bundle-only) but not in the concentrations archive — this is the
    # error path that says "use bundle translation".
    with pytest.raises(KeyError, match="No RCMIP3 concentration source"):
        load_rcmip3_concentrations(["scen7-HC"])


def test_load_rcmip3_concentrations_rejects_empty_scenarios():
    with pytest.raises(ValueError, match="non-empty sequence"):
        load_rcmip3_concentrations([])


def test_load_rcmip3_concentrations_refuses_to_download_when_disabled(tmp_path):
    with pytest.raises(FileNotFoundError, match="not cached"):
        load_rcmip3_concentrations(
            ["ssp245"], cache_dir=tmp_path, download_if_missing=False,
        )


def test_load_rcmip3_concentrations_returns_canonical_variable(tmp_path):
    # Pre-seed cache, load, check the variable name is the canonical
    # ``Atmospheric Concentrations|CO2`` and that scenario filtering works.
    _write_rcmip3_conc_like_csv(
        tmp_path / _SOURCE_RCMIP3_CONC.cache_filename,
        rows=[
            {"Scenario": "ssp245", "Variable": "Atmospheric Concentrations|CO2",
             "Unit": "ppm"},
            {"Scenario": "ssp370", "Variable": "Atmospheric Concentrations|CO2",
             "Unit": "ppm"},
        ],
    )
    run = load_rcmip3_concentrations(
        ["ssp245"], cache_dir=tmp_path, download_if_missing=False,
    )
    assert run["scenario"].tolist() == ["ssp245"]
    assert run["variable"].tolist() == ["Atmospheric Concentrations|CO2"]
    assert run["unit"].tolist() == ["ppm"]


def test_load_rcmip3_concentrations_strips_fgas_parent_paths(tmp_path):
    # Same parent-path harmonisation that emissions go through: the
    # CSV's ``Atmospheric Concentrations|F-Gases|HFC|HFC125`` row should
    # come back as the canonical flat name.
    _write_rcmip3_conc_like_csv(
        tmp_path / _SOURCE_RCMIP3_CONC.cache_filename,
        rows=[
            {
                "Scenario": "ssp245",
                "Variable": "Atmospheric Concentrations|F-Gases|HFC|HFC125",
                "Unit": "ppt",
            },
            {
                "Scenario": "ssp245",
                "Variable": "Atmospheric Concentrations|Montreal Gases|CCl4",
                "Unit": "ppt",
            },
            # Bare CO2 row to confirm coexistence (regression guard).
            {
                "Scenario": "ssp245",
                "Variable": "Atmospheric Concentrations|CO2",
                "Unit": "ppm",
            },
        ],
    )
    run = load_rcmip3_concentrations(
        ["ssp245"], cache_dir=tmp_path, download_if_missing=False,
    )
    assert set(run["variable"]) == {
        "Atmospheric Concentrations|CO2",
        "Atmospheric Concentrations|HFC125",
        "Atmospheric Concentrations|CCl4",
    }


def test_load_rcmip3_concentrations_filters_to_requested_scenarios(tmp_path):
    _write_rcmip3_conc_like_csv(
        tmp_path / _SOURCE_RCMIP3_CONC.cache_filename,
        rows=[
            {"Scenario": "ssp245"},
            {"Scenario": "ssp370"},
            {"Scenario": "1pctCO2"},
        ],
    )
    run = load_rcmip3_concentrations(
        ["ssp245", "1pctCO2"], cache_dir=tmp_path, download_if_missing=False,
    )
    assert sorted(run["scenario"].unique()) == ["1pctCO2", "ssp245"]


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
