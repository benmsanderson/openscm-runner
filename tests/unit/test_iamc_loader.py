"""Unit tests for the IAMC-format scenario loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from openscm_runner.scenarios import CANONICAL_VARIABLES, load_iamc
from openscm_runner.scenarios.iamc_loader import (
    _canonicalise_unit,
    _canonicalise_variable,
)


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "scenarios.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_xlsx(tmp_path: Path, rows: list[dict], sheet: str = "data") -> Path:
    path = tmp_path / "scenarios.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name=sheet, index=False)
    return path


def _row(model="MOD", scenario="SCN", region="World", variable="Emissions|CH4",
         unit="Mt CH4/yr", **years):
    base = {
        "Model": model, "Scenario": scenario, "Region": region,
        "Variable": variable, "Unit": unit,
    }
    base.update({str(y): v for y, v in years.items()})
    return base


def test_canonicalise_rcmip_parent_paths():
    assert _canonicalise_variable("Emissions|F-Gases|HFC|HFC125") == "Emissions|HFC125"
    assert _canonicalise_variable("Emissions|F-Gases|PFC|CF4") == "Emissions|CF4"
    assert (
        _canonicalise_variable("Emissions|Montreal Gases|CCl4") == "Emissions|CCl4"
    )


def test_canonicalise_sci_parent_paths():
    assert _canonicalise_variable("Emissions|HFC|HFC125") == "Emissions|HFC125"
    assert _canonicalise_variable("Emissions|HFC|HFC32") == "Emissions|HFC32"


def test_canonicalise_hfc4310_rename_handles_both_spellings():
    assert _canonicalise_variable("Emissions|HFC|HFC43-10") == "Emissions|HFC4310mee"
    assert _canonicalise_variable("Emissions|F-Gases|HFC|HFC4310") == "Emissions|HFC4310mee"


def test_canonicalise_legacy_species_renames():
    assert _canonicalise_variable("Emissions|SOx") == "Emissions|Sulfur"
    assert _canonicalise_variable("Emissions|NMVOC") == "Emissions|VOC"


def test_canonicalise_unit_rewrites_hfc4310_dashed_form():
    # SCI exports HFC4310mee rows with this unit string; the dash is
    # not a valid pint symbol character, so the loader rewrites to the
    # RCMIP convention which openscm-units can parse.
    assert _canonicalise_unit("kt HFC43-10/yr") == "kt HFC4310mee/yr"


def test_canonicalise_unit_normalises_bare_hfc4310_to_mee_form():
    assert _canonicalise_unit("kt HFC4310/yr") == "kt HFC4310mee/yr"


def test_canonicalise_unit_does_not_double_rewrite_hfc4310mee():
    assert _canonicalise_unit("kt HFC4310mee/yr") == "kt HFC4310mee/yr"


def test_canonicalise_unit_legacy_renames():
    assert _canonicalise_unit("Mt SOx/yr") == "Mt Sulfur/yr"
    assert _canonicalise_unit("Mt NMVOC/yr") == "Mt VOC/yr"


def test_canonicalise_unit_passes_other_strings_through():
    assert _canonicalise_unit("Mt CO2/yr") == "Mt CO2/yr"
    assert _canonicalise_unit("ppm") == "ppm"


def test_load_iamc_unit_rewrite_propagates_to_scmrun(tmp_path):
    # End-to-end: SCI-style HFC4310mee row arrives with the dashed unit,
    # gets relabelled to the canonical variable and the canonical unit
    # in the returned ScmRun.
    path = _write_csv(tmp_path, [
        _row(variable="Emissions|HFC|HFC43-10", unit="kt HFC43-10/yr",
             **{"2020": 1.0}),
    ])
    run = load_iamc(path)
    assert run["variable"].tolist() == ["Emissions|HFC4310mee"]
    assert run["unit"].tolist() == ["kt HFC4310mee/yr"]


def test_canonicalise_co2_sector_splits():
    assert (
        _canonicalise_variable("Emissions|CO2|Energy and Industrial Processes")
        == "Emissions|CO2|MAGICC Fossil and Industrial"
    )
    assert (
        _canonicalise_variable("Emissions|CO2|AFOLU")
        == "Emissions|CO2|MAGICC AFOLU"
    )


def test_canonicalise_leaves_unknown_co2_subcategory_untouched():
    # Falls through harmonisation; subsequent allowlist filter drops it.
    sub = "Emissions|CO2|Energy|Demand|Industry"
    assert _canonicalise_variable(sub) == sub


def test_canonicalise_does_not_touch_non_emissions_variables():
    assert _canonicalise_variable("Primary Energy|Coal") == "Primary Energy|Coal"
    assert _canonicalise_variable(None) is None  # type: ignore[arg-type]


def test_load_iamc_csv_filters_to_world_and_canonical_allowlist(tmp_path):
    path = _write_csv(tmp_path, [
        _row(variable="Emissions|CH4", **{"2020": 300.0}),
        _row(variable="Emissions|CH4", region="R5.2OECD", **{"2020": 100.0}),
        _row(variable="Emissions|HFC|HFC125", unit="kt HFC125/yr", **{"2020": 1.0}),
        _row(variable="Primary Energy|Coal", unit="EJ/yr", **{"2020": 50.0}),
        _row(
            variable="Emissions|CO2|Energy and Industrial Processes",
            unit="Mt CO2/yr", **{"2020": 35000.0},
        ),
    ])
    run = load_iamc(path)
    variables = sorted(run["variable"].unique())
    assert variables == [
        "Emissions|CH4",
        "Emissions|CO2|MAGICC Fossil and Industrial",
        "Emissions|HFC125",
    ]
    assert set(run["region"].unique()) == {"World"}


def test_load_iamc_drops_afolu_nghgi_inventory_variant(tmp_path):
    path = _write_csv(tmp_path, [
        _row(variable="Emissions|CO2|AFOLU", unit="Mt CO2/yr", **{"2020": 5000.0}),
        _row(
            variable="Emissions|CO2|AFOLU [NGHGI]", unit="Mt CO2/yr",
            **{"2020": 5500.0},
        ),
    ])
    run = load_iamc(path)
    assert list(run["variable"]) == ["Emissions|CO2|MAGICC AFOLU"]
    # The [NGHGI] variant gets dropped, not relabelled.
    assert run.values.flatten().tolist() == [5000.0]


def test_load_iamc_scenario_filter(tmp_path):
    path = _write_csv(tmp_path, [
        _row(scenario="A", **{"2020": 1.0}),
        _row(scenario="B", **{"2020": 2.0}),
        _row(scenario="C", **{"2020": 3.0}),
    ])
    run = load_iamc(path, scenarios=["A", "C"])
    assert sorted(run["scenario"].unique()) == ["A", "C"]


def test_load_iamc_xlsx_dispatches_on_extension(tmp_path):
    path = _write_xlsx(tmp_path, [
        _row(**{"2020": 300.0, "2050": 200.0}),
    ])
    run = load_iamc(path, sheet="data")
    assert run["variable"].tolist() == ["Emissions|CH4"]


def test_load_iamc_rejects_unknown_extension(tmp_path):
    bad = tmp_path / "scenarios.parquet"
    bad.write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_iamc(bad)


def test_load_iamc_rejects_missing_iamc_columns(tmp_path):
    path = tmp_path / "broken.csv"
    pd.DataFrame({"scenario": ["A"], "variable": ["Emissions|CH4"], "2020": [1.0]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="missing required IAMC columns"):
        load_iamc(path)


def test_load_iamc_raises_when_filter_leaves_no_rows(tmp_path):
    path = _write_csv(tmp_path, [
        _row(variable="Primary Energy|Coal", unit="EJ/yr", **{"2020": 1.0}),
    ])
    with pytest.raises(ValueError, match="No rows survived filtering"):
        load_iamc(path)


def test_canonical_variables_matches_existing_rcmip_csv_set():
    # Guard against drift: the loader's allowlist should equal the
    # variable set in scripts/rcmip_scen_ssp_world_emissions.csv, which
    # downstream tests already consume.
    rcmip_csv = (
        Path(__file__).parent.parent.parent
        / "scripts"
        / "rcmip_scen_ssp_world_emissions.csv"
    )
    if not rcmip_csv.exists():
        pytest.skip("rcmip_scen_ssp_world_emissions.csv not present")
    rcmip_vars = set(pd.read_csv(rcmip_csv)["Variable"].unique())
    assert rcmip_vars == set(CANONICAL_VARIABLES)
