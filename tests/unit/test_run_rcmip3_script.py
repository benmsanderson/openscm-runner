"""Unit tests for scripts/run_rcmip3.py.

These exercise the CLI parser and the scenario-set resolution logic
without invoking any climate model. The end-to-end smoke test that
actually dispatches FaIRv2 + CICERO-SCM-PY2 lives outside the unit
suite (it needs the calibration bundles).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_runner_module():
    """Import scripts/run_rcmip3.py as a module without installing it."""
    spec = importlib.util.spec_from_file_location(
        "_runner_test_target",
        Path(__file__).parent.parent.parent / "scripts" / "run_rcmip3.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner_module()


def test_parse_args_defaults():
    args = runner._parse_args([])
    assert args.models == ["fair2", "ciceroscmpy2"]
    assert args.members == 10
    assert args.mode == "both"
    assert args.scenarios is None
    assert args.scenario_set is None
    # Default output and cache dirs live under the repo root.
    assert args.output_dir.name == "rcmip3"
    assert args.cache_dir.name == "rcmip3"


def test_parse_args_mode_choices():
    for mode in ("emissions", "concentrations", "both"):
        assert runner._parse_args(["--mode", mode]).mode == mode
    with pytest.raises(SystemExit):
        runner._parse_args(["--mode", "neither"])


def test_scenarios_for_mode_keeps_all_in_emissions_mode():
    inputs = ("ssp245", "esm-flat10", "esm-flat10-zec", "scen7-VL")
    assert runner._scenarios_for_mode(inputs, runner.MODE_EMISSIONS) == inputs


def test_scenarios_for_mode_drops_esm_flat_in_concentrations_mode():
    inputs = ("ssp245", "esm-flat10", "esm-flat10-zec", "scen7-VL", "historical")
    out = runner._scenarios_for_mode(inputs, runner.MODE_CONCENTRATIONS)
    assert "esm-flat10" not in out
    assert "esm-flat10-zec" not in out
    assert "ssp245" in out
    assert "scen7-VL" in out
    assert "historical" in out


def test_parse_args_explicit_overrides():
    args = runner._parse_args(
        [
            "--models", "fair2",
            "--members", "5",
            "--scenarios", "ssp245", "esm-flat10",
            "--output-dir", "/tmp/foo",
        ]
    )
    assert args.models == ["fair2"]
    assert args.members == 5
    assert args.scenarios == ["ssp245", "esm-flat10"]
    assert args.scenario_set is None


def test_parse_args_rejects_unknown_model_alias():
    with pytest.raises(SystemExit):
        runner._parse_args(["--models", "magicc"])


def test_resolve_scenarios_default_is_ssps():
    args = runner._parse_args([])
    resolved = runner._resolve_scenarios(args)
    assert resolved == runner._SCENARIO_SETS["ssps"]


def test_resolve_scenarios_named_set_flat():
    args = runner._parse_args(["--scenario-set", "flat"])
    resolved = runner._resolve_scenarios(args)
    assert len(resolved) == 15  # 3 bases x 5 variants
    assert "esm-flat10" in resolved
    assert "esm-flat10-zec" in resolved
    assert "esm-flat20-rev" in resolved


def test_resolve_scenarios_named_set_all_covers_everything():
    args = runner._parse_args(["--scenario-set", "all"])
    resolved = runner._resolve_scenarios(args)
    # 8 SSPs + 7 scen7 + 15 flat-* + 1 historical = 31
    assert len(resolved) == 31
    assert "ssp119" in resolved
    assert "scen7-VL" in resolved
    assert "esm-flat10-zec" in resolved
    assert "historical" in resolved


def test_resolve_scenarios_explicit_takes_precedence_only_if_other_unset():
    args = runner._parse_args(["--scenarios", "ssp245", "ssp585"])
    resolved = runner._resolve_scenarios(args)
    assert resolved == ("ssp245", "ssp585")


def test_resolve_scenarios_rejects_both_flags():
    # The argparse mutually-exclusive group catches this at parse time,
    # so SystemExit comes from argparse rather than _resolve_scenarios.
    with pytest.raises(SystemExit):
        runner._parse_args(
            ["--scenarios", "ssp245", "--scenario-set", "ssps"]
        )


def test_model_dispatch_table_has_both_modern_adapters():
    # Guard against accidental deletion of either canonical name.
    assert runner._MODEL_DISPATCH["fair2"][0] == "FaIRv2"
    assert runner._MODEL_DISPATCH["ciceroscmpy2"][0] == "CICERO-SCM-PY2"


def test_output_variables_includes_tier1_set():
    # The notebook depends on these names being present in the
    # dispatched output. If you trim _OUTPUT_VARIABLES, update the
    # notebook accordingly.
    assert "Surface Air Temperature Change" in runner._OUTPUT_VARIABLES
    assert "Atmospheric Concentrations|CO2" in runner._OUTPUT_VARIABLES
    assert "Effective Radiative Forcing" in runner._OUTPUT_VARIABLES
    assert "Heat Content|Ocean" in runner._OUTPUT_VARIABLES


def test_safe_dir_matches_netcdf_chunk_writer_convention():
    # We mirror NetCDFChunkWriter._safe_name's regex by hand so the
    # summary-reading codepath can find the chunk on disk. If upstream
    # changes the regex this test will catch the drift.
    from openscm_runner.output import _safe_name
    for name in (
        "ssp245", "esm-flat10-zec", "scen7-VL", "CICERO-SCM-PY2", "FaIRv2",
    ):
        assert runner._safe_dir(name) == _safe_name(name)
