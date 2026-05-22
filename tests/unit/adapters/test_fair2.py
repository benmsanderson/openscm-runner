"""
Unit tests for the FaIRv2 adapter.

These tests do not require the Zenodo calibration bundle (#318/#319-
unrelated) and do not run real FaIR simulations. They cover:

- adapter is importable and registered under the FaIRv2 name
- NativeFairCalibration validates bundle structure (FileNotFoundError
  with a useful message when files are missing)
- the adapter raises NotImplementedError when a cfg lacks the
  native_calibration sidecar key (translated-cfg mode is a follow-up)
- the adapter raises ImportError with a clear message if fair 2.x is
  not installed

End-to-end runs against a real calibration bundle live in
``tests/integration/test_fair2.py`` and are gated on the
``FAIR2_CALIBRATION_PATH`` env var.
"""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from openscm_runner.adapters import FAIR2, get_adapter, get_adapters_classes
from openscm_runner.adapters.fair2_adapter._native_calibration import (
    NativeFairCalibration,
)


def test_fair2_is_registered():
    assert FAIR2.model_name == "FaIRv2"
    assert FAIR2 in get_adapters_classes()
    assert isinstance(get_adapter("FaIRv2"), FAIR2)
    # Case-insensitive lookup
    assert isinstance(get_adapter("fairv2"), FAIR2)


def test_fair2_raises_when_fair_not_installed():
    with patch(
        "openscm_runner.adapters.fair2_adapter.fair2_adapter.HAS_FAIR2", False
    ):
        with pytest.raises(ImportError, match="FaIR 2.x is not installed"):
            FAIR2()


def _make_bundle(tmp_path: Path, missing=()) -> Path:
    """Build a minimal bundle directory; optionally omit named files."""
    files = {
        "calibrated_constrained_parameters.csv": "seed,climate_sensitivity\n0,3.0\n1,2.5\n",
        "species_configs_properties.csv": "name,partition_fraction\nCO2,1.0\n",
    }
    for filename, contents in files.items():
        if filename in missing:
            continue
        (tmp_path / filename).write_text(contents)
    return tmp_path


def test_native_calibration_raises_when_directory_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        NativeFairCalibration(tmp_path / "does_not_exist")


def test_native_calibration_raises_when_path_is_file(tmp_path):
    f = tmp_path / "not_a_dir.txt"
    f.write_text("x")
    with pytest.raises(FileNotFoundError, match="not a directory"):
        NativeFairCalibration(f)


def test_native_calibration_raises_with_missing_required_files(tmp_path):
    _make_bundle(tmp_path, missing=("calibrated_constrained_parameters.csv",))
    with pytest.raises(FileNotFoundError, match="missing required files"):
        NativeFairCalibration(tmp_path)


def test_native_calibration_loads_parameters(tmp_path):
    _make_bundle(tmp_path)
    cal = NativeFairCalibration(tmp_path)
    assert cal.n_members == 2
    assert list(cal.parameters.columns) == ["seed", "climate_sensitivity"]


def test_native_calibration_select_members_all(tmp_path):
    _make_bundle(tmp_path)
    cal = NativeFairCalibration(tmp_path)
    out = cal.select_members(None)
    assert len(out) == 2


def test_native_calibration_select_members_indices(tmp_path):
    _make_bundle(tmp_path)
    cal = NativeFairCalibration(tmp_path)
    out = cal.select_members([1])
    assert len(out) == 1
    assert out.iloc[0]["seed"] == 1


def test_native_calibration_select_members_empty_raises(tmp_path):
    _make_bundle(tmp_path)
    cal = NativeFairCalibration(tmp_path)
    with pytest.raises(ValueError, match="zero members"):
        cal.select_members([])


def test_native_calibration_file_returns_none_for_absent_optional(tmp_path):
    _make_bundle(tmp_path)
    cal = NativeFairCalibration(tmp_path)
    assert cal.file("solar_forcing") is None
    assert cal.file("species_configs") is not None


def test_fair2_run_rejects_cfg_without_native_calibration():
    adapter = FAIR2()
    with pytest.raises(NotImplementedError, match="native_calibration"):
        # We never reach FaIR; the check runs before any setup.
        adapter._run(
            scenarios=None,
            cfgs=[{}],
            output_variables=("Surface Air Temperature Change",),
            output_config=None,
        )


def test_fair2_run_rejects_output_config():
    adapter = FAIR2()
    with pytest.raises(NotImplementedError, match="output_config"):
        adapter._run(
            scenarios=None,
            cfgs=[{"native_calibration": "/does/not/matter"}],
            output_variables=("Surface Air Temperature Change",),
            output_config=("foo",),
        )
