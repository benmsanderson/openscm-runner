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
    """
    Build a minimal bundle directory; optionally omit named files.

    The parameters CSV uses the real bundle's shape: the first column
    is unnamed and contains config labels (FaIR's override_defaults
    looks them up there), and subsequent columns are FaIR parameter
    names.
    """
    files = {
        "calibrated_constrained_parameters.csv": (
            ",climate_sensitivity\n100,3.0\n200,2.5\n"
        ),
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
    # The first CSV column is the row index (config label); the rest
    # are the actual parameters.
    assert list(cal.parameters.columns) == ["climate_sensitivity"]
    # Index carries the config labels from the CSV's first column.
    assert list(cal.parameters.index) == [100, 200]


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
    # Row 1 in the CSV (zero-based positional) is the second member,
    # whose config label is 200 (from the test fixture above).
    assert out.iloc[0]["climate_sensitivity"] == 2.5
    assert out.index[0] == 200


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


def test_fair2_translated_cfg_with_no_climate_configs_raises_useful_error():
    """
    Translated-cfg mode runs without a calibration bundle. If the cfg
    dicts do not supply the climate_configs values FaIR needs, FaIR
    rejects the run; the adapter re-raises with a hint pointing at
    native-calibration mode.
    """
    adapter = FAIR2()
    with pytest.raises(ValueError, match="native_calibration"):
        adapter._run(
            scenarios=None,
            cfgs=[{}],  # empty: nothing populates climate_configs
            output_variables=("Surface Air Temperature Change",),
            output_config=None,
        )


def test_fair2_run_rejects_mixed_native_and_translated_cfgs():
    """
    Each cfg list must be entirely native or entirely translated;
    mixing the two in one call is rejected for clarity.
    """
    adapter = FAIR2()
    with pytest.raises(NotImplementedError, match="all-native or all-translated"):
        adapter._run(
            scenarios=None,
            cfgs=[
                {"native_calibration": "/does/not/matter"},
                {"forcing_4co2": 8.0},
            ],
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


def test_fair2_translated_cfg_warns_on_unknown_parameter_names(caplog):
    """
    Translated-cfg mode logs a WARNING when a cfg dict carries keys
    that are not valid FaIR 2.x climate_configs or species_configs
    names. The check runs before the FaIR simulation so we can verify
    it fires even when the run itself fails.
    """
    import logging

    adapter = FAIR2()
    # The cfg has both a real FaIR parameter (forcing_4co2) and a
    # made-up key (definitely_not_a_fair_parameter). The run will
    # error later due to missing other climate_configs, but the
    # WARN should appear in the log before that.
    with caplog.at_level(
        logging.WARNING,
        logger="openscm_runner.adapters.fair2_adapter.fair2_adapter",
    ):
        with pytest.raises(ValueError):
            adapter._run(
                scenarios=None,
                cfgs=[
                    {
                        "forcing_4co2": 8.0,
                        "definitely_not_a_fair_parameter": 1.0,
                    }
                ],
                output_variables=("Surface Air Temperature Change",),
                output_config=None,
            )

    assert "definitely_not_a_fair_parameter" in caplog.text
    assert "ignored unknown parameter names" in caplog.text


def test_fair2_stochastic_run_default_is_off():
    """
    The AR7-relevant fair-calibrate bundles ship
    `climate_configs['stochastic_run']=True` for every posterior
    member, which adds an AR(1) natural-variability term on top of
    each member's deterministic trajectory. openscm-runner's typical
    use case is comparing medians + spreads across scenarios, which
    is cleaner with parameter-only spread; the adapter overrides
    `stochastic_run` back to False by default. Set
    `fair2_stochastic_run=True` in the cfg to opt back in.

    Pins the contract by inspecting the threaded-through default of
    `_run_one_calibration`; the actual override against
    `f.climate_configs['stochastic_run']` is exercised by the
    integration test (gated on FAIR2_CALIBRATION_PATH) and by the
    cross-model notebook on `modernisation/integration`.
    """
    import inspect

    from openscm_runner.adapters.fair2_adapter.fair2_adapter import (
        _run_one_calibration,
    )
    sig = inspect.signature(_run_one_calibration)
    assert sig.parameters["stochastic_run"].default is False, (
        "_run_one_calibration's stochastic_run kwarg must default to "
        "False so the adapter overrides the calibration's "
        "stochastic_run=True. See module docstring."
    )


def test_fair2_conc_driven_requires_bundle_dir(tmp_path):
    """
    Setting ``fair2_conc_driven=True`` without ``fair2_conc_bundle_dir``
    is a user error - the adapter must raise immediately rather than
    silently fall back to emissions-driven (which would silently
    produce different physics than the user asked for).
    """
    adapter = FAIR2()
    # Build a minimal bundle so the calibration loader doesn't choke
    # before we hit the conc-driven validation.
    _make_bundle(tmp_path)
    with pytest.raises(ValueError, match="fair2_conc_bundle_dir"):
        adapter._run(
            scenarios=None,
            cfgs=[
                {
                    "native_calibration": str(tmp_path),
                    "fair2_conc_driven": True,
                }
            ],
            output_variables=("Surface Air Temperature Change",),
            output_config=None,
        )
