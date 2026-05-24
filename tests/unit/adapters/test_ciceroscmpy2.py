"""
Unit tests for the CICEROSCMPY2 adapter.

These tests do not require ciceroscm 2.x to be installed and do not
run real CICERO-SCM simulations. They cover:

- adapter is importable and registered under the CICERO-SCM-PY2 name
- the adapter raises ImportError with a clear message if ciceroscm is
  not installed, or if a major version other than 2.x is installed
- the adapter rejects cfgs missing required sidecar keys
  (distribution_json, gaspam_file, concentrations_file) and rejects
  empty member_indices
- the adapter raises NotImplementedError when output_config is set

End-to-end runs against a real distribution JSON live in
``tests/integration/test_ciceroscmpy2.py`` and are gated on the
``CICEROSCMPY2_CALIBRATION_PATH`` env var.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from openscm_runner.adapters import CICEROSCMPY2, get_adapter, get_adapters_classes


def test_ciceroscmpy2_is_registered():
    assert CICEROSCMPY2.model_name == "CICERO-SCM-PY2"
    assert CICEROSCMPY2 in get_adapters_classes()
    assert isinstance(get_adapter("CICERO-SCM-PY2"), CICEROSCMPY2)
    # Case-insensitive lookup
    assert isinstance(get_adapter("cicero-scm-py2"), CICEROSCMPY2)


def test_ciceroscmpy2_raises_when_ciceroscm_not_installed():
    with patch(
        "openscm_runner.adapters.ciceroscm_py2_adapter."
        "ciceroscmpy2_adapter.HAS_CICEROSCM_PY2",
        False,
    ):
        with pytest.raises(ImportError, match="ciceroscm is not installed"):
            CICEROSCMPY2()


def test_ciceroscmpy2_raises_when_wrong_major_version():
    """
    A ciceroscm install of major version 1 should be rejected (it
    does not have the DistributionRun API this adapter targets).
    """
    with patch(
        "openscm_runner.adapters.ciceroscm_py2_adapter."
        "ciceroscmpy2_adapter._ciceroscm_major_version",
        return_value=1,
    ):
        with pytest.raises(ImportError, match="requires >=2"):
            CICEROSCMPY2()


def test_ciceroscmpy2_run_rejects_output_config():
    adapter = CICEROSCMPY2()
    with pytest.raises(NotImplementedError, match="output_config"):
        adapter._run(
            scenarios=None,
            cfgs=[{
                "distribution_json": "/does/not/matter",
                "gaspam_file": "/does/not/matter",
                "concentrations_file": "/does/not/matter",
            }],
            output_variables=("Surface Air Temperature Change",),
            output_config=("foo",),
        )


@pytest.mark.parametrize(
    "missing_key",
    ["distribution_json", "gaspam_file", "concentrations_file"],
)
def test_ciceroscmpy2_rejects_cfg_missing_required_sidecar(missing_key):
    adapter = CICEROSCMPY2()
    cfg = {
        "distribution_json": "/does/not/matter",
        "gaspam_file": "/does/not/matter",
        "concentrations_file": "/does/not/matter",
    }
    cfg.pop(missing_key)
    with pytest.raises(ValueError, match=missing_key):
        adapter._run(
            scenarios=None,
            cfgs=[cfg],
            output_variables=("Surface Air Temperature Change",),
            output_config=None,
        )


def test_ciceroscmpy2_rejects_missing_distribution_json(tmp_path):
    """
    After sidecar-key validation, the path is checked for existence
    so the user gets a clear FileNotFoundError rather than a JSON
    decode error from deep inside upstream.
    """
    adapter = CICEROSCMPY2()
    cfg = {
        "distribution_json": str(tmp_path / "does_not_exist.json"),
        "gaspam_file": str(tmp_path / "gaspam.txt"),
        "concentrations_file": str(tmp_path / "conc.txt"),
    }
    with pytest.raises(FileNotFoundError, match="distribution_json"):
        adapter._run(
            scenarios=None,
            cfgs=[cfg],
            output_variables=("Surface Air Temperature Change",),
            output_config=None,
        )


def test_ciceroscmpy2_bundle_mode_skips_splice_sidecar_validation(tmp_path):
    """
    Bundle mode (``cicero_bundle_dir`` set) drops the splice-mode
    requirement for ``gaspam_file`` / ``concentrations_file`` because
    those are derived from the bundle. The adapter should accept a
    cfg with only ``distribution_json`` + ``cicero_bundle_dir`` and
    fail later on the bundle-dir existence check rather than upfront
    on missing splice-mode keys.
    """
    adapter = CICEROSCMPY2()
    samples_json = tmp_path / "samples.json"
    samples_json.write_text("[{\"pamset_udm\": {}, \"pamset_emiconc\": {}}]")
    cfg = {
        "distribution_json": str(samples_json),
        "cicero_bundle_dir": str(tmp_path / "nonexistent_bundle"),
    }
    # Splice-mode missing-key ValueError mentions 'gaspam_file' /
    # 'concentrations_file'; bundle-mode failure is a
    # FileNotFoundError about the bundle directory itself.
    with pytest.raises(FileNotFoundError, match="cicero_bundle_dir"):
        adapter._run(
            scenarios=None,
            cfgs=[cfg],
            output_variables=("Surface Air Temperature Change",),
            output_config=None,
        )


def test_ciceroscmpy2_splice_mode_emits_present_day_bias_warning(
    tmp_path, caplog
):
    """
    Splice mode carries a documented ~0.3-0.5 K present-day warm bias
    because the v1.1.x-bundled ssp245 historical doesn't match v2.x
    calibrations such as draw_samples_500. The adapter must emit a
    ``LOGGER.warning`` whenever splice mode is invoked so the bias is
    not silent. Bundle mode must NOT emit it.

    See project memory ``project-ciceroscm-historical-bias`` for the
    diagnosis (splice mode 1.85 K vs bundle mode 1.50 K at 2024).
    """
    import logging

    import pandas as pd
    import scmdata

    from openscm_runner.adapters.ciceroscm_py2_adapter import (
        ciceroscmpy2_adapter,
    )
    from openscm_runner.adapters.ciceroscm_py2_adapter.ciceroscmpy2_adapter import (
        _build_scendata_list_splice,
    )

    # Splice mode requires a real ScmRun; build a 2-year stub via
    # scmdata so the helper's `time_points.years()` call works.
    df = pd.DataFrame({
        "model": ["MOD"], "scenario": ["SCN"], "region": ["World"],
        "variable": ["Emissions|CO2"], "unit": ["Gt CO2/yr"],
        2020: [40.0], 2021: [40.0],
    })
    scenarios = scmdata.ScmRun(df)

    with caplog.at_level(logging.WARNING, logger=ciceroscmpy2_adapter.__name__):
        try:
            _build_scendata_list_splice(scenarios, {})
        except Exception:
            # Helper will fail later on missing splice files; we only
            # care that the warning fired before the failure.
            ...

    warning_records = [
        r for r in caplog.records
        if r.levelname == "WARNING"
        and "splice mode" in r.getMessage()
    ]
    captured = [r.getMessage() for r in caplog.records]
    assert warning_records, (
        f"Expected a LOGGER.warning containing 'splice mode' when "
        f"_build_scendata_list_splice runs. Got: {captured}"
    )
    msg = warning_records[0].getMessage()
    assert "warm bias" in msg, f"Splice warning should flag warm bias: {msg!r}"
    assert "bundle mode" in msg, (
        f"Splice warning should point users at bundle mode: {msg!r}"
    )


def test_ciceroscmpy2_rejects_empty_member_indices(tmp_path):
    """
    An explicit empty member_indices is almost certainly a user
    mistake (omitting the key is the documented way to run the full
    distribution), so we reject it with a clear error rather than
    silently producing a zero-member run.
    """
    samples_json = tmp_path / "samples.json"
    samples_json.write_text("[{\"pamset_udm\": {}, \"pamset_emiconc\": {}}]")
    adapter = CICEROSCMPY2()
    cfg = {
        "distribution_json": str(samples_json),
        "gaspam_file": str(tmp_path / "gaspam.txt"),
        "concentrations_file": str(tmp_path / "conc.txt"),
        "member_indices": [],
    }

    # scenarios=None triggers an earlier ValueError, so build a tiny
    # stand-in that exposes the attrs the helper uses.
    class _StubScenarios:
        def time_points(self):
            raise AssertionError("not reached")

    # Member-indices validation runs before scenariodata construction
    # in _run_one_distribution, so we can pass scenarios=None as a
    # cheap sentinel; the empty-list check fires first.
    with pytest.raises(ValueError, match="zero members"):
        adapter._run(
            scenarios=_StubScenarios(),
            cfgs=[cfg],
            output_variables=("Surface Air Temperature Change",),
            output_config=None,
        )
