import logging
import re

import pandas as pd
import pytest
import scmdata

import openscm_runner.run
from openscm_runner.adapters import _registered_adapters, register_adapter_class
from openscm_runner.adapters.base import _Adapter


def test_run_out_config_conflict_error(caplog):
    expected_log = (
        "Found model(s) in `out_config` which are not in "
        "`climate_models_cfgs`: {'another model'}"
    )
    with caplog.at_level(logging.WARNING, logger="openscm_runner.run"):
        with pytest.raises(NotImplementedError):
            openscm_runner.run.run(
                climate_models_cfgs={"model_a": ["config list"]},
                scenarios="not used",
                out_config={"another model": ("hi",)},
            )

    assert expected_log in caplog.text


def test_run_out_config_type_error():
    error_msg = re.escape(
        "`out_config` values must be tuples, this isn't the case for "
        "climate_model: 'model_a'"
    )
    with pytest.raises(TypeError, match=error_msg):
        openscm_runner.run.run(
            climate_models_cfgs={"model_a": ["config list"]},
            scenarios="not used",
            out_config={"model_a": "hi"},
        )


def _dummy_result(climate_model):
    return scmdata.ScmRun(
        pd.DataFrame(
            [[1.0, 2.0]],
            index=pd.MultiIndex.from_tuples(
                [
                    (
                        climate_model,
                        "test_scen",
                        "test_iam",
                        "World",
                        "Surface Temperature",
                        "K",
                        0,
                    )
                ],
                names=[
                    "climate_model",
                    "scenario",
                    "model",
                    "region",
                    "variable",
                    "unit",
                    "run_id",
                ],
            ),
            columns=[2020, 2021],
        )
    )


class _DummyAdapterA(_Adapter):
    model_name = "DummyA"

    def _init_model(self, *args, **kwargs):
        pass

    def _run(self, scenarios, cfgs, output_variables, output_config):
        return _dummy_result("DummyA")


class _DummyAdapterB(_Adapter):
    model_name = "DummyB"

    def _init_model(self, *args, **kwargs):
        pass

    def _run(self, scenarios, cfgs, output_variables, output_config):
        return _dummy_result("DummyB")


@pytest.fixture()
def dummy_adapters():
    existing = _registered_adapters.copy()
    register_adapter_class(_DummyAdapterA)
    register_adapter_class(_DummyAdapterB)
    yield
    _registered_adapters.clear()
    _registered_adapters.extend(existing)


def test_serial_dispatch_runs_each_model(dummy_adapters):
    res = openscm_runner.run.run(
        climate_models_cfgs={"DummyA": [{}], "DummyB": [{}]},
        scenarios=None,
        parallel_models=False,
    )
    assert set(res["climate_model"]) == {"DummyA", "DummyB"}


def test_parallel_models_single_model_skips_pool(dummy_adapters, monkeypatch):
    pool_calls = []

    def _spy(*args, **kwargs):
        pool_calls.append((args, kwargs))
        raise AssertionError(
            "ProcessPoolExecutor should not be constructed for a single-model run"
        )

    monkeypatch.setattr("openscm_runner.run.ProcessPoolExecutor", _spy)

    res = openscm_runner.run.run(
        climate_models_cfgs={"DummyA": [{}]},
        scenarios=None,
        parallel_models=True,
    )

    assert res["climate_model"].iloc[0] == "DummyA"
    assert pool_calls == []


def test_parallel_dispatch_uses_process_pool(dummy_adapters, monkeypatch):
    """
    Verify that parallel_models=True with >1 model routes through
    ProcessPoolExecutor. We do not actually fork: we substitute a fake pool
    that runs submitted tasks in-process. This keeps the test deterministic
    across platforms (macOS uses the spawn start method, which would not
    inherit our runtime-registered dummy adapters).
    """
    captured = {}

    class _FakeFuture:
        def __init__(self, func, args):
            self._func = func
            self._args = args

        def result(self):
            return self._func(*self._args)

    class _FakePool:
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers
            captured["submitted"] = []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, func, *args):
            captured["submitted"].append((func.__name__, args[0]))
            return _FakeFuture(func, args)

    monkeypatch.setattr("openscm_runner.run.ProcessPoolExecutor", _FakePool)

    res = openscm_runner.run.run(
        climate_models_cfgs={"DummyA": [{}], "DummyB": [{}]},
        scenarios=None,
        parallel_models=True,
    )

    assert captured["max_workers"] == 2
    assert [name for name, _ in captured["submitted"]] == [
        "_run_one_model",
        "_run_one_model",
    ]
    assert sorted(model for _, model in captured["submitted"]) == ["DummyA", "DummyB"]
    assert set(res["climate_model"]) == {"DummyA", "DummyB"}


def test_output_writer_returns_runresult_and_skips_in_memory_append(
    dummy_adapters, tmp_path
):
    written = []

    def capture_writer(chunk, metadata):
        path = tmp_path / f"{metadata['climate_model']}__{metadata['scenario']}.nc"
        path.write_text("placeholder")
        written.append((metadata["climate_model"], metadata["scenario"]))
        return path

    result = openscm_runner.run.run(
        climate_models_cfgs={"DummyA": [{}], "DummyB": [{}]},
        scenarios=None,
        parallel_models=False,
        output_writer=capture_writer,
    )

    from openscm_runner.output import RunResult

    assert isinstance(result, RunResult)
    assert result.n_chunks == 2
    assert sorted(result.models) == ["DummyA", "DummyB"]
    assert result.scenarios == ["test_scen"]
    assert result.n_runs == 2
    assert sorted(p.name for p in result.chunk_paths) == [
        "DummyA__test_scen.nc",
        "DummyB__test_scen.nc",
    ]
    assert sorted(written) == [
        ("DummyA", "test_scen"),
        ("DummyB", "test_scen"),
    ]


def test_output_writer_works_through_parallel_dispatch(
    dummy_adapters, monkeypatch, tmp_path
):
    captured = {}

    class _FakeFuture:
        def __init__(self, func, args):
            self._func = func
            self._args = args

        def result(self):
            return self._func(*self._args)

    class _FakePool:
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, func, *args):
            return _FakeFuture(func, args)

    monkeypatch.setattr("openscm_runner.run.ProcessPoolExecutor", _FakePool)

    written_paths = []

    def writer(chunk, metadata):
        path = tmp_path / f"{metadata['climate_model']}__{metadata['scenario']}.nc"
        path.write_text("placeholder")
        written_paths.append(path)
        return path

    result = openscm_runner.run.run(
        climate_models_cfgs={"DummyA": [{}], "DummyB": [{}]},
        scenarios=None,
        parallel_models=True,
        output_writer=writer,
    )

    from openscm_runner.output import RunResult

    assert isinstance(result, RunResult)
    assert captured["max_workers"] == 2
    assert len(written_paths) == 2
    assert result.n_chunks == 2


def test_max_model_workers_caps_pool_size(dummy_adapters, monkeypatch):
    captured = {}

    class _FakePool:
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, func, *args):
            class _Fut:
                def result(self_inner):  # noqa: N805
                    return func(*args)

            return _Fut()

    monkeypatch.setattr("openscm_runner.run.ProcessPoolExecutor", _FakePool)

    openscm_runner.run.run(
        climate_models_cfgs={"DummyA": [{}], "DummyB": [{}]},
        scenarios=None,
        parallel_models=True,
        max_model_workers=1,
    )

    assert captured["max_workers"] == 1
