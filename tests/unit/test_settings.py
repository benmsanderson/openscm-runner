import logging
import multiprocessing
import os
from unittest import mock

import pytest

from openscm_runner import settings
from openscm_runner.settings import ConfigLoader, get_worker_count


@mock.patch("openscm_runner.settings.dotenv_values")
def test_config_empty(mock_dotenv, monkeypatch):
    mock_dotenv.return_value = {}
    # This assumes that the envvar TEST_ENV_VAR does not exist
    config = ConfigLoader()
    assert not config.is_loaded

    with pytest.raises(KeyError, match="TEST_ENV_VAR"):
        config["TEST_ENV_VAR"]

    with pytest.raises(KeyError, match="TEST_ENV_VAR"):
        config["test_ENV_var"]


@mock.patch("openscm_runner.settings.dotenv_values")
def test_config_envvar(mock_dotenv, monkeypatch):
    mock_dotenv.return_value = {}
    config = ConfigLoader()

    res = config["PATH"]
    assert res == os.environ["PATH"]

    res = config["paTh"]
    assert res == os.environ["PATH"]


@mock.patch("openscm_runner.settings.dotenv_values")
def test_config_with_dotenv(mock_dotenv, monkeypatch):
    monkeypatch.setenv("TEST", "env_var")
    mock_dotenv.return_value = {"TEST": "env_file", "EXTRA": "yay"}

    config = ConfigLoader()

    assert config["TEST"] == "env_var"
    assert config["EXTRA"] == "yay"
    assert config["extra"] == "yay"

    with pytest.raises(KeyError, match="NOT_THERE"):
        config["not_there"]

    assert config.is_loaded
    mock_dotenv.assert_called_once()
    assert config._config == mock_dotenv.return_value


@mock.patch("openscm_runner.settings.dotenv_values")
def test_config_get(mock_dotenv, monkeypatch):
    monkeypatch.setenv("TEST", "env_var")
    mock_dotenv.return_value = {}

    config = ConfigLoader()

    assert config.get("TEST", "default") == "env_var"
    assert config.get("UNKNOWN_CONFIG", "default") == "default"


_WORKER_ENV_VARS = (
    "FAIR_WORKER_NUMBER",
    "MAGICC_WORKER_NUMBER",
    "CICEROSCM_WORKER_NUMBER",
    "SLURM_CPUS_PER_TASK",
    "OMP_NUM_THREADS",
)


@pytest.fixture()
def fresh_config(monkeypatch):
    """
    Swap the module-level config for a fresh ConfigLoader with empty
    dotenv and clear any worker-count env vars, so each test starts from
    a known state. The module-level get_worker_count resolves config via
    settings.config at call time, so monkeypatching the attribute is
    sufficient.
    """
    monkeypatch.setattr(settings, "dotenv_values", lambda *args, **kwargs: {})
    monkeypatch.setattr(settings, "config", ConfigLoader())
    for var in _WORKER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_get_worker_count_falls_back_to_cpu_count(fresh_config):
    assert get_worker_count("FAIR_WORKER_NUMBER") == multiprocessing.cpu_count()


def test_get_worker_count_uses_omp_num_threads(fresh_config, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert get_worker_count("FAIR_WORKER_NUMBER") == 3


def test_get_worker_count_prefers_slurm_over_omp(fresh_config, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "5")
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert get_worker_count("FAIR_WORKER_NUMBER") == 5


def test_get_worker_count_prefers_override_over_slurm(fresh_config, monkeypatch):
    monkeypatch.setenv("FAIR_WORKER_NUMBER", "2")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("OMP_NUM_THREADS", "12")
    assert get_worker_count("FAIR_WORKER_NUMBER") == 2


def test_get_worker_count_at_least_one(fresh_config, monkeypatch):
    monkeypatch.setenv("FAIR_WORKER_NUMBER", "0")
    assert get_worker_count("FAIR_WORKER_NUMBER") == 1


def test_get_worker_count_ignores_garbage_values(fresh_config, monkeypatch, caplog):
    monkeypatch.setenv("FAIR_WORKER_NUMBER", "not_a_number")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    with caplog.at_level(logging.WARNING, logger="openscm_runner.settings"):
        result = get_worker_count("FAIR_WORKER_NUMBER")
    assert result == 4
    assert "Ignoring non-integer worker count from FAIR_WORKER_NUMBER" in caplog.text
