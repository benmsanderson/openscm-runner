import pandas as pd
import pytest
import scmdata

from openscm_runner.output import (
    NetCDFChunkWriter,
    RunResult,
    _safe_name,
)


pytest.importorskip("netCDF4")


def _make_chunk(climate_model, scenario, run_ids=(0,)):
    rows = []
    index = []
    for run_id in run_ids:
        rows.append([1.0 + run_id, 2.0 + run_id])
        index.append(
            (
                climate_model,
                scenario,
                "test_iam",
                "World",
                "Surface Temperature",
                "K",
                run_id,
            )
        )
    return scmdata.ScmRun(
        pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(
                index,
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


def test_safe_name_replaces_unsafe_chars():
    assert _safe_name("ssp 1/2.6") == "ssp_1_2.6"
    assert _safe_name("MAGICC7.5.3") == "MAGICC7.5.3"
    assert _safe_name("") == "unnamed"
    assert _safe_name("@@@") == "_"


def test_run_result_summary_properties():
    result = RunResult(
        chunk_paths=["/a", "/b", "/c"],
        models=["FaIR", "MAGICC"],
        scenarios=["ssp126", "ssp245", "ssp370"],
        n_runs=42,
    )
    assert result.n_chunks == 3
    assert result.n_models == 2
    assert result.n_scenarios == 3
    assert result.n_runs == 42


def test_netcdf_chunk_writer_writes_expected_path(tmp_path):
    writer = NetCDFChunkWriter(tmp_path)
    chunk = _make_chunk("FaIR", "ssp126")

    path = writer(chunk, {"climate_model": "FaIR", "scenario": "ssp126"})

    assert path == tmp_path / "FaIR" / "ssp126.nc"
    assert path.exists()


def test_netcdf_chunk_writer_sanitises_path_components(tmp_path):
    writer = NetCDFChunkWriter(tmp_path)
    chunk = _make_chunk("FaIRv2.2.4", "ssp 1/2.6")

    path = writer(
        chunk, {"climate_model": "FaIRv2.2.4", "scenario": "ssp 1/2.6"}
    )

    assert path == tmp_path / "FaIRv2.2.4" / "ssp_1_2.6.nc"
    assert path.exists()


def test_netcdf_chunk_writer_round_trip_preserves_data(tmp_path):
    writer = NetCDFChunkWriter(tmp_path)
    chunk = _make_chunk("FaIR", "ssp126", run_ids=(0, 1, 2))

    path = writer(chunk, {"climate_model": "FaIR", "scenario": "ssp126"})
    reread = scmdata.ScmRun.from_nc(path)

    # The round-trip preserves variable, unit, region, run_id and the
    # data values. Other meta (climate_model, model, scenario) round-
    # trips via the extras configured on the writer.
    assert set(reread["variable"]) == {"Surface Temperature"}
    assert sorted(reread["run_id"].tolist()) == [0, 1, 2]
    assert reread.get_unique_meta("climate_model", no_duplicates=True) == "FaIR"
    assert reread.get_unique_meta("scenario", no_duplicates=True) == "ssp126"


def test_netcdf_chunk_writer_falls_back_to_chunk_metadata(tmp_path):
    writer = NetCDFChunkWriter(tmp_path)
    chunk = _make_chunk("FaIR", "ssp126")

    # No explicit metadata dict entries; the writer should read climate_model
    # and scenario from the chunk itself.
    path = writer(chunk, {})

    assert path == tmp_path / "FaIR" / "ssp126.nc"
    assert path.exists()


def test_run_result_iter_chunks(tmp_path):
    writer = NetCDFChunkWriter(tmp_path)
    p1 = writer(
        _make_chunk("FaIR", "ssp126"),
        {"climate_model": "FaIR", "scenario": "ssp126"},
    )
    p2 = writer(
        _make_chunk("FaIR", "ssp245"),
        {"climate_model": "FaIR", "scenario": "ssp245"},
    )

    result = RunResult(
        chunk_paths=[p1, p2],
        models=["FaIR"],
        scenarios=["ssp126", "ssp245"],
        n_runs=2,
    )

    loaded = list(result.iter_chunks())
    assert len(loaded) == 2
    assert set(s for chunk in loaded for s in chunk["scenario"]) == {
        "ssp126",
        "ssp245",
    }
