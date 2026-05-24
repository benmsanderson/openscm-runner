"""
High-level run function
"""
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import scmdata

from .adapters import get_adapter
from .output import RunResult
from .progress import progress

LOGGER = logging.getLogger(__name__)


def _check_out_config(out_config, climate_models_cfgs):
    if out_config is not None:
        unknown_models = set(out_config.keys()) - set(climate_models_cfgs.keys())
        if unknown_models:
            LOGGER.warning(
                "Found model(s) in `out_config` which are not in "
                "`climate_models_cfgs`: %s",
                unknown_models,
            )

        for key, value in out_config.items():
            if not isinstance(value, tuple):
                raise TypeError(
                    f"`out_config` values must be tuples, this isn't the case for "
                    f"climate_model: '{key}'"
                )


def _stream_to_writer(climate_model, model_res, output_writer):
    """
    Split a per-model result into per-scenario chunks for the writer.

    Each chunk is dispatched to ``output_writer`` in turn. Returns a
    list of dicts (one per chunk) carrying the written path, the
    climate_model and scenario it represents, and the number of
    distinct ensemble members in that chunk. The runner aggregates
    these into a :class:`RunResult` summary.
    """
    records = []
    for chunk in model_res.groupby("scenario"):
        scen_name = chunk.get_unique_meta("scenario", no_duplicates=True)
        chunk_runs = (
            len(chunk["run_id"].unique()) if "run_id" in chunk.meta.columns else 1
        )
        path = output_writer(
            chunk, {"climate_model": climate_model, "scenario": scen_name}
        )
        records.append(
            {
                "path": Path(path),
                "climate_model": climate_model,
                "scenario": scen_name,
                "n_runs": chunk_runs,
            }
        )
    return records


def _run_one_model(  # noqa: PLR0913
    climate_model, cfgs, scenarios, output_variables, output_config, output_writer
):
    """
    Run a single climate model.

    Defined at module level so it can be pickled and dispatched to a
    :class:`concurrent.futures.ProcessPoolExecutor` worker when
    ``parallel_models=True`` is set on :func:`run`.

    Returns the full per-model :class:`scmdata.ScmRun` when
    ``output_writer`` is ``None`` (legacy behaviour). When
    ``output_writer`` is supplied, returns a list of chunk records
    (see :func:`_stream_to_writer`). Splitting and writing happen
    inside the worker so the full per-model result never has to be
    returned to the parent process.
    """
    runner = get_adapter(climate_model)
    model_res = runner.run(
        scenarios,
        cfgs,
        output_variables=output_variables,
        output_config=output_config,
    )
    if output_writer is None:
        return model_res
    return _stream_to_writer(climate_model, model_res, output_writer)


def _aggregate_chunk_records(per_model_records):
    """
    Build a :class:`RunResult` from the per-worker chunk records.

    Models are kept in first-seen order; scenarios are deduplicated and
    sorted; total run count is summed across chunks.
    """
    chunk_paths = []
    scenarios_seen = set()
    models_seen = []
    n_runs = 0
    for model_records in per_model_records:
        for rec in model_records:
            chunk_paths.append(rec["path"])
            scenarios_seen.add(rec["scenario"])
            if rec["climate_model"] not in models_seen:
                models_seen.append(rec["climate_model"])
            n_runs += rec["n_runs"]
    return RunResult(
        chunk_paths=chunk_paths,
        models=models_seen,
        scenarios=sorted(scenarios_seen),
        n_runs=n_runs,
    )


def run(
    climate_models_cfgs,
    scenarios,
    output_variables=("Surface Temperature",),
    out_config=None,
    parallel_models=True,
    max_model_workers=8,
    output_writer=None,
):  # pylint: disable=W9006,too-many-arguments,too-many-branches
    """
    Run a number of climate models over a number of scenarios

    Parameters
    ----------
    climate_models_cfgs : dict[str: list]
        Dictionary where each key is a model and each value is the configs
        with which to run the model. The configs are passed to the model
        adapter.

    scenarios : :obj:`pyam.IamDataFrame`
        Scenarios to run

    output_variables : list[str]
        Variables to include in the output

    out_config : dict[str: tuple of str]
        Dictionary where each key is a model and each value is a tuple of
        configuration values to include in the output's metadata.

    parallel_models : bool
        If ``True`` (default), dispatch the requested climate models to a
        top-level :class:`concurrent.futures.ProcessPoolExecutor` so they
        run concurrently. If ``False``, run them serially in the calling
        process; useful for debugging and for callers that depend on a
        specific model dispatch order. Has no effect when only one model
        is requested.

    max_model_workers : int
        Cap on the number of top-level worker processes used when
        ``parallel_models=True``. The actual number used is
        ``min(len(climate_models_cfgs), max_model_workers)``. The cap
        exists so that a run with many models does not spawn one worker
        per model, each of which then spawns its own per-adapter pool.
        Per-adapter parallelism is controlled separately by each
        adapter's worker-count configuration.

    output_writer : callable, optional
        If supplied, each model's result is split into per-scenario
        chunks and dispatched to ``output_writer`` as soon as that model
        finishes, rather than accumulated in memory and returned as one
        :class:`scmdata.ScmRun`. The writer's signature is
        ``writer(chunk: ScmRun, metadata: dict) -> pathlib.Path``;
        ``metadata`` carries at least ``climate_model`` and ``scenario``.
        See :mod:`openscm_runner.output` for the shipped
        :class:`~openscm_runner.output.NetCDFChunkWriter`.

        When set, the cross-model meta-column consistency check is
        skipped (since results have already been written to disk) and
        the return type changes; see below.

    Returns
    -------
    :obj:`scmdata.ScmRun`
        Model output (default, when ``output_writer`` is ``None``).

    :obj:`openscm_runner.output.RunResult`
        Summary of written chunks (when ``output_writer`` is set).
        Carries the list of written paths plus model/scenario/run
        counts. Use :meth:`~openscm_runner.output.RunResult.iter_chunks`
        to read chunks back lazily.

    Raises
    ------
    KeyError
        ``out_config`` has keys which are not in ``climate_models_cfgs``

    TypeError
        A value in ``out_config`` is not a :obj:`tuple`
    """
    _check_out_config(out_config, climate_models_cfgs)

    model_tasks = []
    for climate_model, cfgs in climate_models_cfgs.items():
        if out_config is not None and climate_model in out_config:
            output_config_cm = out_config[climate_model]
            LOGGER.debug(
                "Using output config: %s for %s", output_config_cm, climate_model
            )
        else:
            LOGGER.debug("No output config for %s", climate_model)
            output_config_cm = None
        model_tasks.append(
            (
                climate_model,
                cfgs,
                scenarios,
                output_variables,
                output_config_cm,
                output_writer,
            )
        )

    if parallel_models and len(model_tasks) > 1:
        n_workers = min(len(model_tasks), max_model_workers)
        LOGGER.info(
            "Running %d climate models in parallel with %d top-level workers",
            len(model_tasks),
            n_workers,
        )
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_run_one_model, *task) for task in model_tasks]
            res = [f.result() for f in futures]
    else:
        if parallel_models:
            LOGGER.debug(
                "Only one climate model requested, dispatching serially"
            )
        else:
            LOGGER.info(
                "Running %d climate models serially (parallel_models=False)",
                len(model_tasks),
            )
        res = [
            _run_one_model(*task)
            for task in progress(model_tasks, desc="Climate models")
        ]

    if output_writer is not None:
        return _aggregate_chunk_records(res)

    for i, model_res in enumerate(res):
        if i < 1:
            key_meta = set(model_res.meta.columns.tolist())

        model_meta = set(model_res.meta.columns.tolist())
        climate_model = model_res.get_unique_meta("climate_model")
        if model_meta != key_meta:
            raise AssertionError(
                f"{climate_model} meta: {model_meta}, expected meta: {key_meta}"
            )

    if len(res) == 1:
        LOGGER.info("Only one model run, returning its results")
        scmdf = res[0]
    else:
        LOGGER.info("Appending model results")
        scmdf = scmdata.run_append(res)

    return scmdf
