"""
High-level run function
"""
import logging
from concurrent.futures import ProcessPoolExecutor

import scmdata

from .adapters import get_adapter
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


def _run_one_model(climate_model, cfgs, scenarios, output_variables, output_config):
    """
    Run a single climate model.

    Defined at module level so it can be pickled and dispatched to a
    :class:`concurrent.futures.ProcessPoolExecutor` worker when
    ``parallel_models=True`` is set on :func:`run`.
    """
    runner = get_adapter(climate_model)
    return runner.run(
        scenarios,
        cfgs,
        output_variables=output_variables,
        output_config=output_config,
    )


def run(
    climate_models_cfgs,
    scenarios,
    output_variables=("Surface Temperature",),
    out_config=None,
    parallel_models=True,
    max_model_workers=8,
):  # pylint: disable=W9006,too-many-arguments
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

    Returns
    -------
    :obj:`scmdata.ScmRun`
        Model output

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
            (climate_model, cfgs, scenarios, output_variables, output_config_cm)
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
