"""
Output writers for openscm-runner.

When :func:`openscm_runner.run.run` is called with the optional
``output_writer`` keyword argument, each model's results are streamed
to disk in per-(climate_model, scenario) chunks rather than
accumulated in memory and returned as a single :class:`scmdata.ScmRun`.

The intended use case is ensembles too large to fit in process memory
(e.g. Scenario Compass Initiative scale: of order 1600 scenarios x
many hundreds of ensemble members across several SCMs). The runner
returns a lightweight :class:`RunResult` carrying the written file
paths plus summary metadata so one-call workflows still work.

The writer is any callable with the signature
``writer(chunk: ScmRun, metadata: dict) -> pathlib.Path``. This module
ships :class:`NetCDFChunkWriter` as the recommended default; users
who need a different on-disk format can supply their own callable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import scmdata

LOGGER = logging.getLogger(__name__)


ChunkWriter = Callable[[scmdata.ScmRun, "dict[str, Any]"], Path]
"""
Type alias for the ``output_writer`` callable.

The runner calls the writer once per (climate_model, scenario) chunk.
``metadata`` is a dict with at least ``climate_model`` and ``scenario``
keys. The writer must return the :class:`pathlib.Path` it wrote to so
the runner can record it in the returned :class:`RunResult`.
"""


@dataclass
class RunResult:
    """
    Summary returned by :func:`openscm_runner.run.run` when an
    ``output_writer`` is supplied.

    Use :meth:`iter_chunks` to read chunks back lazily without holding
    the whole result in memory. :meth:`to_scmrun` is provided for
    convenience but defeats the OOM motivation the writer exists to
    solve; use it sparingly and only for small results.

    Attributes
    ----------
    chunk_paths : list of pathlib.Path
        Paths to the chunk files written, in dispatch order.
    models : list of str
        Climate models that produced output, deduplicated, in the order
        each first appeared.
    scenarios : list of str
        Distinct scenarios across all models, sorted.
    n_runs : int
        Total number of distinct model runs across all (climate_model,
        scenario, cfg) combinations.
    """

    chunk_paths: list[Path] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    scenarios: list[str] = field(default_factory=list)
    n_runs: int = 0

    @property
    def n_chunks(self) -> int:
        """Number of chunk files written."""
        return len(self.chunk_paths)

    @property
    def n_models(self) -> int:
        """Number of distinct climate models."""
        return len(self.models)

    @property
    def n_scenarios(self) -> int:
        """Number of distinct scenarios."""
        return len(self.scenarios)

    def iter_chunks(self) -> Iterator[scmdata.ScmRun]:
        """
        Yield each chunk back from disk as an :class:`scmdata.ScmRun`.

        Lazy: only one chunk is held in memory at a time.
        """
        for path in self.chunk_paths:
            yield scmdata.ScmRun.from_nc(path)

    def to_scmrun(self) -> scmdata.ScmRun:
        """
        Load every chunk and concatenate into a single
        :class:`scmdata.ScmRun`.

        Defeats the OOM motivation of streaming to disk in the first
        place. Provided for convenience in small-result workflows only.
        """
        return scmdata.run_append(list(self.iter_chunks()))


# Replace any character that is not a safe filename char with an underscore.
# We deliberately exclude spaces and slashes so paths stay portable.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(name: str) -> str:
    safe = _UNSAFE_FILENAME_CHARS.sub("_", str(name))
    return safe or "unnamed"


class NetCDFChunkWriter:
    """
    Stream per-(climate_model, scenario) chunks to netCDF on disk.

    Files are laid out as ``{output_dir}/{climate_model}/{scenario}.nc``.
    Both ``climate_model`` and ``scenario`` are passed through a filename
    sanitiser that replaces any character outside ``[A-Za-z0-9._-]`` with
    an underscore, so the layout is portable across filesystems.

    Existing files are silently overwritten so a re-run from the same
    ``output_dir`` produces a consistent state.

    The writer is picklable so it can be sent to
    :class:`concurrent.futures.ProcessPoolExecutor` workers when
    ``parallel_models=True`` is set on :func:`openscm_runner.run.run`.
    Each worker writes to a distinct file path (per-climate_model
    subdirectory plus per-scenario file), so writes do not contend.

    Requires the ``netCDF4`` package (``pip install netcdf4``); the
    import is checked at construction time so the error surfaces before
    a run starts.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory to write chunk files into. Subdirectories are created
        on first write as needed.
    dimensions : iterable of str
        Passed through to :meth:`scmdata.ScmRun.to_nc`. Defaults to
        ``("region",)``, matching scmdata's own default.
    extras : iterable of str
        Passed through to :meth:`scmdata.ScmRun.to_nc`. Defaults to
        ``("climate_model", "model", "run_id")`` so each chunk records
        the provenance and ensemble member identifiers needed to glue
        the full result back together.
    """

    def __init__(
        self,
        output_dir,
        dimensions=("region",),
        extras=("climate_model", "model", "run_id"),
    ):
        try:
            import netCDF4  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "NetCDFChunkWriter requires the netCDF4 package. "
                "Install with 'pip install netcdf4'."
            ) from exc

        self.output_dir = Path(output_dir)
        self.dimensions = tuple(dimensions)
        self.extras = tuple(extras)

    def __call__(self, chunk: scmdata.ScmRun, metadata: "dict[str, Any]") -> Path:
        """
        Write ``chunk`` to ``{output_dir}/{climate_model}/{scenario}.nc``.

        Parameters
        ----------
        chunk : ScmRun
            A single (climate_model, scenario) slice of the run output.
        metadata : dict
            At minimum ``climate_model`` and ``scenario`` keys. Values
            from ``metadata`` take precedence over what the chunk itself
            reports via ``get_unique_meta``, so the caller can label
            chunks deterministically even if a model emits multiple
            climate_model strings (e.g. version suffixes).

        Returns
        -------
        pathlib.Path
            The path that was written.
        """
        climate_model = metadata.get(
            "climate_model",
            chunk.get_unique_meta("climate_model", no_duplicates=True),
        )
        scenario = metadata.get(
            "scenario",
            chunk.get_unique_meta("scenario", no_duplicates=True),
        )

        model_dir = self.output_dir / _safe_name(climate_model)
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{_safe_name(scenario)}.nc"

        LOGGER.debug("Writing chunk for %s / %s to %s", climate_model, scenario, path)
        chunk.to_nc(path, dimensions=self.dimensions, extras=self.extras)
        return path
