"""
In-tree workarounds for ``scmdata`` ↔ pandas 3.0 incompatibilities.

Applied once when :mod:`openscm_runner` is imported. The patches edit
``scmdata.groupby`` and ``scmdata._xarray`` in place and are idempotent;
re-importing this module is a no-op once the patches have been applied.

Two pandas 3.0 breakages are addressed. Both surface as soon as
:meth:`openscm_runner.run.run` returns more than one scenario worth of
output, since the runner's per-scenario chunking and any downstream
``ScmRun.to_nc`` write both pass through the affected scmdata code:

1. :func:`scmdata.groupby.RunGroupBy.__init__` uses
   ``numpy.issubdtype(col.dtype, numpy.number)`` to detect numeric
   columns. pandas 3.0 ships ``pd.StringDtype`` as the default for
   inferred string columns, and ``numpy.issubdtype`` raises
   ``TypeError: Cannot interpret '<StringDtype(...)>' as a data type``
   on those. Replace with :func:`pandas.api.types.is_numeric_dtype`,
   which short-circuits the conversion and returns ``False`` cleanly
   for ``StringDtype``.

2. :func:`scmdata._xarray._many_to_one` ends with
   ``checker.groupby(col2).count().max()[0]``. The chained ``.max()``
   returns a :class:`pandas.Series`; pandas 3.0 has removed positional
   indexing on label-indexed Series, so ``[0]`` raises ``KeyError: 0``.
   Replace with ``.iloc[0]``.

Both fixes have been proposed upstream as a single PR against
``openscm/scmdata``; this module exists so that openscm-runner users
on pandas 3.0 are not blocked while we wait for a release. Once a
patched scmdata is on PyPI the module (and the import-time
:func:`apply_scmdata_patches` call) can be removed and the version
pin tightened to whatever release contains the fixes.

Issue tracking removal of this shim:
https://github.com/benmsanderson/openscm-runner/issues/11
"""
from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


def apply_scmdata_patches() -> None:
    """
    Apply both pandas-3.0 compatibility patches to ``scmdata``.

    Idempotent: a second call returns immediately. Safe to call from
    module import. If ``scmdata`` is not installed, returns silently;
    the openscm-runner adapter import that actually needs it will
    raise a clear ImportError elsewhere.
    """
    try:
        import pandas as pd
        from scmdata import _xarray, groupby
    except ImportError:
        return

    if getattr(groupby, "_openscm_runner_patches_applied", False):
        return

    # --- Patch 1: scmdata.groupby uses np.issubdtype on column dtypes ---
    _orig_groupby_init = groupby.RunGroupBy.__init__

    def _patched_groupby_init(self, run, groups, na_fill_value=-10000):
        # Mirror the upstream constructor body exactly, swapping only
        # the dtype check that breaks on pandas 3.0's StringDtype.
        self.run = run
        self.group_keys = groups
        m = run.meta.reset_index(drop=True)
        self.na_fill_value = float(na_fill_value)
        if any(pd.api.types.is_numeric_dtype(m[c]) for c in m):
            if (m == na_fill_value).any(axis=None):
                raise ValueError(
                    "na_fill_value conflicts with data value. "
                    "Choose a na_fill_value not in meta"
                )
            m = m.fillna(na_fill_value)
        self._grouper = m.groupby(list(groups), group_keys=True)

    groupby.RunGroupBy.__init__ = _patched_groupby_init

    # --- Patch 2: scmdata._xarray._many_to_one uses Series positional [0] ---
    def _patched_many_to_one(df, col1, col2):
        # Upstream:
        #     max_count = checker.groupby(col2).count().max()[0]
        # which trips pandas 3.0's removal of positional indexing on
        # label-indexed Series. .iloc[0] is positional and safe.
        checker = df[[col1, col2]].drop_duplicates()
        max_count = checker.groupby(col2).count().max().iloc[0]
        if max_count < 1:  # pragma: no cover  # emergency valve, mirrors upstream
            raise AssertionError
        return max_count == 1

    _xarray._many_to_one = _patched_many_to_one

    groupby._openscm_runner_patches_applied = True
    LOGGER.debug(
        "openscm_runner: applied pandas 3.0 compatibility patches to "
        "scmdata.groupby and scmdata._xarray"
    )
