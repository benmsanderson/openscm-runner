"""In-tree workarounds for ``scmdata`` ↔ pandas-3.0 / xarray-2025+ breakages.

Applied once when :mod:`openscm_runner` is imported. The patches edit
``scmdata.groupby``, ``scmdata._xarray``, ``scmdata.run``, and
``scmdata.netcdf`` in place and are idempotent; re-importing this
module is a no-op once the patches have been applied.

Four breakages are addressed:

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

3. :meth:`scmdata.run.ScmRun.convert_unit` contains an inner
   ``apply_units`` closure that does ``group._df.values[:] = ...``.
   In pandas 3.0, ``DataFrame.values`` always returns a read-only numpy
   array (the change applies even to ``DataFrame.copy()`` results), so
   the in-place write raises ``ValueError: assignment destination is
   read-only``.  Patch ``convert_unit`` to use ``group._df.iloc[:, :]``
   for the assignment, which does not go through the read-only array
   path.

4. :func:`scmdata.netcdf._read_nc` calls
   ``xr.load_dataset(fname, use_cftime=True)``. xarray 2025+
   deprecates the bare ``use_cftime`` kwarg in favour of passing a
   :class:`xarray.coders.CFDatetimeCoder` via ``decode_times``. The
   deprecation fires a FutureWarning on every ``ScmRun.from_nc()``
   call, which floods the comparison-notebook output. Route the read
   through the new API.

All four fixes have been proposed upstream as a single PR against
``openscm/scmdata``; this module exists so that openscm-runner users
on pandas 3.0 are not blocked while we wait for a release. Once a
patched scmdata is on PyPI the module (and the import-time
:func:`apply_scmdata_patches` call) can be removed and the version
pin tightened to whatever release contains the fixes.

Issue tracking removal of this shim:
https://github.com/benmsanderson/openscm-runner/issues/10
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

    # --- Patch 3: convert_unit inner closure uses read-only .values[:] ---
    # scmdata.run imports _get_target and run_append as module-level names;
    # reach them via the module reference so the patch stays in sync with
    # whatever version is actually installed.
    import scmdata.run as _scmdata_run

    _orig_convert_unit = _scmdata_run.ScmRun.convert_unit

    def _patched_convert_unit(self, unit, context=None, inplace=False, **kwargs):
        # Identical to the upstream implementation in scmdata 0.18.0 except
        # the inner apply_units closure replaces
        #   group._df.values[:] = uc.convert_from(group._df.values)
        # with
        #   group._df.iloc[:, :] = uc.convert_from(group._df.values)
        # because pandas 3.0 makes DataFrame.values always read-only.
        from scmdata.units import UnitConverter as _UC

        _get_target = _scmdata_run._get_target
        _run_append = _scmdata_run.run_append

        ret = _get_target(self, inplace)

        to_convert_filtered = ret.filter(**kwargs, log_if_empty=False)
        to_not_convert_filtered = ret.filter(**kwargs, keep=False, log_if_empty=False)

        already_correct_unit = to_convert_filtered.filter(unit=unit, log_if_empty=False)
        if (
            "unit_context" in already_correct_unit.meta_attributes
            and not already_correct_unit.empty
        ):
            self._check_unit_context(already_correct_unit, context)

        to_convert = to_convert_filtered.filter(
            unit=unit, log_if_empty=False, keep=False
        )
        to_not_convert = _run_append([to_not_convert_filtered, already_correct_unit])

        if "unit_context" in to_convert.meta_attributes and not to_convert.empty:
            self._check_unit_context(to_convert, context)

        if context is not None:
            to_convert["unit_context"] = context

        if "unit_context" not in to_not_convert.meta_attributes and context is not None:
            to_not_convert["unit_context"] = None

        def apply_units(group):
            orig_unit = group.get_unique_meta("unit", no_duplicates=True)
            uc = _UC(orig_unit, unit, context=context)
            group._df.iloc[:, :] = uc.convert_from(group._df.values)
            group["unit"] = unit
            return group

        ret = to_convert
        if not to_convert.empty:
            ret = ret.groupby("unit").apply(apply_units)

        ret = _run_append([ret, to_not_convert], inplace=inplace)
        return ret

    _scmdata_run.ScmRun.convert_unit = _patched_convert_unit

    # --- Patch 4: scmdata.netcdf._read_nc uses deprecated use_cftime kwarg ---
    # Upstream:
    #     loaded = xr.load_dataset(fname, use_cftime=True)
    # xarray 2025+ deprecates the bare `use_cftime` kwarg in favour of
    # passing a CFDatetimeCoder via decode_times. The deprecation
    # currently emits a FutureWarning on every from_nc() call (every
    # cell in the comparison notebooks), spamming the output. Mirror
    # the upstream body but route through the new API.
    try:
        import xarray as xr
        from scmdata import netcdf as _scmdata_netcdf

        def _patched_read_nc(cls, fname):
            time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            loaded = xr.load_dataset(fname, decode_times=time_coder)
            dataframe = loaded.to_dataframe()
            dataframe = _scmdata_netcdf._reshape_to_scmrun_dataframe(
                dataframe, loaded
            )
            return _scmdata_netcdf._convert_to_cls_and_add_metadata(
                dataframe, loaded, cls
            )

        _scmdata_netcdf._read_nc = _patched_read_nc
    except (ImportError, AttributeError):
        # xarray older than 2024.09 doesn't have CFDatetimeCoder; leave
        # the warning in place rather than break compatibility.
        pass

    groupby._openscm_runner_patches_applied = True
    LOGGER.debug(
        "openscm_runner: applied pandas 3.0 compatibility patches to "
        "scmdata.groupby, scmdata._xarray, and scmdata.netcdf"
    )
