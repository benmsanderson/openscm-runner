"""
Tests for the pandas 3.0 compatibility patches against ``scmdata``.

The patches are applied at openscm_runner package import time
(see :mod:`openscm_runner._scmdata_patches`), so all that's needed
is to drive the previously-broken code paths and assert they no
longer raise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Importing openscm_runner triggers apply_scmdata_patches() once.
import openscm_runner  # noqa: F401
from openscm_runner._scmdata_patches import apply_scmdata_patches


@pytest.fixture
def small_scmrun():
    """Two scenarios x two variables, with object-dtype strings."""
    import scmdata

    df = pd.DataFrame(
        {
            "model": ["M"] * 4,
            "scenario": ["A", "A", "B", "B"],
            "region": ["World"] * 4,
            "variable": [
                "Emissions|CO2",
                "Emissions|CH4",
                "Emissions|CO2",
                "Emissions|CH4",
            ],
            "unit": ["Gt CO2/yr", "Mt CH4/yr", "Gt CO2/yr", "Mt CH4/yr"],
            2020: [40.0, 350.0, 50.0, 400.0],
            2050: [30.0, 300.0, 60.0, 450.0],
        }
    )
    return scmdata.ScmRun(df)


def test_apply_scmdata_patches_is_idempotent():
    apply_scmdata_patches()
    apply_scmdata_patches()  # should be a no-op, no exception


def test_groupby_does_not_crash_on_stringdtype_meta(small_scmrun):
    # The bug: scmdata.groupby called np.issubdtype on a StringDtype
    # column dtype, which raised TypeError. After the patch the call
    # routes through pd.api.types.is_numeric_dtype and runs cleanly.
    grouped = list(small_scmrun.groupby("scenario"))
    assert sorted(g.get_unique_meta("scenario", no_duplicates=True) for g in grouped) == ["A", "B"]


def test_to_xarray_does_not_crash_on_series_positional_indexing(small_scmrun):
    # The bug: scmdata._xarray._many_to_one ended with .max()[0],
    # which raised KeyError on pandas 3 because Series positional
    # integer indexing was removed. After the patch it uses .iloc[0].
    ds = small_scmrun.to_xarray(dimensions=("region",), extras=("scenario",))
    # to_xarray returns an xarray.Dataset; just confirm it's that shape
    # and that the data is roundtrippable in essentials.
    assert "Emissions|CO2" in ds.data_vars
    assert "Emissions|CH4" in ds.data_vars


def test_groupby_still_detects_numeric_columns_after_patch():
    # Sanity check: the patched is_numeric_dtype shortcut should
    # still flag numeric meta columns (and therefore still apply
    # the fillna(na_fill_value) branch). We do this via the
    # is_numeric_dtype check directly rather than ScmRun, since
    # ScmRun's meta columns are all strings by design.
    s_num = pd.Series([1, 2, 3])
    s_str_object = pd.Series(["a", "b", "c"], dtype="object")
    s_str_pdstr = pd.Series(["a", "b", "c"], dtype="string")
    assert pd.api.types.is_numeric_dtype(s_num) is True
    assert pd.api.types.is_numeric_dtype(s_str_object) is False
    assert pd.api.types.is_numeric_dtype(s_str_pdstr) is False
    # And the original buggy call would raise on the StringDtype one,
    # which is the whole point of the patch.
    with pytest.raises(TypeError):
        np.issubdtype(s_str_pdstr.dtype, np.number)
