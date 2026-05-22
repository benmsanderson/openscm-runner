"""
Module supporting the CICERO-SCM v2.x Python adapter.

Sibling to :mod:`openscm_runner.adapters.ciceroscm_py_adapter` (which
wraps CICERO-SCM v1.1.x via a per-config loop). This package targets
v2.1.0's native parameter-distribution API (``DistributionRun``) and
exposes the v2.x ensemble as a single adapter call.
"""
from .ciceroscmpy2_adapter import CICEROSCMPY2  # noqa: F401
