"""
Thin wrapper to run emissions scenarios with simple climate models
"""
import importlib.metadata

from openscm_runner._scmdata_patches import apply_scmdata_patches

__version__ = importlib.metadata.version(__package__)

# scmdata 0.18.x is incompatible with pandas 3.0 in two places that
# openscm-runner exercises on every multi-scenario run. Patch on
# import so consumers do not have to set pandas options themselves.
# See openscm_runner._scmdata_patches for the diff and removal plan.
apply_scmdata_patches()
