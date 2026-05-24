"""
Thin wrapper to run emissions scenarios with simple climate models
"""
import importlib.metadata

from openscm_runner._scmdata_patches import apply_scmdata_patches

# Patch scmdata before anything else that might import it: the patches
# edit module attributes in place, and downstream imports cache the
# unpatched functions if we let them resolve first.
apply_scmdata_patches()

from .output import NetCDFChunkWriter, RunResult  # noqa: E402

__version__ = importlib.metadata.version(__package__)

__all__ = ["NetCDFChunkWriter", "RunResult", "__version__"]
