"""
Thin wrapper to run emissions scenarios with simple climate models
"""
import importlib.metadata

from .output import NetCDFChunkWriter, RunResult

__version__ = importlib.metadata.version(__package__)

__all__ = ["NetCDFChunkWriter", "RunResult", "__version__"]
