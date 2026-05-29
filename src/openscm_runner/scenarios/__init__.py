"""Scenario input loaders.

This subpackage provides helpers for reading scenario emissions files in
IAMC tabular format (CSV or Excel) and producing the canonical
``scmdata.ScmRun`` shape that openscm-runner adapters consume.

The same loader handles both RCMIP-style CSVs (``F-Gases|HFC|HFC125``
parenting) and Scenario Compass Initiative xlsx releases
(``HFC|HFC125`` parenting); per-species names are normalised to the
flat ``Emissions|HFC125`` form used internally.
"""

from openscm_runner.scenarios.iamc_loader import (
    CANONICAL_VARIABLES,
    load_iamc,
)
from openscm_runner.scenarios.rcmip3 import (
    CONSTRAINT_TARGETS,
    available_concentration_scenarios,
    available_scenarios,
    constraint_targets_dataframe,
    load_rcmip3_concentrations,
    load_rcmip3_emissions,
)

__all__ = [
    "CANONICAL_VARIABLES",
    "CONSTRAINT_TARGETS",
    "available_concentration_scenarios",
    "available_scenarios",
    "constraint_targets_dataframe",
    "load_iamc",
    "load_rcmip3_concentrations",
    "load_rcmip3_emissions",
]
