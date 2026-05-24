"""
RCMIP3 scenario input loader.

Resolves RCMIP3 protocol scenario IDs (``ssp245``, ``esm-flat10-zec``,
``scen7-VL``, ``historical``, …) against the upstream archives that
publish their emissions, downloads/caches the underlying CSVs, and
returns a single :class:`scmdata.ScmRun` whose variable names match
the canonical openscm-runner convention used by the adapter layer.

Two upstream sources are wired in:

* **RCMIP3 protocol inputs**
  (``RCMIP3_input_datafiles/rcmip_phase3_emissions_v1.1.7.csv`` from
  ``gitlab.com/rcmip/rcmip-phase-3``). Covers the CMIP6-era SSPs, the
  ``esm-flat*`` idealised family, historical (CMIP6 + CMIP7 vintage),
  the attribution single-forcing runs, bell-shaped and pulse
  experiments. As of release ``v1.1.7`` the file does **not** yet
  contain the CMIP7 ``scen7-*`` scenarios; those use the second source.

* **CMIP7 ScenarioMIP emissions**
  (``data/continuous_emissions_timeseries_1750_2500.csv`` from
  ``github.com/benmsanderson/scenariomip-paper-plots``, the
  reproducibility companion to van Vuuren et al. 2026). Covers the
  seven ``scen7-{VL,LN,L,ML,M,H,HL}`` markers with their marker IAMs
  baked in.

Both sources publish emissions in the same IAMC-style dialect
(``Emissions|CO2|Energy and Industrial Processes`` /
``Emissions|CO2|AFOLU``, ``Emissions|HFC|HFC125``, …), so the
harmonisation step is exactly :func:`openscm_runner.scenarios.load_iamc`
without any new rename rules.

The constraint-target table from Table 3 of the RCMIP3 protocol paper
(Romero-Prieto et al. 2025) is re-exported here as
:data:`CONSTRAINT_TARGETS` for downstream evaluation notebooks.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

import pandas as pd
import scmdata

from openscm_runner.scenarios.iamc_loader import load_iamc

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Upstream source descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Source:
    """A remote scenario CSV plus the local cache filename to use."""

    name: str
    url: str
    cache_filename: str


_SOURCE_RCMIP3 = _Source(
    name="rcmip3-v1.1.7",
    url=(
        "https://gitlab.com/rcmip/rcmip-phase-3/-/raw/main/"
        "RCMIP3_input_datafiles/rcmip_phase3_emissions_v1.1.7.csv"
    ),
    cache_filename="rcmip_phase3_emissions_v1.1.7.csv",
)

_SOURCE_SCEN7 = _Source(
    name="scenariomip-paper-plots-v1.0.0",
    url=(
        "https://raw.githubusercontent.com/benmsanderson/"
        "scenariomip-paper-plots/main/data/"
        "continuous_emissions_timeseries_1750_2500.csv"
    ),
    cache_filename="scen7_continuous_emissions_1750_2500.csv",
)


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ScenarioSpec:
    """How to fetch one protocol scenario from one upstream CSV.

    Attributes
    ----------
    source
        Which upstream CSV holds this scenario.
    source_scenario
        The scenario string as it appears in that CSV's ``scenario`` column.
    source_model
        Optional IAM-marker filter for sources that publish one scenario
        per IAM (the scen7 CSV does this; the RCMIP3 CSV typically has
        only one model row per scenario already).
    """

    source: _Source
    source_scenario: str
    source_model: Optional[str] = None


# Scenarios from the RCMIP3 v1.1.7 emissions CSV.
# Per the RCMIP3 README, ``esm-sspXXX`` and ``esm-allGHG-sspXXX`` share
# emissions filed under plain ``sspXXX``; we expose only the plain
# ``sspXXX`` protocol IDs here and leave the adapter to pick which
# species subset (CO2-only vs all-GHG) it drives.
_RCMIP3_SSPS = (
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp434",
    "ssp460",
    "ssp534-over",
    "ssp585",
)
_RCMIP3_FLAT_BASE = ("esm-flat7.5", "esm-flat10", "esm-flat20")
_RCMIP3_FLAT_VARIANTS = ("", "-zec", "-cdr", "-nz", "-rev")
_RCMIP3_FLAT = tuple(
    f"{base}{suffix}" for base in _RCMIP3_FLAT_BASE for suffix in _RCMIP3_FLAT_VARIANTS
)
_RCMIP3_HISTORICAL = (
    "historical",
    "historical-cmip6",
    "hist-aer",
    "hist-GHG",
    "hist-CO2",
)


# scen7-* → (long scenario name, marker IAM) per the van Vuuren et al.
# 2026 ScenarioMIP paper. Matches ``scenario_model_match`` in
# benmsanderson/scenariomip-paper-plots:scripts/plotting.py.
_SCEN7_MARKERS = {
    "scen7-VL": ("SSP1 - Very Low Emissions", "REMIND-MAgPIE 3.5-4.11"),
    "scen7-LN": ("SSP2 - Low Overshoot_a", "AIM 3.0"),
    "scen7-L": ("SSP2 - Low Emissions", "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12"),
    "scen7-ML": ("SSP2 - Medium-Low Emissions", "COFFEE 1.6"),
    "scen7-M": ("SSP2 - Medium Emissions", "IMAGE 3.4"),
    "scen7-H": ("SSP3 - High Emissions", "GCAM 8s"),
    "scen7-HL": ("SSP5 - Medium-Low Emissions_a", "WITCH 6.0"),
}


def _build_registry() -> dict[str, _ScenarioSpec]:
    reg: dict[str, _ScenarioSpec] = {}
    for s in _RCMIP3_SSPS + _RCMIP3_FLAT + _RCMIP3_HISTORICAL:
        reg[s] = _ScenarioSpec(source=_SOURCE_RCMIP3, source_scenario=s)
    for protocol_id, (source_scen, source_model) in _SCEN7_MARKERS.items():
        reg[protocol_id] = _ScenarioSpec(
            source=_SOURCE_SCEN7,
            source_scenario=source_scen,
            source_model=source_model,
        )
    return reg


_REGISTRY: dict[str, _ScenarioSpec] = _build_registry()


def available_scenarios() -> tuple[str, ...]:
    """All protocol scenario IDs this loader knows how to resolve."""
    return tuple(sorted(_REGISTRY))


# ---------------------------------------------------------------------------
# Constraint targets (Table 3 of Romero-Prieto et al. 2025, RCMIP3 paper)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ConstraintTarget:
    variable: str
    unit: str
    source: str
    base_period: Optional[tuple[int, int]]
    constraint_period: tuple[int, int]
    central: float
    lower: float
    upper: float


CONSTRAINT_TARGETS: dict[str, _ConstraintTarget] = {
    "GMST_anomaly": _ConstraintTarget(
        variable="Surface Air Temperature Change",
        unit="K",
        source="IGCC2024",
        base_period=(1850, 1900),
        constraint_period=(2014, 2023),
        central=1.19,
        lower=1.0,
        upper=1.4,
    ),
    "OHC_change": _ConstraintTarget(
        variable="Heat Content|Ocean",
        unit="ZJ",
        source="IGCC2024",
        base_period=(1971, 1971),
        constraint_period=(2020, 2020),
        central=435.1,
        lower=315.8,
        upper=560.5,
    ),
    "CO2_concentration": _ConstraintTarget(
        variable="Atmospheric Concentrations|CO2",
        unit="ppm",
        source="NOAA GML",
        base_period=None,
        constraint_period=(2014, 2023),
        central=408.65,
        lower=407.05,
        upper=410.25,
    ),
    "carbon_flux_to_oceans": _ConstraintTarget(
        variable="Net Flux to Atmosphere|CO2|Ocean",
        unit="PgC/yr",
        source="GCB2024",
        base_period=None,
        constraint_period=(2014, 2023),
        central=2.9,
        lower=2.24,
        upper=3.56,
    ),
    "carbon_flux_to_land": _ConstraintTarget(
        variable="Net Flux to Atmosphere|CO2|Land",
        unit="PgC/yr",
        source="GCB2024",
        base_period=None,
        constraint_period=(2014, 2023),
        central=3.2,
        lower=1.72,
        upper=4.68,
    ),
    "aerosol_ERF": _ConstraintTarget(
        variable="Effective Radiative Forcing|Anthropogenic|Aerosol",
        unit="W/m^2",
        source="AR6",
        base_period=(1850, 1900),
        constraint_period=(2005, 2014),
        central=-1.3,
        lower=-2.0,
        upper=-0.6,
    ),
}


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


DEFAULT_CACHE_DIR = Path("configurations") / "rcmip3"


def load_rcmip3_emissions(
    scenarios: Sequence[str],
    *,
    cache_dir: Optional[Path] = None,
    download_if_missing: bool = True,
) -> scmdata.ScmRun:
    """
    Resolve and load emissions for one or more RCMIP3 protocol scenarios.

    Parameters
    ----------
    scenarios
        Protocol scenario IDs (e.g. ``"ssp245"``, ``"esm-flat10-zec"``,
        ``"scen7-VL"``). See :func:`available_scenarios`.
    cache_dir
        Local directory under which to cache the upstream CSVs. Defaults
        to ``configurations/rcmip3/`` relative to the current working
        directory (which matches openscm-runner's existing convention
        for adapter calibration bundles).
    download_if_missing
        When True (default), fetch any missing source CSV from its
        upstream URL on demand. Set to False in CI / offline contexts
        where the cache has been pre-populated.

    Returns
    -------
    scmdata.ScmRun
        One timeseries per (scenario, variable) pair with variable names
        harmonised to the canonical adapter dialect (e.g.
        ``Emissions|CO2|MAGICC Fossil and Industrial``). The
        ``scenario`` meta column carries the protocol ID
        (``"ssp245"``, not ``"SSP2 - Medium Emissions"``).

    Raises
    ------
    KeyError
        If any requested scenario is not in the registry.
    FileNotFoundError
        If a source CSV is missing and ``download_if_missing`` is False.
    """
    if not scenarios:
        raise ValueError("scenarios must be a non-empty sequence of protocol IDs")
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR

    unknown = sorted(set(scenarios) - set(_REGISTRY))
    if unknown:
        raise KeyError(
            f"Unknown RCMIP3 protocol scenarios: {unknown}. "
            f"Known scenarios: {available_scenarios()}"
        )

    by_source: dict[_Source, list[str]] = {}
    for s in scenarios:
        spec = _REGISTRY[s]
        by_source.setdefault(spec.source, []).append(s)

    runs: list[scmdata.ScmRun] = []
    for source, protocol_ids in by_source.items():
        local = _ensure_source_cached(
            source, cache_dir=cache_dir, download=download_if_missing
        )
        runs.append(_load_one_source(local, protocol_ids))

    if len(runs) == 1:
        return runs[0]
    return scmdata.run_append(runs)


def _load_one_source(
    csv_path: Path, protocol_ids: Sequence[str]
) -> scmdata.ScmRun:
    """Load one upstream CSV and return only the requested protocol scenarios.

    Handles the source-scenario → protocol-id rename and the marker-IAM
    filter (for sources like the scen7 CSV that ship one scenario per IAM).
    """
    # Build the per-source filter sets and the inverse renaming we will
    # apply to the returned ScmRun's scenario column.
    source_scenarios: list[str] = []
    source_models: set[str] = set()
    has_model_filter = False
    rename: dict[tuple[str, Optional[str]], str] = {}
    for protocol_id in protocol_ids:
        spec = _REGISTRY[protocol_id]
        source_scenarios.append(spec.source_scenario)
        if spec.source_model is not None:
            has_model_filter = True
            source_models.add(spec.source_model)
        rename[(spec.source_scenario, spec.source_model)] = protocol_id

    run = load_iamc(csv_path, scenarios=source_scenarios)

    if has_model_filter:
        # Filter to just the marker IAMs we asked for, otherwise we
        # would multiply timeseries by every IAM that happens to have
        # produced a same-named scenario.
        run = run.filter(model=list(source_models))

    # Map (source_scenario, source_model) -> protocol_id on the meta.
    df = run.meta.copy()
    keys = list(
        zip(
            df["scenario"].astype(str).tolist(),
            df["model"].astype(str).tolist(),
        )
    )

    def _resolve(scenario_model: tuple[str, str]) -> str:
        scen, model = scenario_model
        if (scen, model) in rename:
            return rename[(scen, model)]
        # No model-scoped entry; fall back to the model-agnostic one
        # (RCMIP3 source).
        return rename[(scen, None)]

    new_scenarios = [_resolve(k) for k in keys]
    run["scenario"] = new_scenarios
    return run


# ---------------------------------------------------------------------------
# Source caching
# ---------------------------------------------------------------------------


def _ensure_source_cached(
    source: _Source, *, cache_dir: Path, download: bool
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / source.cache_filename
    if dest.exists():
        return dest
    if not download:
        raise FileNotFoundError(
            f"RCMIP3 source {source.name!r} is not cached at {dest} and "
            f"download_if_missing=False. Fetch it manually from {source.url}."
        )
    LOGGER.info("Downloading %s → %s", source.url, dest)
    _download(source.url, dest)
    return dest


def _download(url: str, dest: Path) -> None:
    req = Request(url, headers={"Accept": "text/csv,application/octet-stream;q=0.9"})
    with urlopen(req) as resp, dest.open("wb") as fh:  # noqa: S310 (allowlisted hosts)
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            fh.write(chunk)


# ---------------------------------------------------------------------------
# Helpers for callers that want the raw protocol metadata
# ---------------------------------------------------------------------------


def constraint_targets_dataframe() -> pd.DataFrame:
    """Constraint-target table (Table 3 of the RCMIP3 paper) as a DataFrame."""
    rows = []
    for key, ct in CONSTRAINT_TARGETS.items():
        rows.append(
            {
                "id": key,
                "variable": ct.variable,
                "unit": ct.unit,
                "source": ct.source,
                "base_period_start": (
                    ct.base_period[0] if ct.base_period is not None else None
                ),
                "base_period_end": (
                    ct.base_period[1] if ct.base_period is not None else None
                ),
                "constraint_period_start": ct.constraint_period[0],
                "constraint_period_end": ct.constraint_period[1],
                "central": ct.central,
                "lower": ct.lower,
                "upper": ct.upper,
            }
        )
    return pd.DataFrame(rows)
