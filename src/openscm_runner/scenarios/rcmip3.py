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

_SOURCE_RCMIP3_CONC = _Source(
    name="rcmip3-conc-v1.1.7",
    url=(
        "https://gitlab.com/rcmip/rcmip-phase-3/-/raw/main/"
        "RCMIP3_input_datafiles/rcmip_phase3_concentrations_v1.1.7.csv"
    ),
    cache_filename="rcmip_phase3_concentrations_v1.1.7.csv",
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

# Sentinel for scenarios whose emissions live ONLY inside an
# adapter-side calibration bundle (e.g. the CICERO RCMIP-aligned
# bundle's ``{scenario}_em_*`` files), not in any IAMC-format CSV
# the loader knows about. For these scenarios we return a stub
# ScmRun whose sole purpose is to put the protocol-id scenario name
# into the runner's dispatch pipeline so the adapter can resolve
# its own per-scenario bundle file. The stub is NOT physically
# meaningful as scenario input on its own and FaIRv2 cannot
# currently consume it; CICEROSCMPY2 ignores the stub and uses the
# bundle file directly. See _make_bundle_only_stub.
_SOURCE_BUNDLE_ONLY = _Source(
    name="adapter-bundle-only",
    url="",
    cache_filename="",
)

# Repo-shipped IAMC translation of the CICEROSCM rcmip-march2026
# bundle's _em_ files for the two CH4-swap scenarios. Generated
# offline by scripts/translate_cicero_bundle_to_iamc.py and committed
# so the loader doesn't need bundle access at runtime. Identity-
# checked in _ensure_source_cached (no download, no cache).
_SOURCE_BUNDLE_TRANSLATED = _Source(
    name="cicero-bundle-translated-rcmip-march2026",
    url="",
    cache_filename="cicero_rcmip_march2026.csv",
)
_BUNDLE_TRANSLATED_PATH = (
    Path(__file__).parent / "data" / "bundle_translated"
    / _SOURCE_BUNDLE_TRANSLATED.cache_filename
)


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ScenarioSpec:
    """How to fetch one protocol scenario from one (or two) upstream CSVs.

    Attributes
    ----------
    source
        Primary upstream CSV. For all scenarios this carries the
        emissions (or, for CD scenarios loaded via the conc registry,
        the concentrations).
    source_scenario
        The scenario string as it appears in the primary CSV's
        ``scenario`` column.
    source_model
        Optional IAM-marker filter for sources that publish one scenario
        per IAM (the scen7 CSV does this; the RCMIP3 CSV typically has
        only one model row per scenario already).
    secondary_source / secondary_source_scenario
        Optional second upstream CSV, set when the scenario is in the
        protocol's "ED CO2-only" mixed-driving mode. The primary then
        contributes CO2 emissions only (``Emissions|CO2|*``) and the
        secondary contributes non-CO2 concentrations only
        (``Atmospheric Concentrations|*`` excluding CO2). The loader
        concatenates them into a single ScmRun. CICEROSCM v2 cannot
        consume this mixed input (its ``conc_run`` is all-or-nothing
        per scenario, so it falls back to all-GHG ED matching Marit's
        de facto reference); FaIR 2.x consumes both modes natively
        via ``fill_from_pandas``.
    protocol_mode
        One of ``"CD"`` (concentration-driven), ``"ED-CO2-only"``
        (emissions-driven CO2 + concentration-driven non-CO2, the
        protocol's intent), or ``"ED-all-GHG"`` (emissions-driven for
        all species). Adapters should consume this column rather
        than pattern-matching on the scenario name.
    protocol_natural_forcing
        ``"on"`` for runs that include historical solar/volcanic
        natural forcing; ``"off"`` for idealised runs and piControl
        (constant pre-industrial natural forcing throughout).
    protocol_land_use_forcing
        ``"historical"`` for runs that include the historical LUC
        trajectory; ``"constant_zero"`` for idealised runs and
        piControl (no LUC perturbation).
    """

    source: _Source
    source_scenario: str
    source_model: Optional[str] = None
    secondary_source: Optional[_Source] = None
    secondary_source_scenario: Optional[str] = None
    protocol_mode: str = "CD"
    protocol_natural_forcing: str = "on"
    protocol_land_use_forcing: str = "historical"

    @property
    def is_mixed_mode(self) -> bool:
        """True when the scenario sources CO2 emissions + non-CO2 conc separately."""
        return self.secondary_source is not None


# Six protocol profiles — every scenario in the registry matches one of
# them. Pulled out so the registry build below stays a flat lookup
# table rather than a tangle of kwargs.
_PROFILE_CD_REAL = dict(
    protocol_mode="CD",
    protocol_natural_forcing="on",
    protocol_land_use_forcing="historical",
)
_PROFILE_CD_IDEAL = dict(
    protocol_mode="CD",
    protocol_natural_forcing="off",
    protocol_land_use_forcing="constant_zero",
)
_PROFILE_ED_CO2_REAL = dict(
    protocol_mode="ED-CO2-only",
    protocol_natural_forcing="on",
    protocol_land_use_forcing="historical",
)
_PROFILE_ED_CO2_IDEAL = dict(
    protocol_mode="ED-CO2-only",
    protocol_natural_forcing="off",
    protocol_land_use_forcing="constant_zero",
)
_PROFILE_ED_ALLGHG_REAL = dict(
    protocol_mode="ED-all-GHG",
    protocol_natural_forcing="on",
    protocol_land_use_forcing="historical",
)
_PROFILE_ED_ALLGHG_IDEAL = dict(
    protocol_mode="ED-all-GHG",
    protocol_natural_forcing="off",
    protocol_land_use_forcing="constant_zero",
)

_PROTOCOL_META_COLUMNS: tuple[str, ...] = (
    "protocol_mode",
    "protocol_natural_forcing",
    "protocol_land_use_forcing",
)


# Scenarios from the RCMIP3 v1.1.7 emissions CSV.
# Per the RCMIP3 README, ``esm-sspXXX`` and ``esm-allGHG-sspXXX`` share
# emissions filed under plain ``sspXXX``; the registry materialises
# both protocol-id variants from this same source row (the esm-* form
# with non-CO2 zeroed via co2_only=True, the esm-allGHG-* form as-is)
# so callers see distinct ScmRun timeseries per protocol id.
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


# RCMIP3 CSV v1.1.7 also carries these (not just SSPs / flat-* / hist):
# - 3 esm-bell-{750,1000,2000}PgC
# - 2 esm-pi-{CO2pulse,cdr-pulse}
# - 1 esm-piControl
# - 5 sensitivity SSPs (esm-allGHG-ssp370-lowCH4, -lowNTCF, etc.)
# Methane-cycle scenarios (methanemip-*) and 1pctCO2-bgc/-rad are
# intentionally NOT listed (shelved per the modernisation discussion;
# the methane scenarios need scenario-specific MethaneMIP inputs and
# the -bgc/-rad runs need non-standard model configuration).
_RCMIP3_EXTRA_ED = (
    "esm-bell-750PgC", "esm-bell-1000PgC", "esm-bell-2000PgC",
    "esm-pi-CO2pulse", "esm-pi-cdr-pulse",
    "esm-piControl",
    "esm-allGHG-ssp370-lowCH4",
    "esm-allGHG-ssp370-lowNTCF",
    "esm-allGHG-ssp370-lowNTCF-HighCH4",
    "esm-allGHG-ssp534-over-highCH4",
    "esm-allGHG-ssp585-lowCH4",
)


# Bundle-only emissions-driven scenarios: the loader cannot materialise
# these from any IAMC CSV it knows about, so it emits a stub ScmRun whose
# only purpose is to carry the protocol-id scenario name into the runner's
# dispatch pipeline. CICERO's _build_scendata_list_bundle picks up the
# bundle file by name; FaIR cannot consume stubs (these scenarios won't
# run under FaIR until step 5c lands the bundle translator).
#
# After step 5c, the residual bundle-only ED set is only the
# esm-1pct-brch-*PgC scenarios. These are model-specific
# back-calculations of each ESM's own esm-1pctCO2 run (the protocol
# says "model-specific"); CICERO's bundle doesn't ship a
# back-calculation either, so they stay NO_MATCH against Marit's
# reference and the bundle-only stub path is retained as a registry
# placeholder.
_RCMIP3_BUNDLE_ONLY_ED = (
    "esm-1pct-brch-750PgC", "esm-1pct-brch-1000PgC", "esm-1pct-brch-2000PgC",
)


# Bundle-only concentration-driven scenarios. The CICEROSCMPY2 bundle
# ships `{scenario}_conc_*` files for these; in CD mode the adapter
# resolves them by name.
_RCMIP3_BUNDLE_ONLY_CD = (
    # 1pctCO2 family (the -bgc/-rad variants are shelved — they need
    # non-standard model configuration, not just data)
    "1pctCO2", "1pctCO2-4xext", "1pctCO2-cdr",
    # abrupt-* (3)
    "abrupt-0p5xCO2", "abrupt-2xCO2", "abrupt-4xCO2",
    # CD scen7 variants (the "C" suffix marks concentration-driven)
    "scen7-HC", "scen7-HLC", "scen7-LC", "scen7-LNC",
    "scen7-MC", "scen7-MLC", "scen7-VLC",
    # CD pre-industrial control
    "piControl",
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

    # CMIP6 SSPs (CD real-world).
    for s in _RCMIP3_SSPS:
        reg[s] = _ScenarioSpec(
            source=_SOURCE_RCMIP3, source_scenario=s, **_PROFILE_CD_REAL,
        )

    # esm-ssp* / esm-allGHG-ssp* — ED variants of the SSPs, both read
    # from the bare ssp* source row in the RCMIP3 CSV (per the protocol
    # README).
    #
    # The ED-CO2-only variant is the protocol's mixed-mode driving: CO2
    # from emissions, non-CO2 from concentrations. The loader emits
    # both streams in a single ScmRun and FaIR consumes them via
    # fill_from_pandas (mode="emissions" + mode="concentration"). CICERO
    # v2 can't honor mixed mode (conc_run is all-or-nothing upstream),
    # so its adapter detects the mixed-mode signal and falls back to
    # the bundle's bare-name ssp*_em_* file — running full all-GHG ED,
    # which matches Marit's published reference (her esm-ssp245 and
    # esm-allGHG-ssp245 are bit-identical for the same reason).
    for s in _RCMIP3_SSPS:
        reg[f"esm-{s}"] = _ScenarioSpec(
            source=_SOURCE_RCMIP3, source_scenario=s,
            secondary_source=_SOURCE_RCMIP3_CONC, secondary_source_scenario=s,
            **_PROFILE_ED_CO2_REAL,
        )
        reg[f"esm-allGHG-{s}"] = _ScenarioSpec(
            source=_SOURCE_RCMIP3, source_scenario=s, **_PROFILE_ED_ALLGHG_REAL,
        )

    # esm-flat* — idealised constant-emissions CO2-only experiments.
    for s in _RCMIP3_FLAT:
        reg[s] = _ScenarioSpec(
            source=_SOURCE_RCMIP3, source_scenario=s, **_PROFILE_ED_CO2_IDEAL,
        )

    # Historical CD runs (full historical + 3 single-forcing attribution
    # runs hist-aer / hist-CO2 / hist-GHG). All run over the historical
    # window with natural forcing active.
    for s in _RCMIP3_HISTORICAL:
        reg[s] = _ScenarioSpec(
            source=_SOURCE_RCMIP3, source_scenario=s, **_PROFILE_CD_REAL,
        )

    # _RCMIP3_EXTRA_ED breakdown:
    #  - esm-bell-* (3), esm-pi-{CO2pulse,cdr-pulse}, esm-piControl:
    #    idealised ED CO2-only impulse-response and PI control.
    #  - 5 sensitivity SSPs (esm-allGHG-ssp370-lowCH4, etc.):
    #    real-world ED all-GHG perturbations of SSP markers.
    _extra_ed_co2_ideal = (
        "esm-bell-750PgC", "esm-bell-1000PgC", "esm-bell-2000PgC",
        "esm-pi-CO2pulse", "esm-pi-cdr-pulse",
        "esm-piControl",
    )
    _extra_ed_allghg_real = (
        "esm-allGHG-ssp370-lowCH4",
        "esm-allGHG-ssp370-lowNTCF",
        "esm-allGHG-ssp370-lowNTCF-HighCH4",
        "esm-allGHG-ssp534-over-highCH4",
        "esm-allGHG-ssp585-lowCH4",
    )
    for s in _extra_ed_co2_ideal:
        reg[s] = _ScenarioSpec(
            source=_SOURCE_RCMIP3, source_scenario=s, **_PROFILE_ED_CO2_IDEAL,
        )
    for s in _extra_ed_allghg_real:
        reg[s] = _ScenarioSpec(
            source=_SOURCE_RCMIP3, source_scenario=s, **_PROFILE_ED_ALLGHG_REAL,
        )

    # scen7 markers and their ED CO2-only / ED all-GHG siblings.
    #
    # ``esm-scen7-*`` is tagged ED-CO2-only but does NOT get a
    # secondary_source: the RCMIP3 v1.1.7 concentrations archive does
    # not ship scen7-* concentrations (they are CMIP7-only and
    # post-date the archive). Without a non-CO2 concentration
    # trajectory the loader can't materialise the mixed-mode input, so
    # the scenario falls back to its primary emissions source — same
    # data as ``esm-allGHG-scen7-*``. A future bundle translator for
    # scen7-* ``_conc_*`` files (extending step 5c) would unlock the
    # proper mixed-mode pathway here.
    for protocol_id, (source_scen, source_model) in _SCEN7_MARKERS.items():
        reg[protocol_id] = _ScenarioSpec(
            source=_SOURCE_SCEN7, source_scenario=source_scen,
            source_model=source_model, **_PROFILE_CD_REAL,
        )
        reg[f"esm-{protocol_id}"] = _ScenarioSpec(
            source=_SOURCE_SCEN7, source_scenario=source_scen,
            source_model=source_model, **_PROFILE_ED_CO2_REAL,
        )
        reg[f"esm-allGHG-{protocol_id}"] = _ScenarioSpec(
            source=_SOURCE_SCEN7, source_scenario=source_scen,
            source_model=source_model, **_PROFILE_ED_ALLGHG_REAL,
        )

    # Bundle-only scenarios: no IAMC source, but still need protocol
    # metadata so adapters that consume meta cols see the right mode.
    # Categorise per scenario id.
    _bundle_only_profiles: dict[str, dict[str, str]] = {
        # esm-1pct-brch-* (3): model-specific back-calculations, ED
        # CO2-only idealised. No IAMC source exists.
        "esm-1pct-brch-750PgC": _PROFILE_ED_CO2_IDEAL,
        "esm-1pct-brch-1000PgC": _PROFILE_ED_CO2_IDEAL,
        "esm-1pct-brch-2000PgC": _PROFILE_ED_CO2_IDEAL,
        # CD idealised: 1pctCO2 family + abrupt-* + piControl
        "1pctCO2": _PROFILE_CD_IDEAL,
        "1pctCO2-4xext": _PROFILE_CD_IDEAL,
        "1pctCO2-cdr": _PROFILE_CD_IDEAL,
        "abrupt-0p5xCO2": _PROFILE_CD_IDEAL,
        "abrupt-2xCO2": _PROFILE_CD_IDEAL,
        "abrupt-4xCO2": _PROFILE_CD_IDEAL,
        "piControl": _PROFILE_CD_IDEAL,
        # CD scen7 (real-world): scen7-*C inherits the same LUC and
        # natural forcing as bare scen7-*; only the driving (CD vs ED)
        # differs.
        "scen7-HC": _PROFILE_CD_REAL,
        "scen7-HLC": _PROFILE_CD_REAL,
        "scen7-LC": _PROFILE_CD_REAL,
        "scen7-LNC": _PROFILE_CD_REAL,
        "scen7-MC": _PROFILE_CD_REAL,
        "scen7-MLC": _PROFILE_CD_REAL,
        "scen7-VLC": _PROFILE_CD_REAL,
    }
    # ED variants of historical / piControl that source-map onto rows
    # already present in the RCMIP3 emissions CSV. Step 5a moved these
    # out of bundle-only; B-partial adds the secondary_source for the
    # CO2-only variants so they get proper mixed-mode driving via the
    # historical concentrations CSV.
    reg["esm-hist"] = _ScenarioSpec(
        source=_SOURCE_RCMIP3, source_scenario="historical",
        secondary_source=_SOURCE_RCMIP3_CONC,
        secondary_source_scenario="historical",
        **_PROFILE_ED_CO2_REAL,
    )
    reg["esm-hist-cmip6"] = _ScenarioSpec(
        source=_SOURCE_RCMIP3, source_scenario="historical-cmip6",
        secondary_source=_SOURCE_RCMIP3_CONC,
        secondary_source_scenario="historical-cmip6",
        **_PROFILE_ED_CO2_REAL,
    )
    reg["esm-allGHG-hist"] = _ScenarioSpec(
        source=_SOURCE_RCMIP3, source_scenario="historical",
        **_PROFILE_ED_ALLGHG_REAL,
    )
    reg["esm-allGHG-hist-cmip6"] = _ScenarioSpec(
        source=_SOURCE_RCMIP3, source_scenario="historical-cmip6",
        **_PROFILE_ED_ALLGHG_REAL,
    )
    # esm-allGHG-piControl shares the esm-piControl row (PI emissions
    # are ~zero for all anthropogenic species either way, so the mask
    # toggle is functionally moot; the protocol_mode tag still
    # distinguishes them for adapter routing).
    reg["esm-allGHG-piControl"] = _ScenarioSpec(
        source=_SOURCE_RCMIP3, source_scenario="esm-piControl",
        **_PROFILE_ED_ALLGHG_IDEAL,
    )

    # CH4-swap variants of scen7-* (step 5c). These come from the
    # repo-shipped translation of CICERO's bundle _em_ files (see
    # scripts/translate_cicero_bundle_to_iamc.py); no upstream IAMC
    # source publishes them. The source_scenario is the protocol id
    # itself because the translated CSV uses protocol ids as its
    # Scenario column.
    for s in ("esm-allGHG-scen7-H-CH4L", "esm-allGHG-scen7-L-CH4H"):
        reg[s] = _ScenarioSpec(
            source=_SOURCE_BUNDLE_TRANSLATED, source_scenario=s,
            **_PROFILE_ED_ALLGHG_REAL,
        )

    for s in _RCMIP3_BUNDLE_ONLY_ED + _RCMIP3_BUNDLE_ONLY_CD:
        profile = _bundle_only_profiles.get(s)
        if profile is None:
            raise RuntimeError(
                f"Bundle-only scenario {s!r} has no protocol profile "
                f"assigned in _bundle_only_profiles."
            )
        reg[s] = _ScenarioSpec(
            source=_SOURCE_BUNDLE_ONLY, source_scenario=s, **profile,
        )
    return reg


_REGISTRY: dict[str, _ScenarioSpec] = _build_registry()


# Scenarios for which the RCMIP3 official archive ships
# concentrations (rcmip_phase3_concentrations_v1.1.7.csv). Audited
# directly off that CSV's Scenario column on 2026-05-25; covers all
# 8 CMIP6 SSPs (CD), the 5 CD-only idealised families (1pctCO2*,
# abrupt-*), historical variants + attribution runs, piControl, and
# the 5 sensitivity SSPs. The CMIP7 scen7-* concentration variants
# (scen7-HC etc.) are NOT in this archive (those need bundle
# translation in step 5).
_RCMIP3_CONC_SCENARIOS = (
    "1pctCO2", "1pctCO2-4xext", "1pctCO2-cdr",
    "abrupt-0p5xCO2", "abrupt-2xCO2", "abrupt-4xCO2",
    "esm-allGHG-ssp370-lowCH4",
    "esm-allGHG-ssp370-lowNTCF",
    "esm-allGHG-ssp370-lowNTCF-HighCH4",
    "esm-allGHG-ssp534-over-highCH4",
    "esm-allGHG-ssp585-lowCH4",
    "hist-CO2", "hist-GHG", "hist-aer",
    "historical", "historical-cmip6",
    "piControl",
    "ssp119", "ssp126", "ssp245", "ssp370",
    "ssp434", "ssp460", "ssp534-over", "ssp585",
)


def _build_conc_registry() -> dict[str, _ScenarioSpec]:
    """Build the concentration-source registry.

    Every concentration scenario id is also present in the emissions
    ``_REGISTRY``; we inherit the natural/land-use profile from there
    so the same idealised-vs-real-world classification applies in both
    paths. ``protocol_mode`` is always ``"CD"`` for the concentrations
    loader's output — even when the scenario's emissions cousin is an
    ED perturbation experiment (the sensitivity SSPs), the SCM consumes
    these as concentration inputs, i.e. CD mode.
    """
    reg: dict[str, _ScenarioSpec] = {}
    for s in _RCMIP3_CONC_SCENARIOS:
        em_spec = _REGISTRY[s]
        reg[s] = _ScenarioSpec(
            source=_SOURCE_RCMIP3_CONC,
            source_scenario=s,
            protocol_mode="CD",
            protocol_natural_forcing=em_spec.protocol_natural_forcing,
            protocol_land_use_forcing=em_spec.protocol_land_use_forcing,
        )
    return reg


_CONC_REGISTRY: dict[str, _ScenarioSpec] = _build_conc_registry()


def available_scenarios() -> tuple[str, ...]:
    """All protocol scenario IDs this loader knows how to resolve."""
    return tuple(sorted(_REGISTRY))


def available_concentration_scenarios() -> tuple[str, ...]:
    """Protocol scenario IDs for which RCMIP3 ships canonical concentrations.

    These can be loaded via :func:`load_rcmip3_concentrations`. Bundle-
    translated concentrations for the CMIP7 ``scen7-*C`` variants will
    be added in a follow-up step.
    """
    return tuple(sorted(_CONC_REGISTRY))


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

    # Split into three groups: bundle-only stubs, mixed-mode (CO2 em +
    # non-CO2 conc), and primary-only single-source.
    bundle_only_ids: list[str] = []
    mixed_mode_ids: list[str] = []
    primary_by_source: dict[_Source, list[str]] = {}
    for s in scenarios:
        spec = _REGISTRY[s]
        if spec.source is _SOURCE_BUNDLE_ONLY:
            bundle_only_ids.append(s)
        elif spec.is_mixed_mode:
            mixed_mode_ids.append(s)
        else:
            primary_by_source.setdefault(spec.source, []).append(s)

    runs: list[scmdata.ScmRun] = []
    if bundle_only_ids:
        runs.append(_make_bundle_only_stub(bundle_only_ids))
    for source, protocol_ids in primary_by_source.items():
        local = _ensure_source_cached(
            source, cache_dir=cache_dir, download=download_if_missing,
        )
        runs.append(_load_one_source(local, protocol_ids, registry=_REGISTRY))
    for protocol_id in mixed_mode_ids:
        runs.append(_load_mixed_mode_scenario(
            protocol_id, cache_dir=cache_dir, download=download_if_missing,
        ))

    if len(runs) == 1:
        return runs[0]
    return scmdata.run_append(runs)


def load_rcmip3_concentrations(
    scenarios: Sequence[str],
    *,
    cache_dir: Optional[Path] = None,
    download_if_missing: bool = True,
) -> scmdata.ScmRun:
    """
    Resolve and load atmospheric concentrations for one or more RCMIP3
    protocol scenarios.

    Mirror of :func:`load_rcmip3_emissions` for the concentration-driven
    half of the protocol. Source is ``rcmip_phase3_concentrations_v1.1.7
    .csv`` from the RCMIP3 official archive
    (gitlab.com/rcmip/rcmip-phase-3). See
    :func:`available_concentration_scenarios` for the list of scenario
    IDs covered.

    Parameters
    ----------
    scenarios
        Protocol scenario IDs whose concentrations should be loaded
        (e.g. ``"1pctCO2"``, ``"abrupt-4xCO2"``, ``"ssp245"``,
        ``"piControl"``). Must be in
        :func:`available_concentration_scenarios`.
    cache_dir
        Local directory under which to cache the upstream CSV. Defaults
        to ``configurations/rcmip3/`` (the same cache the emissions
        loader uses).
    download_if_missing
        When True (default), fetch the upstream CSV on demand. False is
        the offline / CI path that requires the file to already be in
        ``cache_dir``.

    Returns
    -------
    scmdata.ScmRun
        One timeseries per (scenario, variable) pair. Variable names
        follow the canonical ``Atmospheric Concentrations|{species}``
        convention (the iamc loader strips the upstream
        ``Atmospheric Concentrations|F-Gases|HFC|`` parent paths to
        the flat ``Atmospheric Concentrations|HFC125`` form, matching
        what adapters consume).

    Raises
    ------
    KeyError
        If any requested scenario is not in
        :func:`available_concentration_scenarios`.
    FileNotFoundError
        If the source CSV is missing and ``download_if_missing`` is False.
    """
    if not scenarios:
        raise ValueError("scenarios must be a non-empty sequence of protocol IDs")
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR

    unknown = sorted(set(scenarios) - set(_CONC_REGISTRY))
    if unknown:
        raise KeyError(
            f"No RCMIP3 concentration source for: {unknown}. "
            f"Available concentration scenarios: "
            f"{available_concentration_scenarios()}"
        )

    local = _ensure_source_cached(
        _SOURCE_RCMIP3_CONC,
        cache_dir=cache_dir,
        download=download_if_missing,
    )
    return _load_one_source(local, list(scenarios), registry=_CONC_REGISTRY)


def _make_bundle_only_stub(protocol_ids: Sequence[str]) -> scmdata.ScmRun:
    """Construct a stub ScmRun for scenarios whose emissions live in the adapter bundle.

    Produces a single ``Emissions|CO2|MAGICC Fossil and Industrial``
    timeseries of zeros for 1750-2500 per requested scenario, with
    the canonical meta columns the runner expects. The data is NOT
    physically meaningful as scenario input; the only purpose is to
    put the protocol-id scenario name (and its protocol-mode metadata)
    into the dispatch pipeline so the CICEROSCMPY2 bundle resolver
    can pick up its own ``{scenario}_em_*`` file.

    FaIRv2 cannot currently consume bundle-only stubs (it would run
    with zero anthropogenic CO2). Future work: extend the loader to
    materialise emissions for these scenarios by reading the bundle
    file directly, so FaIR can share the path.
    """
    years = list(range(1750, 2501))
    rows = []
    for scenario in protocol_ids:
        spec = _REGISTRY[scenario]
        row = {
            "model": "(bundle-only stub)",
            "scenario": scenario,
            "region": "World",
            "variable": "Emissions|CO2|MAGICC Fossil and Industrial",
            "unit": "Mt CO2/yr",
            "protocol_mode": spec.protocol_mode,
            "protocol_natural_forcing": spec.protocol_natural_forcing,
            "protocol_land_use_forcing": spec.protocol_land_use_forcing,
        }
        row.update({str(y): 0.0 for y in years})
        rows.append(row)
    return scmdata.ScmRun(pd.DataFrame(rows))


def _load_one_source(
    csv_path: Path,
    protocol_ids: Sequence[str],
    *,
    registry: dict[str, _ScenarioSpec],
) -> scmdata.ScmRun:
    """Load one upstream CSV and return one ScmRun per requested protocol id.

    Handles the source-scenario → protocol-id rename, the marker-IAM
    filter (for sources like the scen7 CSV that ship one scenario per
    IAM), and the per-protocol-id co2_only mask. Multiple protocol ids
    may resolve to the same source row (e.g. ``ssp245``, ``esm-ssp245``,
    ``esm-allGHG-ssp245`` all read from the bare ``ssp245`` row);
    each gets its own copy of the loaded timeseries with the scenario
    column renamed and the appropriate transforms applied.

    The registry is passed in explicitly so the same code serves both
    the emissions (``_REGISTRY``) and concentrations (``_CONC_REGISTRY``)
    paths.
    """
    # Group protocol ids by the source row they read from. This lets
    # us read the CSV once even when several protocol ids share a row.
    by_row: dict[tuple[str, Optional[str]], list[str]] = {}
    for pid in protocol_ids:
        spec = registry[pid]
        by_row.setdefault((spec.source_scenario, spec.source_model), []).append(pid)

    source_scenarios = sorted({s for s, _ in by_row})
    source_models = sorted({m for _, m in by_row if m is not None})

    base = load_iamc(csv_path, scenarios=source_scenarios)
    if source_models:
        # Filter to just the marker IAMs we asked for, otherwise we
        # would multiply timeseries by every IAM that happens to have
        # produced a same-named scenario.
        base = base.filter(model=source_models)

    # Materialise one ScmRun per protocol id with rename applied.
    runs: list[scmdata.ScmRun] = []
    for (source_scen, source_model), pids in by_row.items():
        filt: dict[str, object] = {"scenario": source_scen}
        if source_model is not None:
            filt["model"] = source_model
        row_run = base.filter(**filt)
        for pid in pids:
            per_id = row_run.copy()
            per_id["scenario"] = pid
            spec = registry[pid]
            _apply_protocol_metadata(per_id, spec)
            runs.append(per_id)

    if len(runs) == 1:
        return runs[0]
    return scmdata.run_append(runs)


def _apply_protocol_metadata(run: scmdata.ScmRun, spec: _ScenarioSpec) -> None:
    """Stamp the three protocol-metadata columns onto a single-scenario ScmRun.

    Mutates ``run`` in place. The metadata is constant across every row
    of the scenario (mode/natural/LU are scenario-level, not
    variable-level), so we broadcast the spec's values across the
    ScmRun's meta. Adapters that consume the meta cols should always
    see one unique value per scenario.
    """
    run["protocol_mode"] = spec.protocol_mode
    run["protocol_natural_forcing"] = spec.protocol_natural_forcing
    run["protocol_land_use_forcing"] = spec.protocol_land_use_forcing


# The two CO2 emissions variables that survive the mixed-mode split.
# Everything else from the primary (emissions) source gets dropped;
# their concentration trajectories come from the secondary source.
_CO2_EMISSIONS_VARIABLES: frozenset[str] = frozenset({
    "Emissions|CO2|MAGICC Fossil and Industrial",
    "Emissions|CO2|MAGICC AFOLU",
})


def _load_mixed_mode_scenario(
    protocol_id: str, *, cache_dir: Path, download: bool,
) -> scmdata.ScmRun:
    """Load a single ED-CO2-only scenario in protocol-correct mixed mode.

    The protocol's "ED CO2-only" driving mode (``esm-ssp245``,
    ``esm-hist``, etc.) wants CO2 from emissions input and all other
    species from concentration input — within a single run. This helper
    reads the spec's primary source (emissions CSV) for the two CO2
    sector splits and the secondary source (concentrations CSV) for
    every non-CO2 species, then concatenates them into one ScmRun with
    the protocol id stamped on the ``scenario`` column.

    FaIR 2.x consumes this directly via two ``fill_from_pandas`` calls
    (one for ``mode="emissions"``, one for ``mode="concentration"``).
    CICEROSCM v2 cannot — its ``conc_run`` is all-or-nothing per
    scenario — so the CICEROSCM adapter detects the mixed-mode signal
    and falls back to the bundle's bare-name ``_em_*`` file, running
    full all-GHG ED (matching Marit's de facto reference).
    """
    spec = _REGISTRY[protocol_id]
    if not spec.is_mixed_mode:
        raise RuntimeError(
            f"_load_mixed_mode_scenario called on {protocol_id!r}, which "
            f"has no secondary_source (not in mixed mode)."
        )

    em_path = _ensure_source_cached(
        spec.source, cache_dir=cache_dir, download=download,
    )
    conc_path = _ensure_source_cached(
        spec.secondary_source, cache_dir=cache_dir, download=download,
    )

    # Primary: read the emissions row, keep only the two CO2 sector
    # splits (anything else gets replaced by the concentration twin).
    em_run = load_iamc(
        em_path, scenarios=[spec.source_scenario],
        variables=list(_CO2_EMISSIONS_VARIABLES),
    )
    if spec.source_model is not None:
        em_run = em_run.filter(model=spec.source_model)

    # Secondary: read the conc row, keep everything except
    # ``Atmospheric Concentrations|CO2`` (we drive CO2 by emissions).
    conc_run = load_iamc(
        conc_path, scenarios=[spec.secondary_source_scenario],
    )
    conc_run = conc_run.filter(
        variable="Atmospheric Concentrations|CO2", keep=False,
    )

    # Rename scenario col on both to the protocol id and stamp meta.
    em_run["scenario"] = protocol_id
    conc_run["scenario"] = protocol_id
    _apply_protocol_metadata(em_run, spec)
    _apply_protocol_metadata(conc_run, spec)

    return scmdata.run_append([em_run, conc_run])


# ---------------------------------------------------------------------------
# Source caching
# ---------------------------------------------------------------------------


def _ensure_source_cached(
    source: _Source, *, cache_dir: Path, download: bool
) -> Path:
    # Repo-shipped translated bundle: no cache, no download. Generated
    # offline by scripts/translate_cicero_bundle_to_iamc.py.
    if source is _SOURCE_BUNDLE_TRANSLATED:
        if not _BUNDLE_TRANSLATED_PATH.exists():
            raise FileNotFoundError(
                f"Translated bundle CSV missing from package: "
                f"{_BUNDLE_TRANSLATED_PATH}. Re-run "
                f"scripts/translate_cicero_bundle_to_iamc.py to "
                f"regenerate it."
            )
        return _BUNDLE_TRANSLATED_PATH
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
