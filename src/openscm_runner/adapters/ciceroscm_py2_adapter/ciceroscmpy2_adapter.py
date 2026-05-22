"""
Adapter for CICERO-SCM v2.x (DistributionRun mode, v1 scope).

This adapter is a sibling to
:class:`openscm_runner.adapters.ciceroscm_py_adapter.CICEROSCMPY`
(which wraps the v1.1.x release with a per-config loop) and does not
replace it. The v1.1.x adapter is kept working for downstream users
that depend on its outputs; this one targets the AR7-relevant v2.1.0
release and its native ``DistributionRun`` ensemble API.

**Scope of this first version**

DistributionRun mode only. Each entry in the per-model ``cfgs`` list
must carry a ``distribution_json`` sidecar key pointing at a CICERO-SCM
parameter-distribution JSON (a posterior of ``pamset_udm`` +
``pamset_emiconc`` dicts, e.g. ``draw_samples_500.json``). The runner
hands the JSON to upstream's :class:`ciceroscm.parallel.DistributionRun`
and lets it dispatch the ensemble in parallel (upstream already manages
its own ProcessPool sized by the ``max_workers`` sidecar key).

Translated-cfg mode (one parameter dict per ensemble member, like the
v1.1.x adapter) is **not** implemented in this version. The v2.1.0
DistributionRun API is the path AR7-relevant CICERO Monte-Carlo runs
take, and a translated path adds bookkeeping that is not warranted by
any current use case.

**Per-cfg sidecar keys**

- ``distribution_json`` (required): filesystem path to a parameter-
  posterior JSON readable by ``DistributionRun``. Either a flat list
  of cfg dicts or a ``{"meta_info": ..., "configurations": [...]}``
  envelope is accepted (upstream handles both).
- ``gaspam_file`` (required): filesystem path to a gases parameter
  file (e.g. ``gases_vupdate_2022_AR6.txt``). Defines molecular
  weights, lifetimes and radiative properties per species; the
  scenario emissions header must list a superset of the gaspam's
  species or CICERO-SCM raises.
- ``concentrations_file`` (required): filesystem path to a historical
  concentrations file covering ``nystart`` to ``emstart``. CICERO-SCM
  drives concentrations directly over that range; emissions take over
  from ``emstart`` onward.
- ``member_indices`` (optional, default = all members in JSON):
  sequence of zero-based ``int`` selecting which rows of the
  parameter posterior to run. ``range(N)`` gives "first N members".
- ``max_workers`` (optional): forwarded to
  ``DistributionRun.run_over_distribution``. Defaults to
  :func:`openscm_runner.settings.get_worker_count` with
  ``CICEROSCM_WORKER_NUMBER`` as the override env var.
- ``nystart`` / ``nyend`` / ``emstart`` (optional): time bounds
  forwarded to CICERO-SCM. Defaults: ``nystart=1750``,
  ``nyend=max(scenario_year)``, ``emstart=min(scenario_year)``.
- ``sunvolc`` (optional, default ``1``): enables CICERO-SCM's bundled
  solar / volcanic / land-use forcing inputs (``solar_IPCC.txt``,
  ``meanVOLCmnd_ipcc_NH.txt``, ``IPCC_LUCalbedo.txt`` from the
  ``ciceroscm`` package). Set to ``0`` for a sun/volcanic-free run.

**Species coverage and the ssp245 fallback**

CICERO-SCM models a longer emissions species list than most
openscm-runner scenario inputs carry (notably the legacy halocarbons
CFC-11/12/113/114/115, HCFC-22/141b/123/142b, H-1211/1301/2402,
CH3Br, CCl4, CH3CCl3, and the biomass-burning aerosol subspecies
BMB_AEROS_BC / BMB_AEROS_OC). When the user's ``scenarios`` ScmRun
omits a species CICERO-SCM expects, the splice machinery in
:func:`openscm_runner.adapters.utils.cicero_utils.make_scenario_common.COMMONSFILEWRITER.make_printoutframe`
falls back to the bundled
``adapters/ciceroscm_adapter/utils_templates/ssp245_em_RCMIP.txt``
historical trajectory for those species across the full scenario
window, regardless of which SSP the user is actually running. This
is inherited unchanged from the v1.1.x adapter.

The fallback is exact for ssp245 runs and a small effect for the
other SSPs (halocarbons are largely Montreal-Protocol-controlled
and similar across SSPs). For a richer baseline, replace the
omitted species in your input ScmRun directly; an
``historical_emissions_file`` sidecar key to swap the fallback CSV
is on the follow-up list.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from scmdata import ScmRun, run_append

from ...settings import config
from ..base import _Adapter
from ..ciceroscm_py_adapter.make_scenario_data import SCENARIODATAGETTER
from ._compat import HAS_CICEROSCM_PY2, _ciceroscm_major_version, cscmpy2

try:
    # `get_worker_count` lands with the SLURM-aware worker sizing PR
    # (modernisation/worker-counts); fall through to the simpler env-
    # var lookup when reviewing this branch alone against main.
    from ...settings import get_worker_count
except ImportError:  # pragma: no cover
    def get_worker_count(override_env_var: str | None = None) -> int:
        """Fallback worker-count lookup (override env var > cpu_count)."""
        if override_env_var and (val := config.get(override_env_var, None)):
            return max(int(val), 1)
        return max(os.cpu_count() or 1, 1)

LOGGER = logging.getLogger(__name__)

# Bundled historical emissions used by the SCENARIODATAGETTER splice
# (CICERO-SCM expects emissions starting from nystart). Lives under
# the v1.1.x adapter's utils_templates directory and is reused here to
# avoid a redundant copy; if you need a different historical baseline,
# fork SCENARIODATAGETTER to accept an explicit path.
_BUNDLED_HISTORICAL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "ciceroscm_adapter", "utils_templates"
)


class CICEROSCMPY2(_Adapter):
    """
    Adapter for running CICERO-SCM v2.x with a parameter-distribution JSON.

    Registered under ``model_name = "CICERO-SCM-PY2"`` (distinct from
    the existing ``"CICERO-SCM-PY"`` which wraps v1.1.x).
    """

    model_name = "CICERO-SCM-PY2"

    def _init_model(self, *args, **kwargs):
        if not HAS_CICEROSCM_PY2:
            raise ImportError(
                "ciceroscm is not installed. Run 'pip install \"ciceroscm>=2,<3\"' "
                "or 'pip install openscm-runner[ciceroscmpy2]'. Note this "
                "conflicts with the v1.1.x adapter's 'ciceroscm<2' pin; only "
                "one major version of ciceroscm can be installed at a time."
            )
        major = _ciceroscm_major_version()
        if major < 2:
            raise ImportError(
                f"ciceroscm major version {major} is installed but the "
                "CICEROSCMPY2 adapter requires >=2. Either upgrade "
                "('pip install \"ciceroscm>=2,<3\"') or use the v1.1.x "
                "adapter (CICEROSCMPY) instead."
            )

    def _run(self, scenarios, cfgs, output_variables, output_config):
        if output_config is not None:
            raise NotImplementedError(
                "`output_config` not implemented for CICEROSCMPY2"
            )

        for cfg in cfgs:
            for key in ("distribution_json", "gaspam_file", "concentrations_file"):
                if key not in cfg:
                    raise ValueError(
                        f"CICEROSCMPY2 cfg is missing required sidecar key "
                        f"{key!r}. See the adapter docstring for the full "
                        "list of supported sidecar keys."
                    )

        results = []
        for cfg_index, cfg in enumerate(cfgs):
            LOGGER.info(
                "Running CICEROSCMPY2 (cfg %d/%d) with distribution %s",
                cfg_index + 1,
                len(cfgs),
                cfg["distribution_json"],
            )
            results.append(_run_one_distribution(scenarios, cfg, output_variables))

        out = run_append(results)
        out["climate_model"] = f"CICERO-SCM-PY{self.get_version()}"
        return out

    @classmethod
    def get_version(cls):
        """Return the installed ciceroscm version (e.g. ``"2.1.0"``)."""
        if not HAS_CICEROSCM_PY2:
            import sys as _sys

            raise ImportError(
                "ciceroscm is not installed (this Python interpreter "
                f"is {_sys.version.split()[0]}; ciceroscm 2.x requires "
                "Python >= 3.10). Run 'pip install \"ciceroscm>=2,<3\"' "
                "or 'pip install openscm-runner[ciceroscmpy2]'."
            )
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("ciceroscm")
        except PackageNotFoundError:
            # Fall back to runtime __version__ (a setuptools_scm dev
            # string for source installs); better than nothing.
            return cscmpy2.__version__


def _run_one_distribution(scenarios, cfg: dict[str, Any], output_variables) -> ScmRun:
    """
    Dispatch one ``DistributionRun`` for the cfg.

    Builds scenariodata dicts from ``scenarios``, applies any
    ``member_indices`` subset to the JSON cfgs, and returns the
    concatenated ScmRun across all scenarios and members.
    """
    from ciceroscm.parallel.distributionrun import DistributionRun

    distribution_json = cfg["distribution_json"]
    if not os.path.exists(distribution_json):
        raise FileNotFoundError(
            f"CICEROSCMPY2 cfg `distribution_json` not found: "
            f"{distribution_json}"
        )

    # Validate cheap cfg invariants before doing scenariodata work.
    member_indices = cfg.get("member_indices")
    if member_indices is not None:
        member_indices = list(member_indices)
        if not member_indices:
            raise ValueError(
                "CICEROSCMPY2 cfg `member_indices` selected zero members; "
                "omit the key (or pass a non-empty sequence) to run the "
                "full distribution."
            )

    scendata_list = _build_scendata_list(scenarios, cfg)

    # distro_config is unused when json_file_name exists (upstream
    # checks os.path.exists before falling back to .make_config_lists).
    dist = DistributionRun(distro_config=None, json_file_name=distribution_json)

    if member_indices is not None:
        dist.cfgs = [dist.cfgs[i] for i in member_indices]

    max_workers = cfg.get("max_workers")
    if max_workers is None:
        max_workers = get_worker_count("CICEROSCM_WORKER_NUMBER")

    LOGGER.info(
        "CICEROSCMPY2 dispatching %d members x %d scenarios with "
        "max_workers=%d",
        len(dist.cfgs),
        len(scendata_list),
        max_workers,
    )

    result = dist.run_over_distribution(
        scendata_list, list(output_variables), max_workers=max_workers
    )
    # DistributionRun returns a pandas DataFrame (concatenated upstream);
    # wrap it as ScmRun so the adapter's caller gets the documented type.
    return ScmRun(result) if not isinstance(result, ScmRun) else result


def _build_scendata_list(scenarios, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert ``scenarios`` (ScmRun) into the list of dicts upstream wants.

    One dict per (model, scenario) pair. The emissions DataFrame is
    built by SCENARIODATAGETTER (reused from the v1.1.x adapter),
    which splices user emissions on top of the bundled ssp245
    historical emissions.
    """
    if scenarios is None:
        raise ValueError(
            "CICEROSCMPY2 requires `scenarios` to be provided (an "
            "emissions ScmRun). Forcing-only runs are not yet supported."
        )

    nystart = int(cfg.get("nystart", 1750))
    scenario_years = scenarios.time_points.years()
    nyend = int(cfg.get("nyend", max(scenario_years)))
    emstart = int(cfg.get("emstart", min(scenario_years)))
    sunvolc = int(cfg.get("sunvolc", 1))

    sdatagetter = SCENARIODATAGETTER(_BUNDLED_HISTORICAL_DIR, nystart, nyend)

    scendata_list: list[dict[str, Any]] = []
    timeseries = scenarios.timeseries(time_axis="year")
    for (scenario_name, model_name), scen_df in timeseries.groupby(
        ["scenario", "model"]
    ):
        emissions_df = sdatagetter.get_scenario_data(scen_df, nystart)
        scendata: dict[str, Any] = {
            "nystart": nystart,
            "nyend": nyend,
            "emstart": emstart,
            "sunvolc": sunvolc,
            "scenname": scenario_name,
            "gaspam_file": cfg["gaspam_file"],
            "concentrations_file": cfg["concentrations_file"],
            "emissions_data": emissions_df,
        }
        # Pass through any explicit forcing-file overrides if the user
        # wants to replace ciceroscm's bundled solar / volcanic / LUC
        # data. Forwarded as-is; CICEROSCM validates these on init.
        for opt_key in (
            "rf_sun_file", "rf_volc_n_file", "rf_volc_s_file",
            "rf_volc_file", "rf_luc_file", "nat_ch4_file", "nat_n2o_file",
        ):
            if opt_key in cfg:
                scendata[opt_key] = cfg[opt_key]
        LOGGER.debug(
            "CICEROSCMPY2 scendata for scenario=%s model=%s years=%d-%d",
            scenario_name,
            model_name,
            nystart,
            nyend,
        )
        scendata_list.append(scendata)

    return scendata_list
