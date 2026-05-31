"""
Tests for the CICEROSCMPY2 adapter's output-variable validator.

The validator (see
:mod:`openscm_runner.adapters.ciceroscm_py2_adapter._output_variables`)
fails fast on output variable names CICERO-SCM v2.x cannot produce,
instead of letting upstream silently drop them. These tests check that
the supported-set snapshot covers the spot-checked variables, that
structurally-unsupported requests yield a categorised message with the
documented reason, and that simple typos surface as "not recognised"
rather than the structural-limit bucket.
"""
from __future__ import annotations

import pytest

from openscm_runner.adapters.ciceroscm_py2_adapter._output_variables import (
    CARBON_CYCLE_VARIABLES,
    SUPPORTED_VARIABLES,
    all_supported,
    validate_output_variables,
)


@pytest.mark.skipif(
    not SUPPORTED_VARIABLES,
    reason="ciceroscm not installed; the validator no-ops in this case",
)
class TestValidator:
    """Behavioural tests that depend on the live upstream snapshot."""

    def test_supported_set_covers_core_outputs(self):
        # The protocol-relevant minimum the adapter has to be able to
        # produce. Drift-guard: if upstream ever drops one of these the
        # test fails before users discover the silent drop in a run.
        for must in (
            "Surface Air Temperature Change",
            "Effective Radiative Forcing",
            "Effective Radiative Forcing|Anthropogenic|CO2",
            "Atmospheric Concentrations|CO2",
            "Heat Uptake",
        ):
            assert must in SUPPORTED_VARIABLES, must

    def test_supported_set_covers_per_species_fgases(self):
        # F-Gas hierarchy uses the deep RCMIP path (HFC|HFC125 etc.).
        for must in (
            "Effective Radiative Forcing|Anthropogenic|F-Gases|HFC|HFC125",
            "Atmospheric Concentrations|F-Gases|HFC|HFC125",
        ):
            assert must in SUPPORTED_VARIABLES, must

    def test_carbon_cycle_variables_covered(self):
        for must in (
            "Carbon Flux|Land",
            "Carbon Flux|Ocean",
            "Airborne fraction CO2",
        ):
            assert must in CARBON_CYCLE_VARIABLES, must

    def test_all_supported_is_union(self):
        assert all_supported() == SUPPORTED_VARIABLES | CARBON_CYCLE_VARIABLES

    def test_validate_accepts_supported_variables(self):
        validate_output_variables(
            [
                "Surface Air Temperature Change",
                "Atmospheric Concentrations|CO2",
                "Carbon Flux|Land",  # carbon-cycle path also OK
            ]
        )

    def test_validate_accepts_empty_request(self):
        validate_output_variables([])
        validate_output_variables(())

    def test_validate_flags_sea_level_with_structural_reason(self):
        with pytest.raises(ValueError) as exc:
            validate_output_variables(
                ["Surface Air Temperature Change", "Sea Level Change"]
            )
        msg = str(exc.value)
        assert "Sea Level Change" in msg
        assert "no sea-level module" in msg
        # The valid variable should not appear in the error.
        assert "Surface Air Temperature Change" not in msg

    def test_validate_flags_sea_level_subentry_via_prefix(self):
        with pytest.raises(ValueError) as exc:
            validate_output_variables(["Sea Level Change|Glaciers"])
        assert "no sea-level module" in str(exc.value)

    def test_validate_flags_input_side_forcings(self):
        with pytest.raises(ValueError) as exc:
            validate_output_variables(
                ["Effective Radiative Forcing|Natural|Solar"]
            )
        assert "input-side forcing" in str(exc.value)

    def test_validate_flags_back_calc_co2_sector_split(self):
        with pytest.raises(ValueError) as exc:
            validate_output_variables(
                ["Emissions|CO2|MAGICC Fossil and Industrial"]
            )
        msg = str(exc.value)
        assert "not attributed by sector" in msg

    def test_validate_categorises_typos_separately_from_structural(self):
        with pytest.raises(ValueError) as exc:
            validate_output_variables(
                ["Sea Level Change", "Surface Air Temperture Change"]  # typo
            )
        msg = str(exc.value)
        assert "Not recognised" in msg
        assert "Surface Air Temperture Change" in msg
        # Make sure the typo wasn't bucketed under a structural reason.
        typo_index = msg.index("Surface Air Temperture Change")
        not_recognised_index = msg.index("Not recognised")
        assert not_recognised_index < typo_index

    def test_validate_message_lists_total_supported_count(self):
        with pytest.raises(ValueError) as exc:
            validate_output_variables(["Sea Level Change"])
        assert f"Total supported: {len(all_supported())}" in str(exc.value)


@pytest.mark.skipif(
    SUPPORTED_VARIABLES,
    reason="ciceroscm installed; the no-ciceroscm path doesn't apply",
)
def test_validate_noops_when_ciceroscm_not_installed():
    # When upstream isn't importable, the snapshot is empty and the
    # validator must not raise — _compat.py owns the canonical
    # ImportError for that case.
    validate_output_variables(["anything", "at all"])
