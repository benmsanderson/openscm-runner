# Phase B scorecard against Marit's RCMIP3 reference

Output from `scripts/validate_against_marit.py` against the full
registry, 10 members per scenario, after the Phase B registry
expansion. Comparison variable: `Surface Air Ocean Blended Temperature
Change`. Matched run_id per member.

Raw output: `out/rcmip3/scorecard_phase_b.txt` (gitignored under
`out/`; re-generate with `scripts/validate_against_marit.py
--output-dir out/rcmip3`).

## Summary

| label | count | notes |
|---|---|---|
| PASS  | 10 | All CD SSPs that match, the bare and esm- historicals, the attribution runs, ssp245 bit-exact |
| WARN  | 20 | esm-flat* family (15) + four warm SSPs + esm-hist-cmip6 |
| FAIL  | 60 | See categorisation below |
| NO_OURS_RUN | 2 | (the 4 shelved scenarios, minus 2 NO_MARIT_REF and 3 NO_MATCH-from-format-bug) |
| NO_MATCH | 3 | esm-1pct-brch-*PgC: Marit CSV column layout differs (no "id" col), our loader misreads variable column |
| NO_MARIT_REF | 2 | 1pctCO2-bgc, 1pctCO2-rad (shelved per user) |

## Failure categorisation

The 60 FAILs cluster into four structural issues:

### 1. Bundle-file lookup needs prefix/suffix stripping (38 scenarios)

The CICEROSCMPY2 bundle uses **bare** scenario names for its
`_em_*` / `_conc_*` files, with the protocol's ED/CD/CO2-only/all-GHG
distinctions encoded in cfg rather than filenames:

```
configurations/ciceroscm/rcmip-march2026/
    ssp245_em_gases_vupdate_*.txt        # used by ssp245 (CD), esm-ssp245 (ED CO2-only), esm-allGHG-ssp245 (ED all-GHG)
    ssp245_conc_gases_vupdate_*.txt      # used for CD mode
    scen7-H_em_gases_vupdate_*.txt       # used by scen7-H, scen7-HC, esm-scen7-H, esm-allGHG-scen7-H
    scen7-H_conc_gases_vupdate_*.txt
```

Our adapter currently looks up `{scenario_name}_em_*` and falls back
to `historical_em_*` if missing. The latter has historical
trajectories forward-filled at 2023, producing the large diffs.

**Affected**:
- `esm-ssp*` (8 scenarios; CO2-only ED variants of SSPs)
- `esm-allGHG-ssp*` (8 scenarios)
- `esm-scen7-*` (7 scenarios; the bundle has only `scen7-H_em_*` etc., no `esm-scen7-H_em_*`)
- `esm-allGHG-scen7-*` (7 scenarios; same)
- `scen7-*C` (7 scenarios; the "C" suffix marks CD intent but the bundle file is just `scen7-H_conc_*`)
- `esm-1pct-brch-*PgC` (3; would benefit from bundle lookup but Marit CSV parse-issue also blocks NO_MATCH)
- `esm-hist-cmip6`, `esm-allGHG-hist*` (3)

**Diff magnitudes**: 1-9 K (i.e. dominated by the wrong-file fallback)

**Fix**: extend the CICEROSCMPY2 adapter's `_build_scendata_list_bundle`
to strip protocol prefixes/suffixes when resolving bundle files:

```python
def _bundle_basename(scenario_name: str) -> str:
    # Try the exact name first (esm-allGHG-ssp370-lowCH4 has its own file),
    # then strip esm-allGHG-, then esm-, then a trailing C from scen7-*C.
    candidates = [scenario_name]
    if scenario_name.startswith("esm-allGHG-"):
        candidates.append(scenario_name[len("esm-allGHG-"):])
    elif scenario_name.startswith("esm-"):
        candidates.append(scenario_name[len("esm-"):])
    if scenario_name.startswith("scen7-") and scenario_name.endswith("C"):
        # scen7-HC -> scen7-H, scen7-MLC -> scen7-ML, scen7-LNC -> scen7-LN
        candidates.append(scenario_name[:-1])
    return candidates
```

Then loop `_pick_bundle_file(bundle_dir, [f"{c}_em_{gases_ep}" for c in candidates])`.

### 2. CO2-only ED mode needs non-CO2 emission masking (8 scenarios)

Even with the bundle-file lookup fixed, `esm-ssp245` (CO2-only ED) and
`esm-allGHG-ssp245` (all-GHG ED) point at the SAME `ssp245_em_*`
bundle file. The distinction is whether non-CO2 anthropogenic
emissions are active (all-GHG) or held at pre-industrial (CO2-only).

**Affected**: the `esm-ssp*` and `esm-scen7-*` families (15
scenarios; expected diff sign: warm — full GHG emissions where Marit
ran CO2-only). The all-GHG variants `esm-allGHG-*` should be fine
once the bundle-file lookup is fixed.

**Fix**: similar to the existing FaIRv2 `co2_only_scenarios` logic
in `build_emissions_df`, but applied at the CICEROSCMPY2 side via a
cfg flag that zeros non-CO2 columns in the emissions DataFrame
before passing to CICEROSCMPY2.

### 3. CD-only idealised forcing setup differs from Marit (6 scenarios)

`1pctCO2`, `1pctCO2-4xext`, `1pctCO2-cdr`, `abrupt-0p5xCO2`,
`abrupt-2xCO2`, `abrupt-4xCO2`: diffs 0.5 - 1.0 K. Magnitudes are
similar to what we saw with `esm-flat*` before the four-layer
idealised-suppression fixes (sunvolc / emstart / natemis / LUC).

**Hypothesis**: at least one of the four idealised suppressions
doesn't match Marit's actual reference setup for these CD-only
experiments. The `_is_idealised` rule fires on `1pctco2` and
`abrupt` so we're applying all four; Marit's reference may apply
fewer or differently. Most likely candidates:
- sunvolc=1 in her CD idealised runs (we set 0)
- different LUC file (she might use historical, not constant_zero)

**Fix**: align with Marit's run setup. Worth a 10-minute test
toggling `sunvolc` and `rf_luc_file` independently on `abrupt-4xCO2`
to confirm.

### 4. Small residuals around the WARN threshold (8 scenarios)

`esm-bell-*` (3), `esm-pi-CO2pulse`, `esm-pi-cdr-pulse`,
`piControl`, `esm-piControl`, `esm-allGHG-piControl`: max diffs
0.2 - 1.9 K, mean diffs -0.1 to +1.2 K.

`piControl` runs should produce essentially zero drift (constant
pre-industrial forcing throughout). A +1.2 K mean drift suggests the
model isn't actually being run at pre-industrial — likely the same
historical_concentration_fallback issue as the bundle-name lookup,
just smaller magnitude because the perturbation is smaller.

**Fix**: likely subsumed by the bundle-lookup fix.

## Per-failure-category PR plan

| PR | Scenarios fixed | Diff against Marit |
|---|---|---|
| B.1 — bundle-file prefix/suffix stripping in CICEROSCMPY2 | ~38 | Most large-diff failures collapse to ~100 mK (similar to esm-flat WARN level) |
| B.2 — CO2-only ED mask | ~15 (already fixed by B.1 + this) | Closes the `esm-ssp*` / `esm-scen7-*` gap |
| B.3 — idealised CD forcing alignment with Marit | 6 (1pctCO2, abrupt-*) | TBD; depends on which suppression Marit doesn't apply |
| B.4 — esm-1pct-brch CSV parser fix | 3 | Just an output-format edge case in the scorecard |

PR B.1 is the biggest unlock — most FAILs come from there. PR B.2 is
small and only needed for the CO2-only ED variants.

## Reproducing this scorecard

```bash
# (requires CICEROSCMPY2 bundle at $CICEROSCMPY2_BUNDLE_DIR/rcmip-march2026/)
scripts/run_rcmip3.py --models ciceroscmpy2 --members 10 \
    --scenario-set all --mode both --output-dir out/rcmip3
scripts/validate_against_marit.py --output-dir out/rcmip3
```

Wallclock for the sweep on a 2024 M3 laptop: ~5 min for ED, ~100 min
for CD (the carbon-cycle back-calculation runs ~30-50x per-member in
CD per the CICEROSCMPY2 adapter's `_OUTPUT_VARIABLES` doc).
