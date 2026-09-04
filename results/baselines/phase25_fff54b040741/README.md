# Phase 2.5 calibration result

This directory freezes the lightweight scientific outputs from the 2024
all-state/DC calibration with configuration digest `fff54b040741`. The run was
produced from commit `b04ab0b` and stopped before frequent subgraph mining.

## Decision

**HOLD before gSpan.**

The bootstrap-stable education-primary adjusted rule produced 94 graphs, but
they remained too dense for the prespecified technical target:

- median 31 of 45 possible edges;
- median density 0.689;
- maximum density 0.889;
- 45 of 94 graphs at or above the 0.70 density caution threshold.

The decision is based on sparsity, stability, recurrence, and comparability.
SES permutation significance is explicitly excluded from the decision.

## Main findings

- Education produced 94 eligible paired graphs across 47 jurisdictions.
- Income produced 96 eligible paired graphs across 48 jurisdictions.
- Median education Kish effective sample size was 1,976.9; the minimum was
  295.7, with one eligible population below 500.
- All 2,807 adjusted education candidate edges completed 200 valid bootstrap
  replicates.
- Sign stability ranged from 0.935 to 1.000, so every adjusted candidate edge
  passed the 0.90 rule. The stability filter therefore provided no sparsity.
- Thirty bootstrap percentile intervals included zero. Stability selection
  used the prespecified sign proportion, not an interval-bound criterion.
- The conservative `phi_012` rule met the technical sparsity target with a
  median of 18 edges and maximum density of 0.60, but it did not receive the
  prespecified stability bootstrap and is not promoted as the primary rule.
- Density-adjusted similarity remained positive, showing recurrent structure
  beyond what graph density alone predicts.
- Paired within-state lower/higher label swaps found same-SES similarity above
  the conditional null for all reported rules (`p = 0.0001`). This is an
  empirical result, not a rule-selection criterion.
- Forward and reverse adjusted logistic directions were concordant for
  4,219/4,230 education pairs and 4,303/4,320 income pairs. Both OR directions
  remain diagnostics for undirected, non-causal edges.

## California lower-education diagnostic

The original California lower-SES outlier was rule-sensitive:

- `phi_008`: 21 edges, density 0.467, robust raw-similarity z = -5.24;
- `phi_012`: 15 edges, density 0.333, density-adjusted robust z = -0.36;
- adjusted stable: 24 edges, density 0.533.

Ten phi estimates were within 0.02 below the original 0.08 threshold.
California also had lower prevalence than same-SES peers for other cancer,
arthritis, and current asthma. Denominators were adequate, node validity was
at least 98.3%, and the shared coding registry provides no evidence of a
state-specific coding error. The outlier is therefore most consistent with
prevalence composition and threshold sensitivity rather than instability or
poor effective sample size.

## Archive scope

The archive includes configurations, the manifest, viability result,
population registry, compressed full edge and structural-similarity tables,
bootstrap stability, rule statistics, permutation summaries, outlier
diagnostics, and figures. Raw BRFSS data, harmonized respondent-level data,
and serialized graphs are not versioned here.

`SHA256SUMS` records byte-level provenance, including PNGs. Automated tests
assert scientific values and configuration identity, not exact PNG hashes.
