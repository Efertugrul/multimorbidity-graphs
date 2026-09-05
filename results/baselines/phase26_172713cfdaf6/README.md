# Phase 2.6 phi-stability calibration result

This directory freezes the aggregate scientific outputs from the 2024
education-primary calibration with configuration digest `172713cfdaf6`. The
run was produced from clean commit `6663f67` and stopped before frequent
subgraph mining.

## Decision

**HOLD before gSpan.**

Eight of nine prespecified technical criteria passed. The sole failure was
selection stability: 1,057 of 1,733 raw `phi >= 0.12` edges met
`P_boot(phi >= 0.12) >= 0.90`, a retention fraction of 0.610 versus the fixed
0.80 gate.

The stable graphs remained structurally usable:

- 94 eligible state/DC × education-SES graphs;
- median 11 edges, range 2–21;
- median density 0.244 and maximum density 0.467;
- median raw Jaccard 0.538;
- median density-adjusted Jaccard z-score 5.364;
- 16 variably recurrent edges.

Topology continuity, adjusted-model concordance, graph count, sparsity,
meaningful edge count, heterogeneity/comparability, and recurrent-backbone
criteria all passed.

## Threshold-location diagnostic

Point phi and bootstrap stability were strongly associated
(`Spearman rho = 0.892`). Instability was concentrated near the threshold:

- all 205 edges from `.12–.13` failed the 0.90 stability gate;
- 199 of 201 edges from `.13–.14` failed;
- 156 of 190 edges from `.14–.15` failed;
- 560 of all 676 unstable edges (82.8%) had point `phi < 0.15`;
- stability against the old `.12` cutoff reached 87.2% at `.16–.18` and
  96.1% at `.18–.20`.

This supports threshold placement rather than graph methodology as the
immediate calibration problem. It does not validate `phi >= 0.15`: the
reported probabilities remain `P_boot(phi >= 0.12)`. A prespecified `.15`
experiment must recompute `P_boot(phi >= 0.15)` while retaining the 0.90
edge-stability and 0.80 retention gates.

## Adjusted validation

Of the 1,733 raw phi edges, 1,732 (99.94%) were positive in both directions of
the frozen age/sex-adjusted models, and 1,577 (91.00%) met their adjusted FDR
rule. All 1,057 stable phi edges were positive in both directions; 1,024
(96.88%) met the adjusted FDR rule. Adjusted models were validation only and
did not select Phase 2.6 edges.

## Archive scope

The archive contains configurations, manifest, viability result, aggregate
population and edge tables, bootstrap diagnostics, topology and concordance
summaries, structural-similarity evidence, figures, and the run log. Raw
BRFSS data, harmonized respondent-level data, R checkpoints, and serialized
graphs are not versioned.

`SHA256SUMS` records byte-level provenance, including PNGs. Automated tests
assert scientific values and configuration identity rather than exact image
hashes.
