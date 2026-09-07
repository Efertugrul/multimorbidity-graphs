# Corrected Phase 3 exploratory motif baseline

This archive freezes the 2024 education-only Phase 3 result produced by config
digest `50bda756d7f8` at source commit
`5a5846ff21e09393953af51310e59ca2b14a4e27`.

It supersedes [`phase3_b755affa0537`](../phase3_b755affa0537/). The earlier
archive remains available only as an audit record.

## Corrections

- Motif robustness uses an independent 500-replicate respondent-bootstrap
  evaluation bank with seed `20240905`. The Phase 2.6 selection bank used seed
  `20240902`.
- Required dyads must be eligible in both SES graphs for a state to enter a
  motif's support denominator.
- Motifs require at least 40 paired-state opportunities.
- Fixed-edge nulls condition on each graph's eligible dyad count and edge
  count.
- Degree-preserving swaps cannot enter ineligible dyads.
- SES permutations use the motif-specific paired-state set, one common
  Rademacher sign per state across all motifs, and report Monte Carlo standard
  errors.

There were 39 ineligible graph–dyad cells across 16 graphs. None involved the
required edges of the retained frequent motifs, so all 1,098 retained motifs
were evaluable in all 47 paired states. Opportunity correction therefore left
the motif family unchanged but corrected density expectations and made the
workflow valid for motifs with incomplete dyad opportunity.

## Audit status

- Phase 2.6 raw-edge retention criterion: **HOLD**
- Phase 2.6 structural feasibility: **GO**
- Corrected Phase 3 execution: **EXPLORATORY_COMPLETE**
- Automatic Phase 3 motif GO gate: not prespecified
- 2023 replication: not run

The corrected evidence supports motif-stage structural feasibility and later
prospective replication. It does not establish confirmatory or causal SES
findings.

## Frozen definitions

- 94 graphs from 47 eligible U.S. states/DC with paired education SES strata
- Frozen edges: point weighted phi >= 0.12 and
  `P_boot(phi >= 0.12) >= 0.90`
- Connected, undirected, non-induced, disease-labeled motifs
- Motif size: 3–5 nodes
- Pooled minimum support: 10%, 20%, and 30%
- Primary pooled support: 20%
- Backend: `fast-gspan` 0.1.3 using gBolt gSpan

The primary bootstrap analysis estimates conditional retention of edges in the
frozen graph. It is not full stable-pipeline rediscovery, which would require
an outer survey bootstrap with an inner bootstrap that re-estimates every
edge's 0.90 selection frequency.

## Main results

- 1,098 motifs at 10% support, 542 at 20%, and 342 at 30%
- 448 distinct graph-occurrence classes, 518 closed motifs, and 114 maximal
  motifs
- At 20% support, 499/542 motifs had
  `P_boot(support >= 20%) >= 0.80`; 484/542 had probability >= 0.90
- Frozen-edge gSpan reruns had median 20%-support vocabulary retention 0.950
  and a 2.5th percentile of 0.887
- All motifs had positive support Z-scores against eligible-dyad fixed-edge
  nulls; 94.5% were positive against constrained degree-preserving nulls
- Three of 542 primary motifs had paired density-residualized maxT
  permutation `P < 0.05`; these comparisons remain exploratory

The degree-null chains completed all requested accepted swaps. Across graphs,
the 200 realizations produced 3–200 unique graphs, with median 193; every graph
changed in at least 74% of realizations.

## Raw-threshold sensitivity

The separate sensitivity permits every eligible edge with replicate phi >=
0.12 to enter. At 20% support it produced a median 4,999.5 motifs, including
4,457.5 not in the frozen vocabulary, and median motif-set Jaccard 0.108.
This is evidence that the raw threshold is not a substitute for the stable
graph definition.

## Interpretation limits

Motifs are recurring configurations of pairwise disease associations. They do
not show that respondents jointly have all motif diseases or establish a
higher-order interaction.

The within-state label-swap analysis estimates an education-stratum structural
contrast under within-state exchangeability. Education is not randomized, the
graphs are not age/sex-standardized, and density conditioning does not remove
edge-detection differences caused by sample size or disease prevalence.

Density-null p-values are selection-conditional because motifs were mined from
the observed graphs. The bootstrap probability cutpoints are descriptive and
are not automatic selection gates.

## Archive scope

The archive contains frozen configuration snapshots, the graph database and
index, enriched motif catalog, motif shape and redundancy summaries,
independent fixed-vocabulary and discovery-set stability, raw-threshold
sensitivity, both density nulls, degree-chain diagnostics, paired SES
permutations, figures, run log, report, manifest, and a comparison with the
superseded result. `SHA256SUMS` covers every archived file except itself.
