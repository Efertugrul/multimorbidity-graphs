# Phase 3 exploratory motif baseline

This archive freezes the 2024 education-only Phase 3 result produced by config
digest `b755affa0537` at commit
`bd2d7290fc36cfe4a464e2fdb1c58720aec529d2`.

## Audit status

- Phase 2.6 raw-edge retention criterion: **HOLD**
- Phase 2.6 structural feasibility: **GO**
- Phase 3 execution: **EXPLORATORY_COMPLETE**
- Automatic Phase 3 motif GO gate: not prespecified
- 2023 replication: not run

The exploratory evidence supports freezing a candidate motif set for later
replication. It does not convert the motif or SES results into confirmatory
findings.

## Frozen graph and mining definitions

- 94 graphs from 47 eligible U.S. states/DC with paired education SES strata
- Frozen edges: point weighted phi >= 0.12 and
  `P_boot(phi >= 0.12) >= 0.90`
- Connected, undirected, non-induced, labeled motifs
- Motif size: 3–5 nodes
- Pooled minimum support: 10%, 20%, and 30%
- Primary pooled support: 20%
- Backend: `fast-gspan` 0.1.3 using gBolt gSpan

## Main results

- 1,098 motifs at 10% support, 542 at 20%, and 342 at 30%
- 448 distinct graph-occurrence classes, 518 closed motifs, and 114 maximal
  motifs
- At 20% support, 497/542 motifs had
  `P_boot(support >= 20%) >= 0.80`; 479/542 had probability >= 0.90
- Frozen-edge gSpan reruns had median 20%-support vocabulary retention 0.950
  and a 2.5th percentile of 0.885
- All motifs had positive support Z-scores against fixed-node/edge-count nulls;
  94.6% were positive against degree-preserving edge-swap nulls
- Three of 542 primary motifs had paired density-residualized maxT
  permutation `P < 0.05`; these comparisons remain exploratory

## Bootstrap estimands

The primary robustness analysis allows only edges in the frozen stable graph
to drop when their aligned survey-bootstrap phi is below 0.12. Unselected
edges cannot enter.

The separate raw-threshold sensitivity allows every eligible edge with
replicate phi >= 0.12 to enter. At 20% support, those realizations produced a
median 4,930 motifs, 4,388 novel relative to baseline, and motif-set Jaccard
0.110. This sensitivity result is not the primary stable-graph estimand.

## Archive scope

The archive contains the frozen graph database and index, enriched motif
catalog, motif-shape and redundancy summaries, fixed-vocabulary and
discovery-set stability results, raw-threshold sensitivity, both density
nulls, paired SES permutation results, figures, run log, report, and manifest.
`SHA256SUMS` covers every archived file except itself.

Density-null p-values are selection-conditional because the motifs were mined
from the observed graphs. Independent 2023 testing is required for
replication.
