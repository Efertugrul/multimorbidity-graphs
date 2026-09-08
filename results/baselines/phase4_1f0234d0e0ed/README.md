# BRFSS 2023 independent-year replication

This archive freezes the Phase 4 result produced by config digest
`1f0234d0e0ed` from 433,323 BRFSS 2023 records.

## Audit identity

- Frozen protocol: `phase4_2023_cc5bbcc08283`
- Protocol SHA-256:
  `cc5bbcc0828396fe028317e9901f76cf5ab36dc12fff7d7a429333485245ccff`
- Execution release: `phase4-2023-replication-v1.0.1`
- Execution commit: `c9da1ea70d1cbf48fcbcaa17d3871baddbf2760a`
- Documented deviation: `PH4-D001`
- Motif discovery in 2023: not performed

The original protocol and `v1.0.0` tag remain unchanged. The first execution
stopped during pre-harmonization runtime validation because YAML represented
Survey version 4.5 as a number and the runtime represented the same version as
text. `PH4-D001` authorized string normalization for version comparison only.
No graph, motif, SES statistic, or analytic result existed when the deviation
was defined.

## Graph database

- 47 paired jurisdictions and 94 education-stratified graphs
- 951 selected stable edges
- All 94 graphs completed every point-eligible dyad bootstrap
- Median selected edges: 11 in lower SES and 9 in higher SES
- Median eligible-dyad density: 25.0% in lower SES and 20.0% in higher SES
- District of Columbia and Nevada failed the frozen paired sample-size rule
- Kentucky and Pennsylvania had no 2023 reporting data

## Methodological replication

Of the 484 frozen 2024 motifs, 448 reproduced at least 20% pooled support
with at least 40 motif-evaluable paired jurisdictions: 92.6% retention.

- 2024–2023 support Spearman correlation: 0.914
- Median absolute support difference: 6.38 percentage points
- Motif evaluability range: 46–47 paired jurisdictions
- Three-node motifs: 42/42 reproduced
- Four-node motifs: 124/134 reproduced
- Five-node motifs: 282/308 reproduced
- Frozen occurrence-family representatives: 275/299 reproduced

All three focal confirmatory motifs passed the methodological
pooled-support reproduction criterion. This does not imply that their frozen
SES contrasts replicated.

## Confirmatory SES replication

The primary analysis used 47 equally weighted paired jurisdictions, the frozen
lower-minus-higher density-residual contrast, 99,999 common Rademacher sign
vectors, one-sided frozen directions, and Holm correction across three tests.

- `H1`, higher-SES arthritis–IHD–diabetes–kidney–other-cancer tree:
  effect `-0.04490`, one-sided `p = 0.26384`, Holm `p = 0.26384`;
  **not replicated**.
- `H2`, higher-SES IHD–diabetes–kidney path:
  effect `-0.17053`, one-sided `p = 0.05308`, Holm `p = 0.10616`;
  **not replicated**.
- `H3`, lower-SES arthritis–COPD–current-asthma–depression–diabetes tree:
  effect `+0.27425`, one-sided `p = 0.00105`, Holm `p = 0.00315`;
  **replicated**.

All three effects were direction-concordant, but direction alone was not the
frozen replication rule. `H2` remains a failed replication despite its smaller
raw p-value. The descriptive 45-jurisdiction matched sensitivity analysis had
the same Holm-threshold pattern—H3 only—and cannot alter the primary decisions.

No aggregate study-level success gate was preregistered. The result is
therefore reported hypothesis by hypothesis: one of three SES signals
replicated.

## Interpretation

The 2024 multimorbidity motif vocabulary shows strong temporal reproducibility
within BRFSS. Confirmatory structural SES contrasts are selective rather than
universal: the lower-SES five-condition motif replicated, while the two
higher-SES signals did not meet their frozen Holm-corrected criteria.

This is independent-year replication within BRFSS, not external cohort
validation. The graph contrasts are observational, are not age/sex
standardized, and do not establish causal SES effects or respondent-level
higher-order disease combinations.

`SHA256SUMS` covers every archived file except itself.
