# Frozen 2023 replication protocol

This directory freezes the Phase 4 protocol before any 2023 analytic data are
read. The protocol is confirmatory and cannot use 2023 gSpan output, support
screens, directions, or p-values to choose what is tested.

The content-derived lock ID is recorded in `protocol.lock.json`.
Release tag: `phase4-2023-replication-v1.0.0`.

## Two frozen replication sets

The methodological vocabulary contains 484 exact 2024 motifs that had at
least 20% pooled support and independent-bootstrap
`P(support >= 20%) >= 0.90`. In 2023, every motif is evaluated directly
without rerunning gSpan. Reproduction requires at least 20% pooled support
among at least 40 motif-evaluable paired jurisdictions. Unevaluable motifs
remain in the denominator and count as not reproduced.

The 484 computational motifs reduce to 299 2024 graph-occurrence families.
Raw motifs remain the reproducibility target. Occurrence-family
representatives and closed motifs are the preferred interpretive units; they
are not additional hypothesis filters.

## Confirmatory hypotheses

The confirmatory family contains exactly three one-sided hypotheses:

1. `H1`, `motif_a79cab0e1b2c2fbf`: the
   arthritis–ischemic-heart-disease–diabetes–kidney–other-cancer tree is more
   frequent in higher SES graphs after density standardization.
2. `H2`, `motif_eeb4d747c66beaa4`: the
   ischemic-heart-disease–diabetes–kidney path is more frequent in higher SES
   graphs after density standardization.
3. `H3`, `motif_07019d963bc595b3`: the
   arthritis–COPD–current-asthma–depression–diabetes tree is more frequent in
   lower SES graphs after density standardization.

The effect is always lower-minus-higher density residual. Direction signs are
`-1`, `-1`, and `+1`, respectively. The primary test uses a studentized
Rademacher wild sign-flip with 99,999 common sign vectors, one-sided plus-one
Monte Carlo p-values, and Holm correction across the fixed three-test family
at alpha 0.05. A hypothesis with fewer than 40 evaluable state pairs receives
p = 1 and fails replication. No aggregate study-level success rule is defined.

Only the year-specific eligible 2023 jurisdiction analysis can determine
replication. The matched-2024-jurisdiction analysis is descriptive sensitivity
analysis and cannot rescue or reverse a primary decision.

This is not automatically an exact randomization test: exactness would require
componentwise sign symmetry of the state contrasts or within-state
lower/higher label exchangeability. Its stated justification is the
asymptotic studentized Rademacher wild bootstrap for an equally weighted mean
jurisdiction-level contrast at the mean-zero boundary. That argument assumes
independent jurisdiction contrasts, finite second moments, and no dominating
jurisdiction. Education strata are observational, so the exchangeability
condition is not asserted.

## Frozen graph construction

- Ten conditions and their 2023 coding are frozen in `conditions.yaml`.
- SES is education only: `_EDUCAG` 1–2 versus 3–4.
- Population strata require at least 1,000 unweighted respondents.
- Nodes require 500 valid observations and 30 cases.
- Dyads require 500 complete observations, 30 cases for each condition, and
  10 co-occurring cases.
- Stable edges require point weighted phi at least 0.12 and 500-replicate
  survey-bootstrap `P(phi >= 0.12)` at least 0.90.
- Bootstrap replicate weights are generated within state with the frozen
  strata, PSU, weight, lonely-PSU, nesting, and MSE settings.
- Every point-eligible dyad must have all 500 valid bootstrap estimates. A
  failure makes that population graph unusable; if either SES graph is
  unusable, the state is excluded from both primary and sensitivity analyses.
  Point-ineligible dyads are not submitted to the bootstrap.

The primary jurisdiction set is every prespecified state/DC jurisdiction that
passes the frozen paired eligibility rules in 2023. A sensitivity analysis
uses its intersection with the 47 jurisdictions in the frozen 2024 graph
database. Territories are prohibited. Any motif additionally requires every
one of its required edges to be eligible in both SES graphs. Other dyads among
its nodes do not affect motif evaluability.

## Motifs and density standardization

Motifs are connected, disease-labeled, undirected, non-induced, exact
required-edge patterns with three to five nodes. Extra host-graph edges do not
invalidate occurrence. Canonical node and edge order, compact JSON encoding,
full SHA-256, and truncated motif IDs are frozen in the registries.

For an evaluable motif with `r` required edges in a graph with `N` eligible
dyads and `m` selected stable edges, expected occurrence is
`choose(N-r, m-r) / choose(N, m)`. The analysis uses observed minus expected
occurrence and equal weighting across paired jurisdictions. If `m < r`,
observed occurrence, expected occurrence, and the residual are all zero.
`N < r` means the motif is not evaluable; `m > N` is a fatal integrity error.

This standardization controls graph edge opportunity and density. It does not
remove differences in edge-detection precision caused by sample size or
disease prevalence, does not identify a causal education effect, and does not
turn pairwise motifs into respondent-level higher-order disease combinations.
It is not age/sex standardized, and the sign-flip endpoint treats the selected
2023 graphs as fixed rather than propagating graph-selection uncertainty into
the p-value. The favorable 2024 bootstrap result was conditional frozen-edge
retention, not full graph-pipeline stability.

## Frozen execution

The only target command is:

```shell
multimorbidity-motifs phase4
```

It loads `analysis_phase4.yaml`, `graph_phase4.yaml`, and `conditions.yaml`,
rejects any mismatch with their locked copies, requires the frozen Python and
R package versions, and verifies downloader metadata, the frozen CDC URL, XPT
hash and byte count, exactly 433,323 records, and all required variables. It
then constructs the 2023 graphs once, evaluates all 484 motifs without gSpan,
and runs the three fixed directional tests. There are no skip,
threshold-tuning, motif-selection, or alternate-SES flags. Execution is
refused unless `HEAD` is the clean commit named by the annotated release tag.

## Integrity and amendments

`protocol.lock.json` identifies the payload, including this README.
`SHA256SUMS` covers every file except itself. Checksums detect changes but are
not external proof of a pre-access freeze; the containing commit and annotated
release tag must be pushed before any 2023 analytic data are accessed.
Validation is available through:

```python
from mm_motifs.replication import validate_replication_protocol

validate_replication_protocol("protocols/phase4_2023_replication")
```

Any amendment requires a new protocol version, reason, timestamp, lock, and
git tag before 2023 results are examined. The files here must never be edited
in place after the freeze.
