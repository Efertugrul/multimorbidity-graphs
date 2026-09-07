# Multimorbidity Motifs

This repository builds population-specific multimorbidity association graphs
from BRFSS data to evaluate whether frequent subgraph mining is a viable next
step for identifying socioeconomic signatures of multimorbidity.

Current scope: completed Phases 0–3 plus a frozen Phase 4 replication protocol.
Phase 3 performs exploratory frequent-subgraph discovery, motif robustness
checks, density nulls, and paired SES comparison. No 2023 analytic data were
used to create the replication freeze.

## Scientific boundary

A graph represents a population stratum, currently `state × SES category`.
Nodes are harmonized chronic conditions. Edges are independently estimated
disease-pair associations within each population.

The Phase 2 estimator is a descriptive survey-weighted phi coefficient.
`_LLCPWT` is used in prevalence and association estimates, while `_STSTR` and
`_PSU` are retained in the harmonized data and population registry. Phase 2
does not produce design-based standard errors or inferential p-values. Its
purpose is to assess graph heterogeneity and density before investing in motif
mining.

Edges represent statistical disease-association structure, not causal links,
biological pathways, or individual disease trajectories.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The equivalent Conda environment, including R 4.4 and the R `survey` package
required for Phases 2.5–2.6, is defined in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate multimorbidity-motifs
```

## Run Phases 1–2

Download the official 2024 CDC BRFSS SAS transport file:

```bash
multimorbidity-motifs download
```

Run the audit and prototype separately:

```bash
multimorbidity-motifs audit
multimorbidity-motifs prototype
```

Or run download, audit, and prototype together:

```bash
multimorbidity-motifs phase02
```

The official 2024 source is
<https://www.cdc.gov/brfss/annual_data/annual_2024.html>. Raw and derived
respondent-level data are ignored by Git.

## Run Phase 2.5

Phase 2.5 uses separate configuration files so the five-state Phase 2 run
remains reproducible:

```bash
multimorbidity-motifs phase25
```

Adjusted models checkpoint by state. Re-running the same configuration resumes
an interrupted point-estimate or bootstrap stage from completed states.

The primary geography is the available U.S. states plus DC; Guam, Puerto Rico,
and the U.S. Virgin Islands are excluded. A state enters a given SES analysis
only when both its lower and higher strata have at least 1,000 respondents.
Every population registry also reports Kish effective sample size.
The 1,000 threshold is an unweighted eligibility rule; effective sample size
is reported separately and is not silently substituted for raw `n`. Pairing
is evaluated independently for each SES definition, and cross-definition
sensitivity uses only jurisdictions eligible under both definitions.

Education is primary. Income is a non-bootstrap sensitivity analysis. The
three prespecified point-estimate rules are:

- survey-weighted phi at least `0.08`;
- survey-weighted phi at least `0.12`;
- age/sex-adjusted survey logistic association with positive OR, lower 95%
  confidence bound above 1, and within-graph BH `q <= 0.05`.

Adjusted logistic models use `_LLCPWT`, `_STSTR`, and `_PSU` through R
`survey::svyglm`. The graph uses a fixed condition-registry direction and
records the reverse-direction OR as a concordance diagnostic. Edges remain
undirected statistical associations; neither direction is interpreted
causally.
Lonely PSUs, including domain-induced lonely PSUs, use the survey package's
`adjust` convention and are recorded in the run manifest.

Education adjusted-edge candidates receive 200 bootstrap replicate fits.
The calibration-stable rule additionally requires at least 90% of replicate
ORs above 1. Two hundred replicates are for calibration only; paper-relevant
edges or motifs should be reconfirmed with 500–1,000 replicates.
Wilson intervals are reported to expose uncertainty in each estimated
stability proportion.

Similarity outputs include raw edge-set Jaccard and a null-standardized value
from the exact random-edge null conditional on both graph edge counts. The SES
permutation test independently swaps lower and higher labels within each state
while preserving the paired graphs. Its null is that graph structure is
unrelated to SES label conditional on state.

## Run Phase 2.6

Phase 2.6 tests the sparse `phi >= 0.12` rule without changing its threshold:

```bash
multimorbidity-motifs phase26
```

It uses education-primary graphs and 500 design-aware bootstrap replicates.
Every disease pair reports point phi, `P(phi > 0)`, `P(phi >= 0.12)`, and the
bootstrap median and percentile interval. The stable graph requires both
point `phi >= 0.12` and selection stability at least `0.90`.

The `0.10`, `0.12`, and `0.14` graphs are compared only to test local topology
continuity; they are not candidate thresholds for post hoc selection. Frozen
Phase 2.5 adjusted models validate whether retained phi edges remain positive
after age/sex adjustment. They do not generate or filter Phase 2.6 graphs.

## Run Phase 3

Phase 3 mines the frozen bootstrap-stable `.12` graphs:

```bash
multimorbidity-motifs phase3 --workers 8
```

The graph definition remains point `phi >= 0.12` and
`P_boot(phi >= 0.12) >= 0.90`. The failed Phase 2.6 80% raw-edge retention
criterion remains recorded as **HOLD**, while structural feasibility is
recorded separately as **GO**.

The gBolt gSpan backend mines connected, undirected, non-induced motifs of
3–5 nodes at pooled support levels of 10%, 20%, and 30%. Support uses only
states where every required dyad is eligible in both SES graphs, with at
least 40 paired states required. Node labels are canonical diseases and every
edge has one association label. A motif records recurring pairwise
association structure; it is not evidence that respondents jointly have all
motif diseases.

Phase 3 generates an independent 500-replicate evaluation bank, separate from
the Phase 2.6 bank that selected frozen edges. In the primary robustness
analysis, frozen stable edges may drop when their replicate phi falls below
`.12`, but unselected edges cannot enter. This estimates conditional
edge-retention robustness, not full stable-pipeline selection stability.
gSpan is rerun on all aligned realizations. A separate raw-threshold
sensitivity allows every eligible edge crossing `.12` to enter. Neither
analysis performs a nested outer/inner re-estimation of the 0.90 filter.

Density diagnostics condition on each graph's eligible dyad set and edge
count, with degree-preserving edge swaps constrained to eligible dyads as a
sensitivity analysis.

SES comparisons are exploratory, use density-residualized motif occurrence,
and swap lower/higher labels only within motif-evaluable paired
jurisdictions. They estimate an education-stratum structural contrast under
within-state exchangeability, not a causal SES effect. MaxT and BH adjustments
are reported. Density conditioning does not remove SES differences in
edge-detection precision caused by sample size or disease prevalence. Motif
discovery never uses SES direction or p-values, and frozen motifs still
require 2023 replication.

## Frozen 2023 replication protocol

The preregistered, machine-validated protocol is frozen under
[`protocols/phase4_2023_replication/`](protocols/phase4_2023_replication/).
It contains exact condition, SES, graph, bootstrap, motif, density,
jurisdiction, permutation, direction, and multiplicity definitions.

Two non-overlapping roles are fixed:

- 484 independently bootstrap-robust 2024 motifs form the methodological
  vocabulary. Their exact 2023 support is evaluated without rerunning gSpan.
- Three exact, directional 2024 SES motifs form the confirmatory family.
  They use a one-sided studentized Rademacher wild sign-flip and Holm
  correction across three tests. Only the primary 2023 jurisdiction set can
  determine replication; the matched-jurisdiction sensitivity cannot rescue
  a failure.

The 484 raw motifs are computational patterns, not 484 independent biological
phenomena. They are grouped into 299 occurrence-equivalence families, with
family representatives and closed motifs reserved for interpretation. This
grouping does not alter the exact motif-level replication targets.

The sign-flip endpoint is exact only under componentwise state-contrast sign
symmetry or within-state SES-label exchangeability; its stated justification
is an asymptotic jurisdiction-level wild bootstrap, not randomization of
observational education groups. The graphs are not age/sex standardized, and
graph-selection uncertainty is not fully propagated into the confirmatory
p-values.

Any protocol amendment requires a new version, lock, commit, and release tag
before 2023 results are examined. In-place changes fail checksum validation.
The frozen runner is `multimorbidity-motifs phase4`; it exposes no config,
year, threshold, or motif-selection overrides. It also requires matching
downloader metadata, the frozen CDC URL, XPT hash and byte count, 433,323
records, and all required variables before analysis.

Execution deviation `PH4-D001` was recorded after the first 2023 run stopped
at pre-harmonization runtime validation. YAML had parsed the frozen Survey
package version as a number while the runtime reported the same version as
text. Release `phase4-2023-replication-v1.0.1` authorizes string normalization
for version comparison only; no scientific, graph, motif, or inferential logic
changed, and no analytic results existed when the deviation was defined.

## Configuration

- `configs/conditions.yaml`: year-specific condition registry and coding rules
- `configs/analysis.yaml`: data source, survey variables, SES definitions,
  prototype states, sample-size rules, and output roots
- `configs/graph.yaml`: association estimator, edge/node criteria,
  visualization, and viability thresholds
- `configs/analysis_phase25.yaml`: all-state/DC and dual-SES calibration scope
- `configs/graph_phase25.yaml`: fixed edge scenarios, bootstrap, permutation,
  and neutral GO/HOLD criteria
- `configs/graph_phase26.yaml`: fixed phi stability, threshold-neighborhood,
  adjustment-concordance, and final gate criteria
- `configs/graph_phase3.yaml`: frozen graph input, gSpan support spectrum,
  motif bootstrap, density nulls, and paired SES permutation settings
- `configs/analysis_phase4.yaml`: locked education-only 2023 target scope
- `configs/graph_phase4.yaml`: locked graph and confirmatory endpoint rules

The provisional default uses ten conditions, California, Florida, Michigan,
New York, and Texas, and two education strata:

- lower: did not graduate high school or graduated high school
- higher: attended or graduated from college/technical school

These are exploratory categories, not irreversible analytical decisions. An
income-based alternative is already configured.

The condition registry also records a provisional cross-year availability,
prevalence, and analytic-missingness selection policy. The Phase 1 table shows
the observed criteria rather than silently excluding candidates.

The default edge rule requires weighted phi at least `0.08`, at least 500
complete observations, at least 30 cases of each condition, and at least 10
unweighted co-occurrences. These values are configuration, not claims of
clinical or inferential significance.

## Phase 1 outputs

Audit outputs are written to:

```text
results/audit/year=2024/run=<configuration digest>/
```

They include:

- `variable_dictionary.csv`
- `candidate_chronic_conditions.csv`
- `ses_variables.csv`
- `survey_design_variables.csv`
- `state_sample_sizes.csv`
- `disease_prevalence.csv`
- `missingness_report.csv`
- `manifest.json`
- `run.log`

The missingness report distinguishes raw system missingness from analytic
missingness after configured unknown/refused codes are applied.

## Phase 2 outputs

Prototype outputs are written to:

```text
results/prototype/year=2024/run=<configuration digest>/
```

They include:

- `population_registry.csv`
- `edge_table.csv`
- `disease_prevalence_by_graph.csv`
- `graph_statistics.csv`
- `structural_similarity.csv`
- inspectable GraphML and node-link JSON graph files
- one network image per eligible population and a Jaccard heatmap
- `viability_report.json`
- `manifest.json`
- `run.log`

The motif occurrence matrix is not created because it belongs after the
Phase 2 viability decision.

## Phase 2.5 outputs

Calibration outputs are written to:

```text
results/phase25/year=2024/run=<configuration digest>/full/
```

`--skip-bootstrap` writes an isolated `point_only/` report. R model caches are
content-addressed by the data, jobs, script, parameters, and R package
versions, so the two report modes cannot mix stale artifacts.

Principal outputs include:

- paired eligibility and Kish effective sample sizes;
- weighted-phi and survey-adjusted pair tables;
- 200-replicate edge-stability estimates;
- graph statistics and edge support by rule;
- raw and density-adjusted similarity;
- the within-state SES-label-swap permutation distribution;
- education-versus-income sensitivity;
- California lower-SES outlier diagnostics;
- a neutral pre-gSpan `viability_report.json`.

The GO/HOLD decision uses graph count, sparsity, edge stability, recurrent
structure, and heterogeneous-but-comparable graph structure. SES permutation
significance is reported but explicitly excluded from rule selection and the
technical GO/HOLD decision.
The decision is scoped to the configured bootstrap-stable primary rule.
Alternative threshold rules receive separate technical profiles and are not
promoted without their own stability calibration.

## Phase 2.5 calibration result

The completed `fff54b040741` run returned **HOLD before gSpan**. All 2,807
adjusted education candidate edges passed 200-replicate sign stability, so the
stable rule remained too dense: median 31 of 45 edges and maximum density
0.889. The `phi_012` comparison was technically sparse (median 18 edges,
maximum density 0.60), but was not stability-calibrated and is not promoted as
the primary rule.

The lightweight evidence, figures, checksums, and interpretation are frozen in
[`results/baselines/phase25_fff54b040741/`](results/baselines/phase25_fff54b040741/).
No frequent subgraph mining has been run.

## Phase 2.6 outputs

Final-gate outputs are written to:

```text
results/phase26/year=2024/run=<configuration digest>/full/
```

They include edge-level phi bootstrap stability, raw and stable graph
databases, threshold-transition diagnostics, adjusted-model concordance,
density-adjusted similarity, recurrent edge support, figures, and a neutral
`viability_report.json`. A GO requires all prespecified technical gates;
neither SES separation nor adjusted-model significance is used to tune the
graph rule.

## Phase 2.6 calibration result

The completed `172713cfdaf6` run returned **HOLD before gSpan**. Eight of nine
technical criteria passed, but only 1,057 of 1,733 raw `phi >= 0.12` edges
met the fixed 0.90 bootstrap-stability rule, a retention fraction of 0.610
versus the prespecified 0.80 gate.

Instability was concentrated near the cutoff: 560 of 676 unstable edges had
point `phi < 0.15`. This supports a prespecified stronger-threshold experiment,
not post hoc promotion of `.15`. The current diagnostic estimates
`P_boot(phi >= 0.12)`; a `.15` graph must be evaluated with newly computed
`P_boot(phi >= 0.15)` while keeping both stability gates unchanged.

The aggregate evidence, figures, checksums, and interpretation are frozen in
[`results/baselines/phase26_172713cfdaf6/`](results/baselines/phase26_172713cfdaf6/).
That archive itself stops before frequent subgraph mining.

## Phase 3 outputs

Exploratory motif outputs are written to:

```text
results/phase3/year=2024/run=<configuration digest>/full/
```

They include the gSpan database and label mapping, motif and redundancy
catalogs, support spectra, replicate-specific edge masks, motif-support
bootstrap distributions, gSpan discovery-set stability, fixed-density and
degree-preserving nulls, paired density-adjusted SES permutations, figures,
and an explicit exploratory report. Selection-conditional null probabilities
are diagnostics rather than confirmatory p-values.

The fixed-edge and degree-preserving null summaries are not the headline
validation because motifs were selected for observed recurrence. The primary
reproducibility evidence is the independent survey-bootstrap evaluation, with
independent-year replication prespecified as the confirmatory test.

## Reproducibility

Output directories are deterministic hashes of all three configuration files.
Each manifest records the dataset year, condition registry, population
definition, edge rule, sample-size threshold, software version, Git commit
when available, random seed, timestamp, and the explicit phase boundary.

The 2023 paths and stable condition mappings are configured for later
replication, but 2023 is not run as part of the current milestone:

```bash
multimorbidity-motifs download --year 2023
```

## Tests

```bash
pytest
```

Review `viability_report.json`, graph statistics, edge tables, and network
figures before authorizing any Phase 3 implementation. The report emits a
density caution when any prototype graph reaches the configured `0.70`
threshold, even when the median-density viability criterion passes.

The exact original Phase 2 heatmap and lightweight scientific outputs are
archived under `results/baselines/phase2_365b9697fd28/`. PNG checksums are
provenance only; regression tests target scientific tables and values rather
than renderer-dependent image bytes.
