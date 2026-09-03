# Multimorbidity Motifs

This repository builds population-specific multimorbidity association graphs
from BRFSS data to evaluate whether frequent subgraph mining is a viable next
step for identifying socioeconomic signatures of multimorbidity.

Current scope: Phases 0–2.5 only. Phase 2.5 calibrates graph construction and
tests graph-level SES structure before gSpan. The code intentionally stops
before motif discovery, discriminative motif testing, and replication.

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
required for Phase 2.5, is defined in `environment.yml`:

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

## Configuration

- `configs/conditions.yaml`: year-specific condition registry and coding rules
- `configs/analysis.yaml`: data source, survey variables, SES definitions,
  prototype states, sample-size rules, and output roots
- `configs/graph.yaml`: association estimator, edge/node criteria,
  visualization, and viability thresholds
- `configs/analysis_phase25.yaml`: all-state/DC and dual-SES calibration scope
- `configs/graph_phase25.yaml`: fixed edge scenarios, bootstrap, permutation,
  and neutral GO/HOLD criteria

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
