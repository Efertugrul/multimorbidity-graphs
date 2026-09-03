# Multimorbidity Motifs

This repository builds population-specific multimorbidity association graphs
from BRFSS data to evaluate whether frequent subgraph mining is a viable next
step for identifying socioeconomic signatures of multimorbidity.

Current scope: Phases 0–2 only. The code intentionally stops before gSpan,
motif discovery, discriminative motif testing, replication, and sensitivity
analysis.

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

The equivalent Conda environment is defined in `environment.yml`.

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

## Configuration

- `configs/conditions.yaml`: year-specific condition registry and coding rules
- `configs/analysis.yaml`: data source, survey variables, SES definitions,
  prototype states, sample-size rules, and output roots
- `configs/graph.yaml`: association estimator, edge/node criteria,
  visualization, and viability thresholds

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
