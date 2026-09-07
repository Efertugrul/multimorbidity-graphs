args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7) {
  stop("Expected data, jobs, output, replicates, threshold, seed, workers")
}

data_path <- args[[1]]
jobs_path <- args[[2]]
output_path <- args[[3]]
replicate_count <- as.integer(args[[4]])
selection_threshold <- as.numeric(args[[5]])
base_seed <- as.integer(args[[6]])
worker_count <- as.integer(args[[7]])

if (!isTRUE(all.equal(selection_threshold, 0.12))) {
  stop("Phase 2.6 selection threshold must equal 0.12")
}

suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu = "adjust")
options(survey.adjust.domain.lonely = TRUE)

data <- fread(data_path, na.strings = c("", "NA", "<NA>"))
jobs <- fread(jobs_path)

replicate_phi <- function(values, disease_a, disease_b, replicate_weights) {
  source <- as.numeric(values[[disease_a]])
  target <- as.numeric(values[[disease_b]])
  complete <- is.finite(source) & is.finite(target)
  source <- source[complete]
  target <- target[complete]
  weights <- replicate_weights[complete, , drop = FALSE]
  totals <- colSums(weights)
  source_prevalence <- as.numeric(crossprod(source, weights)) / totals
  target_prevalence <- as.numeric(crossprod(target, weights)) / totals
  joint_prevalence <- as.numeric(crossprod(source * target, weights)) / totals
  denominator <- sqrt(
    source_prevalence * (1 - source_prevalence) *
      target_prevalence * (1 - target_prevalence)
  )
  phi <- (
    joint_prevalence - source_prevalence * target_prevalence
  ) / denominator
  phi[!is.finite(phi) | totals <= 0 | denominator <= 0] <- NA_real_
  phi
}

process_state <- function(state_value) {
  state_data <- data[data$state_code == state_value]
  state_jobs <- jobs[jobs$state_code == state_value]
  set.seed(base_seed + as.integer(state_value))
  design <- svydesign(
    id = ~survey_psu,
    strata = ~survey_strata,
    weights = ~survey_weight,
    data = state_data,
    nest = TRUE
  )
  replicate_design <- as.svrepdesign(
    design,
    type = "bootstrap",
    replicates = replicate_count,
    mse = TRUE
  )
  rows <- vector("list", nrow(state_jobs))
  row_index <- 1L
  for (category_value in sort(unique(state_jobs$ses_category))) {
    category_jobs <- state_jobs[ses_category == category_value]
    ses_column <- paste0("ses_", category_jobs$ses_definition[[1]])
    domain_index <- !is.na(replicate_design$variables[[ses_column]]) &
      replicate_design$variables[[ses_column]] == category_value
    domain <- replicate_design[domain_index, ]
    replicate_weights <- weights(domain, type = "analysis")
    if (is.null(dim(replicate_weights))) {
      replicate_weights <- matrix(replicate_weights, ncol = 1)
    }
    for (job_index in seq_len(nrow(category_jobs))) {
      job <- category_jobs[job_index]
      values <- replicate_phi(
        domain$variables,
        job$disease_a,
        job$disease_b,
        replicate_weights
      )
      finite <- values[is.finite(values)]
      interval <- if (length(finite)) {
        quantile(
          finite,
          probs = c(0.025, 0.5, 0.975),
          na.rm = TRUE,
          names = FALSE
        )
      } else {
        c(NA_real_, NA_real_, NA_real_)
      }
      status <- if (length(finite) == replicate_count) {
        "ok"
      } else {
        "partial"
      }
      rows[[row_index]] <- data.table(
        graph_id = job$graph_id,
        state_code = state_value,
        ses_definition = job$ses_definition,
        ses_category = category_value,
        disease_a = job$disease_a,
        disease_b = job$disease_b,
        point_phi = job$point_phi,
        p_phi_gt_zero = if (length(finite)) mean(finite > 0) else NA_real_,
        p_phi_ge_012 = if (length(finite)) {
          mean(finite >= selection_threshold)
        } else {
          NA_real_
        },
        bootstrap_selection_mask = paste0(
          "b",
          paste(
            ifelse(
              is.finite(values),
              ifelse(values >= selection_threshold, "1", "0"),
              "x"
            ),
            collapse = ""
          )
        ),
        bootstrap_phi_median = interval[[2]],
        bootstrap_phi_025 = interval[[1]],
        bootstrap_phi_975 = interval[[3]],
        bootstrap_replicates_requested = replicate_count,
        bootstrap_replicates_valid = length(finite),
        bootstrap_status = status,
        bootstrap_message = ""
      )
      row_index <- row_index + 1L
    }
  }
  rbindlist(rows, fill = TRUE)
}

state_codes <- sort(unique(jobs$state_code))
checkpoint_directory <- paste0(output_path, ".states")
dir.create(checkpoint_directory, recursive = TRUE, showWarnings = FALSE)
checkpoint_path <- function(state_value) {
  file.path(checkpoint_directory, sprintf("state_%02d.csv", state_value))
}
row_keys <- function(value) {
  paste(value$graph_id, value$disease_a, value$disease_b, sep = "\r")
}
checkpoint_valid <- function(state_value) {
  path <- checkpoint_path(state_value)
  if (!file.exists(path)) {
    return(FALSE)
  }
  saved <- tryCatch(fread(path), error = function(value) NULL)
  if (is.null(saved)) {
    return(FALSE)
  }
  expected <- jobs[jobs$state_code == state_value]
  required <- c(
    "bootstrap_status",
    "bootstrap_replicates_valid",
    "bootstrap_selection_mask"
  )
  if (!all(required %in% names(saved))) {
    return(FALSE)
  }
  status_valid <- (
    saved$bootstrap_status == "ok" &
      saved$bootstrap_replicates_valid == replicate_count
  ) | (
    saved$bootstrap_status == "partial" &
      saved$bootstrap_replicates_valid >= 0L &
      saved$bootstrap_replicates_valid < replicate_count
  ) | (
    saved$bootstrap_status == "state_error" &
      saved$bootstrap_replicates_valid == 0L
  )
  valid_mask_count <- nchar(gsub(
    "[^01]",
    "",
    substring(saved$bootstrap_selection_mask, 2L)
  ))
  identical(sort(row_keys(saved)), sort(row_keys(expected))) &&
    all(!is.na(saved$bootstrap_status)) &&
    all(!is.na(saved$bootstrap_replicates_valid)) &&
    all(!is.na(saved$bootstrap_selection_mask)) &&
    all(status_valid) &&
    all(saved$bootstrap_replicates_valid >= 0L) &&
    all(saved$bootstrap_replicates_valid <= replicate_count) &&
    all(nchar(saved$bootstrap_selection_mask) == replicate_count + 1L) &&
    all(substr(saved$bootstrap_selection_mask, 1L, 1L) == "b") &&
    all(valid_mask_count == saved$bootstrap_replicates_valid)
}
state_failure <- function(state_value, value) {
  failed <- copy(jobs[jobs$state_code == state_value])
  failed[, `:=`(
    p_phi_gt_zero = NA_real_,
    p_phi_ge_012 = NA_real_,
    bootstrap_selection_mask = paste0(
      "b",
      paste(rep("x", replicate_count), collapse = "")
    ),
    bootstrap_phi_median = NA_real_,
    bootstrap_phi_025 = NA_real_,
    bootstrap_phi_975 = NA_real_,
    bootstrap_replicates_requested = replicate_count,
    bootstrap_replicates_valid = 0L,
    bootstrap_status = "state_error",
    bootstrap_message = conditionMessage(value)
  )]
  failed
}
checkpoint_worker <- function(state_value) {
  result <- tryCatch(
    process_state(state_value),
    error = function(value) state_failure(state_value, value)
  )
  path <- checkpoint_path(state_value)
  temporary <- paste0(path, ".", Sys.getpid(), ".tmp")
  fwrite(result, temporary, na = "")
  if (!file.rename(temporary, path)) {
    stop(paste("Unable to persist state checkpoint", state_value))
  }
  path
}

completed <- state_codes[vapply(state_codes, checkpoint_valid, logical(1))]
pending <- setdiff(state_codes, completed)
if (length(pending) > 0) {
  parallel::mclapply(
    pending,
    checkpoint_worker,
    mc.cores = max(1L, worker_count),
    mc.preschedule = FALSE
  )
}
missing <- state_codes[!vapply(state_codes, checkpoint_valid, logical(1))]
if (length(missing) > 0) {
  stop(paste("Incomplete state checkpoints:", paste(missing, collapse = ", ")))
}

result <- rbindlist(
  lapply(state_codes, function(value) fread(checkpoint_path(value))),
  fill = TRUE
)
setorder(result, graph_id, disease_a, disease_b)
temporary_output <- paste0(output_path, ".", Sys.getpid(), ".tmp")
fwrite(result, temporary_output, na = "")
if (!file.rename(temporary_output, output_path)) {
  stop("Unable to publish phi bootstrap result")
}
