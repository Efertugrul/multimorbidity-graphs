args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 8) {
  stop("Expected mode, data, jobs, output, confidence, replicates, seed, workers")
}

mode <- args[[1]]
data_path <- args[[2]]
jobs_path <- args[[3]]
output_path <- args[[4]]
confidence_level <- as.numeric(args[[5]])
replicate_count <- as.integer(args[[6]])
base_seed <- as.integer(args[[7]])
worker_count <- as.integer(args[[8]])

suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu = "adjust")
options(survey.adjust.domain.lonely = TRUE)

data <- fread(data_path, na.strings = c("", "NA", "<NA>"))
jobs <- fread(jobs_path)

fit_point <- function(design, outcome, exposure) {
  formula <- as.formula(
    sprintf("%s ~ %s + factor(age_group) + factor(sex)", outcome, exposure)
  )
  warning_messages <- character()
  fit <- tryCatch(
    withCallingHandlers(
      svyglm(formula, design = design, family = quasibinomial()),
      warning = function(value) {
        warning_messages <<- c(warning_messages, conditionMessage(value))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(value) value
  )
  if (inherits(fit, "error")) {
    return(list(
      status = "error",
      message = conditionMessage(fit),
      estimate = NA_real_,
      standard_error = NA_real_,
      confidence_low = NA_real_,
      confidence_high = NA_real_,
      p_value = NA_real_,
      converged = FALSE
    ))
  }
  coefficients <- coef(fit)
  covariance <- vcov(fit)
  summary_coefficients <- coef(summary(fit))
  if (!(exposure %in% names(coefficients))) {
    return(list(
      status = "coefficient_missing",
      message = paste(unique(warning_messages), collapse = " | "),
      estimate = NA_real_,
      standard_error = NA_real_,
      confidence_low = NA_real_,
      confidence_high = NA_real_,
      p_value = NA_real_,
      converged = isTRUE(fit$converged)
    ))
  }
  estimate <- unname(coefficients[[exposure]])
  standard_error <- sqrt(unname(covariance[exposure, exposure]))
  degrees_freedom <- max(1, fit$df.residual)
  critical_value <- qt((1 + confidence_level) / 2, degrees_freedom)
  p_columns <- grep("^Pr\\(", colnames(summary_coefficients), value = TRUE)
  p_value <- if (length(p_columns)) {
    unname(summary_coefficients[exposure, p_columns[[1]]])
  } else {
    NA_real_
  }
  list(
    status = "ok",
    message = paste(unique(warning_messages), collapse = " | "),
    estimate = estimate,
    standard_error = standard_error,
    confidence_low = estimate - critical_value * standard_error,
    confidence_high = estimate + critical_value * standard_error,
    p_value = p_value,
    converged = isTRUE(fit$converged)
  )
}

fit_bootstrap <- function(design, outcome, exposure) {
  formula <- as.formula(
    sprintf("%s ~ %s + factor(age_group) + factor(sex)", outcome, exposure)
  )
  warning_messages <- character()
  fit <- tryCatch(
    withCallingHandlers(
      svyglm(
        formula,
        design = design,
        family = quasibinomial(),
        return.replicates = TRUE
      ),
      warning = function(value) {
        warning_messages <<- c(warning_messages, conditionMessage(value))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(value) value
  )
  if (inherits(fit, "error")) {
    return(list(
      status = "error",
      message = conditionMessage(fit),
      estimate = NA_real_,
      valid_replicates = 0L,
      positive_stability = NA_real_,
      bootstrap_low = NA_real_,
      bootstrap_high = NA_real_
    ))
  }
  replicate_values <- fit$replicates
  coefficient_names <- names(coef(fit))
  exposure_index <- match(exposure, coefficient_names)
  if (is.null(dim(replicate_values)) || is.na(exposure_index)) {
    return(list(
      status = "replicate_coefficient_missing",
      message = paste(unique(warning_messages), collapse = " | "),
      estimate = if (is.na(exposure_index)) NA_real_ else unname(coef(fit)[[exposure_index]]),
      valid_replicates = 0L,
      positive_stability = NA_real_,
      bootstrap_low = NA_real_,
      bootstrap_high = NA_real_
    ))
  }
  values <- replicate_values[, exposure_index]
  values <- values[is.finite(values)]
  interval <- if (length(values)) {
    quantile(values, probs = c(0.025, 0.975), na.rm = TRUE, names = FALSE)
  } else {
    c(NA_real_, NA_real_)
  }
  list(
    status = "ok",
    message = paste(unique(warning_messages), collapse = " | "),
    estimate = unname(coef(fit)[[exposure]]),
    valid_replicates = length(values),
    positive_stability = if (length(values)) mean(values > 0) else NA_real_,
    bootstrap_low = interval[[1]],
    bootstrap_high = interval[[2]]
  )
}

process_state_point <- function(state_value) {
  state_data <- data[data$state_code == state_value]
  state_jobs <- jobs[jobs$state_code == state_value]
  design <- svydesign(
    id = ~survey_psu,
    strata = ~survey_strata,
    weights = ~survey_weight,
    data = state_data,
    nest = TRUE
  )
  rows <- vector("list", nrow(state_jobs))
  for (index in seq_len(nrow(state_jobs))) {
    job <- state_jobs[index]
    ses_column <- paste0("ses_", job$ses_definition)
    domain_index <- !is.na(design$variables[[ses_column]]) &
      design$variables[[ses_column]] == job$ses_category
    domain <- design[domain_index, ]
    forward <- fit_point(
      domain,
      job$target_condition,
      job$source_condition
    )
    reverse <- fit_point(
      domain,
      job$source_condition,
      job$target_condition
    )
    rows[[index]] <- data.table(
      graph_id = job$graph_id,
      state_code = state_value,
      ses_definition = job$ses_definition,
      ses_category = job$ses_category,
      source_condition = job$source_condition,
      target_condition = job$target_condition,
      model_status = forward$status,
      model_message = forward$message,
      log_odds_ratio = forward$estimate,
      standard_error = forward$standard_error,
      confidence_low_log = forward$confidence_low,
      confidence_high_log = forward$confidence_high,
      p_value = forward$p_value,
      converged = forward$converged,
      reverse_model_status = reverse$status,
      reverse_model_message = reverse$message,
      reverse_log_odds_ratio = reverse$estimate,
      reverse_standard_error = reverse$standard_error,
      reverse_p_value = reverse$p_value,
      reverse_converged = reverse$converged,
      direction_concordant = is.finite(forward$estimate) &&
        is.finite(reverse$estimate) &&
        sign(forward$estimate) == sign(reverse$estimate)
    )
  }
  rbindlist(rows, fill = TRUE)
}

process_state_bootstrap <- function(state_value) {
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
  for (index in seq_len(nrow(state_jobs))) {
    job <- state_jobs[index]
    ses_column <- paste0("ses_", job$ses_definition)
    domain_index <- !is.na(replicate_design$variables[[ses_column]]) &
      replicate_design$variables[[ses_column]] == job$ses_category
    domain <- replicate_design[domain_index, ]
    result <- fit_bootstrap(
      domain,
      job$target_condition,
      job$source_condition
    )
    rows[[index]] <- data.table(
      graph_id = job$graph_id,
      state_code = state_value,
      ses_definition = job$ses_definition,
      ses_category = job$ses_category,
      source_condition = job$source_condition,
      target_condition = job$target_condition,
      bootstrap_status = result$status,
      bootstrap_message = result$message,
      bootstrap_log_odds_ratio = result$estimate,
      bootstrap_replicates_requested = replicate_count,
      bootstrap_replicates_valid = result$valid_replicates,
      positive_stability = result$positive_stability,
      bootstrap_low_log = result$bootstrap_low,
      bootstrap_high_log = result$bootstrap_high
    )
  }
  rbindlist(rows, fill = TRUE)
}

state_codes <- sort(unique(jobs$state_code))
worker <- if (mode == "point") process_state_point else process_state_bootstrap
checkpoint_directory <- paste0(output_path, ".states")
dir.create(checkpoint_directory, recursive = TRUE, showWarnings = FALSE)
checkpoint_path <- function(state_value) {
  file.path(checkpoint_directory, sprintf("state_%02d.csv", state_value))
}
row_keys <- function(value) {
  paste(
    value$graph_id,
    value$source_condition,
    value$target_condition,
    sep = "\r"
  )
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
  if (!identical(sort(row_keys(saved)), sort(row_keys(expected)))) {
    return(FALSE)
  }
  status_column <- if (mode == "point") "model_status" else "bootstrap_status"
  status_column %in% names(saved) &&
    !any(saved[[status_column]] == "state_error")
}
state_failure <- function(state_value, value) {
  failed <- copy(jobs[jobs$state_code == state_value])
  if (mode == "point") {
    failed[, `:=`(
      model_status = "state_error",
      model_message = conditionMessage(value),
      log_odds_ratio = NA_real_,
      standard_error = NA_real_,
      confidence_low_log = NA_real_,
      confidence_high_log = NA_real_,
      p_value = NA_real_,
      converged = FALSE,
      reverse_model_status = "state_error",
      reverse_model_message = conditionMessage(value),
      reverse_log_odds_ratio = NA_real_,
      reverse_standard_error = NA_real_,
      reverse_p_value = NA_real_,
      reverse_converged = FALSE,
      direction_concordant = FALSE
    )]
  } else {
    failed[, `:=`(
      bootstrap_status = "state_error",
      bootstrap_message = conditionMessage(value),
      bootstrap_log_odds_ratio = NA_real_,
      bootstrap_replicates_requested = replicate_count,
      bootstrap_replicates_valid = 0L,
      positive_stability = NA_real_,
      bootstrap_low_log = NA_real_,
      bootstrap_high_log = NA_real_
    )]
  }
  failed
}
checkpoint_worker <- function(state_value) {
  result <- tryCatch(
    worker(state_value),
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
setorder(result, graph_id, source_condition, target_condition)
temporary_output <- paste0(output_path, ".", Sys.getpid(), ".tmp")
fwrite(result, temporary_output, na = "")
if (!file.rename(temporary_output, output_path)) {
  stop("Unable to publish model result")
}
