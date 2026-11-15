# Compare saved forecasts on identical origin/horizon pairs; no model fitting.
# Run from the repository root: Rscript R/evaluation/evaluate.R
# Requires data.table, gzip (for compressed CSVs), and base R unzip support.
# EMA is an operational reference with unknown issue times, not horizon-matched.
suppressPackageStartupMessages(library(data.table))
source("R/utils.R")

EvaluateForecasts <- function() {
  input <- "output/forecasts"
  output <- "output/tables"
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  benchmark <- read_gz(file.path(input, "benchmarks.csv.gz"),
                       colClasses = c(datetime = "character"))
  lasso <- read_gz(file.path(input, "lasso.csv.gz"),
                   colClasses = c(datetime = "character"))
  # Importance and tuning CSVs are diagnostics, not additional forecasts.
  lightgbm <- read_gz(file.path(input, "lgbm.csv.gz"),
                      colClasses = c(datetime = "character"))
  seasonal <- fread(file.path(input, "sarima_holt_winters_forecast_results.csv"),
                    colClasses = c(origin = "character",
                                   target_datetime = "character"))
  forest <- read_gz(file.path(input, "rf_bagging.csv.gz"),
                   colClasses = c(datetime = "character",
                                  target_datetime = "character"))
  setnames(seasonal, "origin", "datetime")
  setnames(forest, c("horizon", "actual_mw", "rf_forecast_mw",
                     "bagging_forecast_mw"),
           c("h", "actual", "random_forest", "bagging"))
  keys <- c("datetime", "h")
  datasets <- list(benchmarks = benchmark, lasso = lasso,
                   seasonal = seasonal, forests = forest, lightgbm = lightgbm)
  for (name in names(datasets)) {
    d <- datasets[[name]]
    if (anyDuplicated(d[, keys, with = FALSE])) {
      stop("Duplicate origin/horizon keys in ", name)
    }
    if (anyNA(d[, keys, with = FALSE])) stop("Missing keys in ", name)
  }
  # Check explicit target timestamps against the common Singapore origin clock.
  for (d in list(seasonal, forest)) {
    expected <- format(as.POSIXct(d$datetime, tz = "Asia/Singapore") +
                         d$h * 3600, "%Y-%m-%d %H:%M:%S",
                       tz = "Asia/Singapore")
    if (anyNA(d$target_datetime) || any(expected != d$target_datetime)) {
      stop("Target timestamps do not match origin + horizon.")
    }
  }
  columns <- list(lasso = c("lasso", "post_lasso"),
                  seasonal = c("sarima", "hw24", "period"),
                  forests = c("random_forest", "bagging"),
                  lightgbm = "lgbm")
  paired <- copy(benchmark)
  actual_checks <- list()
  for (name in names(columns)) {
    d <- copy(datasets[[name]][, c(keys, "actual", columns[[name]]),
                                with = FALSE])
    setnames(d, "actual", "other_actual")
    paired <- merge(paired, d, by = keys)
    difference <- max(abs(paired$actual - paired$other_actual))
    if (!is.finite(difference) || difference > 1e-6) {
      stop("Actual demand mismatch for ", name, ": ", difference)
    }
    actual_checks[[name]] <- difference
    paired[, other_actual := NULL]
  }
  models <- c("snaive168", "snaive24", "ar_bic", "lasso", "post_lasso",
              "random_forest", "bagging", "sarima", "hw24", "lgbm", "ema")
  finite <- Reduce(`&`, lapply(paired[, c("actual", models), with = FALSE],
                               is.finite))
  coverage <- paired[, .(matched_rows = .N), by = h]
  kept <- paired[finite, .(scored_rows = .N), by = h]
  coverage <- merge(coverage, kept, by = "h", all.x = TRUE)
  coverage[is.na(scored_rows), scored_rows := 0L]
  coverage[, excluded_nonfinite := matched_rows - scored_rows]
  for (name in names(datasets)) {
    count <- datasets[[name]][, .N, by = h]
    setnames(count, "N", paste0(name, "_rows"))
    coverage <- merge(coverage, count, by = "h", all = TRUE)
  }
  paired <- paired[finite]
  if (!nrow(paired)) stop("No common finite forecasts.")
  Score <- function(d, label) {
    rbindlist(lapply(models, function(model) {
      d[, .(n = .N, first_origin = min(datetime), last_origin = max(datetime),
             rmse_mw = sqrt(mean((actual - get(model))^2)),
             mae_mw = mean(abs(actual - get(model))),
             mape_pct = if (any(actual == 0)) NA_real_ else
               100 * mean(abs((actual - get(model)) / actual))), by = h][
                 , `:=`(model = model, sample = label,
                        reference_only = model == "ema")]
    }))
  }
  scores <- Score(paired, "all")
  for (period_name in sort(unique(paired$period))) {
    scores <- rbind(scores, Score(paired[period == period_name], period_name))
  }
  scores[, rank_rmse := NA_integer_]
  scores[reference_only == FALSE,
         rank_rmse := frank(rmse_mw, ties.method = "min"), by = .(sample, h)]
  scores[, `:=`(rank_mae = NA_integer_, rank_mape = NA_integer_)]
  scores[reference_only == FALSE, `:=`(
    rank_mae = frank(mae_mw, ties.method = "min"),
    rank_mape = frank(mape_pct, ties.method = "min")), by = .(sample, h)]
  setorder(scores, sample, h, reference_only, rmse_mw)
  fwrite(scores, file.path(output, "all_models_comparison.csv"))
  fwrite(coverage, file.path(output, "all_models_coverage.csv"))
  cat("Matched demand differences:", unlist(actual_checks), "\n")
  print(coverage)
  for (sample_name in c("all", "test")) {
    cat("\nMAPE (%), sample:", sample_name, "\n")
    print(dcast(scores[sample == sample_name], h ~ model, value.var = "mape_pct"))
    cat("\nBest non-reference model per horizon:\n")
    print(scores[sample == sample_name & rank_rmse == 1L,
                 .(h, model, n, rmse_mw, mae_mw, mape_pct,
                   first_origin, last_origin)])
  }
  invisible(scores)
}

EvaluateForecasts()
