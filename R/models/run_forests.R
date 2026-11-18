# Run from repository root: Rscript R/models/run_forests.R
# Input: data/processed/features.csv.gz (build_dataset.R -> build_features.R).
# RF and bagging always use the same predictor construction as LightGBM:
# p=57 at h=1/6/12, p=56 at h=24 (daily comparator duplicates current demand).
# Direct forecasts use 13,140 rolling training pairs and weekly (168 h) refits,
# from 2018-07-16 00:00 through target time 2026-09-30 23:00, Asia/Singapore.
# Fixed settings: 500 trees, minimum split-node size 5, seed 4308;
# RF mtry=floor(p/3), classical bagging mtry=p. There is no tuning.
# Forecasts and impurity importance are collected in memory and written at the
# end. No checkpoints, saved tree objects or automatic resume are used; every
# invocation starts from the first refit and replaces outputs on completion.
# Default outputs: output/forecasts/rf_bagging.csv.gz and three CSV tables in
# output/tables/ (comparison, importance by refit, mean importance).
# Optional RF_ settings below apply to BOTH models; RF_MAX_REFITS=0 runs all.
# Use FIXED_OUTPUT_DIR for a separate experiment. For a short isolated check:
# FIXED_OUTPUT_DIR=/tmp/forest-smoke RF_NUM_TREES=10 RF_MAX_REFITS=1 Rscript R/models/run_forests.R
# See README.md for output details and the weather-imputation look-ahead limit.
suppressPackageStartupMessages(library(data.table))
source("R/utils.R")
source("R/models/forest_features.R")

local({
  setting <- function(name, default) Sys.getenv(paste0("RF_", name), default)
  integer_setting <- function(name, default, minimum = 1L) {
    value <- suppressWarnings(as.numeric(setting(name, default)))
    if (length(value) != 1L || !is.finite(value) || value < minimum ||
        value != floor(value) || value > .Machine$integer.max)
      stop("RF_", name, " must be an integer >= ", minimum)
    as.integer(value)
  }
  INPUT <- "data/processed/features.csv.gz"
  TZ <- "Asia/Singapore"
  HORIZONS <- suppressWarnings(as.numeric(strsplit(setting("HORIZONS", "1,6,12,24"), ",")[[1]]))
  if (!length(HORIZONS) || anyNA(HORIZONS) ||
      any(!HORIZONS %in% c(1L, 6L, 12L, 24L)) || anyDuplicated(HORIZONS))
    stop("RF_HORIZONS must list distinct supported horizons: 1,6,12,24.")
  HORIZONS <- as.integer(HORIZONS)
  TRAIN_HOURS <- integer_setting("TRAIN_HOURS", "13140")
  REFIT_HOURS <- integer_setting("REFIT_HOURS", "168")
  FIRST_ORIGIN <- setting("FIRST_ORIGIN", "2018-07-16 00:00:00")
  LAST_TARGET <- setting("LAST_TARGET", "2026-09-30 23:00:00")
  NUM_TREES <- integer_setting("NUM_TREES", "500")
  MIN_NODE_SIZE <- integer_setting("MIN_NODE_SIZE", "5")
  MAX_REFITS <- integer_setting("MAX_REFITS", "0", 0L)
  if (!requireNamespace("ranger", quietly = TRUE)) stop("Install ranger first.")

  root <- Sys.getenv("FIXED_OUTPUT_DIR", "output/forecasts")
  canonical <- !nzchar(Sys.getenv("FIXED_OUTPUT_DIR"))
  table_root <- if (canonical) "output/tables" else root
  forecast_path <- file.path(root, if (canonical) "rf_bagging.csv.gz" else "forecasts.csv")
  comparison_path <- file.path(table_root, if (canonical) "rf_bagging_comparison.csv" else "comparison.csv")
  summary_path <- file.path(table_root, if (canonical) "rf_bagging_importance_summary.csv" else "importance_summary.csv")
  detail_path <- file.path(table_root, if (canonical) "rf_bagging_importance_by_refit.csv.gz" else "importance_by_refit.csv.gz")

  if (!file.exists(INPUT)) stop("Feature file not found: ", INPUT,
    "\nRun Rscript R/data_processing/build_features.R first.")
  # Preserve local clock strings; parsing them as UTC would shift the window.
  d <- read_gz(INPUT, colClasses = c(datetime = "character"))
  required <- c("datetime", "y", "usable", paste0("y_h", HORIZONS),
                paste0("nem_h", HORIZONS))
  missing <- setdiff(required, names(d))
  if (length(missing)) stop("Missing required feature columns: ", paste(missing, collapse = ", "))
  d[, datetime_posix := as.POSIXct(datetime, format = "%Y-%m-%d %H:%M:%S", tz = TZ)]
  if (anyNA(d$datetime_posix)) stop("Could not parse one or more datetimes.")
  setorder(d, datetime_posix)
  if (any(as.numeric(diff(d$datetime_posix), units = "hours") != 1))
    stop("Feature timestamps must form a contiguous hourly series.")
  if (!any(d$usable == 1L)) stop("No rows are marked usable in the feature file.")

  # Always match lgbm.R's DROP/X0 and make_X features. Weekdays are categorical.
  base_predictors <- ForestBasePredictors(d)
  if (!length(base_predictors) ||
      !all(vapply(d[, ..base_predictors], is.numeric, logical(1))))
    stop("Expected numeric base predictors before converting weekday factors.")
  d[, dow := factor(dow, levels = 0:6)]
  horizon_features <- horizon_predictors <- list()
  for (h in HORIZONS) {
    features <- BuildForestHorizonFeatures(d, h)
    observed_target <- d$usable == 1L & !is.na(d[[paste0("y_h", h)]])
    if (any(!complete.cases(features[observed_target])))
      stop("Missing horizon-specific features for observed targets at h=", h)
    horizon_features[[as.character(h)]] <- features
    horizon_predictors[[as.character(h)]] <- c(base_predictors, names(features))
  }

  first_origin <- as.POSIXct(FIRST_ORIGIN, format = "%Y-%m-%d %H:%M:%S", tz = TZ)
  last_target <- as.POSIXct(LAST_TARGET, format = "%Y-%m-%d %H:%M:%S", tz = TZ)
  if (is.na(last_target)) stop("RF_LAST_TARGET must be a valid local datetime.")
  last_time <- min(max(d$datetime_posix), last_target)
  if (is.na(first_origin) || first_origin > last_time ||
      !first_origin %in% d$datetime_posix)
    stop("RF_FIRST_ORIGIN must match an available feature timestamp.")
  origins <- seq(first_origin, last_time, by = REFIT_HOURS * 3600)
  if (MAX_REFITS > 0L) origins <- head(origins, MAX_REFITS)

  FitForest <- function(train, target, predictors, method) {
    p <- length(predictors)
    # Bagging considers all p predictors at every split; RF randomly offers a
    # subset. These are chosen settings, not an OOB-selected optimum.
    mtry <- if (method == "bagging") p else max(1L, as.integer(floor(p / 3)))
    # sample.fraction=1 with replacement draws n rows per tree, including repeats.
    # Variance splits reduce squared error. Node size controls splitting, not
    # minimum leaf size; max.depth is omitted, leaving depth unrestricted.
    # Unordered weekday factors are ordered using training responses by ranger;
    # their integer labels are not imposed as an ordinal scale.
    fit <- ranger::ranger(
      dependent.variable.name = target,
      data = train[, c(target, predictors), with = FALSE],
      num.trees = NUM_TREES, mtry = mtry, min.node.size = MIN_NODE_SIZE,
      replace = TRUE, sample.fraction = 1, splitrule = "variance", seed = 4308L,
      respect.unordered.factors = "order", importance = "impurity")
    # OOB MSE is diagnostic only; bootstrap OOB is not chronological validation.
    if (!is.finite(fit$prediction.error)) stop("Non-finite OOB MSE.")
    fit
  }

  cat("Predictors by horizon:", paste(paste0(names(horizon_predictors), "h=",
    lengths(horizon_predictors)), collapse = ", "), "\n")
  cat(length(origins), "weekly refits |", TRAIN_HOURS, "training pairs |",
      NUM_TREES, "trees | no tuning or resume\n")
  forecast_rows <- importance_rows <- list()
  pair <- 0L
  for (i in seq_along(origins)) {
    for (h in HORIZONS) {
      origin <- origins[i]
      target <- paste0("y_h", h)
      predictors <- horizon_predictors[[as.character(h)]]
      # Pairs (X_s, y_{s+h}) end at s=T-h: the latest training label is y_T,
      # already observed at refit T. Features update hourly between weekly fits.
      last_train_time <- origin - h * 3600
      train_start <- last_train_time - (TRAIN_HOURS - 1L) * 3600
      train_idx <- which(d$usable == 1L & d$datetime_posix >= train_start &
        d$datetime_posix <= last_train_time & !is.na(d[[target]]))
      predict_idx <- which(d$usable == 1L & d$datetime_posix >= origin &
        d$datetime_posix < origin + REFIT_HOURS * 3600 &
        d$datetime_posix + h * 3600 <= last_time & !is.na(d[[target]]))
      if (!length(predict_idx)) next
      if (length(train_idx) != TRAIN_HOURS)
        stop("Expected ", TRAIN_HOURS, " training rows at ", format(origin, tz = TZ), " h=", h)
      train <- d[train_idx]
      test <- d[predict_idx]
      features <- horizon_features[[as.character(h)]]
      train[, (names(features)) := features[train_idx]]
      test[, (names(features)) := features[predict_idx]]
      started <- proc.time()[["elapsed"]]
      cat(sprintf("Refit %d/%d | %s | h=%d: fitting RF and bagging...\n",
                  i, length(origins), format(origin, tz = TZ), h))
      rf <- FitForest(train, target, predictors, "random_forest")
      bag <- FitForest(train, target, predictors, "bagging")
      pair <- pair + 1L
      forecast_rows[[pair]] <- data.table(
        horizon = h, refit_origin = format(origin, "%Y-%m-%d %H:%M:%S", tz = TZ),
        datetime = test$datetime,
        target_datetime = format(test$datetime_posix + h * 3600, "%Y-%m-%d %H:%M:%S", tz = TZ),
        actual_mw = test[[target]],
        rf_forecast_mw = predict(rf, data = test[, predictors, with = FALSE])$predictions,
        ema_forecast_mw = test[[paste0("nem_h", h)]],
        train_n = length(train_idx), rf_mtry = rf$mtry, rf_oob_mse = rf$prediction.error,
        feature_set = "lightgbm_aligned", predictor_count = length(predictors),
        bagging_forecast_mw = predict(bag, data = test[, predictors, with = FALSE])$predictions,
        bagging_mtry = bag$mtry, bagging_oob_mse = bag$prediction.error)
      # Importance comes from the fitted prediction trees. Retain only scores,
      # not tree objects. Impurity reductions describe splits, not causal effects.
      importance_rows[[pair]] <- rbindlist(lapply(c("random_forest", "bagging"), function(name) {
        fit <- if (name == "random_forest") rf else bag
        scores <- fit$variable.importance
        if (!identical(names(scores), predictors) || any(!is.finite(scores)))
          stop("Missing or invalid feature importance.")
        data.table(refit_origin = format(origin, "%Y-%m-%d %H:%M:%S", tz = TZ),
          horizon = h, model = name, feature = names(scores), importance = unname(scores),
          method = "impurity", mtry = fit$mtry, feature_set = "lightgbm_aligned")
      }))
      cat(sprintf("  Completed pair %d | RF mtry=%d, bagging mtry=%d | %.1fs\n",
        pair, rf$mtry, bag$mtry, proc.time()[["elapsed"]] - started))
      rm(rf, bag)
    }
  }
  if (!pair) stop("No forecasts were produced; check the origin/target settings.")

  # Write final results only: every invocation recomputes the requested schedule.
  forecasts <- rbindlist(forecast_rows)
  importance <- rbindlist(importance_rows)
  summary <- importance[, .(completed_refits = .N, mean_importance = mean(importance)),
                        by = .(horizon, model, feature, method)]
  setorder(summary, horizon, model, -mean_importance)
  # Reuse the standalone comparison script on temporary model CSVs. These are
  # deleted on exit and are never read as saved state by subsequent runs.
  staging <- tempfile("forest-comparison-")
  dir.create(staging)
  on.exit(unlink(staging, recursive = TRUE), add = TRUE)
  shared <- c("horizon", "refit_origin", "datetime", "target_datetime", "actual_mw", "train_n")
  for (model in c("rf", "bagging")) {
    column <- paste0(model, "_forecast_mw")
    rows <- forecasts[, c(shared, column), with = FALSE]
    setnames(rows, column, "forecast_mw")
    fwrite(rows, file.path(staging, paste0(model, ".csv")))
  }
  status <- system2(file.path(R.home("bin"), "Rscript"),
    c("R/evaluation/compare_forests.R", shQuote(file.path(staging, "rf.csv")),
      shQuote(file.path(staging, "bagging.csv")), shQuote(file.path(staging, "comparison.csv"))))
  if (status != 0L) stop("Fixed-mtry comparison failed.")
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  dir.create(table_root, recursive = TRUE, showWarnings = FALSE)
  fwrite(forecasts, forecast_path)
  fwrite(importance, detail_path)
  fwrite(summary, summary_path)
  if (!file.copy(file.path(staging, "comparison.csv"), comparison_path, overwrite = TRUE))
    stop("Could not write comparison results.")
  cat("Saved", pair, "paired fits to", forecast_path, "\n")
  cat("Comparison:", comparison_path, "\nImportance:", summary_path, "\n")
})
