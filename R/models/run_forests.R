# Fixed RF uses floor(p/3); classical bagging uses every predictor.
# Rolling direct forecasts with 500 trees and minimum split-node size 5.
# Saves paired forecasts, importance and resumable checkpoints.
# Optional FIXED_OUTPUT_DIR; RF_MAX_REFITS/RF_NUM_TREES etc. still work.
suppressPackageStartupMessages(library(data.table))
source("R/utils.R")
source("R/models/forest_features.R")

local({
  mode <- Sys.getenv("RF_FEATURE_MODE", if (Sys.getenv("RF_TARGET_CALENDAR", "1") == "0") "legacy" else "lightgbm")
  default_root <- switch(mode, lightgbm = "output/forecasts",
    target_calendar = "output/target_calendar_forests/fixed",
    legacy = "output/legacy_forests/fixed", stop("Invalid feature mode."))
  root <- Sys.getenv("FIXED_OUTPUT_DIR", default_root)
  canonical <- mode == "lightgbm" && !nzchar(Sys.getenv("FIXED_OUTPUT_DIR"))
  table_root <- if (canonical) "output/tables" else root
  state_root <- if (canonical) "output/checkpoints/rf_bagging" else root
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  dir.create(table_root, recursive = TRUE, showWarnings = FALSE)
  dir.create(state_root, recursive = TRUE, showWarnings = FALSE)
  summary_path <- file.path(table_root, if (canonical) "rf_bagging_importance_summary.csv" else "importance_summary.csv")
  detail_path <- file.path(table_root, if (canonical) "rf_bagging_importance_by_refit.csv.gz" else "importance_by_refit.csv.gz")
  comparison_path <- file.path(table_root, if (canonical) "rf_bagging_comparison.csv" else "comparison.csv")
  # Engine files are temporary; expose only paired forecasts and comparison.
  staging <- tempfile("fixed-forests-")
  dir.create(staging)
  on.exit(unlink(staging, recursive = TRUE), add = TRUE)
  forecast_path <- file.path(root, if (canonical) "rf_bagging.csv" else "forecasts.csv")
  # The uncompressed file is an ignored working checkpoint. A completed,
  # compressed copy is published for Git and the evaluation scripts.
  published_path <- if (canonical) paste0(forecast_path, ".gz") else NULL
  if (canonical && !file.exists(forecast_path) && file.exists(published_path)) {
    temporary <- paste0(forecast_path, ".tmp")
    status <- system2(find_gzip(), c("-dc", shQuote(published_path)), stdout = temporary)
    if (status != 0L || !file.rename(temporary, forecast_path))
      stop("Could not restore working forecasts from compressed results.")
  }
  saved <- if (file.exists(forecast_path)) data.table::fread(forecast_path,
    colClasses = c(refit_origin = "character", datetime = "character",
                   target_datetime = "character")) else NULL
  if (!is.null(saved)) {
    backup <- file.path(state_root, "forecasts.csv.before-resume.bak")
    if (!file.exists(backup) && !file.copy(forecast_path, backup))
      stop("Could not back up existing forecasts.")
    if (anyDuplicated(saved[, .(horizon, datetime)])) stop("Duplicate saved forecast keys.")
  }
  paths <- c(
    RF_OUTPUT = file.path(staging, "rf.csv"),
    RF_METRICS_OUTPUT = file.path(staging, "rf_metrics.csv"),
    RF_OOB_OUTPUT = file.path(staging, "rf_oob.csv"),
    BAG_OUTPUT = file.path(staging, "bagging.csv"),
    BAG_METRICS_OUTPUT = file.path(staging, "bagging_metrics.csv"),
    BAG_OOB_OUTPUT = file.path(staging, "bagging_oob.csv"))
  old <- Sys.getenv(names(paths), unset = NA_character_)
  on.exit({
    Sys.unsetenv(names(old)[is.na(old)])
    if (any(!is.na(old))) do.call(Sys.setenv, as.list(old[!is.na(old)]))
  }, add = TRUE)
  do.call(Sys.setenv, as.list(paths))

  RESUME_INIT <- function() {
    config <- list(input_md5 = unname(tools::md5sum(INPUT)), feature_mode = FEATURE_MODE,
      predictors = horizon_predictors, train_hours = TRAIN_HOURS, refit_hours = REFIT_HOURS,
      trees = NUM_TREES, node_size = MIN_NODE_SIZE, seed = 4308L,
      first_origin = FIRST_ORIGIN, last_target = LAST_TARGET,
      ranger_version = as.character(utils::packageVersion("ranger")))
    manifest <- file.path(state_root, "resume_config.rds")
    if (file.exists(manifest) && !identical(readRDS(manifest), config))
      stop("Resume configuration changed; use a different FIXED_OUTPUT_DIR.")
    if (!file.exists(manifest)) {
      if (!is.null(saved) && (NUM_TREES != 500L || MIN_NODE_SIZE != 5L))
        stop("Existing experiment used 500 trees and node size 5.")
      saveRDS(config, manifest)
    }
    if (!is.null(saved)) {
      for (model in c("rf", "bagging")) {
        rows <- data.table::copy(saved)
        rows[, c(paste0(if (model == "rf") "bagging" else "rf",
                        c("_forecast_mw", "_mtry", "_oob_mse"))) := NULL]
        data.table::setnames(rows, paste0(model, c("_forecast_mw", "_mtry", "_oob_mse")),
                            c("forecast_mw", "mtry", "oob_mse"))
        # fwrite appends by position: both restored files must match the
        # engine's column order, including bagging's renamed prediction fields.
        data.table::setcolorder(rows, c("horizon", "refit_origin", "datetime",
          "target_datetime", "actual_mw", "forecast_mw", "ema_forecast_mw",
          "train_n", "mtry", "oob_mse", "feature_set", "predictor_count"))
        data.table::fwrite(rows, if (model == "rf") OUTPUT else BAG_OUTPUT)
      }
      cat("Resuming:", nrow(unique(saved[, .(horizon, refit_origin)])), "saved paired fits.\n")
    }
  }
  RESUME_SKIP <- function(h, origin, predict_idx, train_idx) {
    if (is.null(saved)) return(FALSE)
    stamp <- format(origin, "%Y-%m-%d %H:%M:%S", tz = TZ)
    rows <- saved[horizon == h & refit_origin == stamp]
    if (!nrow(rows)) return(FALSE)
    p <- length(horizon_predictors[[as.character(h)]])
    if (!identical(rows$datetime, d$datetime[predict_idx]) ||
        any(rows$train_n != length(train_idx)) ||
        any(rows$feature_set != FEATURE_SET) || any(rows$predictor_count != p) ||
        any(rows$rf_mtry != max(1L, floor(p / 3))) || any(rows$bagging_mtry != p) ||
        any(!is.finite(rows$rf_forecast_mw)) || any(!is.finite(rows$bagging_forecast_mw)) ||
        any(abs(rows$actual_mw - d[[paste0("y_h", h)]][predict_idx]) > 1e-6))
      stop("Saved fit does not match expected rows/settings: ", stamp, " h=", h)
    id <- paste0(gsub("[^0-9]", "", stamp), "_h", h)
    checkpoint <- file.path(state_root, "importance_checkpoints", paste0(id, ".rds"))
    if (!file.exists(checkpoint)) {
      cat("Backfilling importance:", stamp, "h=", h, "\n")
      return(FALSE)
    }
    importance <- tryCatch(readRDS(checkpoint), error = function(e) NULL)
    if (is.null(importance) || nrow(importance) != 2L * p ||
        any(!is.finite(importance$importance))) return(FALSE)
    TRUE
  }
  RESUME_FORECAST_EXISTS <- function(h, origin) {
    if (is.null(saved)) return(FALSE)
    stamp <- format(origin, "%Y-%m-%d %H:%M:%S", tz = TZ)
    any(saved$horizon == h & saved$refit_origin == stamp)
  }
  importance_dir <- file.path(state_root, "importance_checkpoints")
  dir.create(importance_dir, recursive = TRUE, showWarnings = FALSE)
  PAIR_WRITER <- function(rf, bag, rf_model, bag_model) {
    shared <- c("horizon", "refit_origin", "datetime", "target_datetime",
                "actual_mw", "ema_forecast_mw", "train_n")
    if (!identical(rf[, shared, with = FALSE], bag[, shared, with = FALSE]))
      stop("Cannot consolidate: model forecast rows do not match.")
    paired <- data.table::copy(rf)
    data.table::setnames(paired, c("forecast_mw", "mtry", "oob_mse"),
                        c("rf_forecast_mw", "rf_mtry", "rf_oob_mse"))
    paired[, `:=`(bagging_forecast_mw = bag$forecast_mw,
                  bagging_mtry = bag$mtry, bagging_oob_mse = bag$oob_mse)]
    previous <- if (is.null(saved)) NULL else saved[
      horizon == paired$horizon[1] & refit_origin == paired$refit_origin[1]]
    if (!is.null(previous) && nrow(previous)) {
      if (!identical(previous$datetime, paired$datetime) ||
          any(abs(previous$rf_forecast_mw - paired$rf_forecast_mw) > 1e-6) ||
          any(abs(previous$bagging_forecast_mw - paired$bagging_forecast_mw) > 1e-6))
        stop("Importance refit does not reproduce saved predictions; forecasts unchanged.")
    }
    # Impurity importance uses the trees already fitted for these predictions.
    importance <- data.table::rbindlist(lapply(c("random_forest", "bagging"), function(name) {
      fit <- if (name == "random_forest") rf_model else bag_model
      scores <- fit$variable.importance
      if (length(scores) != paired$predictor_count[1] || any(!is.finite(scores)))
        stop("Missing or invalid feature importance.")
      data.table::data.table(refit_origin = paired$refit_origin[1],
        horizon = paired$horizon[1], model = name, feature = names(scores),
        importance = unname(scores), method = "impurity", mtry = fit$mtry,
        feature_set = paired$feature_set[1])
    }))
    id <- paste0(gsub("[^0-9]", "", paired$refit_origin[1]), "_h", paired$horizon[1])
    checkpoint <- file.path(importance_dir, paste0(id, ".rds"))
    saveRDS(importance, paste0(checkpoint, ".tmp"))
    if (!file.rename(paste0(checkpoint, ".tmp"), checkpoint))
      stop("Could not save importance checkpoint.")
    if (!is.null(previous) && nrow(previous)) {
      cat("Backfilled importance; preserved forecasts:", paired$refit_origin[1],
          "h=", paired$horizon[1], "\n")
      return(invisible(NULL))
    }
    # Commit a complete paired fit atomically, preserving the prior file on interruption.
    temporary <- paste0(forecast_path, ".tmp")
    if (file.exists(forecast_path) && !file.copy(forecast_path, temporary, overwrite = TRUE))
      stop("Could not stage forecast checkpoint.")
    data.table::fwrite(paired, temporary, append = file.exists(forecast_path))
    if (!file.rename(temporary, forecast_path)) stop("Could not commit forecast checkpoint.")
    cat("Saved paired fit:", paired$refit_origin[1], "h=", paired$horizon[1], "\n")
  }

  ENSEMBLE_FITTER <- function(data, target, p, method, num.trees,
                              min.node.size, seed = 4308L) {
    mtry <- if (method == "bagging") p else max(1L, as.integer(floor(p / 3)))
    if (mtry > p) stop("Requested mtry exceeds available predictors.")
    model <- ranger::ranger(
      dependent.variable.name = target, data = data, mtry = mtry,
      num.trees = num.trees, min.node.size = min.node.size,
      replace = TRUE, sample.fraction = 1, splitrule = "variance", seed = seed, respect.unordered.factors = "order",
      importance = "impurity")
    if (!is.finite(model$prediction.error)) stop("Non-finite OOB MSE.")
    cat(sprintf("    Fixed mtry=%d, OOB MSE=%.3f (no tuning)\n",
                mtry, model$prediction.error))
    trace <- data.frame(mtry = mtry, oob_mse = model$prediction.error,
                        oob_rmse_mw = sqrt(model$prediction.error),
                        relative_improvement = NA_real_, selected = TRUE)
    list(model = model, trace = trace, bagging_model = NULL)
  }
  # Prepare horizon-specific data, resume completed pairs, and fit weekly models.
  fit_ensemble <- ENSEMBLE_FITTER
  METHOD <- "random_forest"
  COMPARE_BAGGING <- TRUE
  PREFIX <- "RF"
  setting <- function(name, default) Sys.getenv(paste0(PREFIX, "_", name), default)
  calendar_setting <- setting("TARGET_CALENDAR", "1")
  if (!calendar_setting %in% c("0", "1")) stop("TARGET_CALENDAR must be 0 or 1.")
  USE_TARGET_CALENDAR <- calendar_setting == "1"
  FEATURE_MODE <- setting("FEATURE_MODE", if (USE_TARGET_CALENDAR) "lightgbm" else "legacy")
  if (!FEATURE_MODE %in% c("lightgbm", "target_calendar", "legacy")) {
    stop("FEATURE_MODE must be lightgbm, target_calendar or legacy.")
  }
  USE_TARGET_CALENDAR <- FEATURE_MODE != "legacy"
  FEATURE_SET <- switch(FEATURE_MODE, lightgbm = "lightgbm_aligned",
    target_calendar = "origin_and_target_calendar", legacy = "origin_calendar_only")
  DEFAULT_ROOT <- switch(FEATURE_MODE, lightgbm = "output/lgbm_aligned_forests",
    target_calendar = "output/target_calendar_forests", legacy = "output/legacy_forests")

  INPUT <- "data/processed/features.csv.gz"
  OUTPUT <- setting("OUTPUT", file.path(DEFAULT_ROOT, paste0(METHOD, ".csv")))
  METRICS_OUTPUT <- setting("METRICS_OUTPUT", file.path(DEFAULT_ROOT, paste0(METHOD, "_metrics.csv")))
  OOB_OUTPUT <- setting("OOB_OUTPUT", file.path(DEFAULT_ROOT, paste0(METHOD, "_oob.csv")))
  BAG_OUTPUT <- Sys.getenv("BAG_OUTPUT", file.path(DEFAULT_ROOT, "bagging.csv"))
  BAG_OOB_OUTPUT <- Sys.getenv("BAG_OOB_OUTPUT", file.path(DEFAULT_ROOT, "bagging_oob.csv"))
  BAG_METRICS_OUTPUT <- Sys.getenv("BAG_METRICS_OUTPUT", file.path(DEFAULT_ROOT, "bagging_metrics.csv"))
  TZ <- "Asia/Singapore"
  HORIZONS <- as.integer(strsplit(setting("HORIZONS", "1,6,12,24"), ",")[[1]])
  TRAIN_HOURS <- as.integer(setting("TRAIN_HOURS", "13140"))
  REFIT_HOURS <- as.integer(setting("REFIT_HOURS", "168"))
  FIRST_ORIGIN <- setting("FIRST_ORIGIN", "2018-07-16 00:00:00")
  LAST_TARGET <- setting("LAST_TARGET", "2026-09-30 23:00:00")
  NUM_TREES <- as.integer(setting("NUM_TREES", "500"))
  MIN_NODE_SIZE <- as.integer(setting("MIN_NODE_SIZE", "5"))
  MAX_REFITS <- as.integer(setting("MAX_REFITS", "0"))

  if (!file.exists(INPUT)) {
    stop("Feature file not found: ", INPUT,
         "\nRun Rscript R/data_processing/build_features.R first.")
  }
  if (!length(HORIZONS) || anyNA(HORIZONS) || any(HORIZONS < 1L)) {
    stop("RF_HORIZONS must be a comma-separated list of positive integers.")
  }
  if (anyNA(c(TRAIN_HOURS, REFIT_HOURS, NUM_TREES, MIN_NODE_SIZE, MAX_REFITS)) ||
      TRAIN_HOURS < 1L || REFIT_HOURS < 1L || NUM_TREES < 1L ||
      MIN_NODE_SIZE < 1L || MAX_REFITS < 0L) {
    stop("Training, refit, tree, node-size, and max-refit settings must be valid positive values (MAX_REFITS may be 0).")
  }

  # fread can otherwise interpret local clock strings as UTC POSIXct values.
  d <- read_gz(INPUT, colClasses = c(datetime = "character"))
  required <- c("datetime", "y", "usable", paste0("y_h", HORIZONS),
                paste0("nem_h", HORIZONS))
  missing <- setdiff(required, names(d))
  if (length(missing)) stop("Missing required feature columns: ",
                            paste(missing, collapse = ", "))

  d[, datetime_posix := as.POSIXct(datetime, format = "%Y-%m-%d %H:%M:%S",
                                   tz = TZ)]
  if (anyNA(d$datetime_posix)) stop("Could not parse one or more datetimes.")
  setorder(d, datetime_posix)
  if (any(as.numeric(diff(d$datetime_posix), units = "hours") != 1))
    stop("Feature timestamps must form a contiguous hourly series.")

  target_cols <- grep("^(y_h|nem_h)", names(d), value = TRUE)
  predictors <- ForestBasePredictors(d, FEATURE_MODE)
  if (!length(predictors)) stop("No predictor columns found.")
  if (!all(vapply(d[, ..predictors], is.numeric, logical(1)))) {
    stop("All feature predictors are expected to be numeric.")
  }
  target_calendar <- list()
  base_predictors <- predictors
  horizon_predictors <- list()
  TARGET_CALENDAR_COLS <- character()
  if (FEATURE_MODE == "lightgbm") d[, dow := factor(dow, levels = 0:6)]
  if (USE_TARGET_CALENDAR) {
    # Build once per horizon, rather than copying the full feature matrix weekly.
    for (h in HORIZONS) {
      calendar <- BuildForestHorizonFeatures(d, h, FEATURE_MODE)
      observed_target <- d$usable == 1L & !is.na(d[[paste0("y_h", h)]])
      if (any(!complete.cases(calendar[observed_target]))) {
        stop("Missing target-calendar values for observed targets at h=", h)
      }
      target_calendar[[as.character(h)]] <- calendar
      horizon_predictors[[as.character(h)]] <- c(base_predictors, names(calendar))
    }
    TARGET_CALENDAR_COLS <- c("hr_tgt", "dow_tgt", "holiday_tgt", "workday_tgt")
  } else {
    for (h in HORIZONS) horizon_predictors[[as.character(h)]] <- base_predictors
  }

  usable <- d[usable == 1L]
  if (!nrow(usable)) stop("No rows are marked usable in the feature file.")
  first_origin <- as.POSIXct(FIRST_ORIGIN, format = "%Y-%m-%d %H:%M:%S", tz = TZ)
  last_target <- as.POSIXct(LAST_TARGET, format = "%Y-%m-%d %H:%M:%S", tz = TZ)
  if (is.na(last_target)) stop("LAST_TARGET must be a valid local datetime.")
  last_time <- min(max(d$datetime_posix), last_target)
  if (is.na(first_origin) || first_origin > last_time ||
      !first_origin %in% d$datetime_posix)
    stop("FIRST_ORIGIN must match an available feature timestamp.")
  origins <- seq(from = first_origin, to = last_time,
                 by = REFIT_HOURS * 3600)
  if (MAX_REFITS > 0L) origins <- head(origins, MAX_REFITS)
  if (!length(origins)) stop("No forecast origins fit in the available sample.")

  dir.create(dirname(OUTPUT), recursive = TRUE, showWarnings = FALSE)
  dir.create(dirname(METRICS_OUTPUT), recursive = TRUE, showWarnings = FALSE)
  dir.create(dirname(OOB_OUTPUT), recursive = TRUE, showWarnings = FALSE)
  if (file.exists(OUTPUT)) file.remove(OUTPUT)
  if (file.exists(OOB_OUTPUT)) file.remove(OOB_OUTPUT)
  if (COMPARE_BAGGING) {
    for (path in c(BAG_OUTPUT, BAG_OOB_OUTPUT, BAG_METRICS_OUTPUT)) {
      dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
      if (file.exists(path)) file.remove(path)
    }
  }

  if (exists("RESUME_INIT", inherits = FALSE)) RESUME_INIT()

  cat("features:", nrow(d), "rows; predictors by horizon:",
      paste(paste0(names(horizon_predictors), "h=", lengths(horizon_predictors)),
            collapse = ", "), "\n")
  cat("feature set:", FEATURE_SET, "\n")
  cat("horizons:", paste(HORIZONS, collapse = ", "),
      "| train window:", TRAIN_HOURS, "hours | refit:", REFIT_HOURS, "hours\n")
  cat("forecast origins:", length(origins), "from",
      format(origins[1], "%Y-%m-%d %H:%M", tz = TZ), "to",
      format(tail(origins, 1), "%Y-%m-%d %H:%M", tz = TZ), "\n")

  for (i in seq_along(origins)) {
    for (h in HORIZONS) {
    predictors <- horizon_predictors[[as.character(h)]]
    target <- paste0("y_h", h)
    benchmark <- paste0("nem_h", h)
    cat("\nHorizon", h, "hour(s)\n")

      origin <- origins[i]
      last_train_time <- origin - h * 3600
      # Exactly TRAIN_HOURS training pairs ending at s=T-h, as in benchmarks.R.
      train_start <- last_train_time - (TRAIN_HOURS - 1L) * 3600
      forecast_end <- origin + REFIT_HOURS * 3600

      train_idx <- which(d$usable == 1L &
                           d$datetime_posix >= train_start &
                           d$datetime_posix <= last_train_time &
                           !is.na(d[[target]]))
      predict_idx <- which(d$usable == 1L &
                             d$datetime_posix >= origin &
                             d$datetime_posix < forecast_end &
                           d$datetime_posix + h * 3600 <= last_time &
                             !is.na(d[[target]]))
      if (!length(predict_idx)) next
      if (length(train_idx) != TRAIN_HOURS) {
        stop("Expected a full window of ", TRAIN_HOURS, " training rows at ",
             format(origin, "%Y-%m-%d %H:%M", tz = TZ),
             " for horizon ", h, ".")
      }

      if (exists("RESUME_SKIP", inherits = FALSE) &&
          RESUME_SKIP(h, origin, predict_idx, train_idx)) next

      forecast_already_saved <- exists("RESUME_FORECAST_EXISTS", inherits = FALSE) &&
        RESUME_FORECAST_EXISTS(h, origin)
      train <- d[train_idx]
      test <- d[predict_idx]
      if (USE_TARGET_CALENDAR) {
        calendar <- target_calendar[[as.character(h)]]
        train[, (names(calendar)) := calendar[train_idx]]
        test[, (names(calendar)) := calendar[predict_idx]]
      }
      fit <- fit_ensemble(
        train[, c(target, predictors), with = FALSE], target, length(predictors),
        method = METHOD, num.trees = NUM_TREES,
        min.node.size = MIN_NODE_SIZE)
      model <- fit$model
      trace <- as.data.table(fit$trace)
      trace[, `:=`(horizon = h, refit_origin = format(origin, tz = TZ),
                   train_n = length(train_idx), method = METHOD)]
      fwrite(trace, OOB_OUTPUT, append = file.exists(OOB_OUTPUT))
      predicted <- predict(model, data = test[, predictors, with = FALSE])$predictions

      result <- data.table(
        horizon = h,
        refit_origin = format(origin, "%Y-%m-%d %H:%M:%S", tz = TZ),
        datetime = test$datetime,
        target_datetime = format(test$datetime_posix + h * 3600,
                                 "%Y-%m-%d %H:%M:%S", tz = TZ),
        actual_mw = test[[target]],
        forecast_mw = predicted,
        ema_forecast_mw = test[[benchmark]],
        train_n = length(train_idx)
      )
      result[, `:=`(mtry = model$mtry, oob_mse = model$prediction.error,
                    feature_set = FEATURE_SET, predictor_count = length(predictors))]
      if (!forecast_already_saved) fwrite(result, OUTPUT, append = file.exists(OUTPUT))
      if (COMPARE_BAGGING) {
        bag <- fit$bagging_model
        if (is.null(bag)) {
          cat("    Fitting comparison model separately\n")
          bag <- fit_ensemble(
            train[, c(target, predictors), with = FALSE], target, length(predictors),
            method = "bagging", num.trees = NUM_TREES,
            min.node.size = MIN_NODE_SIZE)$model
        }
        bag_result <- copy(result)
        bag_result[, `:=`(
          forecast_mw = predict(bag, data = test[, predictors, with = FALSE])$predictions,
          mtry = bag$mtry, oob_mse = bag$prediction.error)]
        if (!forecast_already_saved) fwrite(bag_result, BAG_OUTPUT, append = file.exists(BAG_OUTPUT))
        bag_trace <- data.table(
          mtry = bag$mtry, oob_mse = bag$prediction.error,
          oob_rmse_mw = sqrt(bag$prediction.error), relative_improvement = NA_real_,
          selected = TRUE, horizon = h, refit_origin = format(origin, tz = TZ),
          train_n = length(train_idx), method = "bagging")
        fwrite(bag_trace, BAG_OOB_OUTPUT, append = file.exists(BAG_OOB_OUTPUT))
        if (exists("PAIR_WRITER", inherits = FALSE)) PAIR_WRITER(result, bag_result, model, bag)
      }

      if (i == 1L || i %% 25L == 0L || i == length(origins)) {
        cat(sprintf("  refit %d/%d: %s, train n=%s, forecast n=%s\n",
                    i, length(origins),
                    format(origin, "%Y-%m-%d", tz = TZ),
                    format(length(train_idx), big.mark = ","),
                    format(length(predict_idx), big.mark = ",")))
      }
    }
  }

  predictions <- fread(OUTPUT)
  metrics <- predictions[, .(
    n = .N,
    rmse_mw = sqrt(mean((actual_mw - forecast_mw)^2)),
    mae_mw = mean(abs(actual_mw - forecast_mw)),
    mape_pct = 100 * mean(abs(actual_mw - forecast_mw) / actual_mw),
    ema_rmse_mw = sqrt(mean((actual_mw - ema_forecast_mw)^2)),
    ema_mae_mw = mean(abs(actual_mw - ema_forecast_mw))
  ), by = horizon][order(horizon)]
  fwrite(metrics, METRICS_OUTPUT)

  cat("\nPrediction file:", OUTPUT, "\n")
  cat("Metrics file:", METRICS_OUTPUT, "\n")
  print(metrics)
  if (COMPARE_BAGGING) {
    bag_predictions <- fread(BAG_OUTPUT)
    bag_metrics <- bag_predictions[, .(
      n = .N,
      rmse_mw = sqrt(mean((actual_mw - forecast_mw)^2)),
      mae_mw = mean(abs(actual_mw - forecast_mw)),
      mape_pct = 100 * mean(abs(actual_mw - forecast_mw) / actual_mw),
      ema_rmse_mw = sqrt(mean((actual_mw - ema_forecast_mw)^2)),
      ema_mae_mw = mean(abs(actual_mw - ema_forecast_mw))
    ), by = horizon][order(horizon)]
    fwrite(bag_metrics, BAG_METRICS_OUTPUT)
    print(bag_metrics)
  }
  # Aggregate importance for completed pairs in this experiment.
  importance_files <- list.files(importance_dir, pattern = "[.]rds$", full.names = TRUE)
  if (length(importance_files)) {
    importance <- data.table::rbindlist(lapply(importance_files, readRDS))
    data.table::fwrite(importance, detail_path)
    summary <- importance[, .(completed_refits = .N, mean_importance = mean(importance)),
                          by = .(horizon, model, feature, method)]
    data.table::setorder(summary, horizon, model, -mean_importance)
    data.table::fwrite(summary, summary_path)
  }
  status <- system2(file.path(R.home("bin"), "Rscript"),
    c("R/evaluation/compare_forests.R", shQuote(OUTPUT), shQuote(BAG_OUTPUT),
      shQuote(comparison_path)))
  if (status != 0L) stop("Fixed-mtry comparison failed.")
  if (canonical) {
    temporary <- paste0(published_path, ".tmp")
    status <- system2(find_gzip(), c("-c", shQuote(forecast_path)), stdout = temporary)
    if (status != 0L || !file.rename(temporary, published_path))
      stop("Could not publish compressed forecasts.")
    cat("Published forecasts:", published_path, "\n")
  }
  cat("Consolidated forecasts:", forecast_path, "\n")
  writeLines(c(
    "No tuning: RF uses floor(p/3); classical bagging uses mtry=p.",
    paste("Paired predictions, mtry and OOB MSE:", forecast_path),
    paste("Evaluation metrics by horizon:", comparison_path),
    paste("Predictors by horizon:", paste(names(horizon_predictors),
          lengths(horizon_predictors), sep = "h=", collapse = ", ")),
    "On restart, validated completed paired fits are skipped; forecasts are committed atomically.",
    "comparison.csv summarizes saved forecasts and may be partial until the run finishes.",
    paste("Trees:", NUM_TREES, "| minimum node size:", MIN_NODE_SIZE),
    paste("Training pairs:", TRAIN_HOURS, "| refit hours:", REFIT_HOURS),
    paste("First origin:", FIRST_ORIGIN, "| final target:", LAST_TARGET)
  ), file.path(state_root, "README.txt"))
})
