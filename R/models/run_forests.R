# RF: floor(p/3) = 18; classical bagging: p = 54 with current features.
# Same windows, horizons, 500 trees and node size 5 as the tuned experiment.
# Output: output/forecasts/; no tuned output files are touched.
# Optional FIXED_OUTPUT_DIR; RF_MAX_REFITS/RF_NUM_TREES etc. still work.
local({
  root <- Sys.getenv("FIXED_OUTPUT_DIR", "output/forecasts")
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  # Engine files are temporary; expose only paired forecasts and comparison.
  staging <- tempfile("fixed-forests-")
  dir.create(staging)
  on.exit(unlink(staging, recursive = TRUE), add = TRUE)
  forecast_path <- file.path(root, "forecasts.csv")
  if (file.exists(forecast_path)) unlink(forecast_path)
  paths <- c(
    RF_OUTPUT = file.path(staging, "rf.csv"),
    RF_METRICS_OUTPUT = file.path(staging, "rf_metrics.csv"),
    RF_TUNING_OUTPUT = file.path(staging, "rf_oob.csv"),
    BAG_OUTPUT = file.path(staging, "bagging.csv"),
    BAG_METRICS_OUTPUT = file.path(staging, "bagging_metrics.csv"),
    BAG_TUNING_OUTPUT = file.path(staging, "bagging_oob.csv"))
  old <- Sys.getenv(names(paths), unset = NA_character_)
  on.exit({
    Sys.unsetenv(names(old)[is.na(old)])
    if (any(!is.na(old))) do.call(Sys.setenv, as.list(old[!is.na(old)]))
  }, add = TRUE)
  do.call(Sys.setenv, as.list(paths))

  PAIR_WRITER <- function(rf, bag) {
    shared <- c("horizon", "refit_origin", "datetime", "target_datetime",
                "actual_mw", "ema_forecast_mw", "train_n")
    if (!identical(rf[, shared, with = FALSE], bag[, shared, with = FALSE]))
      stop("Cannot consolidate: model forecast rows do not match.")
    paired <- data.table::copy(rf)
    data.table::setnames(paired, c("forecast_mw", "mtry", "oob_mse"),
                        c("rf_forecast_mw", "rf_mtry", "rf_oob_mse"))
    paired[, `:=`(bagging_forecast_mw = bag$forecast_mw,
                  bagging_mtry = bag$mtry, bagging_oob_mse = bag$oob_mse)]
    data.table::fwrite(paired, forecast_path, append = file.exists(forecast_path))
  }

  ENSEMBLE_FITTER <- function(data, target, p, method, num.trees,
                              min.node.size, seed = 4308L) {
    mtry <- if (method == "bagging") p else max(1L, as.integer(floor(p / 3)))
    if (mtry > p) stop("Requested mtry exceeds available predictors.")
    model <- ranger::ranger(
      dependent.variable.name = target, data = data, mtry = mtry,
      num.trees = num.trees, min.node.size = min.node.size,
      replace = TRUE, sample.fraction = 1, splitrule = "variance", seed = seed)
    if (!is.finite(model$prediction.error)) stop("Non-finite OOB MSE.")
    cat(sprintf("    Fixed mtry=%d, OOB MSE=%.3f (no tuning)\n",
                mtry, model$prediction.error))
    trace <- data.frame(mtry = mtry, oob_mse = model$prediction.error,
                        oob_rmse_mw = sqrt(model$prediction.error),
                        relative_improvement = NA_real_, selected = TRUE)
    list(model = model, trace = trace, bagging_model = NULL)
  }
  RUN_BOTH <- TRUE
  source("R/random_forest.R", local = TRUE)
  status <- system2(file.path(R.home("bin"), "Rscript"),
    c("R/compare_forests.R", shQuote(OUTPUT), shQuote(BAG_OUTPUT),
      shQuote(file.path(root, "comparison.csv"))))
  if (status != 0L) stop("Fixed-mtry comparison failed.")
  cat("Consolidated forecasts:", forecast_path, "\n")
  writeLines(c(
    "No tuning: RF uses floor(p/3); classical bagging uses mtry=p.",
    "forecasts.csv contains both predictions, mtry values and OOB MSE side by side.",
    "comparison.csv contains both models' evaluation metrics by horizon.",
    paste("Predictors:", length(predictors), "| RF mtry:", max(1L, floor(length(predictors)/3)),
          "| bagging mtry:", length(predictors)),
    paste("Trees:", NUM_TREES, "| minimum node size:", MIN_NODE_SIZE),
    paste("Training pairs:", TRAIN_HOURS, "| refit hours:", REFIT_HOURS),
    paste("First origin:", FIRST_ORIGIN, "| final target:", LAST_TARGET)
  ), file.path(root, "README.txt"))
})
