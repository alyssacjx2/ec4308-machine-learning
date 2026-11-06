# Compare only identical forecast origins/targets, never aggregate unmatched runs.
# Optional positional arguments: RF CSV, bagging CSV, output comparison CSV.
library(data.table)
args <- commandArgs(trailingOnly = TRUE)
rf_path <- if (length(args) >= 1L) args[1] else "output/forecasts/random_forest.csv"
bag_path <- if (length(args) >= 2L) args[2] else "output/forecasts/bagging.csv"
out <- if (length(args) >= 3L) args[3] else "output/tables/forest_comparison.csv"
rf <- fread(rf_path)
bag <- fread(bag_path)
keys <- c("horizon", "datetime", "target_datetime")
if (anyDuplicated(rf[, ..keys]) || anyDuplicated(bag[, ..keys]))
  stop("Duplicate forecast keys.")
paired <- merge(rf, bag, by = keys, suffixes = c("_rf", "_bag"))
if (!nrow(paired)) stop("No common forecasts.")
if (nrow(paired) != nrow(rf) || nrow(paired) != nrow(bag))
  warning("Scoring only the intersection of forecast timestamps.")
if (any(paired$actual_mw_rf != paired$actual_mw_bag) ||
    any(paired$train_n_rf != paired$train_n_bag) ||
    any(paired$refit_origin_rf != paired$refit_origin_bag))
  stop("Actuals, training counts or refit schedules differ.")
summary <- paired[, .(
  n = .N, first_origin = min(datetime), last_origin = max(datetime),
  rf_rmse_mw = sqrt(mean((actual_mw_rf - forecast_mw_rf)^2)),
  bag_rmse_mw = sqrt(mean((actual_mw_rf - forecast_mw_bag)^2)),
  rf_mae_mw = mean(abs(actual_mw_rf - forecast_mw_rf)),
  bag_mae_mw = mean(abs(actual_mw_rf - forecast_mw_bag)),
  rf_mape_pct = 100 * mean(abs((actual_mw_rf - forecast_mw_rf) / actual_mw_rf)),
  bag_mape_pct = 100 * mean(abs((actual_mw_rf - forecast_mw_bag) / actual_mw_rf))
), by = horizon][order(horizon)]
summary[, rf_rmse_reduction_pct := 100 * (bag_rmse_mw - rf_rmse_mw) / bag_rmse_mw]
summary[, rmse_winner := fifelse(abs(rf_rmse_mw - bag_rmse_mw) < 1e-9,
                                "tie", fifelse(rf_rmse_mw < bag_rmse_mw, "RF", "bagging"))]
dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
fwrite(summary, out)
print(summary)
cat("Positive RMSE reduction means RF beats bagging. Saved:", out, "\n")
