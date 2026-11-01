# view_benchmarks.R ------------------------------------------------------------
# Loads the saved benchmark forecasts and prints the error table, in seconds.
# Run benchmarks.R (~6 min) only when the benchmark code itself changes; use
# this file to look at the results.
#
#   in : output/forecasts/benchmarks.csv.gz   (from benchmarks.R)

suppressMessages(library(data.table))

bm <- fread("output/forecasts/benchmarks.csv.gz", colClasses = c(datetime = "character"))

RMSE <- function(pred, truth) sqrt(mean((truth - pred)^2))
MAE  <- function(pred, truth) mean(abs(truth - pred))
MAPE <- function(pred, truth) 100 * mean(abs(truth - pred) / truth)

models <- c("snaive168", "snaive24", "ar_bic", "ema")

err <- bm[, .(model = models,
              RMSE  = sapply(models, function(v) RMSE(get(v), actual)),
              MAE   = sapply(models, function(v) MAE(get(v), actual)),
              MAPE  = sapply(models, function(v) MAPE(get(v), actual))), by = h]

cat("forecast origins:", min(bm$datetime), "->", max(bm$datetime), "\n\n")
print(err[, .(h, model, RMSE = round(RMSE, 1), MAE = round(MAE, 1), MAPE = round(MAPE, 3))])
