# view_benchmarks.R ------------------------------------------------------------
# Loads the saved benchmark forecasts and prints the error table, in seconds.
# Run benchmarks.R (~6 min) only when the benchmark code itself changes; use
# this file to look at the results.
#
#   in : output/forecasts/benchmarks.csv.gz   (from benchmarks.R)

suppressMessages(library(data.table))

bm <- fread("output/forecasts/benchmarks.csv.gz", colClasses = c(datetime = "character"))

# ---- errors ------------------------------------------------------------------
# Quick look only. Full evaluation (DM tests, combinations) belongs in evaluate.R.
# RMSE is the headline loss; MAE and MAPE are added because the load-forecasting
# literature reports them almost universally.
RMSE <- function(pred, truth) sqrt(mean((truth - pred)^2))
MAE  <- function(pred, truth) mean(abs(truth - pred))
MAPE <- function(pred, truth) 100 * mean(abs(truth - pred) / truth)

models <- c("snaive168", "snaive24", "ar_bic", "ema")

err <- bm[, .(model = models,
              RMSE  = sapply(models, function(v) RMSE(get(v), actual)),
              MAE   = sapply(models, function(v) MAE(get(v), actual)),
              MAPE  = sapply(models, function(v) MAPE(get(v), actual))), by = h]

cat("=== out-of-sample errors,", min(bm$datetime), "->", max(bm$datetime), "===\n")
print(err[, .(h, model, RMSE = round(RMSE, 1), MAE = round(MAE, 1), MAPE = round(MAPE, 3))])

# ---- best model per horizon --------------------------------------------------
# The same RMSEs side by side, one row per h, so the winner reads across a row.
# Compare within a row only: longer horizons are harder, so errors grow with h.
cat("\n=== RMSE by horizon (MW), lowest = best ===\n")
wide <- dcast(err, h ~ model, value.var = "RMSE")
setkey(wide, NULL)                     # dcast sets a key; drop it so it is not printed
own  <- setdiff(models, "ema")
wide[, best := own[apply(.SD, 1, which.min)], .SDcols = own]
setcolorder(wide, c("h", models, "best"))
print(wide[, lapply(.SD, function(x) if (is.numeric(x) && !is.integer(x)) round(x, 1) else x)])

# ---- AR lag length -----------------------------------------------------------
# Shows whether BIC is pinned at P_MAX - if so, the search limit, not the data,
# is choosing p.
cat("\nAR lag length chosen by BIC:\n")
print(bm[, .(p_min = min(ar_p), p_median = median(ar_p), p_max = max(ar_p)), by = h])
