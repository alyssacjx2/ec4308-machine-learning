# dm_and_combination.R ---------------------------------------------------------
#
#   Adds what evaluate.R does not: (1) a validation-selected equal-weight combination,
#   (2) Linex loss, (3) Diebold-Mariano tests, all on the TEST period, for every model
#   in output/forecasts. No model fitting.
#
#   Combination: for each horizon the 3 non-reference models with the lowest VALIDATION
#   RMSE (Apr-Sep 2025) are averaged with equal weights. The test period plays no part in
#   choosing the members.
#   DM tests: squared-error loss, Newey-West long-run variance, each model against the pre-specified
#   benchmark, the AR(p) with calendar dummies (ar_bic). Negative statistic = row model better.
#   Linex: L(e) = exp(a e) - a e - 1 with e = (actual - forecast) / 100 and a = 0.5, so
#   under-forecasts (e > 0) cost more than over-forecasts.
#
#   in : output/forecasts/{benchmarks,lasso,lgbm}.csv.gz, sarima_holt_winters_forecast_results.csv(.zip),
#        randomforest_bagging.zip
#   out: output/tables/test_metrics_all_models.csv, output/tables/dm_tests_all_models.csv
#
#   Rscript R/evaluation/dm_and_combination.R

suppressPackageStartupMessages(library(data.table))
source("R/utils.R")
FC <- "output/forecasts"; TAB <- "output/tables"
cc <- c(datetime = "character")
bm <- read_gz(file.path(FC, "benchmarks.csv.gz"), colClasses = cc)
la <- read_gz(file.path(FC, "lasso.csv.gz"), colClasses = cc)
lg <- read_gz(file.path(FC, "lgbm.csv.gz"), colClasses = cc)
tmp <- tempfile(); dir.create(tmp)
sz <- file.path(FC, "sarima_holt_winters_forecast_results.csv")
if (!file.exists(sz)) unzip(file.path(FC, "sarima_holt_winters_forecast_results.zip"), exdir = tmp)
sh <- fread(if (file.exists(sz)) sz else file.path(tmp, "sarima_holt_winters_forecast_results.csv"),
            colClasses = c(origin = "character", target_datetime = "character"))
setnames(sh, "origin", "datetime")
unzip(file.path(FC, "randomforest_bagging.zip"), files = "forecasts.csv", exdir = tmp)
fo <- fread(file.path(tmp, "forecasts.csv"), colClasses = c(datetime = "character", target_datetime = "character"))
setnames(fo, c("horizon", "actual_mw", "rf_forecast_mw", "bagging_forecast_mw"), c("h", "actual", "random_forest", "bagging"))

d <- Reduce(function(a, b) merge(a, b, by = c("datetime", "h")),
            list(bm[, .(datetime, h, actual, snaive168, snaive24, ar_bic, ema)],
                 la[, .(datetime, h, lasso, post_lasso)],
                 lg[, .(datetime, h, lgbm)],
                 sh[, .(datetime, h, period, sarima, hw24)],
                 fo[, .(datetime, h, random_forest, bagging)]))
models <- c("ar_bic", "lasso", "post_lasso", "lgbm", "random_forest", "bagging", "sarima", "hw24", "snaive168", "snaive24")

# combination members chosen on validation only
val <- d[period == "validation"]
mem <- rbindlist(lapply(sort(unique(d$h)), function(hh) {
  r <- sapply(models, function(m) sqrt(mean((val[h == hh]$actual - val[h == hh][[m]])^2)))
  data.table(h = hh, members = paste(names(sort(r))[1:3], collapse = "+"))
}))
d[, combination := NA_real_]
for (hh in mem$h) { ms <- strsplit(mem[h == hh, members], "\\+")[[1]]; d[h == hh, combination := rowMeans(.SD), .SDcols = ms] }
print(mem)
fwrite(mem, file.path(TAB, "combination_members.csv"))

dm_test <- function(l1, l2, h = 1) {
  x <- l1 - l2; x <- x[is.finite(x)]; n <- length(x); lag <- max(h - 1L, ceiling(n^(1/3)))
  xb <- mean(x); xc <- x - xb
  g <- function(k) sum(xc[(k + 1):n] * xc[1:(n - k)]) / n
  lrv <- g(0) + 2 * sum(vapply(seq_len(lag), function(k) (1 - k / (lag + 1)) * g(k), 0))
  s <- xb / sqrt(lrv / n); c(stat = s, p = 2 * pt(-abs(s), df = n - 1))
}
linex <- function(e, a = 0.5) { e <- e / 100; mean(exp(a * e) - a * e - 1) }
all_m <- c(models, "combination", "ema")
te <- d[period == "test"]
met <- rbindlist(lapply(sort(unique(te$h)), function(hh) rbindlist(lapply(all_m, function(m) {
  x <- te[h == hh]; e <- x$actual - x[[m]]
  data.table(h = hh, model = m, n = length(e), rmse = sqrt(mean(e^2)), mae = mean(abs(e)),
             mape = 100 * mean(abs(e) / x$actual), bias = mean(e), linex = linex(e))
}))))
fwrite(met, file.path(TAB, "test_metrics_all_models.csv"))

bench <- data.table(h = sort(unique(val$h)), m = "ar_bic")   # pre-specified benchmark: AR(p) with calendar dummies
print(bench)
dm <- rbindlist(lapply(sort(unique(te$h)), function(hh) {
  x <- te[h == hh]; ref <- bench[h == hh, m]; eref <- x$actual - x[[ref]]
  rbindlist(lapply(setdiff(all_m, ref), function(m) {
    e <- x$actual - x[[m]]
    se <- dm_test(e^2, eref^2, hh); ae <- dm_test(abs(e), abs(eref), hh)
    data.table(h = hh, benchmark = ref, model = m, DM_se = se["stat"], p_se = se["p"], DM_ae = ae["stat"], p_ae = ae["p"])
  }))
}))
fwrite(dm, file.path(TAB, "dm_tests_all_models.csv"))
options(width = 200); print(dcast(met, model ~ h, value.var = "rmse")[order(`24`)])
print(dm[, .(h, model, benchmark, DM_se = round(DM_se, 2), p_se = signif(p_se, 2))])
