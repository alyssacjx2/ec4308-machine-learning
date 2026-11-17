# Standalone Holm-adjusted version of dm_and_combination.R.
# Run from repository root: Rscript R/evaluation/forecast_tests.R
# No other script calls this file; no models are refitted.
# Same inputs, validation-selected top-three equal-weight combination, test
# period, AR benchmark, loss differences, Bartlett Newey-West variance and
# two-sided Student-t p-values as dm_and_combination.R.
# Negative statistic favours the row model. Bandwidth is
# max(h - 1, ceiling(n^(1/3))): 21/21/21/23 lags for n=8,760.
# Holm covers all 44 comparisons separately for squared and absolute loss.
# EMA tests are retained to match the original, but unknown issue times mean
# EMA remains an operational reference, not a horizon-matched competitor.
# Output: output/tables/dm_tests_holm.csv; other evaluation files are unchanged.
suppressPackageStartupMessages(library(data.table))
source("R/utils.R")

RunForecastTests <- function() {
  FC <- "output/forecasts"; TAB <- "output/tables"
  cc <- c(datetime = "character")
  bm <- read_gz(file.path(FC, "benchmarks.csv.gz"), colClasses = cc)
  la <- read_gz(file.path(FC, "lasso.csv.gz"), colClasses = cc)
  lg <- read_gz(file.path(FC, "lgbm.csv.gz"), colClasses = cc)
  sh <- read_seasonal_forecasts(FC)
  setnames(sh, "origin", "datetime")
  fo <- read_gz(file.path(FC, "rf_bagging.csv.gz"), colClasses = c(datetime = "character", target_datetime = "character"))
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


  dm_test <- function(l1, l2, h = 1) {
    x <- l1 - l2; x <- x[is.finite(x)]; n <- length(x); lag <- max(h - 1L, ceiling(n^(1/3)))
    xb <- mean(x); xc <- x - xb
    g <- function(k) sum(xc[(k + 1):n] * xc[1:(n - k)]) / n
    lrv <- g(0) + 2 * sum(vapply(seq_len(lag), function(k) (1 - k / (lag + 1)) * g(k), 0))
    s <- xb / sqrt(lrv / n); c(stat = s, p = 2 * pt(-abs(s), df = n - 1))
  }
  all_m <- c(models, "combination", "ema")
  te <- d[period == "test"]
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
  # One family includes all alternatives and horizons in the original script:
  # 9 single models + combination + EMA, at 4 horizons = 44 tests per loss.
  # Squared and absolute losses are adjusted separately.
  dm[, `:=`(p_se_holm = p.adjust(p_se, method = "holm"),
            p_ae_holm = p.adjust(p_ae, method = "holm"), family_size = .N)]
  conclusion <- function(statistic, p) {
    fifelse(p >= 0.05, "no_detectable_difference",
            fifelse(statistic < 0, "model_better", "ar_better"))
  }
  dm[, `:=`(conclusion_se_holm = conclusion(DM_se, p_se_holm),
            conclusion_ae_holm = conclusion(DM_ae, p_ae_holm))]
  dir.create(TAB, recursive = TRUE, showWarnings = FALSE)
  fwrite(dm, file.path(TAB, "dm_tests_holm.csv"))

  print(dm[, .(h, model, DM_se, p_se, p_se_holm, conclusion_se_holm,
                DM_ae, p_ae, p_ae_holm, conclusion_ae_holm)])
  cat("Saved", file.path(TAB, "dm_tests_holm.csv"), "\n")
  invisible(dm)
}

if (sys.nframe() == 0L) RunForecastTests()
