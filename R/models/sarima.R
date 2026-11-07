# sarima.R ---------------------------------------------------------------------
#
#   Seasonal ARIMA benchmark: ARIMA(p,d,q)(P,D,Q)[24] on demand alone (no predictors).
#   The seasonal part works at the daily period (24 h), so it can use "the same hour
#   yesterday" but not the weekly cycle.
#
#   Tuning: the orders (p, d, q, P, Q) are chosen ONCE by AICc (auto.arima) on the first
#   rolling window, i.e. on data up to the first forecast origin (D fixed at 1; p, q <= 3,
#   P, Q <= 2, total order <= 8). The coefficients are then re-estimated at every weekly
#   refit on the latest window by conditional sum of squares. Between refits they are
#   frozen, but the state is updated every hour: at each origin the frozen model is
#   filtered over the last STATE_H hours and run forward recursively, and the forecasts
#   at h = 1, 6, 12, 24 are read off that one 24-step path (one fit serves all horizons).
#
#   in : data/processed/features.csv.gz   (only datetime and y are used)
#   out: output/forecasts/sarima.csv.gz
#        output/tables/sarima_orders.txt          selected orders + first-window coefficients
#        output/tables/sarima_coef_by_refit.csv   coefficients at every weekly refit
#
#   Rscript R/sarima.R            full run (about 1 hour on 2 cores)
#   Rscript R/sarima.R --dry-run  3 refits only, writes *_dryrun files (to check the code)
#
# Output shape matches lasso.R / benchmarks.csv.gz exactly: one row per (forecast origin,
# h), `datetime` is the ORIGIN and `actual` is y_{t+h}, so evaluate.R can merge the model
# files on (datetime, h) without any re-alignment.

suppressMessages({
  library(data.table)
  library(forecast)
})

# ---- 0. settings -------------------------------------------------------------
# THESE MUST MATCH lasso.R / benchmarks.R. If any of H, WINDOW, REFIT or OOS_START differs,
# the forecasts are for different hours and cannot be compared or DM-tested.
IN        <- "data/processed/features.csv.gz"
H         <- c(1L, 6L, 12L, 24L)
WINDOW    <- as.integer(1.5 * 365 * 24) # 13,140 hours (1.5 years) of training data
REFIT     <- 168L                       # refit every week
OOS_START <- "2018-07-16 00:00:00"      # first forecast origin: the first Monday
                                        # with a full window behind it
PERIOD    <- 24L                        # seasonal period (hours)
D_SEAS    <- 1L                         # seasonal differencing
METHOD    <- "CSS"                      # coefficient estimation at each refit
STATE_H   <- 8L * 168L                  # hours the frozen model is filtered over at each origin
N_CORES   <- as.integer(Sys.getenv("N_CORES", max(1L, parallel::detectCores() - 1L)))

# Smoke test: number of refits to run. Inf = full run. `--dry-run` sets it to 3.
MAX_REFITS <- Inf
DRY <- "--dry-run" %in% commandArgs(trailingOnly = TRUE)
if (DRY) MAX_REFITS <- 3L
SFX <- if (DRY) "_dryrun" else ""
OUT <- paste0("output/forecasts/sarima", SFX, ".csv.gz")
TAB <- "output/tables"
set.seed(4308)

# mclapply forks, which Windows cannot do, so fall back to a plain loop there.
par_map <- function(X, FUN) {
  if (N_CORES > 1L && .Platform$OS.type != "windows")
    parallel::mclapply(X, FUN, mc.cores = N_CORES, mc.preschedule = FALSE)
  else lapply(X, FUN)
}

# ---- load --------------------------------------------------------------------
f <- fread(IN, select = c("datetime", "y"), colClasses = c(datetime = "character"))
setorder(f, datetime)
y <- f$y
N <- length(y)

T_first <- match(OOS_START, f$datetime)
if (is.na(T_first)) stop("OOS_START not found in features: ", OOS_START)
if (T_first <= WINDOW) stop("no full window before OOS_START")
HMAX <- max(H)

# ---- 1. orders: AICc on the first window -------------------------------------
# Window ends at the first origin, so nothing at or after the first forecast target is used.
cat("choosing SARIMA orders by AICc on the first window ...\n")
first_win <- (T_first - WINDOW + 1L):T_first
sarima_auto <- auto.arima(ts(y[first_win], frequency = PERIOD),
                          d = NA, D = D_SEAS, max.p = 3, max.q = 3, max.P = 2, max.Q = 2,
                          max.order = 8, seasonal = TRUE, stepwise = TRUE,
                          approximation = TRUE, ic = "aicc")
ord <- arimaorder(sarima_auto)               # c(p, d, q, P, D, Q, m)
desc <- sprintf("ARIMA(%s)(%s)[%d]  AICc = %.1f\n  %s\n",
                paste(ord[1:3], collapse = ","), paste(ord[4:6], collapse = ","), PERIOD,
                sarima_auto$aicc,
                paste(names(coef(sarima_auto)), round(coef(sarima_auto), 3), collapse = "  "))
cat(desc)

# ---- 2. one refit block ------------------------------------------------------
# Re-estimate on the WINDOW ending at the refit origin T0 (every observation is known at
# T0), then run the frozen model over the block's origins t = T0 .. T0 + REFIT - 1.
refits <- seq(T_first, N - 1L, by = REFIT)
if (is.finite(MAX_REFITS)) refits <- head(refits, MAX_REFITS)

sarima_block <- function(T0) {
  t_all <- T0:min(T0 + REFIT - 1L, N - 1L)
  out   <- matrix(NA_real_, length(t_all), length(H))
  fit <- tryCatch(
    Arima(ts(y[(T0 - WINDOW + 1L):T0], frequency = PERIOD),
          order = ord[1:3], seasonal = list(order = ord[4:6], period = PERIOD), method = METHOD),
    error = function(e) NULL)
  if (is.null(fit)) return(list(T0 = T0, t = t_all, pred = out, coef = NULL))
  for (k in seq_along(t_all)) {
    t  <- t_all[k]
    hh <- min(HMAX, N - t)                       # cannot forecast past the end of the data
    p  <- tryCatch(predict(Arima(ts(y[(t - STATE_H + 1L):t], frequency = PERIOD), model = fit),
                           n.ahead = hh)$pred, error = function(e) rep(NA_real_, hh))
    ok <- H <= hh
    out[k, ok] <- as.numeric(p)[H[ok]]
  }
  list(T0 = T0, t = t_all, pred = out, coef = coef(fit))
}

# ---- 3. run all refits -------------------------------------------------------
cat("\nrolling window:", format(WINDOW, big.mark = ","), "hours, refit every", REFIT,
    "h, from", OOS_START, "\n")
cat("refits:", length(refits), "  cores:", N_CORES, "\n")
t0  <- Sys.time()
res <- par_map(refits, sarima_block)
cat(sprintf("done in %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))

bad <- vapply(res, function(r) !is.list(r) || is.null(r$coef), TRUE)
if (any(bad)) warning(sum(bad), " refit(s) failed (forecasts NA)")
res <- res[!bad]

# ---- 4. assemble: one row per (origin, h) ------------------------------------
out <- rbindlist(lapply(seq_along(H), function(j) {
  t    <- unlist(lapply(res, `[[`, "t"))
  pred <- unlist(lapply(res, function(r) r$pred[, j]))
  keep <- t <= N - H[j]                          # last h origins have no y_{t+h}
  data.table(datetime = f$datetime[t[keep]], h = H[j], actual = y[t[keep] + H[j]], sarima = pred[keep])
}))
if (anyNA(out$sarima)) warning(sum(is.na(out$sarima)), " NA forecasts left in the output")

if (!dir.exists(dirname(OUT))) dir.create(dirname(OUT), recursive = TRUE)
if (!dir.exists(TAB)) dir.create(TAB, recursive = TRUE)
fwrite(out, OUT)
writeLines(desc, file.path(TAB, paste0("sarima_orders", SFX, ".txt")))
fwrite(rbindlist(lapply(res, function(r) as.data.table(c(list(refit_origin = f$datetime[r$T0]), as.list(r$coef)))), fill = TRUE),
       file.path(TAB, paste0("sarima_coef_by_refit", SFX, ".csv")))

cat("\n=== out-of-sample error summary ===\n")
for (hh in H) {
  s <- out[h == hh & !is.na(sarima)]
  if (!nrow(s)) next
  cat(sprintf("  h=%2d  n=%s  RMSE %6.1f  MAE %6.1f  MAPE %.3f%%\n", hh, format(nrow(s), big.mark = ","),
      sqrt(mean((s$actual - s$sarima)^2)), mean(abs(s$actual - s$sarima)),
      100 * mean(abs(s$actual - s$sarima) / s$actual)))
}
cat("\nwritten:", OUT, "\n")
