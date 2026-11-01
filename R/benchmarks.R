# benchmarks.R -----------------------------------------------------------------
#
#   1. seasonal random walk (seasonal naive), weekly and daily season
#   2. direct h-step AR(p), p chosen by BIC, with the Covid regime dummies
#
#   in : data/processed/features.csv.gz
#   out: output/forecasts/benchmarks.csv.gz

suppressMessages(library(data.table))

# ---- 0. settings -------------------------------------------------------------
# The teammates' ML scripts must use the same H, WINDOW, REFIT and
# OOS_START, otherwise the forecasts are for different hours and cannot be compared.
IN        <-"data/processed/features.csv.gz"
OUT       <- "output/forecasts/benchmarks.csv.gz"
H         <- c(1L, 6L, 12L, 24L)        # forecast horizons
WINDOW    <- 3L * 365L * 24L            # 26,280 hours of training pairs
REFIT     <- 168L                       # refit every week
P_MAX     <- 24L                        # longest AR lag tried by BIC
OOS_START <- "2020-01-13 00:00:00"      # first forecast origin, a Monday. TEAM DECISION:
                                        # every model must use the same value or the
                                        # forecasts cannot be compared / DM-tested.
DUMS      <- c("cb_strict", "cb_p1", "cb_p2", "cb_p3", "ha_2021")

# ---- load --------------------------------------------------------------------
f <- fread(IN, colClasses = c(datetime = "character")) # read as character to avoid timezone issues
setorder(f, datetime) # sort by datetime
y   <- f$y # target variable, the hourly load
N   <- length(y) # number of hours in the dataset
DUM <- as.matrix(f[, ..DUMS]) # matrix of Covid regime dummies, aligned with y, for AR regressions

T_first <- match(OOS_START, f$datetime)
if (is.na(T_first)) stop("OOS_START not found in features: ", OOS_START)

# Auxiliary error functions
RMSE <- function(pred, truth) sqrt(mean((truth - pred)^2))
MAE  <- function(pred, truth) mean(abs(truth - pred))
MAPE <- function(pred, truth) 100 * mean(abs(truth - pred) / truth)

# ---- 1. seasonal random walk -------------------------------------------------
# Forecast = demand at the same hour one season ago: y_{t+h-168} (last week) or
# y_{t+h-24} (yesterday).
snaive <- function(y, h, s) shift(y, s - h)   # element t is y_{t+h-s}

# ---- 2. AR(p) selected on BIC ------------------------------------------------
# Lag matrix, built with embed() as in the lecture: row t holds
# (y_t, y_{t-1}, ..., y_{t-P_MAX+1}). Padded with NA so row t is origin t.
LAGS <- rbind(matrix(NA_real_, P_MAX - 1L, P_MAX), embed(y, P_MAX))

# Fit the direct h-step AR(p) at refit origin T0, p in 1..P_MAX chosen on BIC.
# Mirrors runAR(type = "bic") in func-ar.R.
runAR <- function(T0, h) {
  tr   <- (T0 - h - WINDOW + 1L):(T0 - h)  # training origins s, so that s + h <= T0
  y_tr <- y[tr + h]
  X    <- LAGS[tr, , drop = FALSE]

  # Dummies are aligned with the TARGET hour, as the lecture aligns dum with y.
  # Covid regimes were announced in advance, so their value at t+h is known at t.
  # A regime absent from this window is an all-zero column that would make the
  # fit singular, so drop it.
  D <- DUM[tr + h, , drop = FALSE]
  D <- D[, colSums(D) > 0, drop = FALSE]

  n  <- length(y_tr)
  bb <- Inf                                # initialise the "best BIC"
  for (i in 1:P_MAX) {                     # try every lag length 1..P_MAX
    Z <- cbind(1, X[, 1:i, drop = FALSE], D)
    # .lm.fit is lm() without the formula overhead (~5x faster). All candidates
    # share the same n, so n*log(RSS/n) + k*log(n) ranks them exactly as BIC().
    m    <- .lm.fit(Z, y_tr)
    crit <- n * log(sum(m$residuals^2) / n) + ncol(Z) * log(n)
    if (crit < bb) {                       # if BIC improved, keep this model
      bb   <- crit
      best <- list(p = i, coef = m$coefficients, dums = colnames(D))
    }
  }
  best
}

# Rolling window: refit every REFIT hours, forecast every origin until the next
# refit. Mirrors ar.rolling.window() in func-ar.R.
ar.rolling.window <- function(h) {
  origins <- T_first:(N - h)               # the last h origins have no y_{t+h}
  refits  <- seq(T_first, N - h, by = REFIT)
  pred    <- rep(NA_real_, length(origins))
  p_sel   <- rep(NA_integer_, length(origins))

  for (k in seq_along(refits)) {
    T0  <- refits[k]
    fit <- runAR(T0, h)
    t   <- T0:min(T0 + REFIT - 1L, N - h)  # origins served by this fit
    Z   <- cbind(1, LAGS[t, 1:fit$p, drop = FALSE],
                 DUM[t + h, fit$dums, drop = FALSE])
    idx <- t - T_first + 1L
    pred[idx]  <- drop(Z %*% fit$coef)
    p_sel[idx] <- fit$p
    if (k %% 50 == 0 || k == length(refits))
      cat(sprintf("  h=%2d  refit %d / %d  (p = %d)\n", h, k, length(refits), fit$p))
  }
  list(origins = origins, pred = pred, p = p_sel)
}

# ---- 3. run all horizons -----------------------------------------------------
# One row per (forecast origin, h), with every benchmark side by side and the
# actual value, saved to disk. Saving matters: the AR run takes ~6 minutes, while
# evaluate.R and view_benchmarks.R can then reload the forecasts in a second.
# EMA's forecast rides along as a reference level, not as a model of ours.
res <- list()
for (h in H) {
  cat("\nh =", h, "\n")
  ar <- ar.rolling.window(h)
  t  <- ar$origins
  res[[as.character(h)]] <- data.table(
    datetime  = f$datetime[t],                       # forecast origin
    h         = h,
    actual    = y[t + h],                            # y_{t+h}
    snaive168 = snaive(y, h, 168L)[t],
    snaive24  = snaive(y, h, 24L)[t],
    ar_bic    = ar$pred,
    ar_p      = ar$p,
    ema       = f[[sprintf("nem_h%d", h)]][t]        # EMA's forecast, for reference
  )
}
out <- rbindlist(res)

if (!dir.exists(dirname(OUT))) dir.create(dirname(OUT), recursive = TRUE)
fwrite(out, OUT)

# ---- 4. errors ---------------------------------------------------------------
# Quick look only. Full evaluation (DM tests, combinations) belongs in evaluate.R.
# RMSE is the headline loss; MAE and MAPE are added because the load-forecasting
# literature reports them almost universally. The p summary shows whether BIC is
# pinned at P_MAX - if so, the search limit, not the data, is choosing p.
cat("\n=== out-of-sample errors,", OOS_START, "onwards ===\n")
err <- out[, {
  m <- c("snaive168", "snaive24", "ar_bic", "ema")
  .(model = m,
    RMSE  = sapply(m, function(v) RMSE(get(v), actual)),
    MAE   = sapply(m, function(v) MAE(get(v), actual)),
    MAPE  = sapply(m, function(v) MAPE(get(v), actual)))
}, by = h]
print(err[, .(h, model, RMSE = round(RMSE, 1), MAE = round(MAE, 1), MAPE = round(MAPE, 3))])

cat("\nAR lag length chosen by BIC (share of refits):\n")
print(out[, .(p_min = min(ar_p), p_median = median(ar_p), p_max = max(ar_p)), by = h])

cat("\nwritten:", OUT, "\n")
