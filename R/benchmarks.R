# benchmarks.R -----------------------------------------------------------------
#
#   1. seasonal random walk (seasonal naive), weekly and daily season
#   2. direct h-step AR(p), p chosen by BIC, with hour-of-day, day-type, their
#      interactions, and Covid regime dummies
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
WINDOW    <- as.integer(1.5 * 365 * 24) # 13,140 hours (1.5 years) of training pairs
REFIT     <- 168L                       # refit every week
P_GRID    <- seq(24L, 168L, by = 24L)   # AR lag lengths tried by BIC: 1 day ... 1 week
P_MAX     <- max(P_GRID)                # longest lag, sets the lag matrix width
OOS_START <- "2018-07-16 00:00:00"      # first forecast origin: the first Monday with a
                                        # full window and 168 lags behind it. TEAM DECISION:
                                        # every model must use the same value or the
                                        # forecasts cannot be compared / DM-tested.
DUMS      <- c("cb_strict", "cb_p1", "cb_p2", "cb_p3", "ha_2021")

# ---- load --------------------------------------------------------------------
f <- fread(IN, colClasses = c(datetime = "character")) # read as character to avoid timezone issues
setorder(f, datetime) # sort by datetime
y   <- f$y # target variable, the hourly load
N   <- length(y) # number of hours in the dataset

# Seasonal dummies: demand's mean depends on the hour and the day, so the AR gets
# 23 hour-of-day dummies (baseline 00:00) and 6 day-type dummies (baseline
# Sunday). A public holiday counts as a Sunday - its load profile looks like one.
dtype <- ifelse(f$is_holiday == 1L, 0L, f$dow)          # dow: 0 = Sunday
HR <- outer(f$hr, 1:23, "==") * 1
DT <- outer(dtype, 1:6, "==") * 1
colnames(HR) <- sprintf("hr%02d", 1:23)
colnames(DT) <- c("mon", "tue", "wed", "thu", "fri", "sat")

# Hour x day-type interactions (23 x 6 = 138), so each day type gets its own
# daily shape - not just the weekday shape shifted up or down. Without them the
# errors kept a weekend/Monday pattern. With the main effects above, this is one
# level per (day type, hour) cell: 167 dummies + intercept for 168 cells.
HXD <- do.call(cbind, lapply(colnames(DT), function(d) {
  m <- HR * DT[, d]; colnames(m) <- paste0(colnames(HR), "_", d); m
}))

DUM <- cbind(as.matrix(f[, ..DUMS]), HR, DT, HXD) # all AR dummies, aligned with y

T_first <- match(OOS_START, f$datetime)
if (is.na(T_first)) stop("OOS_START not found in features: ", OOS_START)

# ---- 1. seasonal random walk -------------------------------------------------
# Forecast = demand at the same hour one season ago: y_{t+h-168} (last week) or
# y_{t+h-24} (yesterday).
snaive <- function(y, h, s) shift(y, s - h)   # element t is y_{t+h-s}

# ---- 2. AR(p) selected on BIC ------------------------------------------------
# Lag matrix, built with embed(): row t holds (y_t, y_{t-1}, ..., y_{t-P_MAX+1}).
# Padded with NA so row t is origin t.
LAGS <- rbind(matrix(NA_real_, P_MAX - 1L, P_MAX), embed(y, P_MAX))

# Fit the direct h-step AR(p) at refit origin T0, p in P_GRID chosen on BIC.
runAR <- function(T0, h) {
  tr   <- (T0 - h - WINDOW + 1L):(T0 - h)  # training origins s, so that s + h <= T0
  y_tr <- y[tr + h] # target values for training, y_{s+h}
  X    <- LAGS[tr, , drop = FALSE] # lagged values for training

  # The calendar and the Covid regimes (announced in advance) are known at t for
  # the target hour t+h, so the dummies use the target hour's values.
  D <- DUM[tr + h, , drop = FALSE] # dummies for the target hour
  D <- D[, colSums(D) > 0, drop = FALSE] # drop all-zero columns

  n  <- length(y_tr) # number of training pairs, same for all candidate models
  bb <- Inf                                # initialise the "best BIC"
  for (i in P_GRID) {                      # try each lag length in the grid
    Z <- cbind(1, X[, 1:i, drop = FALSE], D)
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
# refit.
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
# One row per (forecast origin, h), with every benchmark side by side .
# EMA's forecast rides along as a reference level.
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
cat("\nwritten:", OUT, "- run view_benchmarks.R to see the errors\n")
