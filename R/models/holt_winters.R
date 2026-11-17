# holt_winters.R ---------------------------------------------------------------
#
#   Additive Holt-Winters benchmark (HW24) on demand alone (no predictors): a level plus
#   one daily seasonal cycle of 24 hourly adjustments, no trend (Taylor 2003 error-
#   correction form). Each hour the one-step error e_t updates the level (alpha) and the
#   adjustment of that hour of the day (delta = gamma * (1 - alpha)):
#       e_t = y_t - (l_{t-1} + d_{t-24});  l_t = l_{t-1} + alpha e_t;  d_t = d_{t-24} + delta e_t
#       forecast(t + h) = l_t + d_{slot(t+h)}
#   It has only the daily cycle, so it cannot use the weekly pattern.
#
#   Tuning: alpha and gamma in [0, 1] are chosen at every weekly refit and SEPARATELY FOR
#   EACH HORIZON h (direct forecasting) by minimising the sum of squared h-step-ahead
#   errors over the training window (L-BFGS-B); the first two weeks of the window only
#   set the starting states. Between refits the parameters are frozen but the states
#   update every hour. A one-step criterion would give near-naive day-ahead forecasts,
#   hence one fit per horizon.
#
#   in : data/processed/features.csv.gz   (only datetime and y are used)
#   out: output/forecasts/holt_winters.csv.gz
#        output/tables/holt_winters_params_by_refit.csv   alpha, gamma per refit and horizon
#
#   Rscript R/models/holt_winters.R            full run (about 1.5 hours on 2 cores), daily season M = 24
#   Rscript R/models/holt_winters.R --period=168   weekly season: 168 hourly adjustments (one per
#                                       hour of the week); writes holt_winters168.csv.gz, column hw168
#   Rscript R/models/holt_winters.R --dry-run  3 refits only, writes *_dryrun files (to check the code)
#
# Output shape matches lasso.R / benchmarks.csv.gz exactly: one row per (forecast origin,
# h), `datetime` is the ORIGIN and `actual` is y_{t+h}, so evaluate.R can merge the model
# files on (datetime, h) without any re-alignment.

suppressMessages(library(data.table))

# ---- 0. settings -------------------------------------------------------------
source("R/utils.R")

IN        <- "data/processed/features.csv.gz"
H         <- c(1L, 6L, 12L, 24L)
WINDOW    <- as.integer(1.5 * 365 * 24) # 13,140 hours (1.5 years) of training data
REFIT     <- 168L                       # refit every week
OOS_START <- "2018-07-16 00:00:00"      # first forecast origin: the first Monday
                                        # with a full window behind it
args      <- commandArgs(trailingOnly = TRUE)
M         <- as.integer(sub("^--period=", "", grep("^--period=", args, value = TRUE)[1]))
if (is.na(M)) M <- 24L                  # seasonal period (hours): --period=24 (daily, default) or --period=168 (weekly)
INIT_LEN  <- max(2L * 168L, 4L * M)     # first weeks of each window set the starting states (2 weeks for M = 24, 4 for M = 168)
N_CORES   <- forecast_cores()

# Smoke test: number of refits to run. Inf = full run. `--dry-run` sets it to 3.
MAX_REFITS <- Inf
DRY <- "--dry-run" %in% args
if (DRY) MAX_REFITS <- 3L
SFX <- if (DRY) "_dryrun" else ""
TAG <- if (M == 24L) "" else as.character(M)   # M = 24 keeps the original file names
OUT <- paste0("output/forecasts/holt_winters", TAG, SFX, ".csv.gz")
COL <- paste0("hw", M)
TAB <- "output/tables"

# mclapply forks, which Windows cannot do, so fall back to a plain loop there.
par_map <- function(X, FUN) {
  if (N_CORES > 1L && .Platform$OS.type != "windows")
    parallel::mclapply(X, FUN, mc.cores = N_CORES, mc.preschedule = FALSE)
  else lapply(X, FUN)
}

# ---- load --------------------------------------------------------------------
f <- read_gz(IN, select = c("datetime", "y"), colClasses = c(datetime = "character"))
setorder(f, datetime)
y <- f$y
N <- length(y)

T_first <- match(OOS_START, f$datetime)
if (is.na(T_first)) stop("OOS_START not found in features: ", OOS_START)
if (T_first <= WINDOW) stop("no full window before OOS_START")

# ---- 1. the filter -----------------------------------------------------------
# Runs t_start..t_end; returns, for every t >= fc_from, the h-step forecast made at t
# (state after seeing y_t). The seasonal slot is the absolute hour index, so the daily
# position of the target hour is always right.
hw_filter <- function(y, alpha, delta, l0, d0, t_start, t_end, h, fc_from = t_start) {
  l <- l0; d <- d0
  F <- rep(NA_real_, t_end)
  for (t in t_start:t_end) {
    p <- (t - 1L) %% M + 1L
    e <- y[t] - (l + d[p])
    l <- l + alpha * e
    d[p] <- d[p] + delta * e
    if (t >= fc_from) F[t] <- l + d[(t + h - 1L) %% M + 1L]
  }
  F
}

# ---- 2. one window, one horizon ----------------------------------------------
# Window i0..iE (iE = refit origin). Starting states from the first INIT_LEN hours, then
# alpha, gamma by minimising the squared h-step error over the rest of the window.
fit_hw <- function(i0, iE, h) {
  ini <- i0:(i0 + INIT_LEN - 1L)
  l0  <- mean(y[ini]); slot <- (ini - 1L) %% M + 1L
  d0  <- vapply(seq_len(M), function(k) mean(y[ini][slot == k] - l0), 0)
  t1  <- i0 + INIT_LEN
  obj <- function(p) {
    F <- hw_filter(y, p[1], p[2] * (1 - p[1]), l0, d0, t1, iE, h, fc_from = t1)
    v <- sum((y[(t1 + h):iE] - F[t1:(iE - h)])^2)
    if (is.finite(v)) v else 1e15                # guard: L-BFGS-B needs finite values
  }
  o <- optim(c(0.2, 0.1), obj, method = "L-BFGS-B", lower = c(0, 0), upper = c(1, 1))
  list(alpha = o$par[1], gamma = o$par[2], delta = o$par[2] * (1 - o$par[1]), l0 = l0, d0 = d0, t1 = t1)
}

# ---- 3. one refit block ------------------------------------------------------
# Estimate on the WINDOW ending at the refit origin T0 (all known at T0), then run the
# frozen-parameter filter forward over the block's origins t = T0 .. T0 + REFIT - 1.
refits <- seq(T_first, N - 1L, by = REFIT)
if (is.finite(MAX_REFITS)) refits <- head(refits, MAX_REFITS)

hw_block <- function(T0) {
  t_all <- T0:min(T0 + REFIT - 1L, N - 1L)
  out   <- matrix(NA_real_, length(t_all), length(H))
  par   <- matrix(NA_real_, length(H), 2L)
  i0    <- T0 - WINDOW + 1L
  for (j in seq_along(H)) {
    fi <- fit_hw(i0, T0, H[j])
    Fm <- hw_filter(y, fi$alpha, fi$delta, fi$l0, fi$d0, fi$t1, max(t_all), H[j], fc_from = T0)
    out[, j] <- Fm[t_all]
    par[j, ] <- c(fi$alpha, fi$gamma)
  }
  list(T0 = T0, t = t_all, pred = out, par = par)
}

# ---- 4. run all refits -------------------------------------------------------
cat("\nrolling window:", format(WINDOW, big.mark = ","), "hours, refit every", REFIT,
    "h, from", OOS_START, "\n")
cat("refits:", length(refits), "  cores:", N_CORES, "\n")
t0  <- Sys.time()
res <- par_map(refits, hw_block)
cat(sprintf("done in %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))

bad <- vapply(res, function(r) !is.list(r) || !is.matrix(r$pred), TRUE)
if (any(bad)) warning(sum(bad), " refit(s) failed (forecasts NA)")
res <- res[!bad]

# ---- 5. assemble: one row per (origin, h) ------------------------------------
out <- rbindlist(lapply(seq_along(H), function(j) {
  t    <- unlist(lapply(res, `[[`, "t"))
  pred <- unlist(lapply(res, function(r) r$pred[, j]))
  keep <- t <= N - H[j]                          # last h origins have no y_{t+h}
  d <- data.table(datetime = f$datetime[t[keep]], h = H[j], actual = y[t[keep] + H[j]])
  d[[COL]] <- pred[keep]
  d
}))
if (anyNA(out[[COL]])) warning(sum(is.na(out[[COL]])), " NA forecasts left in the output")

if (!dir.exists(dirname(OUT))) dir.create(dirname(OUT), recursive = TRUE)
if (!dir.exists(TAB)) dir.create(TAB, recursive = TRUE)
fwrite(out, OUT)
fwrite(rbindlist(lapply(res, function(r) data.table(refit_origin = f$datetime[r$T0], h = H,
                                                    alpha = r$par[, 1], gamma = r$par[, 2]))),
       file.path(TAB, paste0("holt_winters", TAG, "_params_by_refit", SFX, ".csv")))

cat("\n=== out-of-sample error summary ===\n")
for (hh in H) {
  s <- out[h == hh & !is.na(out[[COL]])]
  if (!nrow(s)) next
  e <- s$actual - s[[COL]]
  cat(sprintf("  h=%2d  n=%s  RMSE %6.1f  MAE %6.1f  MAPE %.3f%%\n", hh, format(nrow(s), big.mark = ","),
      sqrt(mean(e^2)), mean(abs(e)), 100 * mean(abs(e) / s$actual)))
}
cat("\nwritten:", OUT, "\n")
