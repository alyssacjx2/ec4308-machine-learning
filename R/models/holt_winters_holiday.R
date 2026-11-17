# holt_winters_holiday.R -------------------------------------------------------
#
#   holt_winters.R plus a weekday public-holiday adjustment: four coefficients (MW),
#   one per part of the day, estimated with alpha and gamma on the same per-horizon
#   h-step loss.
#
#   in : data/processed/features.csv.gz   (datetime, y, is_holiday, dow)
#   out: output/forecasts/holt_winters_hol.csv.gz   column hw24h  (--period=168: holt_winters168_hol.csv.gz, hw168h)
#        output/tables/holt_winters{M}_hol_params_by_refit.csv
#
#   Rscript R/models/holt_winters_holiday.R   (or --period=168; --dry-run = 3 refits)

suppressMessages(library(data.table))

# ---- 0. settings -------------------------------------------------------------
source("R/utils.R")

IN        <- "data/processed/features.csv.gz"
H         <- c(1L, 6L, 12L, 24L)
WINDOW    <- as.integer(1.5 * 365 * 24) # 13,140 hours (1.5 years) of training data
REFIT     <- 168L                       # refit every week
OOS_START <- "2018-07-16 00:00:00"      # first forecast origin
args      <- commandArgs(trailingOnly = TRUE)
M         <- as.integer(sub("^--period=", "", grep("^--period=", args, value = TRUE)[1]))
if (is.na(M)) M <- 24L                  # seasonal period in hours (24 or 168)
INIT_LEN  <- max(2L * 168L, 4L * M)     # start of each window only sets the starting states
N_CORES   <- forecast_cores()

MAX_REFITS <- Inf
HOL <- TRUE                              # the weekday public-holiday adjustment is always on in this script
DRY <- "--dry-run" %in% args
if (DRY) MAX_REFITS <- 3L
SFX <- if (DRY) "_dryrun" else ""
TAG <- if (M == 24L) "" else as.character(M)   # M = 24 keeps the original file names
HTAG <- if (HOL) "_hol" else ""
OUT <- paste0("output/forecasts/holt_winters", TAG, HTAG, SFX, ".csv.gz")
COL <- paste0("hw", M, if (HOL) "h" else "")
TAB <- "output/tables"

# mclapply cannot fork on Windows: plain loop there.
par_map <- function(X, FUN) {
  if (N_CORES > 1L && .Platform$OS.type != "windows")
    parallel::mclapply(X, FUN, mc.cores = N_CORES, mc.preschedule = FALSE)
  else lapply(X, FUN)
}

# ---- load --------------------------------------------------------------------
f <- read_gz(IN, select = c("datetime", "y", "is_holiday", "dow"), colClasses = c(datetime = "character"))
setorder(f, datetime)
y <- f$y
N <- length(y)

T_first <- match(OOS_START, f$datetime)
if (is.na(T_first)) stop("OOS_START not found in features: ", OOS_START)
if (T_first <= WINDOW) stop("no full window before OOS_START")

# Holiday index: 0 on an ordinary hour, k = 1..4 on a weekday public holiday in part k of the
# day (00-05, 06-11, 12-17, 18-23). Known from the calendar; padded with zeros beyond the data.
HMAX  <- max(H)
part  <- as.integer(substr(f$datetime, 12, 13)) %/% 6L + 1L
BIDX  <- c(if (HOL) as.integer(f$is_holiday == 1L & f$dow %in% 1:5) * part else integer(N), integer(HMAX))
cat("holiday adjustment:", if (HOL) "weekday public holiday x 4 parts of the day" else "none", "\n")

# ---- 1. the filter -----------------------------------------------------------
# Runs t_start..t_end; returns the h-step forecast made at every t >= fc_from.
# bb = c(0, b_1..b_4) is the holiday effect in MW: the one-step error is taken net of it
# and the forecast adds it back at the target hour.
hw_filter_r <- function(y, alpha, delta, l0, d0, t_start, t_end, h, fc_from = t_start, bb = rep(0, 5)) {
  l <- l0; d <- d0
  F <- rep(NA_real_, t_end)
  for (t in t_start:t_end) {
    p <- (t - 1L) %% M + 1L
    e <- y[t] - (l + d[p] + bb[BIDX[t] + 1L])
    l <- l + alpha * e
    d[p] <- d[p] + delta * e
    if (t >= fc_from) F[t] <- l + d[(t + h - 1L) %% M + 1L] + bb[BIDX[t + h] + 1L]
  }
  F
}

# Same filter in C++ (much faster, same arithmetic); falls back to the R version without Rcpp.
hw_filter <- hw_filter_r
if (requireNamespace("Rcpp", quietly = TRUE)) {
  ok <- tryCatch({
    Rcpp::cppFunction('
      NumericVector hw_cpp(NumericVector y, double alpha, double delta, double l0, NumericVector d0,
                           int t_start, int t_end, int h, int fc_from, int M,
                           IntegerVector bidx, NumericVector bb) {
        double l = l0; NumericVector d = clone(d0);
        NumericVector F(t_end, NA_REAL);
        for (int t = t_start; t <= t_end; t++) {
          int p = (t - 1) % M;
          double e = y[t - 1] - (l + d[p] + bb[bidx[t - 1]]);
          l += alpha * e;
          d[p] += delta * e;
          if (t >= fc_from) F[t - 1] = l + d[(t + h - 1) % M] + bb[bidx[t + h - 1]];
        }
        return F;
      }')
    TRUE }, error = function(e) FALSE)
  if (ok) hw_filter <- function(y, alpha, delta, l0, d0, t_start, t_end, h, fc_from = t_start, bb = rep(0, 5))
    hw_cpp(y, alpha, delta, l0, d0, as.integer(t_start), as.integer(t_end), as.integer(h),
           as.integer(fc_from), as.integer(M), BIDX, bb)
}

# ---- 2. one window, one horizon ----------------------------------------------
# Starting states from the first INIT_LEN hours, then alpha, gamma and the holiday effects on the rest of the window.
fit_hw <- function(i0, iE, h, b_start = NULL) {
  ini <- i0:(i0 + INIT_LEN - 1L)
  t1  <- i0 + INIT_LEN
  sc  <- 100                                       # b is optimised in units of 100 MW
  if (HOL) {                                       # starting holiday effects: holiday hours vs ordinary weekday hours
    w <- t1:iE; wk <- f$dow[w] %in% 1:5 & f$is_holiday[w] == 0L
    b0 <- vapply(1:4, function(k) { hk <- BIDX[w] == k
                                    if (!any(hk)) 0 else mean(y[w][hk]) - mean(y[w][wk & part[w] == k]) }, 0)
    if (is.null(b_start)) b_start <- b0
  } else { b0 <- b_start <- numeric(0) }
  yi <- y[ini] - if (HOL) c(0, b0)[BIDX[ini] + 1L] else 0     # start values net of holiday effects
  l0  <- mean(yi); slot <- (ini - 1L) %% M + 1L
  d0  <- vapply(seq_len(M), function(k) mean(yi[slot == k] - l0), 0)
  nb  <- if (HOL) 4L else 0L
  obj <- function(p) {
    bb <- if (HOL) c(0, p[3:6] * sc) else rep(0, 5)
    F <- hw_filter(y, p[1], p[2] * (1 - p[1]), l0, d0, t1, iE, h, fc_from = t1, bb = bb)
    v <- sum((y[(t1 + h):iE] - F[t1:(iE - h)])^2)
    if (is.finite(v)) v else 1e15                # guard: L-BFGS-B needs finite values
  }
  o <- optim(c(0.2, 0.1, b_start / sc), obj, method = "L-BFGS-B",
             lower = c(0, 0, rep(-15, nb)), upper = c(1, 1, rep(15, nb)))
  list(alpha = o$par[1], gamma = o$par[2], delta = o$par[2] * (1 - o$par[1]), l0 = l0, d0 = d0, t1 = t1,
       b = if (HOL) o$par[3:6] * sc else numeric(0))
}

# ---- 3. one refit block ------------------------------------------------------
# Estimate on the window ending at T0, then run the frozen-parameter filter over the next REFIT origins.
refits <- seq(T_first, N - 1L, by = REFIT)
if (is.finite(MAX_REFITS)) refits <- head(refits, MAX_REFITS)

hw_block <- function(T0) {
  t_all <- T0:min(T0 + REFIT - 1L, N - 1L)
  out   <- matrix(NA_real_, length(t_all), length(H))
  par   <- matrix(NA_real_, length(H), 2L + if (HOL) 4L else 0L)
  i0    <- T0 - WINDOW + 1L
  bs    <- NULL                                    # holiday effects: each horizon starts from the previous solution
  for (j in seq_along(H)) {
    fi <- fit_hw(i0, T0, H[j], bs)
    if (HOL) bs <- fi$b
    Fm <- hw_filter(y, fi$alpha, fi$delta, fi$l0, fi$d0, fi$t1, max(t_all), H[j], fc_from = T0,
                    bb = if (HOL) c(0, fi$b) else rep(0, 5))
    out[, j] <- Fm[t_all]
    par[j, ] <- c(fi$alpha, fi$gamma, fi$b)
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
fwrite(rbindlist(lapply(res, function(r) {
         d <- data.table(refit_origin = f$datetime[r$T0], h = H, alpha = r$par[, 1], gamma = r$par[, 2])
         if (HOL) for (k in 1:4) d[[paste0("hol_part", k)]] <- r$par[, 2L + k]
         d })),
       file.path(TAB, paste0("holt_winters", TAG, HTAG, "_params_by_refit", SFX, ".csv")))

cat("\n=== out-of-sample error summary ===\n")
for (hh in H) {
  s <- out[h == hh & !is.na(out[[COL]])]
  if (!nrow(s)) next
  e <- s$actual - s[[COL]]
  cat(sprintf("  h=%2d  n=%s  RMSE %6.1f  MAE %6.1f  MAPE %.3f%%\n", hh, format(nrow(s), big.mark = ","),
      sqrt(mean(e^2)), mean(abs(e)), 100 * mean(abs(e) / s$actual)))
}
cat("\nwritten:", OUT, "\n")
