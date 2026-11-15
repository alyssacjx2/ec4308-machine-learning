# sarima_direct_holiday.R --------------------------------------------------------------
#
#   Seasonal ARIMA with a SEPARATE FIT FOR EACH HORIZON h = 1, 6, 12, 24, and a choice of
#   seasonal period (24 h = daily, 168 h = weekly).
#
#   Model: ARIMA(p,0,q)(P,1,Q)[S] on demand alone (no predictors). The seasonal difference
#   w_t = y_t - y_{t-S} is taken first; an ARMA model with the usual SARIMA structure
#   (non-seasonal lags times seasonal lags) is then put on w.
#
#   What is different from sarima.R:
#     * sarima.R estimates ONE set of coefficients per week by one-step-ahead conditional
#       sum of squares and iterates it 24 steps. Here the coefficients are re-estimated
#       every week SEPARATELY FOR EACH h, by minimising the squared h-step-ahead forecast
#       error over the training window. The forecast for horizon h still comes from running
#       the ARMA recursion forward h steps (future errors set to zero); only the criterion
#       the coefficients are chosen on changes. Same idea as holt_winters.R, which tunes
#       alpha and gamma per horizon.
#     * The seasonal period is an option (--period=24 or --period=168).
#     * The fit does not call Arima()/predict(): it is a small vectorised routine, because
#       the h-step criterion needs the h-step forecast at every origin of the window at
#       every optimiser step.
#
#   Orders (p, q, P, Q) are chosen ONCE by AICc (auto.arima, D = 1, d = 0) on the first
#   rolling window and cached in output/tables/sarima{S}_direct_orders.rds. Coefficients
#   are kept stationary / invertible through a PACF reparametrisation, started from the
#   auto.arima estimates. Between refits the coefficients are frozen but the state (past
#   errors) is updated every hour: the frozen model is run over the whole window plus the
#   block, so the forecast at origin t only uses data up to t.
#
#   in : data/processed/features.csv.gz   (only datetime and y are used)
#   out: output/forecasts/sarima{S}_direct.csv.gz     column sarima{S}d
#        output/tables/sarima{S}_direct_coef_by_refit.csv
#        output/tables/sarima{S}_direct_orders.rds
#
#   Rscript R/sarima_direct_holiday.R --period=168     full run (weekly season)
#   Rscript R/sarima_direct_holiday.R --period=168 --dry-run   3 refits only, writes *_dryrun files
#   Rscript R/sarima_direct_holiday.R --period=24      same model with a daily season
#   Rscript R/sarima_direct_holiday.R --period=168   (this script always includes the holiday regressors):
#       regression with ARMA errors on the seasonal differences, w_t = b' dx_t + u_t, where
#       x is 1 on a weekday public holiday (separately for four parts of the day) and
#       dx_t = x_t - x_{t-S}. b is estimated together with the ARMA coefficients on the same
#       per-horizon h-step loss. Writes sarima{S}_direct_hol.csv.gz, column sarima{S}dh.
#
# Output shape matches lasso.R / benchmarks.csv.gz exactly: one row per (forecast origin,
# h), `datetime` is the ORIGIN and `actual` is y_{t+h}.

suppressMessages({
  library(data.table)
  library(forecast)
})

# ---- 0. settings -------------------------------------------------------------
# THESE MUST MATCH lasso.R / benchmarks.R (H, WINDOW, REFIT, OOS_START).
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  a <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(a)) sub(paste0("^--", name, "="), "", a[1]) else default
}
IN        <- "data/processed/features.csv.gz"
H         <- c(1L, 6L, 12L, 24L)
WINDOW    <- as.integer(1.5 * 365 * 24) # 13,140 hours (1.5 years) of training data
REFIT     <- 168L                       # refit every week
OOS_START <- "2018-07-16 00:00:00"      # first forecast origin
S         <- as.integer(get_arg("period", 168))  # seasonal period (hours)
BURN      <- 2L * 168L                  # first 2 weeks of the window are not scored (error start-up)
N_CORES   <- as.integer(Sys.getenv("N_CORES", max(1L, parallel::detectCores() - 1L)))

HOL <- TRUE                              # weekday public-holiday regressors are always on in this script
MAX_REFITS <- Inf
DRY <- "--dry-run" %in% args
if (DRY) MAX_REFITS <- 3L
SFX <- if (DRY) "_dryrun" else ""
HTAG <- if (HOL) "_hol" else ""
OUT <- sprintf("output/forecasts/sarima%d_direct%s%s.csv.gz", S, HTAG, SFX)
TAB <- "output/tables"
COL <- sprintf("sarima%dd%s", S, if (HOL) "h" else "")
ORD_FILE <- file.path(TAB, sprintf("sarima%d_direct_orders.rds", S))
set.seed(4308)

par_map <- function(X, FUN) {
  if (N_CORES > 1L && .Platform$OS.type != "windows")
    parallel::mclapply(X, FUN, mc.cores = N_CORES, mc.preschedule = FALSE)
  else lapply(X, FUN)
}

# ---- load --------------------------------------------------------------------
f <- fread(IN, select = c("datetime", "y", "is_holiday", "dow"), colClasses = c(datetime = "character"))
setorder(f, datetime)
y <- f$y
N <- length(y)
T_first <- match(OOS_START, f$datetime)
if (is.na(T_first)) stop("OOS_START not found in features: ", OOS_START)
if (T_first <= WINDOW) stop("no full window before OOS_START")
HMAX <- max(H)
if (HMAX > S) stop("the forecast needs h <= S so that y_{t+h-S} is known at t")
if (!dir.exists(TAB)) dir.create(TAB, recursive = TRUE)

# Holiday regressors: 1 on a public holiday that falls on a weekday (dow 1..5), separately for
# four parts of the day (00-05, 06-11, 12-17, 18-23), so the holiday effect can differ by time of day.
# Known in advance from the calendar, so the value at the TARGET hour is usable at the origin.
# Padded with zeros beyond the data (those rows are never scored).
NB <- if (HOL) 4L else 0L
part <- as.integer(substr(f$datetime, 12, 13)) %/% 6L + 1L
holwd <- as.integer(f$is_holiday == 1L & f$dow %in% 1:5)
X <- matrix(0, N + HMAX, NB)
for (k in seq_len(NB)) X[seq_len(N), k] <- holwd * (part == k)
cat("holiday regressors:", if (HOL) "weekday public holiday x 4 parts of the day" else "none", "\n")

# ---- 1. orders: AICc on the first window (cached) -----------------------------
if (file.exists(ORD_FILE)) {
  sel <- readRDS(ORD_FILE)
} else {
  cat(sprintf("choosing orders by AICc on the first window, S = %d (slow for S = 168, ~8 min) ...\n", S))
  first_win <- (T_first - WINDOW + 1L):T_first
  a <- auto.arima(ts(y[first_win], frequency = S), d = 0, D = 1, max.p = 3, max.q = 3,
                  max.P = 2, max.Q = 2, max.order = 8, seasonal = TRUE, stepwise = TRUE,
                  approximation = TRUE, ic = "aicc")
  o <- arimaorder(a)                            # c(p, d, q, P, D, Q, m)
  cf <- coef(a)
  g <- function(stem, n) if (n) unname(cf[paste0(stem, seq_len(n))]) else numeric(0)
  sel <- list(p = o[[1]], q = o[[3]], P = o[[4]], Q = o[[6]],
              start = c(g("ar", o[[1]]), g("ma", o[[3]]), g("sar", o[[4]]), g("sma", o[[6]])),
              desc = sprintf("ARIMA(%d,0,%d)(%d,1,%d)[%d]  AICc = %.1f", o[[1]], o[[3]], o[[4]], o[[6]], S, a$aicc))
  saveRDS(sel, ORD_FILE)
}
p <- sel$p; q <- sel$q; P <- sel$P; Q <- sel$Q
cat(sel$desc, "\n")
ip <- seq_len(p); iq <- p + seq_len(q); iP <- p + q + seq_len(P); iQ <- p + q + P + seq_len(Q)

# ---- 2. ARMA machinery -------------------------------------------------------
# Unconstrained r  ->  stationary AR-type coefficients via partial autocorrelations
# (Durbin-Levinson); and the inverse. MA polynomial 1 + theta B is invertible iff the AR-type
# polynomial 1 - (-theta) B is stationary, so theta = -transform(r).
pacf2coef <- function(r) {
  n <- length(r); if (!n) return(numeric(0))
  pk <- tanh(r); ph <- pk[1]
  if (n > 1) for (k in 2:n) ph <- c(ph - pk[k] * rev(ph), pk[k])
  ph
}
coef2pacf <- function(ph) {
  n <- length(ph); if (!n) return(numeric(0))
  pk <- numeric(n)
  for (k in n:1) {
    pk[k] <- ph[k]
    if (k > 1) ph <- (ph[1:(k - 1)] + pk[k] * rev(ph[1:(k - 1)])) / (1 - pk[k]^2)
  }
  atanh(pmax(pmin(pk, 0.999), -0.999))
}
par_from_r <- function(r) c(pacf2coef(r[ip]), -pacf2coef(r[iq]), pacf2coef(r[iP]), -pacf2coef(r[iQ]))
r_from_par <- function(pr) c(coef2pacf(pr[ip]), coef2pacf(-pr[iq]), coef2pacf(pr[iP]), coef2pacf(-pr[iQ]))

polymul <- function(a, b) {
  r <- numeric(length(a) + length(b) - 1L)
  for (i in seq_along(a)) r[i:(i + length(b) - 1L)] <- r[i:(i + length(b) - 1L)] + a[i] * b
  r
}
# par = (phi_1..p, theta_1..q, Phi_1..P, Theta_1..Q)  ->  expanded lag polynomials
#   w_t = sum_j A[j] w_{t-j} + e_t + sum_j M[j] e_{t-j}
expand_poly <- function(par) {
  a2 <- numeric(1 + P * S); a2[1] <- 1; if (P) a2[1 + S * seq_len(P)] <- -par[iP]
  m2 <- numeric(1 + Q * S); m2[1] <- 1; if (Q) m2[1 + S * seq_len(Q)] <- par[iQ]
  list(A = -polymul(c(1, -par[ip]), a2)[-1], M = polymul(c(1, par[iq]), m2)[-1])
}

# ws = seasonally differenced series. One-step errors e_t (started at zero), then the
# k-step ARMA forecasts what_{t+k|t} for k = 1..hk at every origin in tt (future errors = 0).
arma_paths <- function(par, ws, tt, hk) {
  ex <- expand_poly(par); A <- ex$A; M <- ex$M
  ia <- which(A != 0); im <- which(M != 0); LA <- length(A); n <- length(ws)
  u <- ws
  for (j in ia) u[(j + 1L):n] <- u[(j + 1L):n] - A[j] * ws[1:(n - j)]
  u[seq_len(LA)] <- 0
  e <- if (length(im)) as.numeric(stats::filter(u, -M, method = "recursive")) else u
  W <- matrix(0, length(tt), hk)
  for (k in seq_len(hk)) {
    acc <- numeric(length(tt))
    for (j in ia) acc <- acc + A[j] * (if (j < k) W[, k - j] else ws[tt + k - j])
    for (j in im) if (j >= k) acc <- acc + M[j] * e[tt + k - j]
    W[, k] <- acc
  }
  W
}
seas_diff <- function(v) v[(S + 1L):length(v)] - v[1:(length(v) - S)]
seas_diff_m <- function(M) M[(S + 1L):nrow(M), , drop = FALSE] - M[1:(nrow(M) - S), , drop = FALSE]

# ---- 3. fit one horizon on one window -----------------------------------------
# Window = y[i0..T0]. Regression with ARMA errors on the seasonal differences:
#   w_t = b' dx_t + u_t,  u_t ~ ARMA,  dx_t = x_t - x_{t-S}   (no regressors when --holiday is off)
# Minimises the mean squared h-step error of w over the window's origins, over the ARMA
# coefficients and b together. The forecast is the ARMA forecast of u plus b' dx at the target hour.
fit_h <- function(ws, DX, h, z0) {
  n <- length(ws); LA <- length(expand_poly(rep(0.1, p + q + P + Q))$A)
  tt <- (LA + BURN + 1L):(n - h)
  np <- p + q + P + Q
  obj <- function(z) {
    b <- z[np + seq_len(NB)] * 100                       # b is optimised in units of 100 MW
    u <- ws - as.numeric(DX %*% b)
    W <- arma_paths(par_from_r(z[seq_len(np)]), u, tt, h)
    v <- mean((ws[tt + h] - W[, h] - as.numeric(DX[tt + h, , drop = FALSE] %*% b))^2)
    if (is.finite(v)) v else 1e15
  }
  o <- optim(z0, obj, method = "BFGS", control = list(maxit = 100, reltol = 1e-7))
  list(par = par_from_r(o$par[seq_len(np)]), b = o$par[np + seq_len(NB)] * 100, z = o$par)
}

# ---- 4. one refit block -------------------------------------------------------
refits <- seq(T_first, N - 1L, by = REFIT)
if (is.finite(MAX_REFITS)) refits <- head(refits, MAX_REFITS)
R0 <- r_from_par(sel$start)

sarima_block <- function(T0) {
  t_all <- T0:min(T0 + REFIT - 1L, N - 1L)
  i0 <- T0 - WINDOW + 1L
  ws_win <- seas_diff(y[i0:T0])
  DX_win <- seas_diff_m(X[i0:T0, , drop = FALSE])
  seg <- i0:max(t_all)
  ws_seg <- seas_diff(y[seg])
  DX_ext <- seas_diff_m(X[i0:(max(t_all) + HMAX), , drop = FALSE])   # regressors known HMAX beyond the last origin
  tt <- (t_all - i0 + 1L) - S                  # origins in ws_seg coordinates
  pred <- matrix(NA_real_, length(t_all), length(H)); cf <- vector("list", length(H))
  b0 <- if (NB) tryCatch(as.numeric(solve(crossprod(DX_win) + diag(1e-3, NB), crossprod(DX_win, ws_win))),
                         error = function(e) rep(0, NB)) else numeric(0)
  z <- c(R0, b0 / 100)                         # start; each horizon starts from the previous one's solution
  for (j in seq_along(H)) {
    fi <- tryCatch(fit_h(ws_win, DX_win, H[j], z), error = function(e) NULL)
    if (is.null(fi)) next
    z <- fi$z
    u <- ws_seg - as.numeric(DX_ext[seq_along(ws_seg), , drop = FALSE] %*% fi$b)
    W <- arma_paths(fi$par, u, tt, H[j])
    pred[, j] <- y[t_all + H[j] - S] + W[, H[j]] + as.numeric(DX_ext[tt + H[j], , drop = FALSE] %*% fi$b)
    cf[[j]] <- c(fi$par, fi$b)
  }
  list(T0 = T0, t = t_all, pred = pred, coef = cf)
}

# ---- 5. run -------------------------------------------------------------------
cat("\nrolling window:", format(WINDOW, big.mark = ","), "hours, refit every", REFIT,
    "h, from", OOS_START, " period S =", S, "\n")
cat("refits:", length(refits), "  cores:", N_CORES, "\n")
t0  <- Sys.time()
res <- par_map(refits, sarima_block)
cat(sprintf("done in %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
bad <- vapply(res, function(r) !is.list(r) || !is.matrix(r$pred), TRUE)
if (any(bad)) warning(sum(bad), " refit(s) failed (forecasts NA)")
res <- res[!bad]

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
fwrite(out, OUT)
nm <- c(if (p) paste0("ar", ip), if (q) paste0("ma", seq_len(q)),
        if (P) paste0("sar", seq_len(P)), if (Q) paste0("sma", seq_len(Q)),
        if (NB) paste0("hol_part", seq_len(NB)))
fwrite(rbindlist(lapply(res, function(r) rbindlist(lapply(seq_along(H), function(j)
         if (is.null(r$coef[[j]])) NULL else
           as.data.table(c(list(refit_origin = f$datetime[r$T0], h = H[j]),
                           setNames(as.list(r$coef[[j]]), nm))))))),
       file.path(TAB, sprintf("sarima%d_direct%s_coef_by_refit%s.csv", S, HTAG, SFX)))

cat("\n=== out-of-sample error summary ===\n")
for (hh in H) {
  s <- out[h == hh & !is.na(out[[COL]])]
  if (!nrow(s)) next
  e <- s$actual - s[[COL]]
  cat(sprintf("  h=%2d  n=%s  RMSE %6.1f  MAE %6.1f  MAPE %.3f%%\n", hh, format(nrow(s), big.mark = ","),
      sqrt(mean(e^2)), mean(abs(e)), 100 * mean(abs(e) / s$actual)))
}
cat("\nwritten:", OUT, "\n")
