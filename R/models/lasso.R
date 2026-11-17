# lasso.R ----------------------------------------------------------------------
#
#   1. LASSO via rlasso()  - data-driven penalty (Belloni et al.)
#   2. post-LASSO          - OLS on the variables LASSO selected
#
#   Both come from one rlasso fit per window. The 168 hour-of-week cells are kept in
#   the model but not penalised (PARTIAL_OUT); FALSE penalises them like every other predictor.
#
#   in : data/processed/features.csv.gz
#   out: output/forecasts/lasso.csv.gz   (lasso_penalise_all.csv.gz if PARTIAL_OUT = FALSE)

suppressMessages({
  library(data.table)
  library(hdm)
})

# ---- 0. settings -------------------------------------------------------------
# H, WINDOW, REFIT and OOS_START must match benchmarks.R.
source("R/utils.R")

IN        <- "data/processed/features.csv.gz"
H         <- c(1L, 6L, 12L, 24L)
WINDOW    <- as.integer(1.5 * 365 * 24) # 13,140 hours (1.5 years) of training pairs
REFIT     <- 168L                       # refit every week
OOS_START <- "2018-07-16 00:00:00"      # first forecast origin
DUMS      <- c("cb_strict", "cb_p1", "cb_p2", "cb_p3", "ha_2021")

# TRUE: hour-of-week cells enter unpenalised (partialled out). FALSE: penalised like the rest.
PARTIAL_OUT <- TRUE
OUT <- if (PARTIAL_OUT) "output/forecasts/lasso.csv.gz" else
                        "output/forecasts/lasso_penalise_all.csv.gz"

# Refits per horizon (Inf = full run; set a small number to test).
MAX_REFITS <- Inf

# ---- load --------------------------------------------------------------------
f <- read_gz(IN, colClasses = c(datetime = "character"))
setorder(f, datetime)
y <- f$y
N <- length(y)

T_first <- match(OOS_START, f$datetime)
if (is.na(T_first)) stop("OOS_START not found in features: ", OOS_START)

# ---- 1. design matrix --------------------------------------------------------
#
# Predictors are split by when they are known. ORIGIN-TIME ones (demand history,
# weather, ERA5 lags) are taken at row t. TARGET-TIME ones (calendar terms, holiday,
# Covid regime) are known in advance and taken at row t+h. Weather and ERA5 are
# origin-time: era5_*_lag120 at row t is what the 5-day publication delay makes
# available at t.

target_time <- c("hr", "dow", "mth", "doy", "year", "t_index",
                 "is_weekend", "is_holiday", "is_workday", DUMS)
origin_time <- setdiff(
  names(f),
  c("datetime", "y", "usable", target_time,
    grep("^(y_h|nem_h)", names(f), value = TRUE)))

# y_t is the target column, so add it explicitly.
X_org <- as.matrix(f[, ..origin_time])
X_org <- cbind(y_t = y, X_org)

# Hour-of-week cell (0..167) from the day type, so a public holiday takes the Sunday
# profile. The cells are mutually exclusive dummies, so partialling them out below is
# exact and only needs the cell index.
dtype   <- fifelse(f$is_holiday == 1L, 0L, f$dow)   # 0 = Sunday
hd      <- f$hr * 7L + dtype

# hr and dow stay out of the penalised block: the cells span hr, and is_weekend,
# is_workday and is_holiday cover dow.
tt_keep <- setdiff(target_time, c("hr", "dow"))
X_pen   <- cbind(X_org, as.matrix(f[, ..tt_keep]))

if (!PARTIAL_OUT) {
  # Penalised variant: cells as dummies, one dropped as the reference (rlasso adds the intercept).
  HD <- matrix(0, N, 168L)
  HD[cbind(seq_len(N), hd + 1L)] <- 1
  HD <- HD[, -1L, drop = FALSE]
  colnames(HD) <- paste0("hd_", seq_len(167L))
  X_pen <- cbind(X_pen, HD)
  rm(HD)
}
invisible(gc(verbose = FALSE))

cat("design:", ncol(X_pen), "penalised predictors",
    if (PARTIAL_OUT) "+ 168 unpenalised hour-of-week cells" else "(cells penalised)", "\n")

# ---- 2. partialling out ------------------------------------------------------
#
# Frisch-Waugh-Lovell: residualise y and X on the unpenalised cells, LASSO the
# residuals, then recover the cell effects. As the cells are mutually exclusive
# dummies, residualising is subtracting the cell mean (no 168 x 168 solve, and the
# dummy matrix is never built). is_weekend survives residualisation, which lets
# LASSO tell a Sunday from a weekday holiday in the same cell.

# Cell means of M by group g, one row per level.
cell_means <- function(M, g, levels) {
  s <- rowsum(M, group = g, reorder = FALSE)
  n <- as.vector(table(factor(g, levels = levels)))
  s <- s[match(levels, as.integer(rownames(s))), , drop = FALSE]
  s[is.na(s)] <- 0
  s / pmax(n, 1L)
}

# ---- 3. one window -----------------------------------------------------------
# Fit at refit origin T0. Training origins end at T0-h, so every training target is
# observable at T0 (same construction as runAR() in benchmarks.R).
run_lasso <- function(T0, h) {
  tr   <- (T0 - h - WINDOW + 1L):(T0 - h)
  y_tr <- y[tr + h]
  X_tr <- X_pen[tr, , drop = FALSE]
  X_tr[, colnames(X_pen) %in% tt_keep] <-
    X_pen[tr + h, colnames(X_pen) %in% tt_keep, drop = FALSE]

  g <- lv <- M <- NULL
  if (PARTIAL_OUT) {
    g  <- hd[tr + h]                      # cell of the TARGET hour
    lv <- sort(unique(g))
    M  <- cell_means(cbind(y_tr, X_tr), g, lv)
    pos  <- match(g, lv)
    y_tr <- y_tr - M[pos, 1L]
    X_tr <- X_tr - M[pos, -1L, drop = FALSE]
  }

  # Drop columns constant within this window.
  keep <- which(apply(X_tr, 2L, function(z) !anyNA(z) && var(z) > 0))
  X_tr <- X_tr[, keep, drop = FALSE]

  # Residualised data have mean zero, so no intercept; the cells carry it.
  fit <- rlasso(X_tr, y_tr, post = FALSE, intercept = !PARTIAL_OUT)

  cf <- coef(fit)
  if (length(cf) == ncol(X_tr) + 1L) { b0 <- unname(cf[1L]); b <- cf[-1L] }
  else                               { b0 <- 0;              b <- cf }
  sel <- which(abs(b) > 0)

  # post-LASSO: OLS on the selected set; qr() pivoting guards against collinear selections.
  if (length(sel) > 0L) {
    Z <- if (PARTIAL_OUT) X_tr[, sel, drop = FALSE] else cbind(1, X_tr[, sel, drop = FALSE])
    q   <- qr(Z)
    piv <- q$pivot[seq_len(q$rank)]
    pb  <- rep(0, ncol(Z))
    pb[piv] <- qr.coef(qr(Z[, piv, drop = FALSE]), y_tr)
    pb[is.na(pb)] <- 0
  } else {
    pb <- if (PARTIAL_OUT) numeric(0) else mean(y_tr)
  }

  list(keep = keep, b0 = b0, b = b, sel = sel, pb = pb, n_sel = length(sel),
       lv = lv, M = M)
}

# Unpenalised cell effects for a coefficient vector: cell mean of y minus cell means of X times b.
cell_effects <- function(fit, beta_full) {
  as.vector(fit$M[, 1L] - fit$M[, -1L, drop = FALSE] %*% beta_full)
}

# ---- 4. rolling window -------------------------------------------------------
lasso.rolling.window <- function(h) {
  origins <- T_first:(N - h)                   # last h origins have no y_{t+h}
  refits  <- seq(T_first, N - h, by = REFIT)
  if (is.finite(MAX_REFITS)) refits <- head(refits, MAX_REFITS)

  pred_l <- rep(NA_real_, length(origins))
  pred_p <- rep(NA_real_, length(origins))
  n_sel  <- rep(NA_integer_, length(origins))

  t0 <- Sys.time()
  for (k in seq_along(refits)) {
    T0  <- refits[k]
    fit <- run_lasso(T0, h)
    t   <- T0:min(T0 + REFIT - 1L, N - h)      # origins served by this fit
    idx <- t - T_first + 1L

    Xn <- X_pen[t, , drop = FALSE]
    Xn[, colnames(X_pen) %in% tt_keep] <-
      X_pen[t + h, colnames(X_pen) %in% tt_keep, drop = FALSE]
    Xk <- Xn[, fit$keep, drop = FALSE]

    xb  <- if (length(fit$sel)) drop(Xk[, fit$sel, drop = FALSE] %*% fit$b[fit$sel]) else rep(0, length(t))
    xbp <- if (length(fit$sel)) drop(Xk[, fit$sel, drop = FALSE] %*%
                                     tail(fit$pb, length(fit$sel))) else rep(0, length(t))

    if (PARTIAL_OUT) {
      # Add back the unpenalised cell effect for the TARGET hour's cell.
      bf <- numeric(ncol(fit$M) - 1L); bf[fit$keep[fit$sel]] <- fit$b[fit$sel]
      a  <- cell_effects(fit, bf)
      bfp <- numeric(ncol(fit$M) - 1L); bfp[fit$keep[fit$sel]] <- tail(fit$pb, length(fit$sel))
      ap  <- cell_effects(fit, bfp)
      gp <- match(hd[t + h], fit$lv)
      # cell unseen in this window -> mean of the cell effects
      pred_l[idx] <- fifelse(is.na(gp), mean(a, na.rm = TRUE), a[gp])  + xb
      pred_p[idx] <- fifelse(is.na(gp), mean(ap, na.rm = TRUE), ap[gp]) + xbp
    } else {
      pred_l[idx] <- fit$b0 + xb
      pred_p[idx] <- fit$pb[1L] + xbp
    }
    n_sel[idx] <- fit$n_sel

    if (k %% 25 == 0 || k == length(refits)) {
      el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
      cat(sprintf("  h=%2d  refit %3d / %3d  (%d selected, %.1f min elapsed, ~%.0f min left)\n",
                  h, k, length(refits), fit$n_sel, el,
                  el / k * (length(refits) - k)))
    }
  }
  list(origins = origins, lasso = pred_l, post = pred_p, n_sel = n_sel)
}

# ---- 5. run all horizons -----------------------------------------------------
cat("\nrolling window:", format(WINDOW, big.mark = ","), "hours, refit every",
    REFIT, "h, from", OOS_START, "\n")
cat("hour-of-week cells:", if (PARTIAL_OUT) "UNPENALISED (partialled out)" else "penalised", "\n")
cat("refits per horizon:", length(seq(T_first, N - 1L, by = REFIT)), "\n")
if (is.finite(MAX_REFITS)) cat("*** SMOKE TEST: capped at", MAX_REFITS, "refits ***\n")

res <- list()
for (h in H) {
  cat("\nh =", h, "\n")
  r <- lasso.rolling.window(h)
  t <- r$origins
  res[[as.character(h)]] <- data.table(
    datetime   = f$datetime[t],                # forecast origin
    h          = h,
    actual     = y[t + h],                     # y_{t+h}
    lasso      = r$lasso,
    post_lasso = r$post,
    n_sel      = r$n_sel
  )
}
out <- rbindlist(res)
out <- out[!is.na(lasso)]                      # drop origins past MAX_REFITS

if (!dir.exists(dirname(OUT))) dir.create(dirname(OUT), recursive = TRUE)
fwrite(out, OUT)

cat("\n=== out-of-sample error summary ===\n")
# `hh`, not `h`: inside out[...] a bare h is the column.
for (hh in H) {
  s <- out[h == hh & !is.na(actual)]
  if (!nrow(s)) next
  cat(sprintf("  h=%2d  n=%s  LASSO RMSE %6.1f  MAE %6.1f  MAPE %.3f%%  |  post RMSE %6.1f  |  median vars %d\n",
      hh, format(nrow(s), big.mark = ","),
      sqrt(mean((s$actual - s$lasso)^2)), mean(abs(s$actual - s$lasso)),
      100 * mean(abs(s$actual - s$lasso) / s$actual),
      sqrt(mean((s$actual - s$post_lasso)^2)),
      as.integer(median(s$n_sel))))
}
cat("\nwritten:", OUT, "\n")
