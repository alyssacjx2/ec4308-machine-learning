# lasso.R ----------------------------------------------------------------------
#
#   1. LASSO via rlasso()  - tuning-free, data-driven penalty (Belloni et al.)
#   2. post-LASSO          - OLS on the variables LASSO selected
#
#   in : data/processed/features.csv.gz
#   out: output/forecasts/lasso.csv.gz
#
# Output shape matches benchmarks.csv.gz exactly: one row per (forecast origin,
# h), `datetime` is the ORIGIN and `actual` is y_{t+h}. evaluate.R can then
# merge the model files on (datetime, h) without any re-alignment.
#
# Both variants come out of ONE rlasso fit per window. post-LASSO is by
# definition OLS on the LASSO-selected set, so refitting rlasso with post=TRUE
# would double the runtime and risk a different selection.

suppressMessages({
  library(data.table)
  library(hdm)
})

# ---- 0. settings -------------------------------------------------------------
# THESE MUST MATCH benchmarks.R. If any of H, WINDOW, REFIT or OOS_START differs,
# the forecasts are for different hours and cannot be compared or DM-tested.
IN        <- "data/processed/features.csv.gz"
H         <- c(1L, 6L, 12L, 24L)
WINDOW    <- as.integer(1.5 * 365 * 24) # 13,140 hours (1.5 years) of training pairs
REFIT     <- 168L                       # refit every week
OOS_START <- "2018-07-16 00:00:00"      # first forecast origin: the first Monday
                                        # with a full window behind it
DUMS      <- c("cb_strict", "cb_p1", "cb_p2", "cb_p3", "ha_2021")

# Keep the 167 hour-of-week cells in the model but EXEMPT from the penalty
# (Denis, consultation): we are confident they belong, so there is nothing for
# LASSO to decide about them, and including them inflates p - which raises the
# penalty for every predictor we DO want judged. See section 1 below.
#
# FALSE reproduces the original design, where the dummies are penalised like
# everything else. Denis suggested running both if time allows; the two write to
# different files so they can be compared.
PARTIAL_OUT <- TRUE
OUT <- if (PARTIAL_OUT) "output/forecasts/lasso.csv.gz" else
                        "output/forecasts/lasso_penalise_all.csv.gz"

# Smoke test: set to a small number (e.g. 5) to run a few refits per horizon and
# confirm the script works before committing to the full run. Inf = full run.
MAX_REFITS <- Inf

# ---- load --------------------------------------------------------------------
f <- fread(IN, colClasses = c(datetime = "character"))
setorder(f, datetime)
y <- f$y
N <- length(y)

T_first <- match(OOS_START, f$datetime)
if (is.na(T_first)) stop("OOS_START not found in features: ", OOS_START)

# ---- 1. design matrix --------------------------------------------------------
#
# The split below is the whole no-leakage argument, so it is worth stating.
# Standing at origin t and forecasting y_{t+h}, two kinds of predictor exist:
#
#   ORIGIN-TIME (stochastic) - we know these only up to t. Demand history,
#   weather, ERA5. Taken at row t.
#
#   TARGET-TIME (deterministic) - these are known for any future date the moment
#   we look at a calendar. What hour and weekday t+h falls on, whether it is a
#   gazetted holiday, which Covid regime was in force. Taken at row t+h.
#
# Using target-time calendar terms is not leakage and it matters: demand at t+h
# depends far more on what hour t+h IS than on what hour t was. benchmarks.R
# applies the same reasoning to the Covid dummies (DUM[tr + h, ]).
#
# Weather is deliberately origin-time. A genuine operational forecast would use
# a weather FORECAST for t+h; we have no such series, so we use the last observed
# reading instead. That is honest but it does handicap the longer horizons -
# worth a sentence in the report.
#
# ERA5 must also be origin-time. era5_*_lag120 at row t holds the value from
# t-120h, which is exactly what the 5-day publication delay makes available at t.
# Reading row t+h would hand the model ERA5 from t+h-120h: for h=24 that is
# t-96h, still unpublished at t. That would be leakage.

target_time <- c("hr", "dow", "mth", "doy", "year", "t_index",
                 "is_weekend", "is_holiday", "is_workday", DUMS)
origin_time <- setdiff(
  names(f),
  c("datetime", "y", "usable", target_time,
    grep("^(y_h|nem_h)", names(f), value = TRUE)))

# y_t itself is known at t and is the single strongest predictor, but it is the
# target column rather than a feature column, so add it explicitly.
X_org <- as.matrix(f[, ..origin_time])
X_org <- cbind(y_t = y, X_org)

# The hour-of-week cell, 0..167. Integer hr and dow are dropped from the
# penalised block: the 168 cells already span every hour effect, every weekday
# effect and their interaction, so keeping them too would be exactly collinear.
hd      <- f$hr * 7L + f$dow
tt_keep <- setdiff(target_time, c("hr", "dow"))
X_pen   <- cbind(X_org, as.matrix(f[, ..tt_keep]))

if (!PARTIAL_OUT) {
  # Penalised variant: materialise the cells as dummies and append them, with
  # one dropped as the reference category (rlasso supplies the intercept).
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
# Frisch-Waugh-Lovell. To fit  y = D g + X b + e  with D unpenalised and X
# penalised, residualise both y and X on D, LASSO the residuals to get b, then
# recover g from the residuals of y - X b.
#
# The useful shortcut: D is a COMPLETE SET OF MUTUALLY EXCLUSIVE DUMMIES (every
# hour falls in exactly one of the 168 cells), so D'D is diagonal and
# residualising on D is just subtracting the cell mean. No 168x168 solve, no QR,
# and the 85,440 x 167 dummy matrix never has to exist - we only need the cell
# index. That is why PARTIAL_OUT = TRUE is the cheaper branch, not the dearer one.
#
# Note is_weekend is a sum of hour-of-week cells, so residualising sends it to
# exactly zero; the constant-column filter in run_lasso() then drops it. That is
# correct - once the cells are in the model unpenalised, is_weekend adds nothing.

# Cell means of a matrix, returned as (G x ncol) with the row order of `levels`.
cell_means <- function(M, g, levels) {
  s <- rowsum(M, group = g, reorder = FALSE)
  n <- as.vector(table(factor(g, levels = levels)))
  s <- s[match(levels, as.integer(rownames(s))), , drop = FALSE]
  s[is.na(s)] <- 0
  s / pmax(n, 1L)
}

# ---- 3. one window -----------------------------------------------------------
# Fit at refit origin T0. Training origins s run over a WINDOW-long block ending
# at T0-h, so that every training target y_{s+h} satisfies s+h <= T0 and was
# therefore observable at T0. Same construction as runAR() in benchmarks.R.
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

  # Columns constant within this window carry no information and upset the
  # penalty normalisation - drop them for this window only.
  keep <- which(apply(X_tr, 2L, function(z) !anyNA(z) && var(z) > 0))
  X_tr <- X_tr[, keep, drop = FALSE]

  # On residualised data both sides are exactly cell-mean zero, hence overall
  # mean zero, so no intercept is needed; D already carries it.
  fit <- rlasso(X_tr, y_tr, post = FALSE, intercept = !PARTIAL_OUT)

  cf <- coef(fit)
  if (length(cf) == ncol(X_tr) + 1L) { b0 <- unname(cf[1L]); b <- cf[-1L] }
  else                               { b0 <- 0;              b <- cf }
  sel <- which(abs(b) > 0)

  # post-LASSO: plain OLS on the selected set. qr() pivoting guards against a
  # rank-deficient selection, which happens when LASSO keeps two columns that
  # are collinear inside this particular window.
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

# Rebuild the unpenalised cell effects for a given coefficient vector:
#   g_c = mean over training rows in cell c of (y - X b)
# which, because D is a complete set of mutually exclusive dummies, is just the
# cell mean of y minus the cell means of X times b.
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
      # a cell unseen in this window (should not happen at 3 years) -> grand mean
      a[is.na(a)] <- mean(a, na.rm = TRUE); ap[is.na(ap)] <- mean(ap, na.rm = TRUE)
      pred_l[idx] <- a[gp]  + xb
      pred_p[idx] <- ap[gp] + xbp
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
for (h in H) {
  s <- out[h == ..h & !is.na(actual)]
  if (!nrow(s)) next
  cat(sprintf("  h=%2d  n=%s  LASSO RMSE %6.1f  MAE %6.1f  MAPE %.3f%%  |  post RMSE %6.1f  |  median vars %d\n",
      h, format(nrow(s), big.mark = ","),
      sqrt(mean((s$actual - s$lasso)^2)), mean(abs(s$actual - s$lasso)),
      100 * mean(abs(s$actual - s$lasso) / s$actual),
      sqrt(mean((s$actual - s$post_lasso)^2)),
      as.integer(median(s$n_sel))))
}
cat("\nwritten:", OUT, "\n")
