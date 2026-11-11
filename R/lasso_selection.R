# lasso_selection.R ------------------------------------------------------------
# Which predictors does LASSO actually keep?
#
#   in : data/processed/features.csv.gz
#   out: output/tables/lasso_selection.csv.gz        (one row per refit x variable)
#        output/tables/lasso_selection_frequency.csv (summary, one row per variable x h)
#
# WHY THIS EXISTS SEPARATELY
#
# lasso.R saves `n_sel` - how many variables survived each refit - but never
# which ones. That cannot be recovered from the saved forecasts: each refit
# serves exactly 168 consecutive origins and the unpenalised block has exactly
# 168 hour-of-week cells, so over one week each cell occurs once and the system
# is perfectly saturated. Any X*beta can be absorbed into the cell effects, so
# beta is not identified from the predictions alone.
#
# Refitting is the only route. rlasso is deterministic - the penalty is computed
# analytically, not cross-validated - so this reproduces the SAME selections the
# full run made, exactly rather than approximately. We therefore only need a
# subsample of refit origins to estimate selection frequency: STRIDE = 10 gives
# ~43 refits per horizon in about five minutes, against 429 and ~50 minutes.
#
# Settings below are copied from lasso.R and must stay in step with it, or this
# describes a different model from the one that produced the forecasts.

suppressMessages({
  library(data.table)
  library(hdm)
})
source("R/utils.R")

IN        <- "data/processed/features.csv.gz"
OUT_LONG  <- "output/tables/lasso_selection.csv.gz"
OUT_FREQ  <- "output/tables/lasso_selection_frequency.csv"

H           <- c(1L, 6L, 12L, 24L)
WINDOW      <- as.integer(1.5 * 365 * 24)
REFIT       <- 168L
OOS_START   <- "2018-07-16 00:00:00"
DUMS        <- c("cb_strict", "cb_p1", "cb_p2", "cb_p3", "ha_2021")
PARTIAL_OUT <- TRUE

STRIDE <- 10L   # fit every STRIDE-th refit origin. 1 = every refit (~50 min).

f <- fread(IN, colClasses = c(datetime = "character"))
setorder(f, datetime)
y <- f$y
N <- length(y)
T_first <- match(OOS_START, f$datetime)
if (is.na(T_first)) stop("OOS_START not found in features: ", OOS_START)

# ---- design, identical to lasso.R -------------------------------------------
target_time <- c("hr", "dow", "mth", "doy", "year", "t_index",
                 "is_weekend", "is_holiday", "is_workday", DUMS)
origin_time <- setdiff(
  names(f),
  c("datetime", "y", "usable", target_time,
    grep("^(y_h|nem_h)", names(f), value = TRUE)))

X_org   <- cbind(y_t = y, as.matrix(f[, ..origin_time]))
hd      <- f$hr * 7L + f$dow
tt_keep <- setdiff(target_time, c("hr", "dow"))
X_pen   <- cbind(X_org, as.matrix(f[, ..tt_keep]))
invisible(gc(verbose = FALSE))

cell_means <- function(M, g, levels) {
  s <- rowsum(M, group = g, reorder = FALSE)
  n <- as.vector(table(factor(g, levels = levels)))
  s <- s[match(levels, as.integer(rownames(s))), , drop = FALSE]
  s[is.na(s)] <- 0
  s / pmax(n, 1L)
}

# ---- one window: fit and return the selected set ----------------------------
# Identical to run_lasso() in lasso.R up to the rlasso call; from there we keep
# the coefficient names instead of the predictions.
select_once <- function(T0, h) {
  tr   <- (T0 - h - WINDOW + 1L):(T0 - h)
  y_tr <- y[tr + h]
  X_tr <- X_pen[tr, , drop = FALSE]
  X_tr[, colnames(X_pen) %in% tt_keep] <-
    X_pen[tr + h, colnames(X_pen) %in% tt_keep, drop = FALSE]

  if (PARTIAL_OUT) {
    g   <- hd[tr + h]
    lv  <- sort(unique(g))
    M   <- cell_means(cbind(y_tr, X_tr), g, lv)
    pos <- match(g, lv)
    y_tr <- y_tr - M[pos, 1L]
    X_tr <- X_tr - M[pos, -1L, drop = FALSE]
  }

  keep <- which(apply(X_tr, 2L, function(z) !anyNA(z) && var(z) > 0))
  X_tr <- X_tr[, keep, drop = FALSE]

  fit <- rlasso(X_tr, y_tr, post = FALSE, intercept = !PARTIAL_OUT)
  cf  <- coef(fit)
  b   <- if (length(cf) == ncol(X_tr) + 1L) cf[-1L] else cf
  sel <- which(abs(b) > 0)
  if (!length(sel)) return(NULL)

  # Raw coefficients are not comparable across predictors on different scales,
  # so also report beta * sd(x) within this window: the change in demand, in MW,
  # associated with a one-standard-deviation move in that predictor.
  sds <- apply(X_tr[, sel, drop = FALSE], 2L, sd)
  data.table(variable = names(b)[sel],
             coef     = as.numeric(b[sel]),
             coef_std = as.numeric(b[sel]) * sds)
}

# ---- run ---------------------------------------------------------------------
all_refits <- seq(T_first, N - max(H), by = REFIT)
origins    <- all_refits[seq(1L, length(all_refits), by = STRIDE)]

cat("hour-of-week cells: UNPENALISED (partialled out)\n")
cat("refits available:", length(all_refits), " | sampling every", STRIDE,
    "->", length(origins), "per horizon,", length(origins) * length(H), "fits total\n\n")

res <- list(); t0 <- Sys.time()
for (h in H) {
  for (k in seq_along(origins)) {
    T0 <- origins[k]
    s  <- select_once(T0, h)
    if (is.null(s)) next
    s[, `:=`(h = h, origin = f$datetime[T0])]
    res[[length(res) + 1L]] <- s
    if (k %% 10 == 0 || k == length(origins)) {
      el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
      cat(sprintf("  h=%2d  %3d / %3d   (%d selected, %.1f min elapsed)\n",
                  h, k, length(origins), nrow(s), el))
    }
  }
}
long <- rbindlist(res)

# ---- summarise ---------------------------------------------------------------
n_fits <- long[, uniqueN(origin), by = h]
setnames(n_fits, "V1", "n_refits")

freq <- long[, .(times_selected = .N,
                 mean_coef_std  = mean(coef_std),
                 sd_coef_std    = sd(coef_std),
                 pct_positive   = 100 * mean(coef > 0)),
             by = .(h, variable)]
freq <- merge(freq, n_fits, by = "h")
freq[, pct_selected := round(100 * times_selected / n_refits, 1)]
freq[, mean_coef_std := round(mean_coef_std, 2)]
freq[, sd_coef_std   := round(sd_coef_std, 2)]
freq[, pct_positive  := round(pct_positive, 0)]
setorder(freq, h, -pct_selected, -mean_coef_std)

if (!dir.exists("output/tables")) dir.create("output/tables", recursive = TRUE)
fwrite(long, OUT_LONG)
fwrite(freq[, .(h, variable, pct_selected, times_selected, n_refits,
                mean_coef_std, sd_coef_std, pct_positive)], OUT_FREQ)

# ---- report ------------------------------------------------------------------
cat("\n=== always selected (100% of refits, every horizon) ===\n")
core <- freq[pct_selected == 100, .N, by = variable][N == length(H)]$variable
cat(if (length(core)) paste(" ", core, collapse = "\n") else "  none", "\n")

for (hh in H) {
  cat(sprintf("\n=== h = %d: top 12 by selection frequency ===\n", hh))
  print(freq[h == hh][1:12, .(variable, pct_selected, mean_coef_std, pct_positive)])
}

cat("\n=== how many variables are stable vs marginal ===\n")
print(freq[, .(always      = sum(pct_selected == 100),
               usually     = sum(pct_selected >= 75 & pct_selected < 100),
               sometimes   = sum(pct_selected >= 25 & pct_selected < 75),
               rarely      = sum(pct_selected < 25),
               ever_chosen = .N), by = h][order(h)])

cat("\nwritten:", OUT_LONG, "\n         ", OUT_FREQ, "\n")
