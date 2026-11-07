# view_lgbm.R ------------------------------------------------------------------
# Loads the saved LightGBM forecasts and prints the error tables and figures, in
# seconds. Run lgbm.R (~45 min on 7 cores) only when the model code changes; use
# this file to look at the results.
#
#   in : output/forecasts/lgbm.csv.gz             (from lgbm.R)
#        output/forecasts/lgbm_tuning.csv.gz      (from lgbm.R)
#        output/forecasts/lgbm_importance.csv.gz  (from lgbm.R)
#        output/forecasts/benchmarks.csv.gz       (from benchmarks.R; optional)

suppressMessages({
  library(data.table)
  library(sandwich)                    # NeweyWest() for the DM tests, as in Lecture 4
})

lg  <- fread("output/forecasts/lgbm.csv.gz", colClasses = c(datetime = "character"))
tun <- fread("output/forecasts/lgbm_tuning.csv.gz", colClasses = c(refit = "character"))
imp <- fread("output/forecasts/lgbm_importance.csv.gz", colClasses = c(refit = "character"))
MAX_TREES <- 3000L                     # as in lgbm.R
H <- sort(unique(lg$h))

# Benchmarks on exactly the same (origin, h) rows, if benchmarks.R has been run,
# plus the simple-average combination of LightGBM and the AR (prof's comment:
# check whether averaging the most promising models beats each one alone).
models <- "lgbm"
BM <- "output/forecasts/benchmarks.csv.gz"
if (file.exists(BM)) {
  bm <- fread(BM, colClasses = c(datetime = "character"))
  lg <- merge(lg, bm[, .(datetime, h, actual_bm = actual, ar_bic, snaive168, snaive24, ema)],
              by = c("datetime", "h"))
  stopifnot(max(abs(lg$actual - lg$actual_bm)) < 1e-6)    # same target on every row
  lg[, avg_lgbm_ar := (lgbm + ar_bic) / 2]
  models <- c("lgbm", "ar_bic", "avg_lgbm_ar", "snaive168", "snaive24", "ema")
}

# ---- errors ------------------------------------------------------------------
# MAE is the headline loss, MAPE the scale-free companion (the literature reports
# both almost universally). RMSE is not used. Full evaluation belongs in evaluate.R.
MAE  <- function(pred, truth) mean(abs(truth - pred))
MAPE <- function(pred, truth) 100 * mean(abs(truth - pred) / truth)

err <- lg[, .(model = models,
              MAE   = sapply(models, function(v) MAE(get(v), actual)),
              MAPE  = sapply(models, function(v) MAPE(get(v), actual)),
              ME    = sapply(models, function(v) mean(actual - get(v)))), by = h]

# ME (mean error, actual - forecast) is a diagnostic, not a ranking loss: > 0 means
# the model under-forecasts on average.
cat("=== out-of-sample errors,", min(lg$datetime), "->", max(lg$datetime), "===\n")
print(err[, .(h, model, MAE = round(MAE, 1), MAPE = round(MAPE, 3), ME = round(ME, 1))])

# One row per h, so the winner reads across a row. Compare within a row only:
# longer horizons are harder. EMA is a reference level, not a contender.
own <- setdiff(models, "ema")
for (v in c("MAE", "MAPE")) {
  cat(sprintf("\n=== %s by horizon, lowest = best ===\n", v))
  wide <- dcast(err, h ~ model, value.var = v)
  setkey(wide, NULL)                   # dcast sets a key; drop it so it is not printed
  wide[, best := own[apply(.SD, 1, which.min)], .SDcols = own]
  setcolorder(wide, c("h", models, "best"))
  print(wide[, lapply(.SD, function(x) if (is.numeric(x) && !is.integer(x)) round(x, if (v == "MAE") 1 else 3) else x)])
}

# ---- tuning ------------------------------------------------------------------
# The grid point that won each re-validation. Trees at the cap would mean
# MAX_TREES, not the data, is choosing M.
ch <- tun[, .SD[which.min(val_mae)], by = .(h, refit)]
cat("\n=== settings chosen by recursive validation (share of re-validations) ===\n")
print(ch[, .N, by = .(h, loss = objective, leaves = num_leaves)][
  , share := round(N / sum(N), 2), by = h][order(h, -N)])
cat("\nnumber of trees chosen:\n")
print(ch[, .(re_validations = .N, trees_min = min(trees), trees_median = median(trees),
             trees_max = max(trees), at_cap = sum(trees > 0.95 * MAX_TREES)), by = h])

# ---- DM tests ----------------------------------------------------------------
# Preliminary: absolute-error loss, d_t = |e_benchmark| - |e_lgbm|, so d_t > 0
# when LightGBM is closer. Regression on a constant with a Newey-West standard
# error, as in the Lecture 4 code; the cube-root lag rule and one week of lags.
if ("ar_bic" %in% models) {
  dm <- rbindlist(lapply(H, function(hh) {
    d <- lg[h == hh]
    rbindlist(lapply(c("ar_bic", "snaive168", "avg_lgbm_ar"), function(b) {
      dd <- abs(d$actual - d[[b]]) - abs(d$actual - d$lgbm)
      m  <- lm(dd ~ 1)
      L  <- floor(length(dd)^(1/3))
      data.table(h = hh, versus = b, mean_d = round(coef(m)[[1]], 2),
                 t_cuberoot = round(coef(m)[[1]] / sqrt(NeweyWest(m, lag = L)[1]), 2),
                 t_lag168   = round(coef(m)[[1]] / sqrt(NeweyWest(m, lag = 168L)[1]), 2))
    }))
  }))
  cat("\n=== DM tests, absolute loss: d_t = |e_versus| - |e_lgbm| (> 0: LightGBM better) ===\n")
  print(dm)
}

# ---- figures -----------------------------------------------------------------
# 1. MAE by year of the forecast origin, one panel per h
# 2. mean error by target hour of the week (as in view_benchmarks.R)
# 3. settings chosen at each re-validation
# 4. mean gain importance, top 15 predictors per h
# 5. two weeks of forecasts against actual demand, h = 1 and h = 24
FIG <- "output/figures"
if (!dir.exists(FIG)) dir.create(FIG, recursive = TRUE)
DAYS <- c("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
cols <- setNames(seq_along(own), own)

# grey bands on weekend days and a day-name label under each day; x is hours
# from the first Sunday 00:00 (same helper as view_benchmarks.R)
week_axis <- function(n_days, labels) {
  wkend <- which(rep(DAYS, length.out = n_days) %in% c("Sat", "Sun")) - 1L
  usr <- par("usr")
  rect(wkend * 24, usr[3], (wkend + 1) * 24, usr[4], col = "grey92", border = NA)
  abline(v = seq(24, (n_days - 1) * 24, 24), col = "grey80", lty = 3)
  axis(1, at = seq(12, n_days * 24, 24), labels = labels, tick = FALSE, cex.axis = 0.85)
}

tt <- as.POSIXlt(as.POSIXct(lg$datetime, tz = "UTC") + lg$h * 3600L)   # target hour
lg[, `:=`(how_target = tt$wday * 24L + tt$hour, year = as.integer(substr(datetime, 1, 4)))]

by_year <- lg[, lapply(.SD, function(p) mean(abs(actual - p))), by = .(h, year),
              .SDcols = own][order(h, year)]
png(file.path(FIG, "lgbm_mae_by_year.png"), width = 1400, height = 800)
par(mfrow = c(2, 2), mar = c(3, 5, 3, 1), oma = c(0, 0, 2, 0))
for (hh in H) {
  d <- by_year[h == hh]
  matplot(d$year, as.matrix(d[, ..own]), type = "b", pch = 19, lty = 1, lwd = 2, col = cols,
          ylim = c(0, max(d[, ..own])), xlab = "", ylab = "MAE (MW)", main = paste("h =", hh))
  if (hh == H[1]) legend("right", own, col = cols, lty = 1, lwd = 2, bty = "n")
}
mtext("MAE by year of the forecast origin (2018 from 16 Jul, 2026 to 30 Sep)", outer = TRUE,
      font = 2, cex = 1.2)
invisible(dev.off())

plot_models <- intersect(c("lgbm", "ar_bic"), own)
by_how <- lg[, lapply(.SD, function(p) mean(actual - p)), by = .(h, how_target),
             .SDcols = plot_models][order(h, how_target)]
ylim <- range(by_how[, ..plot_models])
png(file.path(FIG, "lgbm_error_by_hour_of_week.png"), width = 1400, height = 800)
par(mfrow = c(2, 2), mar = c(3, 5, 3, 1), oma = c(0, 0, 2, 0))
for (hh in H) {
  d <- by_how[h == hh]
  matplot(d$how_target, as.matrix(d[, ..plot_models]), type = "n", xaxt = "n",
          ylim = ylim, xlab = "", ylab = "mean error (MW)", main = paste("h =", hh))
  week_axis(7L, DAYS)
  abline(h = 0, col = "grey40")
  matlines(d$how_target, as.matrix(d[, ..plot_models]), lty = 1, lwd = 2,
           col = cols[plot_models])
  if (hh == H[1]) legend("topright", plot_models, col = cols[plot_models], lty = 1, lwd = 2,
                         bty = "n")
}
mtext("Mean forecast error (actual - forecast) by hour of the week", outer = TRUE,
      font = 2, cex = 1.2)
invisible(dev.off())

ch[, refit_t := as.POSIXct(refit, tz = "UTC")]
pch_leaves <- c(`15` = 1, `31` = 16, `63` = 17)
png(file.path(FIG, "lgbm_tuning.png"), width = 1400, height = 800)
par(mfrow = c(2, 2), mar = c(3, 5, 3, 1), oma = c(0, 0, 2, 0))
for (hh in H) {
  d <- ch[h == hh]
  plot(d$refit_t, d$trees, ylim = c(0, MAX_TREES), xlab = "", ylab = "number of trees M",
       main = paste("h =", hh), col = ifelse(d$objective == "l1", "firebrick", "steelblue"),
       pch = pch_leaves[as.character(d$num_leaves)])
  if (hh == H[1]) legend("topright", bty = "n", cex = 0.9,
                         legend = c("squared loss (l2)", "absolute loss (l1)", paste(names(pch_leaves), "leaves")),
                         col = c("steelblue", "firebrick", rep("grey30", 3)),
                         pch = c(15, 15, pch_leaves))
}
mtext("Settings chosen at each re-validation (every 4th weekly refit)", outer = TRUE,
      font = 2, cex = 1.2)
invisible(dev.off())

# mean share of total gain over all refits; a predictor a refit never split on is
# missing from that refit's table, so divide by the number of refits
n_ref <- imp[, .(n = uniqueN(refit)), by = h]
ia <- imp[, .(gain = sum(Gain)), by = .(h, Feature)][n_ref, on = "h"][
  , gain := gain / n][order(h, -gain)]
png(file.path(FIG, "lgbm_importance.png"), width = 1400, height = 900)
par(mfrow = c(2, 2), mar = c(4, 10, 3, 1), oma = c(0, 0, 2, 0))
for (hh in H) {
  top <- head(ia[h == hh], 15)
  barplot(rev(top$gain), names.arg = rev(top$Feature), horiz = TRUE, las = 1, cex.names = 0.9,
          xlim = c(0, 1.05 * max(top$gain)), xlab = "mean share of total gain",
          main = paste("h =", hh))
}
mtext("LightGBM predictor importance (gain), averaged over all weekly refits", outer = TRUE,
      font = 2, cex = 1.2)
invisible(dev.off())

lg[, tgt := format(as.POSIXct(datetime, tz = "UTC") + h * 3600L, "%Y-%m-%d %H:%M:%S")]
setorder(lg, h, datetime)
png(file.path(FIG, "lgbm_two_weeks.png"), width = 1000, height = 700)
par(mfrow = c(2, 1), mar = c(3, 5, 3, 1))
for (hh in intersect(c(1L, 24L), H)) {
  d <- lg[h == hh & tgt >= "2023-03-05 00:00:00" & tgt < "2023-03-19 00:00:00"]
  if (nrow(d) == 0L) next              # e.g. a dry run, which stops in August 2018
  x  <- seq_len(nrow(d)) - 1L
  yr <- range(d$actual, d[, ..plot_models])
  plot(x, d$actual, type = "n", xaxt = "n", xlab = "", ylab = "demand (MW)",
       ylim = yr - c(0.15 * diff(yr), 0),             # room for the legend at the bottom
       main = sprintf("h = %d: forecasts for 5-18 Mar 2023 (weekends shaded)", hh))
  week_axis(14L, paste(rep(DAYS, 2), 5:18))
  lines(x, d$actual, lwd = 4, col = "grey65")         # actual: thick grey, forecasts on top
  for (v in plot_models) lines(x, d[[v]], col = cols[v])
  legend("bottom", c("actual", plot_models), col = c("grey65", cols[plot_models]),
         lwd = c(4, rep(1, length(plot_models))), bty = "n", horiz = TRUE)
}
invisible(dev.off())
cat("\nfigures written to", FIG, "\n")
