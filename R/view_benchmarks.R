# view_benchmarks.R ------------------------------------------------------------
# Loads the saved benchmark forecasts, prints the error tables and AR lag
# lengths, and redraws the figures. Use this file to look at the results.
#
#   in : output/forecasts/benchmarks.csv.gz   (from benchmarks.R)

suppressMessages(library(data.table))

bm <- fread("output/forecasts/benchmarks.csv.gz", colClasses = c(datetime = "character"))

# ---- errors ------------------------------------------------------------------
# RMSE is the headline loss; MAE and MAPE are added because the load-forecasting
# literature reports them almost universally.
RMSE <- function(pred, truth) sqrt(mean((truth - pred)^2))
MAE  <- function(pred, truth) mean(abs(truth - pred))
MAPE <- function(pred, truth) 100 * mean(abs(truth - pred) / truth)

models <- c("snaive168", "snaive24", "ar_bic", "ema")

err <- bm[, .(model = models,
              RMSE  = sapply(models, function(v) RMSE(get(v), actual)),
              MAE   = sapply(models, function(v) MAE(get(v), actual)),
              MAPE  = sapply(models, function(v) MAPE(get(v), actual))), by = h]

cat("out-of-sample forecast origins:", min(bm$datetime), "->", max(bm$datetime), "\n")

# ---- best model per horizon --------------------------------------------------
# The same errors side by side, one table per loss and one row per h, so the
# winner reads across a row. `best` is among our models only; EMA is a reference.
own <- setdiff(models, "ema")
for (loss in c("RMSE", "MAE", "MAPE")) {
  cat(sprintf("\n=== %s by horizon (%s), lowest = best ===\n", loss,
              if (loss == "MAPE") "%" else "MW"))
  wide <- dcast(err, h ~ model, value.var = loss)
  setkey(wide, NULL)                   # dcast sets a key; drop it so it is not printed
  wide[, best := own[apply(.SD, 1, which.min)], .SDcols = own]
  setcolorder(wide, c("h", models, "best"))
  digits <- if (loss == "MAPE") 3 else 1
  print(wide[, lapply(.SD, function(x) if (is.numeric(x) && !is.integer(x)) round(x, digits) else x)])
}

# ---- AR lag length -----------------------------------------------------------
# Shows whether BIC is pinned at P_MAX - if so, the search limit, not the data,
# is choosing p.
cat("\nAR lag length chosen by BIC:\n")
print(bm[, .(p_min = min(ar_p), p_median = median(ar_p), p_max = max(ar_p)), by = h])

# ---- figures -----------------------------------------------------------------
# 1. two ordinary weeks of demand: the daily shape, and weekends below weekdays
#    Reading: every day dips to ~5,100 MW at night and peaks ~7,000 MW in the
#    afternoon; Sat/Sun peak lower (~6,200-6,700). The mean depends on the hour and
#    the day, so the series needs hour, day-type and hour x day-type dummies.
# 2. mean error by target hour of the week (0 = Sun 00:00 ... 167 = Sat 23:00),
#    one panel per h: a flat line = daily AND weekly seasonality captured.
#    Reading: snaive24 forecasts Monday from Sunday (+900 MW) and Saturday from
#    Friday (-600 MW) - it misses the weekly season. snaive168 and the AR are flat
#    at every h, so both capture the daily and weekly seasonality.
FIG <- "output/figures"
if (!dir.exists(FIG)) dir.create(FIG, recursive = TRUE)
DAYS <- c("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")

# grey bands on weekend days and a day-name label under each day; x is hours
# from the first Sunday 00:00
week_axis <- function(n_days, labels) {
  wkend <- which(rep(DAYS, length.out = n_days) %in% c("Sat", "Sun")) - 1L
  usr <- par("usr")
  rect(wkend * 24, usr[3], (wkend + 1) * 24, usr[4], col = "grey92", border = NA)
  abline(v = seq(24, (n_days - 1) * 24, 24), col = "grey80", lty = 3)
  axis(1, at = seq(12, n_days * 24, 24), labels = labels, tick = FALSE, cex.axis = 0.85)
}

two_wk <- bm[h == 1L & datetime >= "2023-03-04 23:00:00" & datetime < "2023-03-18 23:00:00"]
png(file.path(FIG, "demand_two_weeks.png"), width = 1000, height = 400)
plot(seq_len(nrow(two_wk)) - 1L, two_wk$actual, type = "n", xaxt = "n",
     xlab = "", ylab = "demand (MW)",
     main = "Hourly demand, 5-18 Mar 2023 (weekends shaded)")
week_axis(14L, paste(rep(DAYS, 2), 5:18))
lines(seq_len(nrow(two_wk)) - 1L, two_wk$actual)
invisible(dev.off())

tt <- as.POSIXlt(as.POSIXct(bm$datetime, tz = "UTC") + bm$h * 3600L)
bm[, how_target := tt$wday * 24L + tt$hour]
plot_models <- own   # EMA excluded: it forecasts NEM demand, below system demand,
                     # so its errors carry a level offset that is not seasonality
by_how <- bm[, lapply(.SD, function(p) mean(actual - p)), by = .(h, how_target),
             .SDcols = plot_models][order(h, how_target)]
cols <- setNames(seq_along(plot_models), plot_models)
ylim <- range(by_how[, ..plot_models])

png(file.path(FIG, "error_by_hour_of_week.png"), width = 1400, height = 800)
H <- sort(unique(bm$h))
par(mfrow = c(2, 2), mar = c(3, 5, 3, 1), oma = c(0, 0, 2, 0))   # 2 x 2 grid, filled by row
for (hh in H) {
  d <- by_how[h == hh]
  matplot(d$how_target, as.matrix(d[, ..plot_models]), type = "n", xaxt = "n",
          ylim = ylim, xlab = "", ylab = "mean error (MW)", main = paste("h =", hh))
  week_axis(7L, DAYS)
  abline(h = 0, col = "grey40")
  matlines(d$how_target, as.matrix(d[, ..plot_models]), lty = 1, lwd = 2, col = cols)
  if (hh == H[1]) legend("topright", plot_models, col = cols, lty = 1, lwd = 2, bty = "n")
}
mtext("Mean forecast error (actual - forecast) by hour of the week", outer = TRUE,
      font = 2, cex = 1.2)
invisible(dev.off())
cat("\nfigures written to", FIG, "\n")
