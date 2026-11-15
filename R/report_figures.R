# report_figures.R -------------------------------------------------------------
# Figures 1 and 2 for Section 3.2 of the report.
#
#   in : data/processed/features.csv.gz
#   out: output/figures/fig1_demand_daily_mean.png
#        output/figures/fig2_demand_by_hour_daytype.png
#
# Design notes:
#   * Series are identified by direct labels at the line ends, not a legend box,
#     so identity never depends on colour alone. Line types differ as well, so
#     the figures survive greyscale printing.
#   * Palette is the first four categorical slots (blue, orange, aqua, yellow),
#     which clear the CVD and normal-vision separation floors on the adjacent
#     pairlist that line charts use.
#   * Grid and axes are recessive; the data is the only dark ink.

suppressMessages(library(data.table))
source("R/utils.R")

f <- read_gz("data/processed/features.csv.gz", colClasses = c(datetime = "character"))
setorder(f, datetime)
f[, date := as.Date(substr(datetime, 1, 10))]

fig_dir <- "output/figures"
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

BLUE <- "#2a78d6"; ORANGE <- "#eb6834"; AQUA <- "#1baf7a"; YELLOW <- "#eda100"
GRID <- "#e1e0d9"; AXIS <- "#c3c2b7"; INK <- "#111111"; MUTED <- "#555555"

# ---- Figure 1: daily mean demand --------------------------------------------
daily <- f[, .(mw = mean(y)), by = date][order(date)]

png(file.path(fig_dir, "fig1_demand_daily_mean.png"),
    width = 7, height = 3.6, units = "in", res = 300)
par(mar = c(3.2, 4.0, 1.0, 1.0), mgp = c(2.4, 0.6, 0), las = 1,
    col.axis = MUTED, col.lab = INK, family = "sans")

plot(daily$date, daily$mw, type = "n", xlab = "", ylab = "Daily mean demand (MW)",
     axes = FALSE, ylim = range(daily$mw))
# Circuit breaker band, drawn under the data
rect(as.Date("2020-04-07"), par("usr")[3], as.Date("2020-06-01"), par("usr")[4],
     col = "#f0f0ee", border = NA)
abline(h = pretty(daily$mw), col = GRID, lwd = 0.8)
lines(daily$date, daily$mw, col = BLUE, lwd = 1.1)

axis.Date(1, at = seq(as.Date("2017-01-01"), as.Date("2026-01-01"), by = "2 years"),
          format = "%Y", col = AXIS, col.ticks = AXIS, lwd = 0.8)
axis(2, at = pretty(daily$mw), col = AXIS, col.ticks = AXIS, lwd = 0.8)

text(as.Date("2020-05-04"), par("usr")[4] - 0.04 * diff(par("usr")[3:4]),
     "Circuit Breaker", col = MUTED, cex = 0.7, adj = c(0.5, 1))
dev.off()

# ---- Figure 2: mean demand by hour of day, by day type ----------------------
# Day types match the hour-of-week coding used by the LASSO: a public holiday is
# treated separately here precisely to show that it behaves like a Sunday.
f[, daytype := fifelse(is_holiday == 1L & dow %in% 1:5, "Public holiday (Mon-Fri)",
                fifelse(dow == 0L, "Sunday",
                 fifelse(dow == 6L, "Saturday", "Working day")))]

prof <- f[, .(mw = mean(y)), by = .(daytype, hr)][order(daytype, hr)]

series <- c("Working day", "Saturday", "Sunday", "Public holiday (Mon-Fri)")
cols   <- c(BLUE, ORANGE, AQUA, YELLOW)
ltys   <- c(1, 2, 1, 4)
lwds   <- c(2.0, 1.6, 2.0, 2.0)

png(file.path(fig_dir, "fig2_demand_by_hour_daytype.png"),
    width = 7, height = 4.0, units = "in", res = 300)
par(mar = c(3.4, 4.0, 1.0, 8.6), mgp = c(2.4, 0.6, 0), las = 1,
    col.axis = MUTED, col.lab = INK, family = "sans", xpd = NA)

# Limits cover the data with a little headroom; ticks are then drawn only
# inside those limits, so the axis never runs past the data and the data is
# never clipped.
yr <- range(prof$mw) + c(-1, 1) * 0.05 * diff(range(prof$mw))
yt <- pretty(yr, n = 4)
yt <- yt[yt >= yr[1] & yt <= yr[2]]

plot(0:23, rep(mean(yr), 24), type = "n", xlab = "Hour of day",
     ylab = "Mean demand (MW)", axes = FALSE, ylim = yr, xlim = c(0, 23),
     yaxs = "i", xaxs = "i")
abline(h = yt, col = GRID, lwd = 0.8)

for (i in seq_along(series)) {
  s <- prof[daytype == series[i]][order(hr)]
  lines(s$hr, s$mw, col = cols[i], lwd = lwds[i], lty = ltys[i])
}

# Direct labels at the right-hand end, pushed apart where Sunday and the
# holiday profile converge; a short leader joins each label to its line.
ends <- vapply(series, function(s) prof[daytype == s & hr == 23]$mw, numeric(1))
pos  <- ends
gap  <- 0.06 * diff(yr)
o    <- order(ends, decreasing = TRUE)   # top label stays put, rest pushed down
for (k in seq_along(o)[-1])
  if (pos[o[k - 1]] - pos[o[k]] < gap) pos[o[k]] <- pos[o[k - 1]] - gap
for (i in seq_along(series)) {
  segments(23, ends[i], 23.9, pos[i], col = cols[i], lwd = 0.7)
  text(24.2, pos[i], series[i], col = cols[i], cex = 0.72, adj = 0)
}

axis(1, at = seq(0, 21, by = 3), col = AXIS, col.ticks = AXIS, lwd = 0.8)
axis(2, at = yt, col = AXIS, col.ticks = AXIS, lwd = 0.8)
dev.off()

cat("written:\n",
    " ", file.path(fig_dir, "fig1_demand_daily_mean.png"), "\n",
    " ", file.path(fig_dir, "fig2_demand_by_hour_daytype.png"), "\n")

# ---- numbers quoted in the Section 3.2 text ---------------------------------
cat("\n=== values cited in the text ===\n")
w <- dcast(prof, hr ~ daytype, value.var = "mw")
setnames(w, make.names(names(w)))
ph <- w[["Public.holiday..Mon.Fri."]]
cat("working-day trough:", round(min(w[["Working.day"]])), "MW at hour",
    w$hr[which.min(w[["Working.day"]])], "\n")
cat("working-day peak  :", round(max(w[["Working.day"]])), "MW at hour",
    w$hr[which.max(w[["Working.day"]])], "\n")
for (nm in c("Sunday", "Saturday", "Working.day"))
  cat(sprintf("  mean |gap| holiday vs %-12s %5.0f MW\n", nm, mean(abs(w[[nm]] - ph))))
cat("shape correlation, holiday vs Sunday:",
    round(cor(w[["Sunday"]] - mean(w[["Sunday"]]), ph - mean(ph)), 3), "\n")
