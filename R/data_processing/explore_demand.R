# explore_demand.R -------------------------------------------------------------
# Sanity-check the EMA demand series before it becomes the model target.
#
# Checks:
#   1. coverage, duplicates, missing periods
#   2. the 398 half-hours where system demand < NEM demand (flagged in EMA's
#      own README) - are they clustered anywhere that matters?
#   3. outliers and level shifts
#   4. Circuit Breaker (Apr-Jun 2020) visible in the series?
#
# Writes PNGs to output/figures/ and prints diagnostics.
#
# Note on hour assignment: we derive the hour from `period` (1-48), NOT from
# datetime_end. Period 48 ends at 00:00 of the *next* day, so grouping on
# datetime_end would push the last half hour of every day into the wrong date.
#   period 1 ends 00:30 -> hour 0 ;  period 48 ends 00:00 -> hour 23
#   hour = floor((period - 1) / 2)

suppressMessages(library(data.table))
source("R/utils.R")

fig_dir <- "output/figures"
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

# ---- load -------------------------------------------------------------------
fs <- sort(list.files("data/raw/ema", pattern = "csv[.]gz$", full.names = TRUE))
# Force date to character: fread would parse it, and we want to control tz.
d <- rbindlist(lapply(fs, read_gz, colClasses = c(date = "character")))
setnames(d, tolower(names(d)))

cat("=== coverage ===\n")
cat("rows:", format(nrow(d), big.mark = ","), "\n")
cat("date range:", min(d$date), "->", max(d$date), "\n")
cat("unique days:", uniqueN(d$date), "\n")
cat("periods per day (expect 48):",
    paste(sort(unique(d[, .N, by = date]$N)), collapse = ", "), "\n")
cat("duplicate date-period rows:",
    nrow(d) - uniqueN(d, by = c("date", "period")), "\n")
cat("NAs in system_demand_actual_mw:", sum(is.na(d$system_demand_actual_mw)), "\n")

# ---- build an hourly series -------------------------------------------------
d[, hour := (period - 1L) %/% 2L]
d[, datetime := as.POSIXct(paste0(date, " ", sprintf("%02d", hour), ":00:00"),
                           tz = "Asia/Singapore")]

hourly <- d[, .(demand_mw      = mean(system_demand_actual_mw, na.rm = TRUE),
                nem_actual_mw  = mean(nem_demand_actual_mw,    na.rm = TRUE),
                nem_forecast_mw= mean(nem_demand_forecast_mw,  na.rm = TRUE)),
            by = .(datetime, date, hour)]
setorder(hourly, datetime)
cat("hourly rows:", format(nrow(hourly), big.mark = ","), "\n")

cat("\n=== demand distribution (MW) ===\n")
print(summary(hourly$demand_mw))

# ---- 1. the system < NEM anomaly -------------------------------------------
cat("\n=== anomaly: system demand < NEM demand (half-hourly) ===\n")
anom <- d[system_demand_actual_mw < nem_demand_actual_mw]
cat("count:", nrow(anom), "  (EMA README says 398)\n")
if (nrow(anom) > 0) {
  anom[, gap := nem_demand_actual_mw - system_demand_actual_mw]
  cat("largest shortfall:", round(max(anom$gap), 1), "MW\n")
  cat("median shortfall :", round(median(anom$gap), 1), "MW\n")
  cat("as % of demand   :", round(100 * max(anom$gap / anom$system_demand_actual_mw), 2), "%\n")
  cat("\nby year:\n")
  print(anom[, .(n = .N, max_gap_mw = round(max(gap), 1)),
             by = .(yr = substr(date, 1, 4))][order(yr)])
  cat("\nmost affected days:\n")
  print(head(anom[, .N, by = date][order(-N)], 5))
}

# ---- 2. outliers and jumps --------------------------------------------------
cat("\n=== outliers ===\n")
q <- quantile(hourly$demand_mw, c(0, .001, .01, .5, .99, .999, 1), na.rm = TRUE)
print(round(q, 0))
hourly[, jump := demand_mw - shift(demand_mw)]
cat("\nlargest hour-on-hour jumps (MW):\n")
print(head(hourly[order(-abs(jump)), .(datetime, demand_mw = round(demand_mw),
                                        jump = round(jump))], 5))

# ---- 3. plots ---------------------------------------------------------------
daily <- hourly[, .(mean_mw = mean(demand_mw, na.rm = TRUE)), by = date]
daily[, dt := as.Date(date)]
setorder(daily, dt)

png(file.path(fig_dir, "demand_daily_mean.png"), width = 1400, height = 600)
plot(daily$dt, daily$mean_mw, type = "l", col = "steelblue",
     xlab = "", ylab = "Mean demand (MW)",
     main = "Singapore system demand, daily mean, 2016-2026")
abline(v = as.Date(c("2020-04-07", "2020-06-01")), col = "red", lty = 2)
legend("topleft", "Circuit Breaker", col = "red", lty = 2, bty = "n")
dev.off()

png(file.path(fig_dir, "demand_diurnal.png"), width = 900, height = 600)
prof <- hourly[, .(mw = mean(demand_mw, na.rm = TRUE)), by = hour][order(hour)]
plot(prof$hour, prof$mw, type = "b", pch = 19, col = "steelblue",
     xlab = "Hour of day", ylab = "Mean demand (MW)",
     main = "Average diurnal demand profile")
dev.off()

png(file.path(fig_dir, "demand_circuit_breaker.png"), width = 1200, height = 600)
cb <- daily[dt >= as.Date("2020-01-01") & dt <= as.Date("2020-12-31")]
plot(cb$dt, cb$mean_mw, type = "l", col = "steelblue",
     xlab = "", ylab = "Mean demand (MW)", main = "2020: Circuit Breaker period")
abline(v = as.Date(c("2020-04-07", "2020-06-01")), col = "red", lty = 2)
dev.off()

cat("\nplots written to", fig_dir, "\n")
cat("  demand_daily_mean.png, demand_diurnal.png, demand_circuit_breaker.png\n")
