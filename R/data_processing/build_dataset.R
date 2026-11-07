# build_dataset.R --------------------------------------------------------------
# Merges all four sources onto one hourly spine and writes the modelling dataset.
#
#   in : data/interim/*_hourly.csv.gz     (NEA, hourly x station)
#        data/raw/ema/*.csv.gz            (half-hourly demand)
#        data/raw/era5/*.nc               (hourly x 12 grid cells)
#        data/raw/holidays/*.csv          (one row per holiday date)
#        data/processed/station_panel.csv (which stations to use)
#
#   out: data/processed/hourly_dataset.csv.gz
#
# Sample: 2017-01-01 to 2026-09-30 = 85,440 hours. Start is Jan 2017 because
# usable rainfall only begins then (the 2016 file holds 2-3 Dec only).
#
# Timezone discipline: every timestamp in this project is Singapore local time,
# stored as a plain "YYYY-MM-DD HH:MM:SS" string. Both fread and fwrite will
# silently convert POSIXct to UTC given half a chance, which shifts everything
# 8 hours. Parse explicitly, write as character, never let them guess.

suppressMessages({
  library(data.table)
  library(ncdf4)
})
source("R/utils.R")

TZ        <- "Asia/Singapore"
START     <- as.POSIXct("2017-01-01 00:00:00", tz = TZ)
END       <- as.POSIXct("2026-09-30 23:00:00", tz = TZ)
ERA5_LAG  <- 120L   # hours. ERA5 is published ~5 days late, so a real-time
                    # forecast at time t can only see ERA5 from t-120h.
OUT       <- "data/processed/hourly_dataset.csv.gz"

if (!dir.exists("data/processed")) dir.create("data/processed", recursive = TRUE)

fmt <- function(x) format(x, "%Y-%m-%d %H:%M:%S", tz = TZ)

# Circular mean, for averaging wind direction across stations. The naive mean of
# 350 and 10 degrees is 180, which points the opposite way.
circular_mean <- function(deg) {
  rad <- deg * pi / 180
  ang <- atan2(mean(sin(rad), na.rm = TRUE),
               mean(cos(rad), na.rm = TRUE)) * 180 / pi
  (ang + 360) %% 360
}

# ---- 1. hourly spine ---------------------------------------------------------
# Everything left-joins onto this, so missing source hours surface as NA rather
# than as absent rows. That matters: lags assume contiguous hours, and a missing
# row would make a "24-hour lag" silently reach further back than 24 hours.
spine <- data.table(dt = seq(START, END, by = "hour"))
spine[, datetime := fmt(dt)]
cat("spine:", format(nrow(spine), big.mark = ","), "hours",
    min(spine$datetime), "->", max(spine$datetime), "\n")

# ---- 2. EMA demand -----------------------------------------------------------
# Hour comes from `period` (1-48), NOT datetime_end: period 48 ends at 00:00 of
# the following day, so grouping on datetime_end would push the last half hour
# of every day into the wrong date.
#   period 1 ends 00:30 -> hour 0 ;  period 48 ends 00:00 -> hour 23
cat("\nEMA demand ...\n")
ema <- rbindlist(lapply(
  list.files("data/raw/ema", pattern = "csv[.]gz$", full.names = TRUE),
  read_gz, colClasses = c(date = "character")))
setnames(ema, tolower(names(ema)))

ema[, hour := (period - 1L) %/% 2L]
ema[, datetime := paste0(date, " ", sprintf("%02d", hour), ":00:00")]

demand <- ema[, .(demand_mw       = mean(system_demand_actual_mw, na.rm = TRUE),
                  nem_forecast_mw = mean(nem_demand_forecast_mw,  na.rm = TRUE)),
              by = datetime]
cat("  ", format(nrow(demand), big.mark = ","), "hourly rows\n")

# ---- 3. NEA weather ----------------------------------------------------------
# Collapse hourly x station down to one Singapore-wide value per hour, using the
# fixed panel so the series means the same thing in 2017 as in 2026 (the raw
# station set swings between 12 and 35 over the sample).
cat("\nNEA weather ...\n")
panel      <- fread("data/processed/station_panel.csv")
core_ids   <- panel[in_core_panel     == TRUE]$station_id
rain_ids   <- panel[in_rainfall_panel == TRUE]$station_id

# Rainfall gets its own, larger panel: rain is far more local than temperature,
# and ~3x as many gauges report consistently.
nea_spec <- list(
  air_temperature   = list(col = "temp_c",       ids = core_ids, fun = "mean"),
  relative_humidity = list(col = "humidity_pct", ids = core_ids, fun = "mean"),
  wind_speed        = list(col = "wind_speed_kt",ids = core_ids, fun = "mean"),
  wind_direction    = list(col = "wind_dir_deg", ids = core_ids, fun = "circular"),
  # Values are already hourly SUMS per station; across stations we AVERAGE.
  # Summing both ways would inflate rainfall ~35x.
  rainfall          = list(col = "rain_mm",      ids = rain_ids, fun = "mean")
)

nea_list <- list()
for (v in names(nea_spec)) {
  spec <- nea_spec[[v]]
  fs <- list.files("data/interim", pattern = sprintf("^%s_[0-9]{4}_hourly", v),
                   full.names = TRUE)
  d <- rbindlist(lapply(fs, read_gz, colClasses = c(datetime_hour = "character")))
  d <- d[station_id %in% spec$ids]

  agg <- if (spec$fun == "circular") {
    d[, .(val = circular_mean(value), n_st = .N), by = datetime_hour]
  } else {
    d[, .(val = mean(value, na.rm = TRUE), n_st = .N), by = datetime_hour]
  }
  setnames(agg, c("datetime", spec$col, paste0("n_st_", spec$col)))
  nea_list[[v]] <- agg
  cat(sprintf("  %-18s %s hours, %d stations in panel\n",
              v, format(nrow(agg), big.mark = ","), length(spec$ids)))
}

nea <- Reduce(function(a, b) merge(a, b, by = "datetime", all = TRUE), nea_list)

# ---- 4. ERA5 -----------------------------------------------------------------
# valid_time is seconds since 1970 in UTC. Convert to Singapore time before
# merging, then lag by ERA5_LAG hours to respect the publication delay.
cat("\nERA5 ...\n")
nc <- nc_open("data/raw/era5/era5_singapore_2016_2026.nc")
era_vars <- c("ssrd", "sshf", "tcc", "tp", "fdir")

vt  <- ncvar_get(nc, "valid_time")
utc <- as.POSIXct(vt, origin = "1970-01-01", tz = "UTC")

era <- data.table(datetime = fmt(as.POSIXct(format(utc, tz = TZ), tz = TZ)))
for (v in era_vars) {
  a <- ncvar_get(nc, v)
  # ncvar_get returns dimensions in FILE order, which for ERA5 is
  # [longitude, latitude, valid_time] - the reverse of how the CDL header
  # prints them ("ssrd(valid_time, latitude, longitude)"). Averaging over the
  # wrong axis yields 4 numbers that data.table silently recycles 23,556 times,
  # producing a constant column that looks like plausible data. Find the time
  # axis by name and assert its length.
  dn    <- sapply(nc$var[[v]]$dim, function(d) d$name)
  t_ax  <- which(dn == "valid_time")
  stopifnot(length(t_ax) == 1L, dim(a)[t_ax] == length(vt))
  era[[paste0("era5_", v)]] <- apply(a, t_ax, mean, na.rm = TRUE)  # mean over 12 cells
}
nc_close(nc)
cat("  ", format(nrow(era), big.mark = ","), "hours,",
    length(era_vars), "variables averaged over 3x4 grid\n")

# Lag by joining on a shifted key rather than shift(): robust to any gaps.
era[, join_dt := fmt(as.POSIXct(datetime, tz = TZ) + ERA5_LAG * 3600)]
era_lag <- era[, c("join_dt", paste0("era5_", era_vars)), with = FALSE]
setnames(era_lag, "join_dt", "datetime")
setnames(era_lag, paste0("era5_", era_vars),
         paste0("era5_", era_vars, "_lag", ERA5_LAG))
cat("   lagged", ERA5_LAG, "hours\n")

# ---- 5. holidays -------------------------------------------------------------
# Both gazetted and observed days are non-working, so both count. The observed
# (in-lieu) Monday is often the one that actually moves load.
cat("\nholidays ...\n")
hol <- fread("data/raw/holidays/sg_public_holidays_2016_2026.csv",
             colClasses = c(date = "character"))
hol_hours <- CJ(date = unique(hol$date), hour = 0:23)
hol_hours[, datetime := paste0(date, " ", sprintf("%02d", hour), ":00:00")]
hol_hours <- merge(hol_hours, unique(hol[, .(date, holiday, type)]),
                   by = "date", all.x = TRUE)
hol_hours[, is_holiday := 1L]
cat("  ", uniqueN(hol$date), "holiday dates ->",
    format(nrow(hol_hours), big.mark = ","), "hours\n")

# ---- 6. merge ----------------------------------------------------------------
cat("\nmerging ...\n")
dt <- merge(spine[, .(datetime)], demand,  by = "datetime", all.x = TRUE)
dt <- merge(dt, nea,     by = "datetime", all.x = TRUE)
dt <- merge(dt, era_lag, by = "datetime", all.x = TRUE)
dt <- merge(dt, hol_hours[, .(datetime, holiday, holiday_type = type, is_holiday)],
            by = "datetime", all.x = TRUE)
dt[is.na(is_holiday), is_holiday := 0L]
setorder(dt, datetime)

# ---- 7. report ---------------------------------------------------------------
cat("\n=== result ===\n")
cat("rows:", format(nrow(dt), big.mark = ","),
    " (expected", format(nrow(spine), big.mark = ","), ")\n")
cat("cols:", ncol(dt), "\n")
cat("range:", min(dt$datetime), "->", max(dt$datetime), "\n")

na_tab <- data.table(column = names(dt),
                     n_na   = sapply(dt, function(x) sum(is.na(x))))
na_tab[, pct := round(100 * n_na / nrow(dt), 2)]
cat("\nmissing values:\n")
print(na_tab[n_na > 0][order(-n_na)])
if (na_tab[n_na > 0, .N] == 0) cat("  none\n")

fwrite(dt, OUT)
cat("\nwritten:", OUT, "\n")
cat("NOTE: NAs are NOT imputed here. rlasso/ranger reject NAs, so decide an\n")
cat("      imputation rule in build_features.R before constructing lags.\n")
