# aggregate_nea_hourly.R -------------------------------------------------------
# Collapses the raw NEA station files (minute-level, ~20-100 stations) to hourly
# readings per station, writing one gzipped file per variable-year into
# data/interim/.
#
# Design notes:
#
#   * We aggregate ALL stations, not just the fixed panel. Keeping the station
#     dimension means the spatial-aggregation rule stays a one-line decision in
#     build_dataset.R rather than something that needs this 12-minute job re-run.
#
#   * Hour bucketing uses the first 13 characters of the timestamp string
#     ("2017-04-15T19"), which IS the hour bucket in Singapore local time. This
#     avoids parsing ~440M datetimes and sidesteps the UTC trap: every reading
#     carries a +08:00 offset, so converting to UTC would shift everything by
#     eight hours. Singapore has no DST, so there are no ambiguous hours.
#
#   * Aggregation differs by variable:
#       - Rainfall is an accumulation over the interval  -> SUM within the hour
#       - Wind direction is circular                     -> vector mean
#       - Everything else is an instantaneous level      -> MEAN
#
#   * Relative humidity files have 11 columns (no reading_type) while the others
#     have 12. Selecting by name rather than position makes that a non-issue.
#
# Input : data/raw/nea/Historical<Var>acrossSingapore<YYYY>.csv.gz   (55 files)
# Output: data/interim/<var>_<year>_hourly.csv.gz
#         columns: datetime_hour, station_id, value, n_readings
#
# Requires R.utils (fread cannot read .gz without it).

suppressMessages({
  library(data.table)
})
source("R/utils.R")   # read_gz(): streams via gzip instead of memory-mapping

in_dir  <- "data/raw/nea"
out_dir <- "data/interim"
log_path <- "logs/aggregate_nea_log.csv"

for (d in c(out_dir, dirname(log_path))) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

# Wind direction is circular: the naive mean of 350 and 10 degrees is 180, which
# points the opposite way. Average the unit vectors instead.
circular_mean <- function(deg) {
  rad <- deg * pi / 180
  ang <- atan2(mean(sin(rad), na.rm = TRUE),
               mean(cos(rad), na.rm = TRUE)) * 180 / pi
  (ang + 360) %% 360
}

files <- sort(list.files(in_dir, pattern = "csv[.]gz$", full.names = TRUE))
if (length(files) == 0) stop("no input files found in ", in_dir)

cat("aggregating", length(files), "files to hourly x station\n\n")

log_rows <- vector("list", length(files))

for (i in seq_along(files)) {
  f  <- files[i]
  bn <- basename(f)

  variable <- sub("^Historical(.+?)acrossSingapore[0-9]{4}[.]csv[.]gz$", "\\1", bn)
  year     <- sub("^.*acrossSingapore([0-9]{4})[.]csv[.]gz$", "\\1", bn)
  var_slug <- tolower(gsub("([a-z])([A-Z])", "\\1_\\2", variable))

  out_file <- file.path(out_dir, sprintf("%s_%s_hourly.csv.gz", var_slug, year))

  # Resumable: a 12-minute job should not restart from scratch.
  if (file.exists(out_file)) {
    cat(sprintf("[%2d/%2d] skip  %-18s %s (already done)\n",
                i, length(files), variable, year))
    next
  }

  t0 <- Sys.time()

  # colClasses is essential: without it fread parses the ISO8601 timestamp to
  # POSIXct in UTC, shifting every reading 8 hours earlier. Keeping it as a
  # string means substr() below really does give the Singapore-local hour.
  dt <- read_gz(f, select = c("timestamp", "station_id", "reading_value"),
                colClasses = c(timestamp = "character"))

  # First 13 chars = "YYYY-MM-DDTHH" = the local-time hour bucket.
  dt[, hour_key := substr(timestamp, 1, 13)]

  hourly <- if (variable == "Rainfall") {
    dt[, .(value = sum(reading_value, na.rm = TRUE), n_readings = .N),
       by = .(hour_key, station_id)]
  } else if (variable == "WindDirection") {
    dt[, .(value = circular_mean(reading_value), n_readings = .N),
       by = .(hour_key, station_id)]
  } else {
    dt[, .(value = mean(reading_value, na.rm = TRUE), n_readings = .N),
       by = .(hour_key, station_id)]
  }

  # Parse to a real timestamp once, on the small aggregated table.
  hourly[, datetime_hour := as.POSIXct(paste0(sub("T", " ", hour_key), ":00:00"),
                                       tz = "Asia/Singapore")]
  hourly[, hour_key := NULL]
  setcolorder(hourly, c("datetime_hour", "station_id", "value", "n_readings"))
  setorder(hourly, datetime_hour, station_id)

  # Write local time as a plain string. fwrite() serialises POSIXct as ISO8601
  # in UTC by default ("2016-12-31T16:00:00Z" for midnight SGT), which reads
  # back 8 hours out. Hour-of-day is a model feature, so that offset would
  # silently corrupt the hour dummies and peak-period logic.
  hourly[, datetime_hour := format(datetime_hour, "%Y-%m-%d %H:%M:%S",
                                   tz = "Asia/Singapore")]

  fwrite(hourly, out_file)

  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  log_rows[[i]] <- data.table(
    variable    = variable,
    year        = year,
    raw_rows    = nrow(dt),
    hourly_rows = nrow(hourly),
    stations    = uniqueN(hourly$station_id),
    first_hour  = as.character(min(hourly$datetime_hour)),
    last_hour   = as.character(max(hourly$datetime_hour)),
    med_readings_per_hour = as.numeric(median(hourly$n_readings)),
    mins        = round(mins, 2)
  )

  cat(sprintf("[%2d/%2d] ok    %-18s %s  %9s -> %7s rows, %3d stations, %.1f min\n",
              i, length(files), variable, year,
              format(nrow(dt), big.mark = ","),
              format(nrow(hourly), big.mark = ","),
              uniqueN(hourly$station_id), mins))

  rm(dt, hourly); gc(verbose = FALSE)
}

log_dt <- rbindlist(Filter(Negate(is.null), log_rows))
if (nrow(log_dt) > 0) fwrite(log_dt, log_path)

cat("\ndone. ", nrow(log_dt), " files processed, log -> ", log_path, "\n", sep = "")
