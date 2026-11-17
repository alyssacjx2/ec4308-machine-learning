# utils.R ----------------------------------------------------------------------
# Shared helpers. Source this at the top of other scripts:
#   source("R/utils.R")

suppressMessages(library(data.table))

# Locate gzip -----------------------------------------------------------------
# gzip ships with Git for Windows but is usually NOT on PATH in PowerShell, so
# Sys.which() alone is unreliable. Check PATH first, then the usual Git
# locations.
find_gzip <- function() {
  g <- Sys.which("gzip")
  if (nzchar(g)) return(unname(g))

  candidates <- c(
    file.path(Sys.getenv("LOCALAPPDATA"), "Programs", "Git", "usr", "bin", "gzip.exe"),
    "C:/Program Files/Git/usr/bin/gzip.exe",
    "C:/Program Files (x86)/Git/usr/bin/gzip.exe"
  )
  hit <- candidates[file.exists(candidates)]
  if (length(hit) > 0) return(hit[1])

  stop("gzip not found. Install Git for Windows, or add gzip to PATH.")
}

# Read a gzipped CSV by streaming ----------------------------------------------
# fread(file.csv.gz) decompresses to a temp file and MEMORY-MAPS it, which needs
# a contiguous block the size of the decompressed data (~1 GB for these files).
# On a 6 GB machine that fails after a few files as the address space fragments:
#   "could not memory map it ... not enough contiguous virtual memory"
#
# Piping through gzip streams instead: ~200 MB of R memory regardless of file
# size, at the cost of being a little slower.
read_gz <- function(path, select = NULL, colClasses = NULL) {
  if (!file.exists(path)) stop("Input file not found: ", path)
  gz <- find_gzip()
  cmd <- paste(shQuote(gz), "-dc", shQuote(normalizePath(path, winslash = "/")))
  fread(cmd = cmd, select = select, colClasses = colClasses, showProgress = FALSE)
}

# CPU detection can return NA in restricted sessions. Keep model scripts runnable
# there, and use N_CORES only for parallel scheduling, not model specification.
forecast_cores <- function() {
  detected <- parallel::detectCores()
  default <- if (length(detected) == 1L && is.finite(detected))
    max(1L, detected - 1L) else 1L
  value <- suppressWarnings(as.numeric(Sys.getenv("N_CORES", default)))
  if (length(value) != 1L || !is.finite(value) || value < 1 ||
      value != floor(value) || value > .Machine$integer.max)
    stop("N_CORES must be a positive integer.")
  as.integer(value)
}

# Bridge model outputs to evaluation without requiring a manually assembled ZIP.
# Fresh sarima.R / holt_winters.R outputs take priority when BOTH are present.
# Otherwise accept the supplied legacy combined CSV or ZIP. Both routes return
# the same column schema. Periods are assigned by TARGET time, not origin time.
read_seasonal_forecasts <- function(directory = "output/forecasts") {
  sarima_path <- file.path(directory, "sarima.csv.gz")
  hw_path <- file.path(directory, "holt_winters.csv.gz")
  generated <- file.exists(c(sarima_path, hw_path))
  if (any(generated) && !all(generated))
    stop("Run both R/models/sarima.R and R/models/holt_winters.R before evaluation.")
  if (all(generated)) {
    arima <- read_gz(sarima_path, colClasses = c(datetime = "character"))
    hw <- read_gz(hw_path, colClasses = c(datetime = "character"))
    for (x in list(arima, hw)) {
      if (anyDuplicated(x[, .(datetime, h)])) stop("Duplicate seasonal forecast keys.")
    }
    result <- merge(arima[, .(datetime, h, actual, sarima)],
                    hw[, .(datetime, h, actual_hw = actual, hw24)],
                    by = c("datetime", "h"))
    if (nrow(result) != nrow(arima) || nrow(result) != nrow(hw) ||
        any(!is.finite(result$actual)) || any(!is.finite(result$actual_hw)) ||
        any(abs(result$actual - result$actual_hw) > 1e-6))
      stop("Seasonal forecast coverage or actual demand does not match.")
    result[, actual_hw := NULL]
    result[, target_datetime := format(as.POSIXct(datetime, tz = "Asia/Singapore") +
      h * 3600, "%Y-%m-%d %H:%M:%S", tz = "Asia/Singapore")]
    result[, period := fifelse(target_datetime < "2025-04-01 00:00:00", "pre_validation",
      fifelse(target_datetime < "2025-10-01 00:00:00", "validation", "test"))]
    setnames(result, "datetime", "origin")
    message("Seasonal inputs: freshly generated sarima.csv.gz and holt_winters.csv.gz")
    return(result)
  }
  combined <- file.path(directory, "sarima_holt_winters_forecast_results.csv")
  if (!file.exists(combined)) {
    archive <- sub("[.]csv$", ".zip", combined)
    if (!file.exists(archive)) stop("Missing seasonal forecasts. Run the SARIMA and ",
      "Holt-Winters scripts or supply ", archive)
    temporary <- tempfile("seasonal-input-"); dir.create(temporary)
    on.exit(unlink(temporary, recursive = TRUE), add = TRUE)
    unzip(archive, files = basename(combined), exdir = temporary)
    combined <- file.path(temporary, basename(combined))
  }
  fread(combined, colClasses = c(origin = "character", target_datetime = "character"))
}

# IMPORTANT when reading raw NEA files -----------------------------------------
# The raw timestamp looks like "2017-01-01T00:00:59+08:00". fread recognises
# ISO8601 and silently parses it to POSIXct *in UTC*, so that value comes back as
# "2016-12-31 16:00:59" - the whole series shifted 8 hours earlier.
#
# Always force it to stay a string:
#   read_gz(f, select = c("timestamp", ...), colClasses = c(timestamp = "character"))
#
# then the first 13 characters ("2017-01-01T00") are the Singapore-local hour.
