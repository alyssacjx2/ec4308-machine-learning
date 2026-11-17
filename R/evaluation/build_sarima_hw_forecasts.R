# build_sarima_hw_forecasts.R ---------------------------------------------------
#
#   Packages the per-model SARIMA / Holt-Winters forecast files into the two archives
#   that combination_scenarios_and_gr.R reads. No model is estimated here: this only joins,
#   checks and renames what the model scripts already wrote.
#
#   in : output/forecasts/ (written by the model scripts)
#          sarima24_direct.csv.gz       sarima_direct.R --period=24         column sarima24d
#          sarima168_direct.csv.gz      sarima_direct.R --period=168        column sarima168d
#          sarima24_direct_hol.csv.gz   sarima_direct_holiday.R --period=24   column sarima24dh
#          sarima168_direct_hol.csv.gz  sarima_direct_holiday.R --period=168  column sarima168dh
#          holt_winters.csv.gz          holt_winters.R                      column hw24
#          holt_winters168.csv.gz       holt_winters.R --period=168         column hw168
#          holt_winters_hol.csv.gz      holt_winters_holiday.R              column hw24h
#          holt_winters168_hol.csv.gz   holt_winters_holiday.R --period=168 column hw168h
#          sarima.csv.gz                sarima.R (original recursive SARIMA; only for the second archive)
#   out: output/forecasts/sarima_hw_forecasts.zip
#          forecasts_without_holiday.csv  datetime,h,actual,sarima24_direct,sarima168_direct,hw24,hw168
#          forecasts_with_holiday.csv     datetime,h,actual,sarima24_direct_holiday,sarima168_direct_holiday,
#                                         hw24_holiday,hw168_holiday
#        output/forecasts/sarima_holt_winters_forecast_results.zip   (only if sarima.csv.gz is present)
#          sarima_holt_winters_forecast_results.csv  origin,target_datetime,h,period,actual,sarima,hw24
#
#   Run from the repository root, after the model scripts and before combination_scenarios_and_gr.R:
#       Rscript R/evaluation/build_sarima_hw_forecasts.R
#
# Rows are one per (forecast origin `datetime`, horizon `h`); `actual` is y_{t+h}. Period labels use
# TARGET time: validation 1 Apr-30 Sep 2025, test 1 Oct 2025-30 Sep 2026 (as in R/utils.R).

suppressPackageStartupMessages(library(data.table))
source("R/utils.R")

FC <- "output/forecasts"
H  <- c(1L, 6L, 12L, 24L)
cc <- c(datetime = "character")

# ---- 1. read the eight per-horizon model files --------------------------------------------
# file name -> (column written by the model script, column name expected by the scenario script)
spec <- list(
  without = list(
    list(file = "sarima24_direct.csv.gz",      from = "sarima24d",    to = "sarima24_direct"),
    list(file = "sarima168_direct.csv.gz",     from = "sarima168d",   to = "sarima168_direct"),
    list(file = "holt_winters.csv.gz",         from = "hw24",         to = "hw24"),
    list(file = "holt_winters168.csv.gz",      from = "hw168",        to = "hw168")),
  with = list(
    list(file = "sarima24_direct_hol.csv.gz",  from = "sarima24dh",   to = "sarima24_direct_holiday"),
    list(file = "sarima168_direct_hol.csv.gz", from = "sarima168dh",  to = "sarima168_direct_holiday"),
    list(file = "holt_winters_hol.csv.gz",     from = "hw24h",        to = "hw24_holiday"),
    list(file = "holt_winters168_hol.csv.gz",  from = "hw168h",       to = "hw168_holiday")))

read_model <- function(s) {
  path <- file.path(FC, s$file)
  if (!file.exists(path)) stop("Missing ", path, ". Run the model script listed in the header of this file.")
  x <- read_gz(path, colClasses = cc)
  need <- c("datetime", "h", "actual", s$from)
  if (!all(need %in% names(x))) stop(s$file, " lacks column(s): ", paste(setdiff(need, names(x)), collapse = ", "))
  if (anyDuplicated(x[, .(datetime, h)])) stop(s$file, ": duplicate (datetime, h) keys")
  if (anyNA(x[[s$from]])) stop(s$file, ": NA forecasts")
  if (!setequal(unique(x$h), H)) stop(s$file, ": horizons are not ", paste(H, collapse = "/"))
  x[, .SD, .SDcols = need][, setnames(.SD, s$from, s$to)]
}

# join a list of model files on (datetime, h); the actuals must agree to rounding error
join_models <- function(specs) {
  parts <- lapply(specs, read_model)
  keys <- parts[[1]][, .(datetime, h)]
  out <- copy(parts[[1]])
  for (i in seq_along(parts)[-1]) {
    p <- parts[[i]]
    if (nrow(p) != nrow(out) || !identical(sort(paste(p$datetime, p$h)), sort(paste(out$datetime, out$h))))
      stop(specs[[i]]$file, ": origins/horizons differ from ", specs[[1]]$file)
    m <- merge(out[, .(datetime, h, actual)], p[, .(datetime, h, actual_b = actual)], by = c("datetime", "h"))
    if (max(abs(m$actual - m$actual_b)) > 1e-6) stop(specs[[i]]$file, ": actual demand differs from ", specs[[1]]$file)
    out <- merge(out, p[, !"actual"], by = c("datetime", "h"))
  }
  setorder(out, h, datetime)
  out
}

nh <- join_models(spec$without)
wh <- join_models(spec$with)
if (!identical(nh[, .(datetime, h)], wh[, .(datetime, h)]) || max(abs(nh$actual - wh$actual)) > 1e-6)
  stop("The with- and without-holiday tables are not on the same origins / actuals.")

# ---- 2. coverage report ---------------------------------------------------------------------
target <- format(as.POSIXct(nh$datetime, tz = "Asia/Singapore") + nh$h * 3600, "%Y-%m-%d %H:%M:%S", tz = "Asia/Singapore")
per <- fifelse(target < "2025-04-01 00:00:00", "pre_validation", fifelse(target < "2025-10-01 00:00:00", "validation", "test"))
cat("rows:", nrow(nh), "\n"); print(dcast(data.table(h = nh$h, period = per)[, .N, by = .(h, period)], h ~ period, value.var = "N"))
n_test <- data.table(h = nh$h, per)[per == "test", .N, by = h]
if (any(n_test$N != 8760L)) stop("Test period must have 8,760 target hours per horizon; found ", paste(n_test$N, collapse = "/"))

# ---- 3. write the archives ------------------------------------------------------------------
make_zip <- function(zipfile, files) {
  if (file.exists(zipfile)) file.remove(zipfile)
  zipfile <- file.path(normalizePath(dirname(zipfile)), basename(zipfile))
  if (requireNamespace("zip", quietly = TRUE)) {
    zip::zipr(zipfile, files, mode = "cherry-pick")
  } else if (nzchar(Sys.which("zip"))) {
    utils::zip(zipfile, files, flags = "-j -q")
  } else stop("Neither the 'zip' package nor a zip command is available.")
}
tmp <- tempfile("sarima-hw-"); dir.create(tmp)
f1 <- file.path(tmp, "forecasts_without_holiday.csv"); f2 <- file.path(tmp, "forecasts_with_holiday.csv")
fwrite(nh[, .(datetime, h, actual, sarima24_direct, sarima168_direct, hw24, hw168)], f1)
fwrite(wh[, .(datetime, h, actual, sarima24_direct_holiday, sarima168_direct_holiday, hw24_holiday, hw168_holiday)], f2)
make_zip(file.path(FC, "sarima_hw_forecasts.zip"), c(f1, f2))
cat("written:", file.path(FC, "sarima_hw_forecasts.zip"), "\n")

# second archive: original recursive SARIMA (column `sarima`) and Holt-Winters (24 h) with period labels
if (all(file.exists(file.path(FC, c("sarima.csv.gz", "holt_winters.csv.gz"))))) {
  sh <- read_seasonal_forecasts(FC)            # freshly generated files take precedence (see R/utils.R)
  setcolorder(sh, c("origin", "target_datetime", "h", "period", "actual", "sarima", "hw24"))
  setorder(sh, h, origin)                      # same row order as the other archive: by horizon, then origin
  f3 <- file.path(tmp, "sarima_holt_winters_forecast_results.csv")
  fwrite(sh, f3)
  make_zip(file.path(FC, "sarima_holt_winters_forecast_results.zip"), f3)
  cat("written:", file.path(FC, "sarima_holt_winters_forecast_results.zip"), "\n")
} else {
  cat("sarima.csv.gz not found: sarima_holt_winters_forecast_results.zip left unchanged.\n")
}
unlink(tmp, recursive = TRUE)
