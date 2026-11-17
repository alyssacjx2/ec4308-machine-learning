# build_features.R ------------------------------------------------------------
# Imputes the weather gaps, then builds the model matrix for direct multi-horizon
# forecasting at h = 1, 6, 12, 24.
#
#   in : data/processed/hourly_dataset.csv.gz   (85,440 hours, from build_dataset.R)
#   out: data/processed/features.csv.gz
#
# Two jobs, in order, because lags are meaningless across holes:
#
#   1. IMPUTATION. Weather is 1.3-2.5% missing, concentrated in a few multi-day
#      blocks (worst: 696h of rainfall, Aug 2018) plus ~40 short runs each.
#      Short gaps are interpolated, long ones take a month-by-hour climatology,
#      and every filled cell is flagged so the models can discount it.
#
#   2. FEATURES. Calendar, demand lags and trailing means, weather and its lags,
#      ERA5 (already lagged 120h upstream), and phased Covid dummies.
#
# WHAT IS DELIBERATELY NOT HERE
#
#   * The 168 hour x day-of-week interaction dummies. Materialising them would
#     add 168 near-empty columns to every row; `hr` and `dow` are kept as
#     integers and model scripts expand them per window with model.matrix(), which
#     is cheaper and keeps this file committable.
#
#   * nem_forecast_mw as a predictor. It is EMA's own operational forecast and
#     the headline benchmark (RMSE 179 MW, MAPE 2.29%). Feeding it in as an
#     input would make "can ML beat EMA's forecast" unanswerable, so it travels
#     as nem_h<h> - the benchmark's prediction of the same target - and nothing
#     else. Add it to the feature set only as a separate, clearly-labelled run.

suppressMessages(library(data.table))
source("R/utils.R")

IN   <- "data/processed/hourly_dataset.csv.gz"
OUT  <- "data/processed/features.csv.gz"
TZ   <- "Asia/Singapore"
H    <- c(1L, 6L, 12L, 24L)   # forecast horizons
MAX_INTERP <- 6L              # gaps this long or shorter get interpolated
CDH_BASE   <- 24              # cooling-degree-hour base temperature, deg C

d <- read_gz(IN, colClasses = c(datetime = "character"))
setorder(d, datetime)
cat("loaded", format(nrow(d), big.mark = ","), "hours\n")

# Lags and trailing means below use shift()/frollmean(), which count ROWS, not
# clock time. That is only equivalent to counting hours if the spine is exactly
# contiguous, so check rather than assume.
step <- as.numeric(diff(as.POSIXct(d$datetime, tz = TZ)), units = "hours")
if (any(step != 1)) stop("spine is not contiguous: ", sum(step != 1), " bad steps")

d[, hr  := as.integer(substr(datetime, 12, 13))]
d[, mth := as.integer(substr(datetime,  6,  7))]

# ---- 1. imputation ----------------------------------------------------------
#
# Linear interpolation across a short gap in a smooth series is close to free.
# Across a 696-hour gap it would invent a month of weather, so long gaps instead
# take the mean for that month-and-hour, which carries the monsoon and diurnal
# cycles without inventing variation around them.
#
# Climatology is month x hour (288 cells, ~300 observations each) rather than
# day-of-year x hour (8,784 cells, ~10 observations each). Singapore's seasonal
# cycle is weak enough that the finer grid buys nothing but noise.
#
# LEAKAGE, stated plainly: the climatology is computed over the whole sample, so
# a gap in 2018 is filled using a mean that includes 2024. This is a real but
# contained in-sample leak - it touches only PREDICTORS, never the target
# (demand_mw has zero missing values). Nevertheless, future weather information
# can make forecasting accuracy optimistic through the imputed predictors. To
# remove it entirely, move this block inside the rolling-window loop in
# each model script and recompute climatology using available history only.
# Two-sided short-gap interpolation also uses future observations and would
# need a backward-looking replacement in a strictly real-time exercise.

# Fill only NA runs no longer than `maxlen`, and only where the run has real
# observations on both sides - an unbounded run at either end has nothing to
# interpolate between.
fill_short <- function(x, maxlen) {
  na <- is.na(x)
  if (!any(na) || all(na)) return(x)
  r <- rle(na)
  e <- cumsum(r$lengths); s <- e - r$lengths + 1L
  lin <- approx(which(!na), x[!na], xout = seq_along(x), rule = 1)$y
  for (k in which(r$values & r$lengths <= maxlen)) {
    if (s[k] == 1L || e[k] == length(x)) next
    x[s[k]:e[k]] <- lin[s[k]:e[k]]
  }
  x
}

# Fill whatever is still NA with the month-by-hour mean of the observed values.
# Written as an explicit keyed lookup returning a row-aligned vector, rather than
# a join-update: `get(v)` inside a `:=` in a join context resolves unpredictably.
fill_clim <- function(d, v) {
  x   <- d[[v]]
  obs <- !is.na(x)
  cl  <- d[obs, .(.cl = mean(.SD[[1L]])), by = .(mth, hr), .SDcols = v]
  setkey(cl, mth, hr)
  lk  <- cl[.(d$mth, d$hr), .cl]        # length nrow(d), in d's row order
  x[!obs] <- lk[!obs]
  d[, (v) := x]
  invisible(d)
}

# Rainfall is excluded from interpolation on purpose. It is zero in 68% of hours
# and arrives as short convective bursts, so there is no smooth path between two
# observations to interpolate along: straight-lining from 0 mm to 5 mm would
# fabricate two hours of moderate rain that did not happen. Every rainfall gap,
# short or long, takes the climatological mean instead.
imp_spec <- list(
  temp_c        = list(interp = TRUE,  circular = FALSE),
  humidity_pct  = list(interp = TRUE,  circular = FALSE),
  wind_speed_kt = list(interp = TRUE,  circular = FALSE),
  wind_dir_deg  = list(interp = TRUE,  circular = TRUE),
  rain_mm       = list(interp = FALSE, circular = FALSE)
)

cat("\nimputing:\n")
for (v in names(imp_spec)) {
  spec <- imp_spec[[v]]
  n0 <- sum(is.na(d[[v]]))
  d[, (paste0("imp_", v)) := as.integer(is.na(get(v)))]   # flag BEFORE filling

  if (spec$circular) {
    # Interpolating degrees would run the wrong way round the circle through
    # 0/360. Decompose, fill each component, recombine.
    d[, `:=`(.si = sin(get(v) * pi / 180), .co = cos(get(v) * pi / 180))]
    d[, `:=`(.si = fill_short(.si, MAX_INTERP), .co = fill_short(.co, MAX_INTERP))]
    fill_clim(d, ".si"); fill_clim(d, ".co")
    d[, (v) := (atan2(.si, .co) * 180 / pi + 360) %% 360]
    d[, c(".si", ".co") := NULL]
  } else {
    if (spec$interp) d[, (v) := fill_short(get(v), MAX_INTERP)]
    fill_clim(d, v)
  }

  cat(sprintf("  %-14s %5d NA -> %d remaining  (flagged in imp_%s)\n",
              v, n0, sum(is.na(d[[v]])), v))
}
d[, imp_any := as.integer(rowSums(.SD) > 0), .SDcols = patterns("^imp_")]
cat("  hours with any imputed weather:", sum(d$imp_any),
    sprintf("(%.2f%%)\n", 100 * mean(d$imp_any)))

# ---- 2. features ------------------------------------------------------------
f <- d[, .(datetime, y = demand_mw)]

# calendar. `dow`: 0 = Sunday. t_index is a linear trend - demand grew ~30%
# across the sample, though a 3-year rolling window absorbs most of that.
p <- as.POSIXct(d$datetime, tz = TZ)
f[, `:=`(hr      = d$hr,
         dow     = as.integer(format(p, "%w")),
         mth     = d$mth,
         doy     = as.integer(format(p, "%j")),
         year    = as.integer(substr(d$datetime, 1, 4)),
         t_index = seq_len(.N))]
f[, is_weekend    := as.integer(dow %in% c(0L, 6L))]
f[, is_holiday    := d$is_holiday]
f[, is_workday    := as.integer(is_weekend == 0L & is_holiday == 0L)]

# demand history. y itself is a legitimate predictor: standing at time t we know
# demand through t and forecast t+h. frollmean(align="right") likewise includes
# the current hour, so no window reaches into the future.
for (L in c(1L, 2L, 3L, 24L, 25L, 48L, 168L, 169L))
  f[, (sprintf("dem_lag%d", L)) := shift(y, L)]
for (W in c(3L, 24L, 168L))
  f[, (sprintf("dem_ma%d", W)) := frollmean(y, W, align = "right")]
f[, dem_d1  := y - dem_lag1]      # recent momentum
f[, dem_d24 := y - dem_lag24]     # change vs the same hour yesterday

# weather, contemporaneous. Wind direction enters as sin/cos: 359 deg and 1 deg
# are adjacent on the compass but maximally far apart as raw numbers.
f[, `:=`(temp_c        = d$temp_c,
         humidity_pct  = d$humidity_pct,
         wind_speed_kt = d$wind_speed_kt,
         wind_sin      = sin(d$wind_dir_deg * pi / 180),
         wind_cos      = cos(d$wind_dir_deg * pi / 180),
         rain_mm       = d$rain_mm)]

# Air-conditioning load responds to temperature non-linearly - demand barely
# moves below ~24 C and climbs steeply above it - so give the linear models a
# hinge and a square to work with. Trees would find this themselves.
f[, cdh     := pmax(temp_c - CDH_BASE, 0)]
f[, temp_sq := temp_c^2]

for (L in c(1L, 24L)) f[, (sprintf("temp_lag%d", L)) := shift(temp_c, L)]
for (L in c(1L, 2L, 3L)) f[, (sprintf("rain_lag%d", L)) := shift(rain_mm, L)]
f[, temp_ma24 := frollmean(temp_c, 24L, align = "right")]
f[, temp_ma72 := frollmean(temp_c, 72L, align = "right")]   # thermal inertia
f[, rain_ma24 := frollmean(rain_mm, 24L, align = "right")]

# ERA5, already lagged 120h in build_dataset.R. Converted to conventional units
# here: fluxes from J/m2 accumulated over the hour to mean W/m2, precipitation
# from metres to mm. One hour carries ssrd = -1.6 J/m2 from GRIB packing noise,
# so the solar terms are floored at zero.
f[, era5_ssrd_wm2 := pmax(d$era5_ssrd_lag120, 0) / 3600]
f[, era5_fdir_wm2 := pmax(d$era5_fdir_lag120, 0) / 3600]
f[, era5_sshf_wm2 := d$era5_sshf_lag120 / 3600]   # sign is meaningful, no floor
f[, era5_tcc      := d$era5_tcc_lag120]
f[, era5_tp_mm    := d$era5_tp_lag120 * 1000]

# Covid. The EDA showed a ~600 MW drop with GRADUAL recovery, so a single binary
# dummy would misdate the return to normal. These are the actual restriction
# regimes; the models decide which ones mattered.
dte <- as.Date(substr(d$datetime, 1, 10))
f[, cb_strict := as.integer(dte >= as.Date("2020-04-07") & dte <= as.Date("2020-06-01"))]
f[, cb_p1     := as.integer(dte >= as.Date("2020-06-02") & dte <= as.Date("2020-06-18"))]
f[, cb_p2     := as.integer(dte >= as.Date("2020-06-19") & dte <= as.Date("2020-12-27"))]
f[, cb_p3     := as.integer(dte >= as.Date("2020-12-28") & dte <= as.Date("2021-05-15"))]
f[, ha_2021   := as.integer((dte >= as.Date("2021-05-16") & dte <= as.Date("2021-06-13")) |
                            (dte >= as.Date("2021-07-22") & dte <= as.Date("2021-08-18")))]

# imputation flags, carried through so models can discount filled hours
for (v in c(paste0("imp_", names(imp_spec)), "imp_any")) f[, (v) := d[[v]]]

# ---- 3. targets and benchmark ----------------------------------------------
# Direct forecasting: features at t predict demand at t+h. nem_h<h> is EMA's own
# forecast OF THAT SAME TARGET HOUR, which is what the benchmark comparison
# needs. EMA issue times are unavailable: this is an operational reference,
# not evidence that EMA information was available at any particular origin.
for (h in H) {
  f[, (sprintf("y_h%d",   h)) := shift(d$demand_mw,      h, type = "lead")]
  f[, (sprintf("nem_h%d", h)) := shift(d$nem_forecast_mw, h, type = "lead")]
}

# The first 168 rows have incomplete lags and the last 24 have no y_h24. Flagged
# rather than dropped, so the row index still lines up with the hourly spine.
feat_cols <- setdiff(names(f), c("datetime", grep("^(y_h|nem_h)", names(f), value = TRUE)))
f[, usable := as.integer(complete.cases(f[, ..feat_cols]))]

cat("\n=== result ===\n")
cat("rows:", format(nrow(f), big.mark = ","), " cols:", ncol(f), "\n")
cat("usable rows (all features present):", format(sum(f$usable), big.mark = ","),
    sprintf("(%.2f%%)\n", 100 * mean(f$usable)))
cat("first usable:", f[usable == 1][1]$datetime, "\n")
cat("predictors (excl. datetime/targets/usable):", length(setdiff(feat_cols, "usable")), "\n")

na <- data.table(col = names(f), n = sapply(f, function(x) sum(is.na(x))))
cat("\ncolumns still holding NA (expect only lag/lead edges):\n")
print(na[n > 0][order(-n)])

cat("\nbenchmark on usable rows:\n")
for (h in H) {
  cmp <- f[usable == 1 & !is.na(get(sprintf("y_h%d", h))) &
             !is.na(get(sprintf("nem_h%d", h)))]
  yy <- cmp[[sprintf("y_h%d", h)]]; nn <- cmp[[sprintf("nem_h%d", h)]]
  cat(sprintf("  h=%2d  n=%s  EMA RMSE %6.1f MW  MAE %6.1f  MAPE %.3f%%\n",
              h, format(nrow(cmp), big.mark = ","), sqrt(mean((yy - nn)^2)),
              mean(abs(yy - nn)), 100 * mean(abs(yy - nn) / yy)))
}

fwrite(f, OUT)
cat("\nwritten:", OUT, "\n")
cat("NOTE: hour x day-of-week interactions are NOT materialised here - expand\n")
cat("      them per window in forecast.R with model.matrix(~ factor(hr):factor(dow)).\n")
