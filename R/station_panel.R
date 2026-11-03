# Station panel analysis -------------------------------------------------------
# For every NEA variable-year file, count readings per station. Then work out
# which stations report consistently across the whole sample, so the group can
# decide on a fixed panel vs taking all available stations.
#
# Sample starts Jan 2017 (group decision), so 2016 is reported but excluded from
# the "present in every year" test.

suppressMessages(library(data.table))
source("R/utils.R")   # read_gz(): streams via gzip instead of memory-mapping

# Create output dirs defensively - they have been removed by zipping before.
for (d in c("data/interim", "data/processed")) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

files <- list.files("data/raw/nea", pattern = "csv[.]gz$", full.names = TRUE)
cat("scanning", length(files), "files...\n\n")

inv <- vector("list", length(files))
meta <- list()

for (i in seq_along(files)) {
  f  <- files[i]
  bn <- basename(f)
  variable <- sub("^Historical(.+?)acrossSingapore\\d{4}\\.csv\\.gz$", "\\1", bn)
  year     <- as.integer(sub("^.*acrossSingapore(\\d{4})\\.csv\\.gz$", "\\1", bn))

  dt <- read_gz(f, select = c("station_id", "station_name",
                              "location_latitude", "location_longitude"))

  inv[[i]] <- dt[, .(n_readings = .N), by = station_id][
    , `:=`(variable = variable, year = year)]

  # keep one row of metadata per station (first seen wins)
  m <- unique(dt, by = "station_id")[, .(station_id, station_name,
                                         lat = location_latitude,
                                         lon = location_longitude)]
  meta[[length(meta) + 1]] <- m

  cat(sprintf("  [%2d/%2d] %-18s %d  %2d stations, %s rows\n",
              i, length(files), variable, year,
              nrow(inv[[i]]), format(nrow(dt), big.mark = ",")))
  rm(dt); gc(verbose = FALSE)
}

inv  <- rbindlist(inv)
meta <- unique(rbindlist(meta), by = "station_id")

fwrite(inv,  "data/interim/station_inventory.csv")
fwrite(meta, "data/interim/station_metadata.csv")

SAMPLE_YEARS <- 2017:2026
n_years <- length(SAMPLE_YEARS)

cat("\n================ PER-VARIABLE ================\n")
for (v in sort(unique(inv$variable))) {
  sub <- inv[variable == v & year %in% SAMPLE_YEARS]
  per <- sub[, .(years_present = uniqueN(year),
                 total_readings = sum(n_readings)), by = station_id]
  full <- per[years_present == n_years]
  cat(sprintf("\n%-18s  %2d stations ever | %2d present in all %d years\n",
              v, nrow(per), nrow(full), n_years))
  cat("   stations per year: ")
  cat(paste(sub[, .(n = uniqueN(station_id)), by = year][order(year)]$n, collapse = " "))
  cat("\n")
}

cat("\n================ FIXED PANEL CANDIDATES ================\n")
# stations present in every year AND for every variable
per_var <- inv[year %in% SAMPLE_YEARS,
               .(years_present = uniqueN(year)), by = .(variable, station_id)]
full_by_var <- per_var[years_present == n_years]
n_vars <- uniqueN(inv$variable)
panel <- full_by_var[, .(n_variables = .N), by = station_id][n_variables == n_vars]

cat(sprintf("\nStations reporting ALL %d variables in ALL %d years: %d\n\n",
            n_vars, n_years, nrow(panel)))

if (nrow(panel) > 0) {
  out <- merge(panel, meta, by = "station_id")
  tot <- inv[year %in% SAMPLE_YEARS & station_id %in% panel$station_id,
             .(readings = sum(n_readings)), by = station_id]
  out <- merge(out, tot, by = "station_id")[order(-readings)]
  print(out[, .(station_id, station_name, lat, lon,
                readings = format(readings, big.mark = ","))])
} else {
  cat("None. Fixed panel not viable across all variables - consider per-variable panels.\n")
}

cat("\n================ PER-VARIABLE PANEL SIZES ================\n")
print(full_by_var[, .(stations_in_all_years = .N), by = variable][order(variable)])

# ---- Write the panel as an explicit artifact --------------------------------
# build_dataset.R reads this rather than hardcoding station codes, so the panel
# choice is visible, version-controlled, and changeable in one place.
#
# Two panels, because rainfall behaves differently: rain is far more spatially
# variable than temperature, and there are ~3x as many gauges reporting
# consistently, so restricting rainfall to the core panel would throw away real
# information.

core_ids <- panel$station_id
rain_ids <- full_by_var[variable == "Rainfall"]$station_id

all_ids <- union(core_ids, rain_ids)
out_panel <- meta[station_id %in% all_ids][
  , `:=`(in_core_panel     = station_id %in% core_ids,
         in_rainfall_panel = station_id %in% rain_ids)]
setorder(out_panel, -in_core_panel, station_id)

fwrite(out_panel, "data/processed/station_panel.csv")

cat(sprintf("\ncore panel     (all %d variables, all %d years): %d stations\n",
            n_vars, n_years, length(core_ids)))
cat(sprintf("rainfall panel (rainfall, all %d years)       : %d stations\n",
            n_years, length(rain_ids)))

cat("\nwritten:\n")
cat("  data/interim/station_inventory.csv\n")
cat("  data/interim/station_metadata.csv\n")
cat("  data/processed/station_panel.csv   <- read this in build_dataset.R\n")
