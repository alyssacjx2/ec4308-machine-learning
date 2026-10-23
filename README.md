# Forecasting Singapore's hourly electricity demand

EC4308 Machine Learning and Economic Forecasting — Group 10.

Direct multi-horizon forecasts of Singapore system demand at **h = 1, 6, 12 and
24 hours**, comparing regularised linear methods and tree ensembles against
time-series benchmarks on a 3-year rolling window with weekly refitting.

**EMA's own operational forecast** sets the reference level at RMSE 179.4 MW /
MAPE 2.288%. Read that number carefully: EMA publishes one forecast per period
and the data does not record how far ahead it was made, so it is a **near-term**
operational forecast, not a 24-hour-ahead one. Scored against `y_h<h>` it returns
the identical RMSE at every horizon, because each horizon is the same
(actual, forecast) series merely shifted.

So it is a reference level — "the accuracy achievable with near-real-time
information" — and **not** a horizon-matched benchmark. Beating it at h = 1 is a
fair fight; losing to it at h = 24 says nothing about the model. The
horizon-matched comparators are **seasonal-naive and AR(p) by BIC**, which
degrade with h the way our models do. Stating EMA's 179 MW as a 24-hour-ahead
benchmark in the report would invite a fair criticism.

---

## Repository layout

```
R/                  pipeline scripts, run in the order below
docs/
  DATA.md           the four sources, coverage, conventions, 13 known quirks
  nea_dataset_manifest.json
data/               NOT in the repo, except as noted in DATA.md
  raw/              as downloaded
  interim/          hourly x station NEA aggregates (generated)
  processed/        merged dataset and model matrix (generated)
logs/               stdout from the long extraction and aggregation runs
output/figures/     plots
```

**Read [`docs/DATA.md`](docs/DATA.md) before touching the data.** Its "Known
quirks" section records the traps that have already cost time: the eight-hour
UTC shift, the netCDF axis order that silently produces constant columns, and
the station panel changing composition mid-sample.

## Setup

R 4.4.2. Not on PATH by default on Windows:

```powershell
$env:Path += ";$env:LOCALAPPDATA\Programs\R\R-4.4.2\bin\x64"
```

```r
install.packages(c("data.table", "R.utils", "ncdf4", "hdm", "ranger"))
```

`R.utils` is required — `data.table::fread()` cannot read `.gz` without it.
`R/utils.R` also needs **gzip**, which it locates automatically from a Git for
Windows install; it streams compressed files rather than memory-mapping them,
which is what makes the 44 GB of raw NEA data tractable on 6 GB of RAM.

## Pipeline

Run from the project root. Steps 1–2 are slow and only need running once —
their outputs are committed, so most people can start at step 3.

| # | Script | Does | Time |
|---|---|---|---|
| 1 | `station_panel.R` | Picks the fixed station panel → `data/processed/station_panel.csv` | ~1 min |
| 2 | `aggregate_nea_hourly.R` | 55 raw NEA files → hourly × station in `data/interim/` | ~12 min |
| 3 | `build_dataset.R` | Merges all four sources onto one hourly spine | ~3 min |
| 4 | `build_features.R` | Imputes weather gaps, builds the model matrix | ~1 min |
| 5 | `forecast.R` | Rolling-window engine — **not written yet** | — |
| 6 | `evaluate.R` | RMSE/MAE/MAPE, Diebold-Mariano, combination — **not written yet** | — |

```bash
Rscript R/build_dataset.R
Rscript R/build_features.R
```

`aggregate_nea_hourly.R` is resumable — it skips variable-years already present
in `data/interim/`, so an interrupted run can simply be restarted.

`explore_demand.R` is standalone EDA on the demand series and is not part of the
pipeline.

## Dataset

`build_dataset.R` produces **85,440 contiguous hours, 2017-01-01 to 2026-09-30**.
The sample starts in 2017 because usable rainfall does (the 2016 NEA rainfall
file holds only 2–3 December).

| Source | Contributes |
|---|---|
| EMA | `demand_mw` (target), `nem_forecast_mw` (benchmark) |
| NEA stations | temperature, humidity, wind speed/direction, rainfall |
| ERA5 | solar radiation, sensible heat flux, cloud cover, precipitation — **lagged 120h** |
| MOM/time.sg | public holidays, gazetted vs observed |

The **120-hour ERA5 lag** is deliberate: ERA5 reanalysis publishes about five
days late, so a model standing at time *t* genuinely cannot see it. Set
`ERA5_LAG <- 0L` in `build_dataset.R` to treat it as contemporaneous — defensible
for an ex-post study, but it would overstate real-time accuracy.

`build_features.R` then adds calendar terms, demand lags and trailing means,
cooling-degree-hours, weather lags, and phased Covid dummies. Two choices worth
knowing about:

- **Hour × day-of-week interactions are not materialised.** 168 near-empty
  columns per row is wasteful; `hr` and `dow` stay as integers and `forecast.R`
  expands them per window with `model.matrix()`.
- **`nem_forecast_mw` is not a predictor.** It is the headline benchmark, so
  feeding it in as an input would make "can ML beat EMA's forecast"
  unanswerable. It travels as `nem_h<h>` — the benchmark's own prediction of the
  same target hour — and nothing else.

Weather gaps (1.3–2.5%, concentrated in a few multi-day NEA outages) are
interpolated when ≤ 6 hours, filled with a month-by-hour climatology when longer,
and flagged in `imp_<var>` either way. The target has no missing hours at all, so
no fill rule can leak into forecast accuracy through `y`. Reasoning and the
bounded in-sample leak are documented in the script header.

## What's in `features.csv.gz`

85,440 rows × 65 columns. **`forecast.R` loads this file and nothing else.**

| Group | Columns |
|---|---|
| Key | `datetime` (Asia/Singapore local string), `y` (demand MW at t) |
| Targets | `y_h1`, `y_h6`, `y_h12`, `y_h24` — demand at t+h |
| Benchmark | `nem_h1/6/12/24` — EMA's forecast of that same target hour |
| Calendar | `hr`, `dow` (0 = Sun), `mth`, `doy`, `year`, `t_index`, `is_weekend`, `is_holiday`, `is_workday` |
| Demand history | `dem_lag1/2/3/24/25/48/168/169`, `dem_ma3/24/168`, `dem_d1`, `dem_d24` |
| Weather | `temp_c`, `humidity_pct`, `wind_speed_kt`, `wind_sin`, `wind_cos`, `rain_mm` |
| Weather derived | `cdh` (cooling-degree-hours, base 24 °C), `temp_sq`, `temp_lag1/24`, `rain_lag1/2/3`, `temp_ma24/72`, `rain_ma24` |
| ERA5 (lagged 120h) | `era5_ssrd_wm2`, `era5_fdir_wm2`, `era5_sshf_wm2`, `era5_tcc`, `era5_tp_mm` |
| Covid | `cb_strict`, `cb_p1`, `cb_p2`, `cb_p3`, `ha_2021` — mutually exclusive |
| Imputation flags | `imp_temp_c`, `imp_humidity_pct`, `imp_wind_speed_kt`, `imp_wind_dir_deg`, `imp_rain_mm`, `imp_any` |
| Row filter | `usable` |

54 predictors in total, before the hour × day-of-week interactions are expanded.

**Filter on `usable == 1 & !is.na(y_h<h>)`.** `usable` covers features only — it
excludes the first 169 hours, whose lags are incomplete — while `y_h24` is also
NA for the final 24 rows. Usable counts run 85,270 at h = 1 down to 85,247 at
h = 24.

Two things to know when reading coefficients:

- `wind_dir_deg` is **not** a column. It enters as `wind_sin`/`wind_cos`, because
  359° and 1° are adjacent on the compass but maximally far apart as raw numbers.
  `imp_wind_dir_deg` still flags where it was filled.
- **Climatological rainfall is never exactly zero** (filled cells span
  0.02–1.59 mm) where the observed series is zero in 68% of hours. Those 2,154
  hours lose their zero-inflation structure; `imp_rain_mm` marks them.

## Status

Done: all four sources acquired and verified, station panel fixed, NEA
aggregated to hourly, merged dataset built and verified, feature matrix built
and verified. `features.csv.gz` is ready for modelling.

Next: `forecast.R` — the rolling-window engine (3-year window, weekly refit,
h = 1/6/12/24) over `rlasso`/post-LASSO, random forest and bagging, then
`evaluate.R` for Diebold-Mariano tests and forecast combination.
