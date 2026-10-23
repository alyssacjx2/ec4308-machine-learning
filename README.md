# Forecasting Singapore's hourly electricity demand

EC4308 Machine Learning and Economic Forecasting — Group 10.

Direct multi-horizon forecasts of Singapore system demand at **h = 1, 6, 12 and
24 hours**, comparing regularised linear methods and tree ensembles against
time-series benchmarks on a 3-year rolling window with weekly refitting.

The benchmark that matters is **EMA's own operational forecast**, which achieves
RMSE 179.3 MW / MAPE 2.289% over the sample. Seasonal-naive and AR(p) are the
textbook comparators; EMA's forecast is the demanding one.

---

## Repository layout

```
R/                  pipeline scripts, run in the order below
docs/
  DATA.md           the four sources, coverage, conventions, 13 known quirks
  VERIFICATION.md   what was checked in the merged dataset and what came back
  nea_dataset_manifest.json
data/               NOT in the repo, except as noted in DATA.md
  raw/              as downloaded
  interim/          hourly x station NEA aggregates (generated)
  processed/        merged dataset and model matrix (generated)
logs/               stdout from the long extraction and aggregation runs
output/figures/     plots
```

**Read [`docs/DATA.md`](docs/DATA.md) before touching the data** and
[`docs/VERIFICATION.md`](docs/VERIFICATION.md) before touching the ERA5 code.
Between them they record the traps that have already cost time: the eight-hour
UTC shift, the netCDF axis order that silently produces constant columns, and
the station panel changing composition mid-sample.

## Setup

R 4.4.2. Not on PATH by default on Windows:

```powershell
$env:Path += ";$env:LOCALAPPDATA\Programs\R\R-4.4.2\bin\x64"
```

```r
install.packages(c("data.table", "R.utils", "ncdf4", "hdm", "randomForest"))
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

## Status

Done: all four sources acquired and verified, station panel fixed, NEA
aggregated to hourly, merged dataset verified
([`docs/VERIFICATION.md`](docs/VERIFICATION.md)), feature matrix built.

Next: `forecast.R` — the rolling-window engine (3-year window, weekly refit,
h = 1/6/12/24) over `rlasso`/post-LASSO, random forest and bagging, then
`evaluate.R` for Diebold-Mariano tests and forecast combination.
