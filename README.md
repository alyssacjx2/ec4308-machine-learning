# Forecasting Singapore's hourly electricity demand

EC4308 Machine Learning and Economic Forecasting — Group 10.

## 1. The problem

Singapore's grid operator has to match supply to demand hour by hour, so
accurate short-term load forecasts lower the cost of keeping reserve generation
on standby. We ask whether machine-learning models that use weather, calendar
and Covid-regime information can beat classical time-series forecasts of
Singapore's hourly system demand. Each model produces direct forecasts at
**h = 1, 6, 12 and 24 hours ahead**, using a rolling 1.5-year training window
that is refitted every week. Forecasts run from 16 Jul 2018. Every model is scored on the
same held-out test year (Oct 2025 – Sep 2026), using RMSE, MAE, MAPE and
Diebold-Mariano tests.

## 2. Repository structure

```
R/
  utils.R                    shared helpers (gzip streaming, NEA timestamp handling)
  data_processing/           raw sources -> model matrix
    station_panel.R          picks the fixed NEA weather-station panel
    aggregate_nea_hourly.R   raw NEA files -> hourly x station aggregates
    build_dataset.R          merges all four sources onto one hourly spine
    build_features.R         imputes weather gaps, builds features.csv.gz
    explore_demand.R         standalone EDA (not part of the pipeline)
  models/                    one script per model, all with the same forecast settings
    benchmarks.R             seasonal naive (24h, 168h) + AR(p) by BIC
    sarima.R, sarima_direct.R
    holt_winters.R
    lasso.R                  LASSO and post-LASSO (hdm::rlasso)
    run_forests.R            random forest and bagging (ranger)
    lgbm.R                   LightGBM
  lasso_selection.R          which predictors LASSO keeps across refits
  evaluation/
    evaluate.R               merges every model's forecasts and scores them
    dm_and_combination.R     Diebold-Mariano tests, Linex loss, forecast combination
    compare_forests.R, view_benchmarks.R, view_lgbm.R   quick result viewers
docs/
  DATA.md                    sources, coverage, conventions, known data quirks
  FORESTS.md                 how the random forest / bagging runs were set up
  EC4308 Group 10 Project Proposal.pdf
data/
  raw/                       as downloaded (not committed)
  interim/                   hourly x station NEA aggregates (committed)
  processed/                 merged dataset and features.csv.gz (generated)
output/
  forecasts/                 one file per model: one row per (origin, h)
  tables/                    comparison tables, LASSO selection frequencies
  figures/                   plots
```

## 3. Dataset and loading

The dataset covers **85,440 contiguous hours, from 2017-01-01 to 2026-09-30**. It
starts in 2017 because that is when usable NEA rainfall data begins.

| Source | Contributes |
|---|---|
| EMA | `demand_mw` (the target), `nem_forecast_mw` (EMA's own forecast, used only as a reference) |
| NEA weather stations | temperature, humidity, wind speed/direction, rainfall |
| ERA5 reanalysis | solar radiation, sensible heat flux, cloud cover, precipitation — **lagged 120 h**, because ERA5 is published about five days late |
| MOM / time.sg | public holidays |

`build_features.R` writes **`data/processed/features.csv.gz`** (85,440 rows ×
65 columns, 54 predictors). Every model script reads this one file:

```r
library(data.table)
f <- fread("data/processed/features.csv.gz",
           colClasses = c(datetime = "character"))  # keep Singapore local time as text
```

- **Targets:** `y_h1`, `y_h6`, `y_h12`, `y_h24` hold the demand at t + h.
- **Predictors:** calendar terms (`hr`, `dow`, `mth`, holidays), demand lags and
  moving averages, weather and weather lags, cooling-degree-hours, ERA5 and
  Covid-regime dummies. Weather gaps are interpolated or filled and flagged
  in `imp_*`.
- **Filter on `usable == 1 & !is.na(y_h<h>)`.** The first 169 hours have
  incomplete lags.
- `nem_forecast_mw` is **not** a predictor. It is carried as `nem_h<h>` for
  comparison only.

Read [docs/DATA.md](docs/DATA.md) before changing anything upstream. Its
"Known quirks" section covers three traps: the 8-hour UTC shift when parsing
NEA timestamps, the netCDF axis order, and the change in the station panel
partway through the sample.

## 4. Models

Every model uses the same forecast settings. Otherwise the forecasts would be
for different hours and could not be compared or DM-tested.

| Setting | Value |
|---|---|
| Horizons | h = 1, 6, 12, 24, direct forecasting (one model per h) |
| Training window | rolling, 1.5 years = 13,140 (origin, target) pairs |
| Refit | every 168 hours (weekly) |
| First forecast origin | 2018-07-16 00:00:00 |
| No leakage | a refit at T0 uses only pairs whose target is already observed (s + h ≤ T0) |
| Output | one row per (origin, h): `datetime`, `h`, `actual`, model column(s) |

| Model | Script | Specification |
|---|---|---|
| Seasonal naive | `benchmarks.R` | ŷ<sub>t+h</sub> = y<sub>t+h−s</sub>, with s = 24 and s = 168 |
| AR(p) by BIC | `benchmarks.R` | direct AR with p ∈ {24, 48, …, 168}, plus hour × day-type and Covid dummies (an ARX) |
| SARIMA | `sarima.R` | ARIMA(p,d,q)(P,1,Q)[24]; orders chosen by AICc on the first window, coefficients re-estimated weekly |
| Holt-Winters | `holt_winters.R` | additive, daily season (HW24); α and γ tuned separately for each horizon |
| LASSO / post-LASSO | `lasso.R` | `hdm::rlasso` with a data-driven penalty; the 167 hour-of-week cells are left unpenalised |
| Random forest | `run_forests.R` | 500 trees, mtry = ⌊p/3⌋ = 18, min node size 5 |
| Bagging | `run_forests.R` | the same forest with mtry = p = 54 |
| LightGBM | `lgbm.R` | tree count, size and loss tuned on the last 4 weeks of each window, retuned about monthly |
| Combination | `dm_and_combination.R` | equal-weight average of the 3 models with the lowest validation RMSE (Apr–Sep 2025) |

**EMA's forecast is a reference level, not a horizon-matched benchmark.** EMA
publishes one near-term forecast per period and does not record how far ahead
it was made, so it scores the same at every h. Beating it at h = 1 is a fair
comparison. Losing to it at h = 24 tells us nothing about the model. The
horizon-matched benchmarks are the seasonal-naive models and the AR(p).

## 5. Results

We report three error measures. Lower is better on all of them, and bold
marks the best model at each horizon (EMA excluded).

- **RMSE** (root mean squared error, MW) punishes large misses heavily.
- **MAE** (mean absolute error, MW) is the typical miss in megawatts.
- **MAPE** (mean absolute percentage error, %) is the typical miss as a share
  of demand.

Every model produces forecasts from 16 Jul 2018 onward. The results below are
for the **test year (Oct 2025 – Sep 2026)**, which has 8,760 forecasts at
each horizon. The test year comes after the Apr–Sep 2025 validation window
that was used to pick the combination's members.

Each model is tested against the AR(p) benchmark using squared-error loss and a Newey-West variance, so the DM
tests go with the RMSE table. A negative DM statistic means the model beats
the AR.

**RMSE (MW), with the DM statistic in brackets**

| Model | h = 1 | h = 6 | h = 12 | h = 24 |
|---|---|---|---|---|
| Combination | **33.7** (−4.23\*\*\*) | **89.9** (−2.47\*\*) | **109.0** (−4.59\*\*\*) | 118.8 (−5.23\*\*\*) |
| post-LASSO | 34.8 (1.39) | 94.8 (1.82\*) | 111.9 (−1.69\*) | **117.7** (−4.07\*\*\*) |
| LASSO | 35.0 (1.99\*\*) | 95.0 (2.00\*\*) | 112.1 (−1.61) | 117.8 (−4.22\*\*\*) |
| AR(p) by BIC | 34.5 | 92.3 | 114.9 | 127.7 |
| LightGBM | 40.8 (5.57\*\*\*) | 109.7 (4.22\*\*\*) | 129.0 (3.34\*\*\*) | 144.1 (3.97\*\*\*) |
| Random forest | 57.3 (12.24\*\*\*) | 109.6 (6.36\*\*\*) | 148.4 (8.07\*\*\*) | 169.5 (5.74\*\*\*) |
| Bagging | 57.3 (12.01\*\*\*) | 113.3 (6.59\*\*\*) | 148.5 (7.54\*\*\*) | 186.6 (6.24\*\*\*) |
| *EMA (reference)* | *128.6* | *128.6* | *128.6* | *128.6 (0.20)* |

\* p < 0.10, \*\* p < 0.05, \*\*\* p < 0.01. SARIMA, Holt-Winters and the
seasonal-naive models lose to the AR at every horizon with p < 0.001.

**MAE (MW)**

| Model | h = 1 | h = 6 | h = 12 | h = 24 |
|---|---|---|---|---|
| Combination | 24.1 | **67.6** | **83.8** | 92.1 |
| post-LASSO | 25.6 | 72.5 | 86.9 | **91.2** |
| LASSO | 25.6 | 72.6 | 87.0 | 91.5 |
| AR(p) by BIC | **23.9** | 68.3 | 87.6 | 98.3 |
| LightGBM | 29.4 | 79.3 | 94.9 | 109.3 |
| Random forest | 42.2 | 83.0 | 111.3 | 121.9 |
| Bagging | 41.2 | 84.5 | 108.7 | 129.9 |
| SARIMA | 36.2 | 162.1 | 215.5 | 227.7 |
| Holt-Winters (HW24) | 49.7 | 194.4 | 217.5 | 214.4 |
| Seasonal naive (168h) | 160.5 | 160.5 | 160.5 | 160.5 |
| Seasonal naive (24h) | 204.4 | 204.4 | 204.4 | 204.4 |
| *EMA (reference)* | *115.4* | *115.4* | *115.4* | *115.4* |

**MAPE (%)**

| Model | h = 1 | h = 6 | h = 12 | h = 24 |
|---|---|---|---|---|
| Combination | 0.33 | **0.93** | **1.16** | 1.28 |
| post-LASSO | 0.35 | 1.00 | 1.20 | **1.27** |
| LASSO | 0.35 | 1.00 | 1.20 | 1.28 |
| AR(p) by BIC | **0.33** | 0.94 | 1.22 | 1.37 |
| LightGBM | 0.41 | 1.09 | 1.30 | 1.51 |
| Random forest | 0.58 | 1.14 | 1.54 | 1.68 |
| Bagging | 0.57 | 1.16 | 1.51 | 1.79 |
| SARIMA | 0.50 | 2.26 | 2.99 | 3.14 |
| Holt-Winters (HW24) | 0.70 | 2.74 | 3.05 | 2.97 |
| Seasonal naive (168h) | 2.25 | 2.25 | 2.25 | 2.25 |
| Seasonal naive (24h) | 2.82 | 2.82 | 2.82 | 2.82 |
| *EMA (reference)* | *1.63* | *1.63* | *1.63* | *1.63* |

- **The combination** is the equal-weight average of LASSO, post-LASSO and
  the AR. Each member was chosen by validation RMSE (Apr–Sep 2025), so the
  test year played no part in picking them. It is the only model that
  significantly beats the AR at every horizon on RMSE. On MAE and MAPE it
  also comes out on top at h = 6 and 12. At h = 1 it is level with the AR,
  and at h = 24 post-LASSO is narrowly better.
- **LASSO** significantly beats the AR only at h = 24 (an 8% lower RMSE).
  At h = 1 and 6 it is slightly but significantly worse than the AR.
- **The tree ensembles** are significantly worse than the AR at every
  horizon. LightGBM is the best of them on MAE and MAPE at every horizon.
- **SARIMA and Holt-Winters** use demand alone and model only the daily
  cycle. By h = 12, both are worse than the weekly seasonal-naive forecast.

The numbers above are in
[test_metrics_all_models.csv](output/tables/test_metrics_all_models.csv), which
also has Linex loss. The DM tests are in
[dm_tests_all_models.csv](output/tables/dm_tests_all_models.csv). Scores for
the full sample and the validation window, without the combination, are in
[all_models_comparison.csv](output/tables/all_models_comparison.csv).

## 6. How to run

**Requirements:** R ≥ 4.4 and `gzip` on the PATH. `R/utils.R` streams the
compressed files.

```r
install.packages(c("data.table", "R.utils", "ncdf4", "hdm", "ranger",
                   "lightgbm", "forecast", "sandwich"))
```

Run every command from the repository root. The NEA aggregates in
`data/interim/` are committed, so most people can start at step 3.

```bash
# data (steps 1-2 are slow and only need running once)
Rscript R/data_processing/station_panel.R          # ~1 min
Rscript R/data_processing/aggregate_nea_hourly.R   # ~12 min, resumable
Rscript R/data_processing/build_dataset.R          # ~3 min
Rscript R/data_processing/build_features.R         # ~1 min -> features.csv.gz

# models (each writes to output/forecasts/; add --dry-run where supported to test quickly)
Rscript R/models/benchmarks.R      # ~2.5 h
Rscript R/models/lasso.R           # ~1 h
Rscript R/models/lgbm.R            # ~45 min on 7 cores
Rscript R/models/run_forests.R     # ~2 days (random forest + bagging together)
Rscript R/models/sarima.R          # ~1 h
Rscript R/models/holt_winters.R    # ~1.5 h

# evaluation (no model fitting, runs in seconds)
Rscript R/evaluation/evaluate.R
Rscript R/evaluation/dm_and_combination.R
```

The forecasts used for the results above are already in `output/forecasts/`,
so you can skip the model scripts and go straight to evaluation. `evaluate.R`
reads `sarima_holt_winters_forecast_results.csv` uncompressed, so unzip it
first:

```bash
unzip output/forecasts/sarima_holt_winters_forecast_results.zip -d output/forecasts/
```

On Windows, R is not on the PATH by default:

```powershell
$env:Path += ";$env:LOCALAPPDATA\Programs\R\R-4.4.2\bin\x64"
```
