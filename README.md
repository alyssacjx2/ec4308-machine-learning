# Forecasting Singapore's hourly electricity demand

EC4308 Machine Learning and Economic Forecasting — Group 10.

## 1. The problem

Singapore's grid operator has to match supply to demand hour by hour, so
accurate short-term load forecasts lower the cost of keeping reserve generation
on standby. We ask whether machine-learning models that use weather, calendar
and Covid-regime information can beat classical time-series forecasts of
Singapore's hourly system demand. Models produce forecasts at
**h = 1, 6, 12 and 24 hours ahead**, using a rolling 1.5-year training window
that is refitted every week. Forecasts run from 16 Jul 2018. Every model is scored on the
same held-out test year (Oct 2025 – Sep 2026). MAPE is the headline forecast
metric. RMSE carries the Diebold-Mariano tests, and MAE is reported alongside.

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
    sarima_direct.R          per-horizon SARIMA (the version reported in the main results)
    sarima.R                 original recursive SARIMA, kept for reference only
    sarima_direct_holiday.R  per-horizon SARIMA, 24h or 168h season, holiday adjustment
    holt_winters.R
    holt_winters_holiday.R   Holt-Winters, 24h or 168h season, holiday adjustment
    lasso.R                  LASSO and post-LASSO (hdm::rlasso)
    run_forests.R            random forest and bagging (ranger)
    forest_features.R        horizon-specific predictor set for the forests (sourced by run_forests.R)
    lgbm.R                   LightGBM
  lasso_selection.R          which predictors LASSO keeps across refits
  report_figures.R           report Figures 1-2 (daily mean demand, demand by hour and day type)
  evaluation/
    evaluate.R               merges every model's forecasts and scores them
    dm_and_combination.R     Diebold-Mariano tests, forecast combination
    forecast_tests.R         alternative Holm-adjusted DM tests (single 44-test family; not used in the report)
    combination_scenarios_and_gr.R   combination with the extra variants, Granger-Ramanathan weights
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
65 columns, including 54 engineered base features and current demand). The
actual predictor set is model-specific; the updated forests use 57/56 predictors
after horizon-specific construction. Every model script reads this one file:

```r
source("R/utils.R")
f <- read_gz("data/processed/features.csv.gz",
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
| Horizons | h = 1, 6, 12, 24; every reported model is fitted separately for each horizon (the regressions and ensembles are direct models; SARIMA and Holt-Winters choose their coefficients on the h-step error) |
| Training window | rolling, 1.5 years = 13,140 (origin, target) pairs |
| Refit | every 168 hours (weekly) |
| First forecast origin | 2018-07-16 00:00:00 |
| Training-label availability | a refit at T0 uses only observed targets (s + h ≤ T0); weather imputation still has the documented look-ahead limitation |
| Output | one row per (origin, h): `datetime`, `h`, `actual`, model column(s) |

| Model | Script | Specification |
|---|---|---|
| Seasonal naive | `benchmarks.R` | ŷ<sub>t+h</sub> = y<sub>t+h−s</sub>, with s = 24 and s = 168 |
| AR(p) by BIC | `benchmarks.R` | direct AR with p ∈ {24, 48, …, 168}, plus hour × day-type and Covid dummies (an ARX) |
| SARIMA | `sarima_direct.R` | ARIMA(2,0,1)(2,1,0)[24]; orders chosen once by AICc on the first window, coefficients re-estimated weekly and separately for each h by minimising the squared h-step error |
| Holt-Winters | `holt_winters.R` | additive, daily season (HW24); α and γ tuned separately for each horizon |
| LASSO / post-LASSO | `lasso.R` | `hdm::rlasso` with a data-driven penalty; the 167 hour-of-week cells are left unpenalised |
| Random forest | `run_forests.R` | 500 trees, mtry = ⌊p/3⌋ = 19 (h=1/6/12), 18 (h=24), min node size 5 |
| Bagging | `run_forests.R` | the same forest with mtry = p = 57 (h=1/6/12), 56 (h=24) |
| LightGBM | `lgbm.R` | tree count, size and loss tuned on the last 4 weeks of each window, retuned about monthly |
| Combination | `dm_and_combination.R` | equal-weight average of the 3 models with the lowest validation RMSE (Apr–Sep 2025) |

**EMA's forecast is a reference level, not a horizon-matched benchmark.** EMA
publishes one near-term forecast per period and does not record how far ahead
it was made, so it scores the same at every h. Even at h = 1 its information lead cannot be established, so its errors
should not be interpreted as a formally matched-horizon comparison. The
horizon-matched benchmarks are the seasonal-naive models and the AR(p).

## 5. Results

We report three error measures. Lower is better on all of them, and bold
marks the best model at each horizon (EMA excluded).

- **MAPE** (mean absolute percentage error, %) is the typical miss as a share
  of demand. It is our headline forecast metric because it shows the scale of
  the errors.
- **RMSE** (root mean squared error, MW) punishes large misses heavily. It is
  the loss behind the Diebold-Mariano tests.
- **MAE** (mean absolute error, MW) is the typical miss in megawatts.

Model selection and validation (LightGBM tuning, the combination's members)
use MAE or RMSE. MAPE is used only to report the final results.

Every model produces forecasts from 16 Jul 2018 onward. The results below are
for the **test year (Oct 2025 – Sep 2026)**, which has 8,760 forecasts at
each horizon. The test year comes after the Apr–Sep 2025 validation window
that was used to pick the combination's members.

**MAPE (%)**

| Model | h = 1 | h = 6 | h = 12 | h = 24 |
|---|---|---|---|---|
| Combination | 0.33 | **0.93** | **1.16** | 1.28 |
| post-LASSO | 0.35 | 1.00 | 1.20 | **1.27** |
| LASSO | 0.35 | 1.00 | 1.20 | 1.28 |
| AR(p) by BIC | **0.33** | 0.94 | 1.22 | 1.37 |
| LightGBM | 0.41 | 1.09 | 1.30 | 1.51 |
| Random forest | 0.48 | 1.14 | 1.33 | 1.44 |
| Bagging | 0.47 | 1.24 | 1.42 | 1.52 |
| SARIMA | 0.50 | 2.14 | 2.73 | 2.87 |
| Holt-Winters (HW24) | 0.70 | 2.74 | 3.05 | 2.97 |
| Seasonal naive (168h) | 2.25 | 2.25 | 2.25 | 2.25 |
| Seasonal naive (24h) | 2.82 | 2.82 | 2.82 | 2.82 |
| *EMA (reference)* | *1.63* | *1.63* | *1.63* | *1.63* |

At h = 1 the AR's MAPE (0.327) is just below the combination's (0.330).

Each model is tested against the AR(p) benchmark using squared-error loss and
a Newey-West variance, so the DM tests go with the RMSE table. A negative DM
statistic means the model beats the AR.

**RMSE (MW), with the DM statistic in brackets**

| Model | h = 1 | h = 6 | h = 12 | h = 24 |
|---|---|---|---|---|
| Combination | **33.7** (−4.23\*\*\*) | **89.9** (−2.48\*\*) | **109.0** (−4.63\*\*\*) | 118.8 (−5.44\*\*\*) |
| post-LASSO | 34.8 (1.39) | 94.8 (1.83) | 111.9 (−1.71) | **117.7** (−4.22\*\*\*) |
| LASSO | 35.0 (1.99) | 95.0 (2.00) | 112.1 (−1.63) | 117.8 (−4.39\*\*\*) |
| AR(p) by BIC | 34.5 | 92.3 | 114.9 | 127.7 |
| LightGBM | 40.8 (5.61\*\*\*) | 109.7 (4.25\*\*\*) | 129.0 (3.37\*\*\*) | 144.1 (4.06\*\*\*) |
| Random forest | 48.8 (9.02\*\*\*) | 115.3 (5.53\*\*\*) | 131.1 (3.82\*\*\*) | 136.8 (2.51\*) |
| Bagging | 47.8 (11.06\*\*\*) | 128.5 (5.60\*\*\*) | 145.7 (4.75\*\*\*) | 148.6 (4.21\*\*\*) |
| *EMA (reference)* | *128.6* | *128.6* | *128.6* | *128.6* |

Stars use **Holm-adjusted p-values**, as in the report: squared-error loss,
Bartlett Newey-West variance with 20 lags, Student-t p-values, and Holm
applied within two families: the 36 individual comparisons (9 models × 4
horizons) and the 4 combination comparisons. EMA is not tested. With 168 lags
two conclusions change: the combination at h = 6 (p = 0.056) and LightGBM at
h = 12 (p = 0.079) are no longer significant. These tests are scenario A in
`dm_tests_holm_A_B.zip`. `forecast_tests.R` (`dm_tests_holm.csv`) is an
alternative single 44-test family with 21–23 lags and is not used in the report.

\* p < 0.10, \*\* p < 0.05, \*\*\* p < 0.01 (Holm-adjusted). SARIMA,
Holt-Winters and the seasonal-naive models lose to the AR at every horizon
with Holm p < 0.001.

**MAE (MW)**

| Model | h = 1 | h = 6 | h = 12 | h = 24 |
|---|---|---|---|---|
| Combination | 24.1 | **67.6** | **83.8** | 92.1 |
| post-LASSO | 25.6 | 72.5 | 86.9 | **91.2** |
| LASSO | 25.6 | 72.6 | 87.0 | 91.5 |
| AR(p) by BIC | **23.9** | 68.3 | 87.6 | 98.3 |
| LightGBM | 29.4 | 79.3 | 94.9 | 109.3 |
| Random forest | 35.1 | 83.4 | 96.8 | 103.9 |
| Bagging | 34.3 | 90.2 | 103.7 | 110.0 |
| SARIMA | 36.2 | 153.6 | 197.7 | 207.3 |
| Holt-Winters (HW24) | 49.7 | 194.4 | 217.5 | 214.4 |
| Seasonal naive (168h) | 160.5 | 160.5 | 160.5 | 160.5 |
| Seasonal naive (24h) | 204.4 | 204.4 | 204.4 | 204.4 |
| *EMA (reference)* | *115.4* | *115.4* | *115.4* | *115.4* |

- **The combination** is the equal-weight average of LASSO, post-LASSO and
  the AR. Each member was chosen by validation RMSE (Apr–Sep 2025), so the
  test year played no part in picking them. It is the only model that
  significantly beats the AR at every horizon on RMSE, although the h = 6
  result does not survive 168 lags. On MAPE and MAE it also comes out on top
  at h = 6 and 12. At h = 1 it is level with the AR, and at h = 24
  post-LASSO is narrowly better.
- **LASSO and post-LASSO** significantly beat the AR only at h = 24 (about
  8% lower RMSE). At h = 1, 6 and 12 there is no statistically detectable
  difference after the Holm correction (adjusted p = 0.27), even though they
  are slightly worse than the AR at h = 1 and 6.
- **The tree ensembles** are significantly worse than the AR at every
  horizon, except the random forest at h = 24 (Holm p = 0.084). LightGBM is
  the best of them on every metric at h = 1, 6 and 12; the random forest is
  the best of them at h = 24.
- **SARIMA and Holt-Winters** use demand alone and model only the daily
  cycle. By h = 12, both are worse than the weekly seasonal-naive forecast.

The numbers above are in
[test_metrics_all_models.csv](output/tables/test_metrics_all_models.csv),
except the SARIMA row: that file's `sarima` row is the original recursive
`sarima.R` (MAPE 2.26 / 2.99 / 3.14 at h = 6 / 12 / 24).
The per-horizon SARIMA scores (`sarima24`) and the Holm-adjusted DM tests
are in `test_metrics_A_B.zip` and `dm_tests_holm_A_B.zip` in
[output/tables/](output/tables/).
[dm_tests_all_models.csv](output/tables/dm_tests_all_models.csv) holds the
unadjusted tests, and its SARIMA row is the original recursive `sarima.R`. Scores for
the full sample and the validation window, without the combination, are in
[all_models_comparison.csv](output/tables/all_models_comparison.csv).

### 5.1 Holiday-adjusted and weekly-season SARIMA / Holt-Winters

`sarima_direct_holiday.R` and `holt_winters_holiday.R` re-run the two
classical models with a separate fit for each horizon, a choice of daily
(24 h) or weekly (168 h) season, and (`_hol`) a weekday public-holiday
adjustment. The SARIMA (24h) row is the same per-horizon fit as in the main
tables. Test year, Holm-adjusted DM tests against the AR (squared error, 20
Newey-West lags, 60 individual comparisons in the extension case):

| Model | MAPE h = 1 / 6 / 12 / 24 | RMSE h = 1 / 6 / 12 / 24 |
|---|---|---|
| SARIMA (24h) | 0.50 / 2.14 / 2.73 / 2.87 | 51.6 / 216.6 / 281.2 / 273.2 |
| SARIMA (24h) + holiday | 0.50 / 2.10 / 2.64 / 2.78 | 51.6 / 210.8 / 270.2 / 262.1 |
| SARIMA (168h) | 0.43 / 1.34 / 1.70 / 1.91 | 47.6 / 150.8 / 188.5 / 200.9 |
| SARIMA (168h) + holiday | 0.43 / 1.16 / 1.42 / 1.63 | 46.9 / 116.9 / 138.6 / 152.2 |
| Holt-Winters (24h) + holiday | 0.69 / 2.52 / 2.94 / 2.91 | 65.7 / 229.2 / 272.7 / 295.9 |
| Holt-Winters (168h) | 0.45 / 1.29 / 1.46 / 1.61 | 45.4 / 135.5 / 149.8 / 163.2 |
| Holt-Winters (168h) + holiday | 0.43 / 1.05 / **1.19** / **1.33** | 43.4 / 100.1 / 111.1 / 123.0 |
| *AR(p) by BIC, for reference* | *0.33 / 0.94 / 1.22 / 1.37* | *34.5 / 92.3 / 114.9 / 127.7* |

- **The weekly season matters most.** At h ≥ 6, moving from a 24 h to a
  168 h season cuts MAPE by about a third for SARIMA and about half for
  Holt-Winters.
- **The holiday adjustment helps further**, mostly at h ≥ 6.
- **Weekly Holt-Winters with holidays** uses demand alone, yet it has a lower
  MAPE and MAE than the AR at h = 12 and 24 (MAE 84.6 vs 87.6 MW and 95.0 vs
  98.3 MW). Its RMSE is also lower there, but the DM tests cannot tell it
  apart from the AR (Holm p = 0.44). Every other variant is significantly
  worse than the AR at every horizon.

Adding the six new variants (all rows above except SARIMA (24h), which is
already a baseline candidate) to the candidate pool ("scenario B", 16
candidates) does not
change the equal-weight combination: LASSO, post-LASSO and the AR still have
the lowest validation RMSE at every horizon. Scores and tests are in
`test_metrics_A_B.zip`, `dm_tests_holm_A_B.zip` and
`combination_members_A_B.zip` in [output/tables/](output/tables/).

### 5.2 Granger-Ramanathan combination weights

`combination_scenarios_and_gr.R` replaces the equal weights with weights
estimated by regressing actual demand on the member forecasts over the
validation window (Apr–Sep 2025). Three versions are fitted: no intercept
(`noint`), weights summing to one (`sum1`), and with an intercept (`int`).
Each is fitted on three pools: the equal-weight combination's 3 members
(`top3`), all 10 original models (A), and A plus the 6 variants above (B).
As in the report, the **main version has unrestricted weights and no
intercept**, and pool A (the 10 baseline candidates) is the main pool. The
sum-to-one and intercept versions are sensitivity checks; their results are
in the zip files below. Weights are estimated once on validation and then
fixed for the test year:

| Combination | MAPE h = 1 / 6 / 12 / 24 | RMSE h = 1 / 6 / 12 / 24 |
|---|---|---|
| Equal weights, top 3 (Section 5) | 0.330 / 0.932 / 1.158 / 1.282 | 33.7 / 89.9 / 109.0 / 118.8 |
| GR, top 3 | 0.343 / 0.934 / 1.140 / **1.254** | 34.3 / 89.6 / 107.7 / **117.1** |
| **GR, pool A (main)** | 0.326 / 0.882 / 1.095 / 1.262 | 32.8 / 86.7 / 106.3 / 118.1 |
| GR, pool B (extension) | **0.326** / **0.873** / **1.089** / 1.280 | **32.8** / **86.3** / **104.3** / 119.3 |

- **GR on pool A** lowers RMSE relative to equal weights by 2.7%, 3.5% and
  2.5% at h = 1, 6 and 12, with no gain at h = 24 (0.6%). On the RMSE DM test
  (Holm over 9 schemes × 4 horizons, 20 lags) it significantly beats the AR
  at every horizon, but it significantly beats equal weights only at h = 1
  (p < 0.001; p = 0.051, 0.795 and 1.000 at h = 6, 12 and 24).
- **GR leans on the tree models** that the top-three rule excludes. At h = 12
  the largest pool-A weights are LightGBM (0.33), post-LASSO (0.24), AR (0.24)
  and random forest (0.23).
- **GR over the top 3 alone is unstable.** LASSO and post-LASSO are almost
  collinear, so the weights offset each other (at h = 1: post-LASSO 1.40,
  LASSO −0.54, AR 0.14). It is significantly worse than equal weights at
  h = 1 and significantly better at h = 12 (p = 0.003).
- **GR on pool B** beats equal weights at h = 1, 6 and 12 with 20 lags
  (p < 0.001, 0.012 and 0.026), but only h = 1 stays significant with
  168 lags (p = 0.070 and 0.077 at h = 6 and 12).
- These results are **exploratory**. Some weights are negative, up to 16
  weights are fitted on one six-month window of autocorrelated data, and the
  schemes were not chosen in advance.

**Effect on the overall evaluation.** The main ranking in Section 5 still
holds. The combination's members (LASSO, post-LASSO, AR) do not change when
the new SARIMA/Holt-Winters variants are added, because they still have the
lowest validation RMSE. The tree ensembles, and every SARIMA/Holt-Winters
variant except weekly Holt-Winters with holidays, remain significantly worse
than the AR. The two changes are:

1. Estimating the combination weights (GR, pool A) improves on equal weights
   at h = 1 to 12, but significantly so only at h = 1, and adds nothing at
   h = 24.
2. A pure time-series model, weekly Holt-Winters with holidays, now matches
   the AR at h = 12 and 24.

Metrics, weights and DM tests (20 and 168 lags, Holm-adjusted) are in
`granger_ramanathan_metrics.zip`, `granger_ramanathan_weights.zip` and
`granger_ramanathan_dm.zip` in [output/tables/](output/tables/).

### 5.3 What LASSO selects

`lasso_selection.R` refits LASSO at every 10th weekly origin (43 refits per
horizon) and records which predictors it keeps. Its hour-of-week cells now use
day type, so a public holiday takes the Sunday profile, as in `lasso.R`.

- **Demand terms dominate.** The 168 h moving average is kept in every refit
  at every horizon, and current demand (`y_t`) in 81–100%. The 24 h moving
  average is always kept up to h = 12 but only in about half the refits at
  h = 24.
- **Temperature matters more as the horizon grows.** At h = 1 only the
  one-hour lag is always kept. From h = 6 the 24 h and 72 h moving averages
  are kept in 88–100% of refits. Together with squared temperature (kept in
  91–100% at h = 6 and 12), they carry the largest weather coefficients.
- **Rainfall and wind** (current rainfall, wind speed and direction) are kept
  in 74–100% of refits at every horizon, and `is_holiday` in 74–93%.
- Between 16 (h = 1) and 22 (h = 12, 24) predictors are kept in at least half
  the refits.

Frequencies and average standardised coefficients are in
[lasso_selection_frequency.csv](output/tables/lasso_selection_frequency.csv).
[appendix_feature_list.csv](output/tables/appendix_feature_list.csv) describes
every predictor.

## 6. How to run

**Requirements:** R ≥ 4.4 and `gzip` on the PATH. `R/utils.R` streams the
compressed files.

The reviewer can install missing R dependencies with:

```bash
Rscript install_packages.R
```

This lists the project dependencies and installs those absent from the active
R library. It does not install data, change already installed package versions,
or pin versions. For exact package-version restoration, create a `renv.lock`
from the verified modelling environment and provide `renv::restore()` instructions.

Equivalent manual installation for the core pipeline:

```r
install.packages(c("data.table", "ncdf4", "hdm", "ranger",
                   "lightgbm", "forecast"))
# Optional legacy viewers use R.utils and sandwich:
# install.packages(c("R.utils", "sandwich"))
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
Rscript R/models/sarima.R          # ~1 h; recursive SARIMA, read by evaluate.R / dm_and_combination.R
Rscript R/models/sarima_direct.R --period=24   # per-horizon SARIMA in the main results (default is --period=168)
Rscript R/models/holt_winters.R    # ~1.5 h

# evaluation (no model fitting, runs in seconds)
Rscript R/evaluation/evaluate.R
Rscript R/evaluation/dm_and_combination.R
Rscript R/evaluation/forecast_tests.R                 # separate Holm-adjusted original-model tests
Rscript R/evaluation/combination_scenarios_and_gr.R   # reads both SARIMA/HW zips directly
Rscript R/lasso_selection.R                           # ~5 min, refits LASSO on a subsample
Rscript R/report_figures.R                            # writes output/figures/fig1_*.png, fig2_*.png
```

The forecasts used for the results above are already in `output/forecasts/`,
so you can skip the model scripts and go straight to evaluation. The three main
evaluation scripts accept freshly generated `sarima.csv.gz` and
`holt_winters.csv.gz` through `read_seasonal_forecasts()` in `R/utils.R`.
When neither is present, they read the supplied combined seasonal CSV or ZIP
without manual extraction. Generating only one seasonal model is an error,
preventing silent mixing of fresh and archived predictions. Period labels use
TARGET time: validation 1 Apr-30 Sep 2025; test 1 Oct 2025-30 Sep 2026.

For the optional scenario/GR analysis, also supply `sarima_hw_forecasts.zip`.
Its assembly from the individual variant model outputs is not yet automated;
see [the reproducibility audit](docs/REPRODUCIBILITY_AUDIT.md) before attempting
a completely fresh run of that extension.

All commands are project-root-relative. Raw files must be supplied in the exact
layout documented in `docs/DATA.md`; the code does not download them. In
particular, place the yearly EMA CSV.gz files in `data/raw/ema/` (the original
weekly spreadsheets alone are not an input accepted by the current pipeline).
Pin the package versions used for the final submission and retain `sessionInfo()`;
the package installation commands alone do not guarantee identical future fits.

On Windows, R is not on the PATH by default:

```powershell
$env:Path += ";$env:LOCALAPPDATA\Programs\R\R-4.4.2\bin\x64"
```

## 7. Reproducibility audit

See [docs/REPRODUCIBILITY_AUDIT.md](docs/REPRODUCIBILITY_AUDIT.md) for the
17 November 2026 checks, fixes, installed/missing dependencies and remaining
raw-to-results gaps. The data rebuild was verified using cached hourly NEA
aggregates; a fresh re-estimation of every model was not performed.
