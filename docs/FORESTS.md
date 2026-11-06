# Random forest and bagging

## Fixed-mtry alternative

The fixed method is independent of the tuning scripts. Its code dependencies are:

- `R/run_fixed_forests.R`: settings, fixed model fits and rolling forecasts.
- `R/utils.R`: compressed input loading.
- `R/compare_forests.R`: evaluation metrics.

Neither `R/random_forest.R`, `R/tree_tuning.R`, `R/bagging.R` nor
`R/run_forest_comparison.R` is needed to run the fixed method. For a fixed-only
submission, include the three scripts above and this documentation. The existing
`output/fixed_forests/forecasts.csv` and `comparison.csv` are optional saved
results; they are not inputs to model training.

Install `data.table` and `ranger`, provide `gzip` on PATH (or Git for Windows),
and generate `data/processed/features.csv.gz` using the repository's data pipeline.
Run all commands from the repository root. A full rerun replaces the forecast
file; use a separate directory for a quick reproducibility check:

```sh
FIXED_OUTPUT_DIR=output/fixed_forests_check RF_MAX_REFITS=1 RF_NUM_TREES=10 Rscript R/run_fixed_forests.R
```

This check deliberately uses 10 trees for speed; the default full run uses 500.

`Rscript R/run_fixed_forests.R` runs without tuning: RF uses floor(p/3)=18
and classical bagging uses mtry=p=54.
`forecasts.csv` holds both models' predictions, mtry and OOB MSE side by side,
with one row per forecast origin/horizon. It is updated after each paired fit.
`comparison.csv` holds both models' metrics by horizon on completion.
Separate model CSVs are temporary and removed when the script exits.
All other forecasting settings remain the same. Results go exclusively into
`output/fixed_forests/`, leaving the running tuning experiment's files separate.
Set `FIXED_OUTPUT_DIR` to override the destination. Running both experiments
concurrently will share available CPU and slow each one.

Run from the repository root after building `data/processed/features.csv.gz`:

```sh
Rscript R/run_forest_comparison.R
```

This paired runner uses early stopping for RF and fits bagging at mtry=p.
If RF reaches p, it reuses that identical bagging fit; otherwise it fits bagging
separately. It writes both models' predictions,
metrics, comparison and `output/tables/random_forest_mtry_usage.csv` (selection
counts and percentages per horizon). It uses RF_* model settings for both models;
BAG_* settings control bagging output paths only. `RF_COMPARISON_OUTPUT` and
`RF_MTRY_OUTPUT` can set the summary output paths. Each weekly refit processes all four horizons
before advancing to the next week. Final summaries are written on completion.

Alternatively, run each model separately:

```sh
Rscript R/random_forest.R
Rscript R/bagging.R
Rscript R/compare_forests.R
```

Both use exactly 13,140 training pairs (1.5 years), refits every 168 hours,
horizons 1/6/12/24, 500 trees, minimum split-node size 5, bootstrap sampling with replacement and
seed 4308. Predictor selection is unchanged (54 predictors in the current data).
The first forecast origin is 2018-07-16 00:00:00 Asia/Singapore. Evaluation
continues through 2026-09-30 23:00:00 (target time), capped by data availability.
The explicit cutoff prevents any later target hours from entering evaluation
if the input dataset is extended.
For horizon h and refit T, training feature times run from T-h-13,139 hours
through T-h, inclusive: exactly 13,140 pairs, matching benchmarks.R.
Each hourly forecast uses features observed at that hour, with a weekly fitted
model. Training labels must already be observed at the weekly refit.

## Tuning only mtry

At every RF refit and horizon, start at `max(1, floor(p/3))`. Multiply by 1.5,
round upward, and cap at `p`. For p=54 the potential path is 18, 27, 41, 54.
Only mtry changes between candidate fits. Each candidate uses the same seed.

`ranger` reports regression OOB error as MSE. Stop at the first candidate whose
MSE is equal to or higher than the best earlier candidate, or when p is reached.
Retain the lowest-OOB-MSE model tested; ties retain the earlier, smaller mtry.
No tolerance is used: any strictly positive improvement continues the search.
This selects the best tested mtry, not necessarily the best across the full grid.
A later untested candidate could improve after an intermediate worsening.
Only mtry is tuned; all other hyperparameters are fixed within each run.
The bagging fit at p is not added to RF tuning if RF stopped before reaching p.
The trace's relative_improvement compares each candidate to the best earlier MSE
for diagnostic purposes only.

Bagging uses mtry=p, so every predictor is considered at each split. All other
settings and the rolling evaluation engine are shared. RF can select p and
therefore coincide with bagging at a refit.

## Settings and output

RF settings use `RF_`; bagging settings use `BAG_`:
`HORIZONS`, `TRAIN_HOURS`, `REFIT_HOURS`, `FIRST_ORIGIN`, `LAST_TARGET`, `NUM_TREES`, `MIN_NODE_SIZE`,
`MAX_REFITS`, `OUTPUT`, `METRICS_OUTPUT`, `TUNING_OUTPUT`.
`MAX_REFITS=0` means all refits. The former TRAIN_YEARS, REFIT_DAYS and
OOB_TOL settings are no longer used; windows and refits are specified in hours.

Default forecast files are `output/forecasts/random_forest.csv` and
`output/forecasts/bagging.csv`. Metrics and OOB traces are in `output/tables/`.
OOB traces identify the selected candidate at every horizon/refit. Forecast
rows also record selected mtry and OOB MSE. Runs replace their output files.

The comparison script accepts RF file, bagging file and output path as optional
positional arguments. It scores common horizon/origin/target keys, verifies
actual values, training counts and refit dates, and reports RMSE, MAE and MAPE.
Positive `rf_rmse_reduction_pct` means RF has lower RMSE than bagging.

## Short comparison

Use one weekly refit per horizon without overwriting full-run outputs:

```sh
RF_MAX_REFITS=1 RF_OUTPUT=output/forecasts/random_forest_pilot.csv RF_METRICS_OUTPUT=output/tables/random_forest_pilot_metrics.csv RF_TUNING_OUTPUT=output/tables/random_forest_pilot_oob.csv Rscript R/random_forest.R
BAG_MAX_REFITS=1 BAG_OUTPUT=output/forecasts/bagging_pilot.csv BAG_METRICS_OUTPUT=output/tables/bagging_pilot_metrics.csv BAG_TUNING_OUTPUT=output/tables/bagging_pilot_oob.csv Rscript R/bagging.R
Rscript R/compare_forests.R output/forecasts/random_forest_pilot.csv output/forecasts/bagging_pilot.csv output/tables/forest_comparison_pilot.csv
```

This evaluates the first held-out week only; it is not evidence of full-period
superiority. Full runs are substantially slower because RF retunes every week.

## Interpretation limits

OOB is the requested tuning heuristic, not an unbiased time-series forecasting
score: a tree's training rows can occur later than its OOB rows. Use the held-out
chronological errors to compare predictive performance.

The existing feature file uses full-sample weather climatology and two-sided
interpolation. This can make both models' forecasting scores optimistic. These
scripts preserve that preprocessing for a controlled model comparison.

The default training-pair count, weekly refit schedule and first forecast origin
now match `benchmarks.R`. EMA forecast issue times are unknown, so
its stored errors are only an operational reference, not horizon-matched scores.
