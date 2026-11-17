# LightGBM gradient boosted regression trees, one direct model per horizon, on the common rolling window

library(data.table)
library(lightgbm)

# settings set to be the same as the benchmark AR model, to ensure comparability
source("R/utils.R")

IN <- "data/processed/features.csv.gz"
OUT <- "output/forecasts/lgbm.csv.gz"
OUT_TUNE <- "output/forecasts/lgbm_tuning.csv.gz"
OUT_IMP <- "output/forecasts/lgbm_importance.csv.gz"
H <- c(1L, 6L, 12L, 24L) # chosen forecast horizons
WINDOW <- as.integer(1.5*365*24) # 13140 hours (1.5 years) of training pairs
REFIT <- 168L # refit every week
OOS_START <- "2018-07-16 00:00:00" # common first forecast origin among all our models

# LightGBM settings
VAL <- 672L # validation block is worth 4 weeks (24hours * 7 *4)
LR <- 0.05 # shrinkage lambda: fixed, the number of trees is tuned
MAX_TREES <- 3000L # upper limit for the number of trees M
PATIENCE  <- 100L # stop once validation MAE has not improved for this long
GRID      <- CJ(objective  = c("l2", "l1"), # training loss: squared or absolute error
                num_leaves = c(15L, 31L, 63L))  # d + 1 terminal nodes per tree
RETUNE    <- 4L # re-validate at every 4th refit (roughly monthly)
SEED      <- 4308L

CORES     <- 7L
CHUNK     <- 32L  # refits per task; a multiple of RETUNE, so every

# task starts with a re-validation
DRY_RUN   <- "--dry-run" %in% commandArgs(trailingOnly = TRUE)
if (DRY_RUN) {  # separate files, so dry run will not override results
  OUT      <- "output/forecasts/lgbm_dryrun.csv.gz"
  OUT_TUNE <- "output/forecasts/lgbm_tuning_dryrun.csv.gz"
  OUT_IMP  <- "output/forecasts/lgbm_importance_dryrun.csv.gz"
}

# load
f <- read_gz(IN, colClasses = c(datetime = "character")) #character means no timezone shift
setorder(f, datetime)
y <- f$y #target variable
N <- length(y)

T_first <- match(OOS_START, f$datetime)
if (is.na(T_first)) stop("OOS_START not found in features: ", OOS_START)

# 1. predictors
# no hour x day-of-week dummies and no scaling: trees find interactions by splitting twice, and are unaffected by monotone transformations.
# drop irrelevant preditors, like temp_sq, cdh, t_index, year (justified in the report)
DROP <- c("datetime", "usable", grep("^(y|nem)_h[0-9]+$", names(f), value = TRUE),
          "temp_sq", "cdh", "t_index", "year")
X0 <- as.matrix(f[, setdiff(names(f), DROP), with = FALSE])
storage.mode(X0) <- "double"

# Horizon-specific columns, all known at the origin t:
#   the target hour's calendar - fixed in advance; benchmarks.R gives the AR the same target-hour dummies;
#   demand at the target's clock hour a week (and a day) earlier, y_{t+h-168} and
#     y_{t+h-24} - the seasonal-naive forecasts, also inside the AR's 168 lags.
make_X <- function(h) {
  lead <- function(v) shift(f[[v]], h, type = "lead")
  X <- cbind(X0,
             hr_tgt      = lead("hr"),
             dow_tgt     = lead("dow"),
             holiday_tgt = lead("is_holiday"),
             workday_tgt = lead("is_workday"),
             dem_tgt_1w  = shift(y, 168L - h))
  if (h < 24L) X <- cbind(X, dem_tgt_1d = shift(y, 24L - h))  # at h = 24 this is y itself
  X
}
CAT <- c("dow", "dow_tgt") # day of week as categories: one split can isolate {Sat, Sun}

#  2. LightGBM fit at one refit
lgb_params <- function(objective, num_leaves) list(
  objective        = objective,
  metric           = "l1", #validation loss: MAE
  learning_rate    = LR,
  num_leaves       = num_leaves,
  min_data_in_leaf = 20L,
  bagging_fraction = 0.5, #stochastic gradient boosting using half the rows
  bagging_freq     = 1L,  #per tree, redraw at each iteration
  feature_fraction = 0.8, #80% of the predictors available to each tree
  num_threads      = 1L,
  seed             = SEED,
  deterministic    = TRUE, #with force_col_wise: identical reruns
  force_col_wise   = TRUE,
  verbose          = -1L)

# actual LightGBM code
runLGBM <- function(T0, h, X, best = NULL) {
  tr   <- (T0 - h - WINDOW + 1L):(T0 - h)  # training origins s, so that s + h <= T0
  tune <- NULL
  if (is.null(best)) {
    va <- tail(tr, VAL) # validation origins
    fi <- tr[tr + h <= va[1]] # origins for the validation fit
    tune <- rbindlist(lapply(seq_len(nrow(GRID)), function(g) {
      dfi <- lgb.Dataset(X[fi, , drop = FALSE], label = y[fi + h], categorical_feature = CAT)
      dva <- lgb.Dataset.create.valid(dfi, X[va, , drop = FALSE], label = y[va + h])
      fit <- lgb.train(lgb_params(GRID$objective[g], GRID$num_leaves[g]), dfi,
                       nrounds = MAX_TREES, valids = list(val = dva),
                       early_stopping_rounds = PATIENCE, verbose = -1L)
      mae <- lgb.get.eval.result(fit, "val", "l1")   # validation MAE after each tree
      data.table(GRID[g], trees = which.min(mae), val_mae = min(mae))
    }))
    best <- tune[which.min(val_mae)]
  }

  dtr   <- lgb.Dataset(X[tr, , drop = FALSE], label = y[tr + h], categorical_feature = CAT)
  model <- lgb.train(lgb_params(best$objective, best$num_leaves), dtr,
                     nrounds = best$trees, verbose = -1L)
  list(model = model, best = best, tune = tune)
}

#  3. rolling window
# Refit every REFIT hours, each fit forecasts every origin until the next refit.
# `ks` picks which refits to run. Refit k
# re-validates when (k - 1) is a multiple of RETUNE
lgbm.rolling.window <- function(h, ks) {
  X <- make_X(h)
  stopifnot(!anyNA(X[(T_first - h - WINDOW + 1L):(N - h), ]))  # every row a window uses
  refits <- seq(T_first, N - h, by = REFIT)

  out <- tune <- imp <- vector("list", length(ks))
  best <- NULL
  for (j in seq_along(ks)) {
    k      <- ks[j]
    T0     <- refits[k]
    retune <- (k - 1L) %% RETUNE == 0L
    fit    <- runLGBM(T0, h, X, if (retune) NULL else best)
    best   <- fit$best
    t      <- T0:min(T0 + REFIT - 1L, N - h) # origins served by this fit
    out[[j]] <- data.table(
      datetime    = f$datetime[t], # forecast origin
      h           = h,
      actual      = y[t + h], # y_{t+h}
      lgbm        = predict(fit$model, X[t, , drop = FALSE]),
      lgbm_loss   = best$objective, # settings used by the refit that served t
      lgbm_leaves = best$num_leaves,
      lgbm_trees  = best$trees
    )
    if (retune) tune[[j]] <- data.table(h = h, refit = f$datetime[T0], fit$tune)
    imp[[j]] <- data.table(h = h, refit = f$datetime[T0],
                           lgb.importance(fit$model, percentage = TRUE))
  }
  list(out = rbindlist(out), tune = rbindlist(tune), imp = rbindlist(imp))
}

#  4. run all horizons 
# Tasks of CHUNK consecutive refits, longest horizon-work first (h = 1 fits the largest trees), handed to the next free process as each one finishes.
tasks <- rbindlist(lapply(H, function(h) {
  ks <- seq_along(seq(T_first, N - h, by = REFIT))
  if (DRY_RUN) ks <- 1:3
  data.table(h = h, chunk = (ks - 1L) %/% CHUNK, k = ks)
}))[, .(ks = list(k)), by = .(h, chunk)]
cat(nrow(tasks), "tasks on", CORES, "processes\n")

# Forking is unavailable on Windows. Scheduling changes do not alter seeds
# or the horizon-specific training/validation data.
run_tasks <- function(X, FUN) {
  if (.Platform$OS.type == "windows" || CORES == 1L) lapply(X, FUN) else
    parallel::mclapply(X, FUN, mc.cores = CORES, mc.preschedule = FALSE)
}
res <- run_tasks(seq_len(nrow(tasks)), function(i) {
  setDTthreads(1L)
  t0 <- Sys.time()
  r  <- lgbm.rolling.window(tasks$h[i], tasks$ks[[i]])
  cat(sprintf("  h=%2d  refits %3d-%3d done  %5.1f min\n", tasks$h[i], min(tasks$ks[[i]]),
              max(tasks$ks[[i]]), as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  r
})

failed <- which(vapply(res, function(r) inherits(r, "try-error") || is.null(r), logical(1)))
if (length(failed)) stop("tasks failed: ", paste(failed, collapse = ", "), "\n", res[[failed[1]]])

# One row per (forecast origin, h), in the shared output format.
out <- rbindlist(lapply(res, `[[`, "out"))[order(h, datetime)]
if (!dir.exists(dirname(OUT))) dir.create(dirname(OUT), recursive = TRUE)
fwrite(out, OUT)
fwrite(rbindlist(lapply(res, `[[`, "tune"))[order(h, refit)], OUT_TUNE)
fwrite(rbindlist(lapply(res, `[[`, "imp"))[order(h, refit)], OUT_IMP)
cat("\nwritten:", OUT, "- run view_lgbm.R to see the errors\n")
