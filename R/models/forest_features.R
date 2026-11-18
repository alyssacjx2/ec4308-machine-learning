# Known-in-advance target calendar, shared by RF and bagging.
# The caller must sort data onto a contiguous hourly spine first.
BuildForestTargetCalendar <- function(data, horizon) {
  required <- c("hr", "dow", "is_holiday", "is_workday")
  if (!all(required %in% names(data))) {
    stop("Missing target-calendar inputs: ",
         paste(setdiff(required, names(data)), collapse = ", "))
  }
  if (length(horizon) != 1L || is.na(horizon) || horizon < 1L ||
      horizon != as.integer(horizon)) stop("Horizon must be a positive integer.")
  # Lead only deterministic calendar indicators, never future demand/weather.
  data.table::data.table(
    hr_tgt = data.table::shift(data$hr, horizon, type = "lead"),
    dow_tgt = data.table::shift(data$dow, horizon, type = "lead"),
    holiday_tgt = data.table::shift(data$is_holiday, horizon, type = "lead"),
    workday_tgt = data.table::shift(data$is_workday, horizon, type = "lead"))
}

# Same base predictor selection and order as other models.
ForestBasePredictors <- function(data) {
  excluded <- c("datetime", "datetime_posix", "usable",
                grep("^(y_h|nem_h)", names(data), value = TRUE),
                "temp_sq", "cdh", "t_index", "year")
  setdiff(names(data), excluded)
}

BuildForestHorizonFeatures <- function(data, horizon) {
  calendar <- BuildForestTargetCalendar(data, horizon)
  if (!horizon %in% c(1L, 6L, 12L, 24L)) {
    stop("Supported forest horizons are 1, 6, 12 and 24.")
  }
  calendar[, dow_tgt := factor(dow_tgt, levels = 0:6)]
  calendar[, dem_tgt_1w := data.table::shift(data$y, 168L - horizon)]
  # At h=24 the daily comparator is already present as current demand y.
  if (horizon < 24L) {
    calendar[, dem_tgt_1d := data.table::shift(data$y, 24L - horizon)]
  }
  calendar
}
