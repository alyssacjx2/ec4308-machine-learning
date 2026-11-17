# run Rscript install_packages.R from the repository root.
# Installs missing packages, including dependencies. Internet access is required.
# gzip must be installed separately; see README.md for the Windows instructions.

PROJECT_PACKAGES <- c(
  "data.table", # CSV I/O, joins, aggregation and rolling features
  "ncdf4",     # ERA5 netCDF input
  "hdm",       # LASSO and post-LASSO estimation
  "ranger",    # random forest and bagging
  "lightgbm",  # gradient-boosted trees
  "forecast",  # seasonal ARIMA order selection and estimation
  "R.utils",   # direct gzip reads in optional older viewers/selection code
  "sandwich"   # Newey-West calculations in the optional LightGBM viewer
)

InstallProjectPackages <- function() {
  available <- vapply(PROJECT_PACKAGES, requireNamespace, logical(1), quietly = TRUE)
  missing <- PROJECT_PACKAGES[!available]
  if (length(missing)) {
    message("Installing missing packages: ", paste(missing, collapse = ", "))
    install.packages(missing, repos = "https://cloud.r-project.org")
  }
  available <- vapply(PROJECT_PACKAGES, requireNamespace, logical(1), quietly = TRUE)
  if (any(!available)) stop("Packages still unavailable: ",
    paste(PROJECT_PACKAGES[!available], collapse = ", "),
    ". Inspect the installation errors for any required system libraries/toolchain.")
  print(data.frame(package = PROJECT_PACKAGES,
    version = vapply(PROJECT_PACKAGES, function(p) as.character(packageVersion(p)),
                     character(1))), row.names = FALSE)
  message("Package setup complete. This does not download project data or fit models.")
}

if (sys.nframe() == 0L) InstallProjectPackages()
