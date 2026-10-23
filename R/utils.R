# utils.R ----------------------------------------------------------------------
# Shared helpers. Source this at the top of other scripts:
#   source("R/utils.R")

suppressMessages(library(data.table))

# Locate gzip -----------------------------------------------------------------
# gzip ships with Git for Windows but is usually NOT on PATH in PowerShell, so
# Sys.which() alone is unreliable. Check PATH first, then the usual Git
# locations.
find_gzip <- function() {
  g <- Sys.which("gzip")
  if (nzchar(g)) return(unname(g))

  candidates <- c(
    file.path(Sys.getenv("LOCALAPPDATA"), "Programs", "Git", "usr", "bin", "gzip.exe"),
    "C:/Program Files/Git/usr/bin/gzip.exe",
    "C:/Program Files (x86)/Git/usr/bin/gzip.exe"
  )
  hit <- candidates[file.exists(candidates)]
  if (length(hit) > 0) return(hit[1])

  stop("gzip not found. Install Git for Windows, or add gzip to PATH.")
}

# Read a gzipped CSV by streaming ----------------------------------------------
# fread(file.csv.gz) decompresses to a temp file and MEMORY-MAPS it, which needs
# a contiguous block the size of the decompressed data (~1 GB for these files).
# On a 6 GB machine that fails after a few files as the address space fragments:
#   "could not memory map it ... not enough contiguous virtual memory"
#
# Piping through gzip streams instead: ~200 MB of R memory regardless of file
# size, at the cost of being a little slower.
read_gz <- function(path, select = NULL, colClasses = NULL) {
  gz <- find_gzip()
  cmd <- paste(shQuote(gz), "-dc", shQuote(normalizePath(path, winslash = "/")))
  fread(cmd = cmd, select = select, colClasses = colClasses, showProgress = FALSE)
}

# IMPORTANT when reading raw NEA files -----------------------------------------
# The raw timestamp looks like "2017-01-01T00:00:59+08:00". fread recognises
# ISO8601 and silently parses it to POSIXct *in UTC*, so that value comes back as
# "2016-12-31 16:00:59" - the whole series shifted 8 hours earlier.
#
# Always force it to stay a string:
#   read_gz(f, select = c("timestamp", ...), colClasses = c(timestamp = "character"))
#
# then the first 13 characters ("2017-01-01T00") are the Singapore-local hour.
