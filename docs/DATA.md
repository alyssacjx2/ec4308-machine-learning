# Data

Four sources, matching the project proposal. **None of this is in the repo and none of it
ever will be** — see [Conventions](#conventions). Raw data totals ~1.7 GB compressed
(44 GB uncompressed) and is reproducible from the scripts in `R/`.

```
data/
├── raw/
│   ├── nea/        45 × Historical<Var>acrossSingapore<YYYY>.csv.gz   1.6 GB
│   ├── ema/        EMA Compiled Data.zip                              4 MB
│   ├── era5/       era5_singapore_2016_2026.nc                        23 MB
│   └── holidays/   sg_public_holidays_2016_2026.csv                   12 KB
├── interim/        hourly × station aggregates (generated)
└── processed/      final merged modelling dataset (generated)
```

## Sources

### i. EMA — half-hourly system demand
`raw/ema/EMA Compiled Data.zip` — yearly CSVs, **1 Jan 2016 – 30 Sep 2026**, 188,448 rows
(3,926 days × 48 periods). No missing days or periods.

Columns include `system_demand_actual_mw` (primary target), `nem_demand_actual_mw`, and
**`nem_demand_forecast_mw` — EMA's own operational forecast**, which is a far stronger
benchmark than seasonal-naive and should be used as one.

Compiled from 562 weekly EMA spreadsheets; see the README inside the zip for provenance.

### ii. NEA — station weather
`raw/nea/` — 45 gzipped CSVs, 5 variables × 2016–2024, minute-level, per station.

| Variable | Coverage | Note |
|---|---|---|
| Air temperature | 2016–2024 | 2016 starts 15 Apr |
| Relative humidity | 2016–2024 | **11 columns, not 12** |
| Rainfall | 2016–2024 | **2016 holds only 2–3 Dec** |
| Wind speed | 2016–2024 | |
| Wind direction | 2016–2024 | circular — do not average naively |

Source: data.gov.sg bulk annual downloads. All 45 dataset IDs are in
`docs/nea_dataset_manifest.json`; `R/02_extract_nea_weather.R` fetches and aggregates them.

**Still outstanding: 2025 – Sep 2026.** No bulk files exist past 2024; that span must come
from the live API (`https://api.data.gov.sg/v1/environment/<variable>?date=YYYY-MM-DD`,
which returns a full day per call).

### iii. ERA5 — reanalysis
`raw/era5/era5_singapore_2016_2026.nc` — hourly, **1 Jan 2016 00:00 – 30 Sep 2026 23:00**,
94,224 steps, uniform spacing, no gaps. Grid is 3 × 4 cells at 0.25°
(lat 1.00–1.50, lon 103.50–104.25).

| Variable | Long name | Units |
|---|---|---|
| `ssrd` | Surface solar radiation downwards | J m⁻² |
| `sshf` | Surface sensible heat flux | J m⁻² |
| `tcc` | Total cloud cover | 0–1 |
| `fdir` | Total sky direct solar radiation at surface | J m⁻² |
| `tp` | Total precipitation | m |

**Differs from the proposal** — the proposal listed surface net solar radiation (`ssr`) and
high/medium/low cloud cover (`hcc`/`mcc`/`lcc`), none of which are in the file; it gained
`fdir` and `tp` instead. The report's data section should describe what we actually have.

### iv. Public holidays
`raw/holidays/sg_public_holidays_2016_2026.csv` — 144 rows, 2016–2026, 11–15 per year.

Columns: `date`, `day`, `holiday`, `type`, `source`, `note`. `type` distinguishes
**gazetted** from **observed** (in-lieu Mondays when a holiday falls on a Sunday) — the
observed day is the one that moves electricity load, so keep the distinction.

data.gov.sg/MOM only publishes 2020 onwards; 2016–2019 come from time.sg, with 2018–19
cross-checked against MOM. Provenance is recorded per row in the `source` column.

## Conventions

- Nothing under `data/` is committed. GitHub rejects files over 100 MB; the NEA CSVs are
  ~1 GB each uncompressed.
- `data/processed/` is the exception worth revisiting — once the merged hourly dataset
  exists (~20–40 MB) it is small enough to commit, and it is the only thing needed to run
  models.
- Shared raw copy: _[add Drive link]_
- Requires **`R.utils`** — `data.table::fread()` cannot read `.gz` without it.

## Known quirks

Things that will silently corrupt results if missed:

1. **Humidity files have 11 columns**, every other NEA variable has 12 (no `reading_type`).
   Select columns by name, never by position, and never `rbindlist()` raw files across
   variables.
2. **Timestamps carry `+08:00`.** The 2017 air-temperature file's first rows read as
   2016-12-31 in UTC. Bucket on local time or everything shifts eight hours.
3. **The station panel changes over time** — 21 stations in 2016, 12 in mid-2017, 16 now.
   Averaging whatever reports each hour bakes in artificial level shifts. Fix a station
   panel across the sample, or at minimum record the station count per hour.
4. **Rainfall 2016 is two days.** Usable rainfall starts Jan 2017.
5. **Wind direction is circular.** The naive mean of 350° and 10° is 180°, pointing the
   wrong way. Average unit vectors via `atan2`.
6. **ERA5 accumulated variables** (`ssrd`, `sshf`, `fdir`, `tp`) are totals over the hour
   *ending* at the timestamp, in J m⁻², not instantaneous W m⁻². Divide by 3600 for mean flux.
7. **ERA5 is published with a ~5-day delay**, so it cannot be used contemporaneously in a
   real-time forecast — lag it 120h (per Denis's feedback). NEA station readings are
   real-time and can be used at time *t*.
8. **EMA**: 398 half-hours show system demand below NEM demand, which cannot happen by
   definition. Left as published; mention in the report.
