# Gradient Boosting Model to Predict Air Pollution in California

**Author:** Snehitha Gorantla  
**Degree:** Master of Science in Data Science and Analytics  
**Institution:** California State University, Chico  

---

## Overview

This project develops a complete, production-quality pipeline for forecasting ground-level ozone (O₃) concentrations and assessing long-term vegetation exposure risk across California's federal air quality monitoring network. Using over three decades of hourly observations from the U.S. EPA Clean Air Status and Trends Network (CASTNET), the system integrates atmospheric chemistry measurements, geospatial wildfire perimeter data, and low-cost particulate matter sensors into a unified machine learning framework.

Three separate LightGBM gradient-boosted regression models are trained to forecast ozone at **t+1h**, **t+8h**, and **t+24h** horizons, alongside a dedicated binary classifier for predicting National Ambient Air Quality Standard (NAAQS) exceedances at 70 ppb. An interactive Streamlit dashboard visualizes forecasts, vegetation exposure trends, wildfire exposure, and model diagnostics in real time.

---

## Key Results

| Horizon | MAE (ppb) | RMSE (ppb) | R² | vs. Persistence | vs. Climatology |
|---------|-----------|------------|-----|-----------------|-----------------|
| t+1h    | 2.49      | 3.48       | 0.932 | −39.8% MAE    | −71.9% MAE      |
| t+8h    | 4.42      | 6.14       | 0.858 | −50.6% MAE    | −46.7% MAE      |
| t+24h   | 6.24      | 8.21       | 0.753 | −46.5% MAE    | −32.4% MAE      |

**NAAQS Exceedance Classifier (t+8h, test set):** AUC-ROC = 0.918 · Average Precision = 0.628 · F1 = 0.571 · Recall = 54.1%

**Vegetation exposure:** W126 declining at ≈ 0.4 ppm·h/yr network-wide; 7 of 10 sites improving at p < 0.05 (Mann-Kendall). All sites exceed European AOT40 thresholds, making the US EPA W126 metric the appropriate standard for California.

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_ca_data_extraction_combined.ipynb     # Data acquisition and QA/QC
│   ├── 02_ca_wildfire_features.ipynb            # Geospatial fire exposure features
│   ├── 03_ca_feature_engineering.ipynb          # Full feature store (~160 features)
│   ├── 04_ca_vegetation_exposure_gaps.ipynb     # AOT40 / W126 vegetation analysis
│   ├── 05_model_preparation_fixed.ipynb         # Model training and evaluation
│   ├── 06_ca_models_gap_fixes.ipynb             # Bias correction and SARIMA fixes
│   └── 07_ca_evaluation_gaps.ipynb              # Diagnostic evaluation
│
│
├── data/
│   ├── raw/                                      # Downloaded CASTNET CSV exports
│   │   ├── ozone*.csv
│   │   ├── hourly_gas*.csv
│   │   ├── drychem*.csv
│   │   ├── wet_concentration*.csv
│   │   ├── site.csv
│   │   └── fire24_1.gdb/                         # CAL FIRE FRAP geodatabase
│   │
│   └── processed_data/california/               # Pipeline outputs
│       ├── ca_ozone.csv
│       ├── ca_site_metadata.csv
│       ├── ca_site_static_features.csv
│       ├── ca_hourly_gas.csv
│       ├── ca_drychem.csv
│       ├── ca_wet_conc.csv
│       ├── ca_purpleair.csv
│       ├── ca_drydep_weekly.csv
│       └── ca_total_deposition.csv
│
├── features_parquet/                             # Feature store (partitioned by site)
│   └── ca_feature_store.parquet/
│
├── models/                                       # Trained model artifacts
│   ├── lgb_reg_t1.txt
│   ├── lgb_reg_t8.txt
│   ├── lgb_reg_t24.txt
│   ├── lgb_clf_naaqs_t8.txt
│   ├── train_medians.pkl
│   └── eval_results.pkl
│
├── dashboard_data/                               # Dashboard JSON/Parquet exports
│   ├── site_metadata.json
│   ├── model_metrics.json
│   ├── feature_importance.json
│   ├── clf_metadata.json
│   ├── ozone_predictions.parquet
│   ├── annual_vegetation.parquet
│   ├── site_fire_exposure_daily.parquet
│   ├── site_fire_exposure_annual.parquet
│   ├── coverage_audit.csv
│   └── pipeline_status.json
│
├── california_models_patched/                    # Gap-fix model artifacts
│   ├── gap1_bias_corrections.pkl
│   ├── gap2_dev_eval.pkl
│   ├── sarima_forecasts_patched.pkl
│   └── site_coverage_audit.csv
│
└── requirements.txt
```

---

## Pipeline Overview

The project is organized as seven sequential notebooks. Each notebook reads from the outputs of the previous one and writes its own outputs to disk, allowing any stage to be re-run independently.

### Notebook 1 — Data Acquisition and QA/QC
`01_ca_data_extraction_combined.ipynb`

Downloads and validates all CASTNET data streams for California. Key operations:

- Filters national CASTNET exports to California sites using `CaliforniaSiteExtractor`
- Applies QA code filtering (code ≥ 2), invalid flag removal, physical range filters [0, 300 ppb], and deduplication
- Computes Haversine great-circle distances from each site to the Pacific coast and assigns marine influence classification (COASTAL / TRANSITION / INLAND)
- Loads and merges: hourly ozone, hourly gas chemistry (NO, NO₂, NOₓ, NOy, HNO₃, SO₂, CO, NH₃), weekly dry filter-pack chemistry, weekly precipitation chemistry, dry deposition fluxes, PurpleAir PM₂.₅, and plant phenology data
- Writes 12 clean CSVs to `processed_data/california/`

**Three documented bug fixes in wet chemistry loading:** NaN handling in `VALCODE` filtering; schema disambiguation between `WET_CONCENTRATION.csv` and `WETCHEM.csv`; wrong-file detection.

**Sites covered:** 10 active California CASTNET stations spanning 1990–2025.

| Site ID | Name | Region | Elevation (m) | Coast (km) |
|---------|------|--------|---------------|------------|
| SEK402 | Sequoia NP | Sierra Nevada South | 1,920 | 221 |
| SEK430 | Sequoia NP | Sierra Nevada South | 1,920 | 221 |
| YOS404 | Yosemite NP | Sierra Nevada Central | 1,603 | 187 |
| JOT403 | Joshua Tree NP | Mojave Desert | 1,244 | 148 |
| LAV410 | Lassen Volcanic NP | Cascades | 1,756 | 183 |
| TRI193 | Trinity | Northern Coast Range | 744 | 56 |
| PIN414 | Pinnacles NP | Central Coast | 335 | 64 |
| SND152 | San Bernardino NF | Southern CA Mountains | 1,737 | 142 |
| ABF404 | Angeles Big Bear | San Bernardino Mtns | 2,065 | 139 |
| CAN407 | Cabrillo NM | Southern Coast | 80 | 7 |

---

### Notebook 2 — Wildfire Exposure Feature Construction
`02_ca_wildfire_features.ipynb`

Converts CAL FIRE FRAP fire perimeter polygons into model-ready daily and annual exposure time series.

- Reads `fire24_1.gdb` (layer `firep24_1`) using `geopandas` and `fiona`
- Reprojects all geometries to **EPSG:3310** (California Albers Equal Area) for accurate metric-unit operations
- Buffers each monitoring site to a **50 km radius** and computes polygon-on-polygon intersection with each fire perimeter: `gpd.overlay(fires, site_buffers, how="intersection")`
- Annual aggregation: `burned_km2_50km`, `n_fires_50km`, `is_fire_year_50km` (threshold ≥ 5 km²)
- Daily time series via **delta-sweep algorithm** (O(F log F) vs naïve O(F × D)):
  - `+burned_km2` on `alarm_date`; `−burned_km2` on `cont_date + 1`; `cumsum()` reconstructs daily active exposure
- Rolling windows: 7d (acute smoke), 30d (seasonal), 90d (cumulative burden)
- Outputs: `site_fire_exposure_annual.parquet`, `site_fire_exposure_daily.parquet`

**Join to hourly spine:**
```python
df_hourly["date"] = pd.to_datetime(df_hourly["DATE_TIME"]).dt.floor("D")
df_hourly = df_hourly.merge(fire_daily, on=["SITE_ID", "date"], how="left")
```

---

### Notebook 3 — Feature Engineering
`03_ca_feature_engineering.ipynb`

Assembles the complete **~160-feature** matrix used for model training.

**Feature categories:**

| Group | Count | Key features |
|-------|-------|--------------|
| Temporal / calendar | ~24 | hour/month/dayofyear sin+cos pairs, is\_fire\_season, is\_peak\_photo, year\_trend |
| Site static | 6 | ELEVATION\_M, COAST\_DIST\_KM, LAT, LON, MARINE\_CODE, TERRAIN\_CODE |
| Ozone autoregressive | ~50 | Lags at 1/2/3/6/12/24/48/168h; rolling mean/std/min/max at 3/6/12/24/168h windows; diffs at 1/24/168h; velocity; acceleration; 8hr\_max; NAAQS\_exceed |
| Hourly gas | ~40 | NO/NO₂/NOy/SO₂/CO raw + lags + rolling; NOₓ\_total; NO\_NO₂\_ratio; NOy\_minus\_NOx; TEMP\_2M; is\_hot\_day |
| Dry/wet chemistry | ~27 | Weekly filter-pack and precipitation ions, forward-filled via `merge_asof(direction="backward")` |
| PurpleAir PM₂.₅ | ~10 | PM25\_EPA raw + 24/72h rolling; is\_wildfire\_smoke; smoke\_x\_fire\_season |
| Fire exposure | 5 | fire\_km2\_active\_today; n\_active\_fires; 7d/30d/90d rolling sums |
| CA-specific | 5 | high\_pm25; is\_coastal\_site; hour\_x\_coastal; ozone\_annual\_site\_mean; ozone\_site\_train\_mean |

**Leakage prevention:** All site percentiles, diurnal means, and the annual ozone mean feature are computed from training data (≤ 2018) only. The annual mean feature uses the previous year's value so the current year's ozone is never seen at feature construction time.

**Important note on lag features:** Lags at t−48 through t−120 overlap substantially with information already captured by t−24 and hour-of-day features under strong diurnal patterns. The t−24 lag characterizes day-over-day persistence during smoke/stagnation episodes; the t−168 lag captures the weekly emission cycle from weekday vs. weekend traffic.

**Output:** Flat (non-hive-partitioned) Parquet feature store with Snappy compression. `ca_feature_metadata.pkl` records the exact ordered feature list, train/val/test split boundaries, and all config parameters.

---

### Notebook 4 — Vegetation Exposure Analysis
`04_ca_vegetation_exposure_gaps.ipynb`

Computes AOT40 and W126 vegetation exposure metrics and diagnoses three methodological problems in prior analyses.

**Accumulation window:** Growing season (April–September), daylight hours (08:00–20:00 local). Site-years with < 70% of theoretical hours are excluded.

**AOT40:**
$$\text{AOT40} = \sum_{\substack{t \in [\text{Apr--Sep}] \\ 08{:}00 \leq t \leq 20{:}00}} \max(O_{3,t} - 40,\; 0) \quad \text{(ppb·h)}$$

**W126 (Lefohn, 1988):**
$$\text{W126} = \sum_t \frac{O_{3,t}/1000}{1 + 4403 \cdot e^{-0.126 \cdot O_{3,t}}} \quad \text{(ppm·h)}$$

**Three corrected analytical findings:**

1. **Fire-year confound:** A naïve comparison showed fire years had 18% *lower* AOT40, apparently suggesting fires suppress ozone. This was entirely explained by secular trend confounding — fire years fall late in a multi-decade declining record. After within-site Z-score detrending, the fire-year signal was near zero (Mann-Whitney p ≈ 0.95 two-sided), indicating that UV attenuation and precursor injection from smoke approximately cancel at the network scale.

2. **EU threshold saturation:** European AOT40 thresholds (crops: 3,000 ppb·h; forests: 5,000 ppb·h) were calibrated for European background ozone (~25–35 ppb). Every California site exceeds the forest threshold by a factor of ≥6 in every year — the EU metric provides zero differentiation. The US EPA W126 thresholds (low: 7 ppm·h; high: 17 ppm·h) provide meaningful site-level contrast.

3. **Binary label ceiling effect:** A stacked bar chart of EU High/Moderate/Low always showed ≥100% High, giving an apparent flat trend. Mann-Kendall applied to continuous W126 found 7 of 10 sites improving at p < 0.05, with a network decline of ≈ 0.4 ppm·h/yr.

**Trend results (Theil-Sen slope on W126):**

| Site | Slope (ppm·h/yr) | p-value | Direction |
|------|-----------------|---------|-----------|
| SEK402 | −0.62 | 0.003 | Improving |
| YOS404 | −0.58 | 0.004 | Improving |
| SEK430 | −0.51 | 0.008 | Improving |
| JOT403 | −0.44 | 0.012 | Improving |
| SND152 | −0.38 | 0.019 | Improving |
| ABF404 | −0.29 | 0.041 | Improving |
| LAV410 | −0.18 | 0.142 | Stable |
| TRI193 | −0.14 | 0.218 | Stable |
| PIN414 | −0.09 | 0.384 | Stable |
| CAN407 | −0.04 | 0.612 | Stable |

---

### Notebook 5 — Model Training and Evaluation
`05_model_preparation_fixed.ipynb`

Trains three regression models and one binary classifier.

**Temporal split:**
- Train: ≤ 2018 (28 years of CARB regulatory baseline)
- Validation: 2019–2020 (post-Camp Fire wildfire intensification + COVID NOₓ collapse)
- Test: 2021–2025 (post-pandemic, sustained high-fire era)

**Why three separate models (not multi-output):** Feature importance shifts dramatically by horizon. Autoregressive features account for 72% of gain at t+1h but only 21% at t+24h. A shared model would be forced to compromise between these different structures.

**LightGBM regression hyperparameters:**
```python
{
    "objective":         "regression_l1",   # MAE — robust to wildfire spike outliers
    "num_leaves":        127,
    "learning_rate":     0.05,
    "n_estimators":      2000,              # early stopping governs actual count
    "min_child_samples": 30,
    "subsample":         0.8,
    "colsample_bytree":  0.7,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
}
```

**NAAQS binary classifier:**
```python
{
    "objective":       "binary",
    "num_leaves":      63,
    "scale_pos_weight": 5,   # exceedances ≈ 1–5% of hours
}
```

Optimal decision threshold (0.312) is selected by maximizing F1 on the **validation set** and applied to the test set without further adjustment.

**Boundary row exclusion:** For a row at time T with horizon H, if the forecast target T+H crosses a split boundary, the row is excluded from both adjacent splits to prevent any temporal leakage.

**SITE_ID encoding:** Integer-encoded via `LabelEncoder` (not one-hot) so the model generalizes to new sites and learns site-specific baselines from the shared feature space. The encoder is saved in `train_medians.pkl` for inference.

**Baselines:**
- **Persistence:** predict current ozone = future ozone
- **Climatology:** predict training-period mean for (SITE_ID, month, hour)

---

### Notebook 6 — Model Coverage Fixes
`06_ca_models_gap_fixes.ipynb`

Addresses three specific analytical failures identified after the main modeling notebook.

**Historical-only sites (SEK402, CON186, YOS204, LPO010):**
All records fall before the 2019 validation boundary. Standard held-out evaluation is impossible. A leave-last-year-out bias correction is applied: the global model's systematic directional error and monthly seasonal component are estimated on the most recent 12 months of each site's training data and applied as inference-time corrections.

> These corrections are optimistic because the pseudo-validation window is not independent of training. Results for these sites should not be reported alongside held-out evaluation metrics.

**New site without training data (DEV412):**
Walk-forward residual fine-tuning: the first 12 months of available data serve as a calibration window. A small residual LightGBM model (31 leaves, 500 trees) is trained on the global model's errors and added to its predictions.

**SARIMA `ValueWarning: No frequency information`:**
Root cause: `dropna()` → irregular index → `SARIMAX` internally strips `freq`. The commonly suggested `.asfreq("h")` fix does not work — SARIMAX's internal `_get_dates_from_model` strips freq again.

```python
# WRONG — warning still fires after .asfreq():
mod = SARIMAX(series.dropna().asfreq("h"), ...)

# CORRECT — pass numpy array; no DatetimeIndex, no frequency check:
mod = SARIMAX(series.dropna().values, order=(1,1,1), seasonal_order=(1,1,1,24), ...)
fc_index = pd.date_range(start=forecast_start, periods=n, freq="h")
```

SARIMA parameters are cached by config hash so re-runs skip the expensive Kalman filter optimization (~60–120s per site → <1s).

---

### Notebook 7 — Diagnostic Evaluation
`07_ca_evaluation_gaps.ipynb`

Post-hoc investigation of three performance failures identified in Chapter IV.

**t+24h exceedance recall ≈ 0%:**
- Structural miss rate: 60–70% of true exceedances have `y_hat < 70 ppb` regardless of threshold
- Root cause: autoregressive features decay from 72% of gain (t+1h) to 21% (t+24h); afternoon photochemical episodes are not predictable from the prior day's observations alone
- Fix: Platt scaling maps regression output to calibrated probability P(exceed | ŷ); AUC = 0.84 on test set; threshold-free operating point selection

**No-training site extrapolation penalty:**
- RMSE penalty ≈ 1.2–2.4 ppb vs trained-site mean
- Residual correlation identifies the "analogue site" (training site whose error pattern best matches the unseen site)

**LAV410 t+24h R² = 0.401:**
Three competing hypotheses tested:
1. **Fire-season confound (H2):** Fire vs. non-fire R² stratification showed similar skill in both seasons → H2 not the primary driver
2. **Diurnal phase shift (H3):** Model predicts afternoon ozone peak ≈ 2 hours late vs. observations → H3 supported
3. **Cascade chemistry regime (H1):** Systematic summer positive bias consistent with NOₓ-limited photochemistry receiving Pacific Northwest / Canadian wildfire precursors that the VOC-limited training regime misrepresents

---

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

**`requirements.txt`:**
```
numpy
pandas
geopandas
shapely
fiona
pyproj
pyarrow
dask
lightgbm
optuna
scikit-learn
statsmodels
pymannkendall
scipy
matplotlib
seaborn
streamlit
plotly
```

> **Note:** `geopandas` requires GDAL. On macOS: `brew install gdal`. On Linux: `apt-get install gdal-bin libgdal-dev`. On Windows: use the `conda` installer.

---

## Data

**CASTNET (primary):**  
U.S. EPA Clean Air Status and Trends Network  
https://www.epa.gov/castnet/castnet-data  
Download: ozone, hourly gas, dry chemistry, wet chemistry, site metadata, dry deposition CSVs for all years.

**CAL FIRE FRAP (wildfire perimeters):**  
California Department of Forestry and Fire Protection  
https://www.fire.ca.gov/what-we-do/fire-resource-assessment-program  
Download: `fire24_1.gdb` (2024 release, layer `firep24_1`)

**PurpleAir (PM₂.₅ proxy):**  
Available at three Sierra Nevada sites from 2019 onward (SEK430, YOS404, LAV410).  
https://www2.purpleair.com

Place downloaded files in `data/raw/` before running Notebook 1.

---

## Reproducing Results

Run notebooks in order. Each notebook checks for its required inputs and will raise a clear error if a predecessor has not been run.

```bash
jupyter nbconvert --to notebook --execute notebooks/01_ca_data_extraction_combined.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_ca_wildfire_features.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_ca_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_ca_vegetation_exposure_gaps.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_model_preparation_fixed.ipynb
jupyter nbconvert --to notebook --execute notebooks/06_ca_models_gap_fixes.ipynb
jupyter nbconvert --to notebook --execute notebooks/07_ca_evaluation_gaps.ipynb
```

Expected total runtime on a machine with 16 GB RAM and 8 cores: approximately 4–6 hours, dominated by Notebook 3 (feature engineering, ~160 features × 1.7M rows) and Notebook 5 (model training with early stopping).

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| MAE loss, not MSE | Robust to wildfire-driven ozone spike outliers that would dominate MSE |
| Three separate models per horizon | Feature importance shifts fundamentally from AR-dominated (t+1h) to calendar-dominated (t+24h); a shared model would compromise at every horizon |
| Temporal split at 2018 | Captures the 28-year CARB regulatory baseline; keeps 2019–2020 (Camp Fire regime shift + COVID NOₓ collapse) as an out-of-distribution stress test |
| EPSG:3310 for all spatial ops | California Albers Equal Area ensures accurate meter-unit distances and area calculations; WGS84 degrees produce asymmetric buffers at California latitudes |
| Delta sweep for fire exposure | O(F log F) vs naïve O(F × D) for multi-month fires across three decades |
| Intersection area not presence/absence | Continuous burned area within buffer carries more signal than a binary fire/no-fire flag |
| Flat Parquet, not hive-partitioned | Avoids Dask/PyArrow version-skew `KeyError: 'SITE_ID'` bug on partition key reconstruction |
| merge_asof direction="backward" | Prevents leakage: each hourly row sees only completed weekly chemistry samples |
| Lag features clipped before rolling stats | Extreme raw values do not corrupt the full rolling window history |
| W126 over AOT40 for California | EU AOT40 thresholds saturate at "High" for all CA sites in all years; W126 provides actionable differentiation calibrated to North American background ozone levels |

---

## Citation

If you use this code or data pipeline, please cite:

> Gorantla, S. (2026). *Gradient Boosting Model to Predict Air Pollution in California*. Master's Project, California State University, Chico.

---

## License

This project is for academic use. CASTNET data are in the public domain (U.S. EPA). CAL FIRE FRAP data are publicly available under California state open data terms. See individual data source portals for terms of use.
