# Air Pollution Prediction using Time Series Analysis
**Author:** Snehitha Gorantla  

---

# CHAPTER 1: INTRODUCTION

## 1.1 Background and Motivation
Air pollution is one of the most critical environmental challenges of the 21st century, affecting billions of people and causing millions of premature deaths annually. According to the WHO, ambient air pollution results in approximately **4.2 million deaths per year**. Among major pollutants, *ground-level ozone (O₃)* is particularly harmful—negatively impacting respiratory health, reducing crop yields, and damaging ecosystems.

Ground-level ozone is a **secondary pollutant**, formed through nonlinear photochemical reactions involving **NOx**, **VOCs**, and **sunlight**. Its concentration varies across space and time, shaped by meteorology, emissions, and atmospheric chemistry, making accurate ozone prediction both highly challenging and crucial for air quality management.

To monitor such patterns, the U.S. **Clean Air Status and Trends Network (CASTNET)**, established in 1987, provides long-term hourly ozone and atmospheric chemistry observations across 90+ rural sites. This dataset enables advanced modeling and long-term trend assessment.

Traditional statistical forecasting models like **ARIMA** struggle with nonlinear, high-dimensional atmospheric dynamics. Modern ML methods—particularly Gradient Boosting algorithms like LightGBM—offer improved capability to model these complexities.

### Key Challenges in Operational Forecasting
- **Data Quality & Completeness:** Missing values, sensor errors, calibration drift  
- **Feature Engineering:** Requires atmospheric science expertise  
- **Temporal Complexity:** Diurnal, weekly, and seasonal cycles overlap  
- **Spatial Heterogeneity:** Strong regional variability in pollution  
- **Multi-scale Interactions:** Local emissions + regional transport + weather patterns  

This project develops a comprehensive framework integrating preprocessing, feature engineering, and machine learning to achieve **near-perfect ozone predictions**.

---

## 1.2 Problem Statement
Transforming CASTNET datasets into reliable ozone forecasts involves overcoming four major technical obstacles:

### 1. Data Integration & Quality Assurance
Heterogeneous environmental datasets must be harmonized across monitoring sites, pollutant types, and temporal resolutions. This requires robust quality checks for missing data, sensor issues, and metadata inconsistencies.

### 2. Modeling Multi-Scale Temporal Dynamics
Ozone formation is influenced by:
- **Diurnal Cycles** (hourly patterns)
- **Synoptic Variability** (days–weeks)
- **Seasonal/Interannual Trends**

Capturing these requires engineered temporal features capable of representing periodic and long-term behaviors.

### 3. High-Volume Data Processing
With **21+ million hourly observations**, traditional time-series methods become insufficient. Efficient, memory-conscious data pipelines are essential.

### 4. Advanced Feature Engineering
Model success depends on:
- Lagged variables  
- Rolling statistics  
- Differencing  
- Cyclical encodings  

A systematic Python-based pipeline was implemented to generate these features.

---

# CHAPTER 2: DATA PROCESSING

## 2.1 Data Sources
CASTNET data consists of:
- **Hourly Ozone Data** (primary focus)
- **Site Metadata** (latitude, longitude, elevation)

Data spans **1987–2024** across multiple monitoring sites.

---

## 2.2 Data Extraction Architecture
A custom `OzoneDataExtractor` class handles:

1. Recursive scan for `.csv` files  
2. Metadata extraction (path, size, timestamps)  
3. Standardized date parsing and column naming  
4. Site isolation ensuring clean time-series segmentation  

---

## 2.3 Data Cleaning & Quality Control

### Filters Applied
- QA code ≥ 2 (validated data only)
- Invalid quality flags filtered  
- Negative ozone values removed  
- Only essential variables retained  

---

## 2.4 Final Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Records** | 21,667,030 |
| **Training Samples** | 18,193,749 |
| **Validation Samples** | 2,072,797 |
| **Test Samples** | 1,400,484 |
| **Monitoring Sites** | 84 |
| **Date Range** | 1987–2024 |

### Temporal Split Strategy
| Split | Period | Purpose |
|-------|--------|---------|
| Training | ≤ 2019 | Model learning |
| Validation | 2020–2022 | Hyperparameter tuning |
| Test | 2023+ | Final evaluation |

---

# CHAPTER 3: FEATURE ENGINEERING

## 3.1 Overview
A multi-stage pipeline converts raw time series into a supervised ML-ready feature matrix with **67 features**.

---

## 3.2 Temporal Feature Extraction & Cyclical Encoding
Ozone exhibits strong **diurnal** and **seasonal** periodicity. Sine/cosine transformations prevent discontinuities:

```
sin(2πt/T), cos(2πt/T)
```

### Features Generated
- **Diurnal:** hour, hour_sin, hour_cos, is_daytime
- **Weekly:** day_of_week, is_weekend, dow_sin, dow_cos
- **Seasonal:** month, month_sin, month_cos, week_of_year, day_of_year

---

## 3.3 Lagged Variable Generation
Captures autocorrelation at multiple scales:

| Lag Type | Hours |
|----------|-------|
| Short-term | 1, 2, 3 |
| Transitional | 6, 12 |
| Daily | 24, 48 |
| Weekly | 168 |

**Total lag features: 8**

---

## 3.4 Rolling Window Statistics
Windows: **3, 6, 12, 24, 168 hours**

Statistics computed per window:
- Mean, Median, Std
- Min, Max, Range

**Total rolling features: 30**

---

## 3.5 Additional Features
- **Differencing:** Δ1 (hourly change), velocity, acceleration
- **Site Statistics:** Percentiles (p25, p50, p75, p90), above-median flags
- **Spatial:** Latitude, Longitude

---

# CHAPTER 4: MODEL TRAINING

## 4.1 Model Selection
**LightGBM** (Light Gradient Boosting Machine) was selected for:
- Efficient handling of large datasets (18M+ samples)
- Native support for categorical features
- Fast training with histogram-based algorithms
- Strong performance on tabular data

---

## 4.2 Hyperparameter Tuning
**Optuna** framework used for Bayesian optimization:

| Setting | Value |
|---------|-------|
| Trials | 30 |
| Tuning Sample | 20% of training data |
| Optimization Target | Validation RMSE |
| Search Algorithm | TPE Sampler |

---

# CHAPTER 5: MODEL EVALUATION

## 5.1 Overall Performance Metrics

| Metric | Test Set Value |
|--------|----------------|
| **RMSE** | 0.1270 |
| **MAE** | 0.0311 |
| **Median AE** | 0.0145 |
| **R²** | 0.9999 |
| **MAPE** | 0.13% |
| **Pearson r** | 1.0000 |

### Performance Across Splits
| Split | RMSE | MAE | R² |
|-------|------|-----|-----|
| Training | 0.0581 | 0.0306 | 1.0000 |
| Validation | 0.0895 | 0.0274 | 1.0000 |
| Test | 0.1270 | 0.0311 | 0.9999 |

---

## 5.2 Temporal Error Analysis

### RMSE by Hour of Day
| Best Hours | Worst Hours |
|------------|-------------|
| 10:00 (0.092) | 2:00 (0.167) |
| 19:00 (0.096) | 3:00 (0.129) |
| 20:00 (0.097) | 14:00 (0.154) |

### RMSE by Month
| Best Months | Worst Months |
|-------------|--------------|
| March (0.080) | June (0.183) |
| May (0.108) | July (0.180) |
| October (0.105) | February (0.132) |

**Observation:** Higher errors in summer months (June-July) correspond to increased photochemical ozone production and variability.

---

## 5.3 Residual Analysis

| Statistic | Value |
|-----------|-------|
| Mean Residual | -0.0017 |
| Std Residual | 0.1270 |
| Skewness | -18.16 |

The near-zero mean residual indicates an unbiased model. Negative skewness suggests occasional large under-predictions during rapid ozone changes.

---

# CHAPTER 6: CONCLUSIONS

## 6.1 Key Achievements

1. **Exceptional Predictive Accuracy**
   - R² = 0.9999 (explains 99.99% of variance)
   - MAPE = 0.13% (average error of 0.13%)
   - Near-perfect Pearson correlation (r = 1.0000)

2. **Robust Generalization**
   - Consistent performance across 84 monitoring sites
   - 93% of sites achieve RMSE < 0.2
   - Proper temporal train/val/test splits prevent data leakage

3. **Physically Meaningful Features**
   - Feature importance aligns with atmospheric science
   - Short-term ozone history dominates predictions
   - Temporal patterns (diurnal, seasonal) captured effectively

4. **Scalable Infrastructure**
   - Processes 21M+ records efficiently
   - Reproducible pipeline with saved preprocessing statistics
   - Memory-optimized data storage (Parquet format)

---

## 6.2 Limitations

1. **Summer Season Challenges**
   - June/July RMSE ~2x higher than spring months
   - Photochemical variability harder to predict

2. **Residual Skewness**
   - Occasional large errors during rapid ozone changes
   - May benefit from ensemble approaches

3. **Heavy Reliance on Autocorrelation**
   - Model primarily learns "ozone ≈ recent ozone"
   - May struggle with anomalous events

---

# Project Structure

```
MATH699P/
├── Data/
│   ├── processed_data/
│   │   ├── ozone_processed.csv
│   │   ├── site_metadata.csv
│   │   └── model_ready_dask/
│   │       ├── train/
│   │       ├── val/
│   │       ├── test/
│   │       └── preprocessing.pkl
│   └── model_outputs/
│       ├── models/
│       │   ├── lightgbm_ozone_model.txt
│       │   └── lightgbm_ozone_model.pkl
│       ├── results/
│       │   ├── model_metrics.csv
│       │   ├── feature_importance.csv
│       │   ├── training_report.txt
│       │   └── evaluation_report.txt
│       └── plots/
│           ├── lightgbm_results.png
│           ├── timeseries_sample.png
│           └── residual_analysis.png
├── notebooks/
│   ├── data_extraction.ipynb
│   ├── feature_engineering.ipynb
│   ├── model_preparation.ipynb
│   ├── lightgbm_training.ipynb
│   └── model_evaluation.ipynb
└── README.md
```

---

# Requirements

```
numpy
pandas
lightgbm
optuna
scikit-learn
matplotlib
seaborn
scipy
pyarrow
```

---

# Data

U.S. EPA Clean Air Status and Trends Network (CASTNET)

---
