# Air Pollution Prediction using Time Series Analysis
**Author:** Snehitha Gorantla

---

# CHAPTER 1: INTRODUCTION

## 1.1 Background and Motivation
Air pollution is one of the most critical environmental challenges of the 21st century, affecting billions of people and causing millions of premature deaths annually. According to the WHO, ambient air pollution results in approximately **4.2 million deaths per year**. Among major pollutants, *ground-level ozone (O₃)* is particularly harmful—negatively impacting respiratory health, reducing crop yields, and damaging ecosystems.

Ground-level ozone is a **secondary pollutant**, formed through nonlinear photochemical reactions involving **NOx**, **VOCs**, and **sunlight**. Its concentration varies across space and time, shaped by meteorology, emissions, and atmospheric chemistry, making accurate ozone prediction both highly challenging and crucial for air quality management.

To monitor such patterns, the U.S. **Clean Air Status and Trends Network (CASTNET)**, established in 1987, provides long-term hourly ozone and atmospheric chemistry observations across 90+ rural sites. This dataset enables advanced modeling and long-term trend assessment.

Traditional statistical forecasting models like **ARIMA** struggle with nonlinear, high-dimensional atmospheric dynamics. Modern ML/DL methods—Random Forests, Gradient Boosting, LSTMs—offer improved capability to model these complexities.

### Key Challenges in Operational Forecasting
- **Data Quality & Completeness:** Missing values, sensor errors, calibration drift.  
- **Feature Engineering:** Requires atmospheric science expertise.  
- **Temporal Complexity:** Diurnal, weekly, and seasonal cycles overlap.  
- **Spatial Heterogeneity:** Strong regional variability in pollution.  
- **Multi-scale Interactions:** Local emissions + regional transport + weather patterns.  

This project develops a comprehensive framework integrating preprocessing, feature engineering, and ML-ready dataset generation.

---

## 1.2 Problem Statement
Transforming CASTNET datasets into reliable ozone forecasts involves overcoming four major technical obstacles:

### 1. Data Integration & Quality Assurance
Heterogeneous environmental datasets must be harmonized across monitoring sites, pollutant types, and temporal resolutions. This requires robust quality checks for missing data, sensor issues, and metadata inconsistencies.

### 2. Modeling Multi-Scale Temporal Dynamics
Ozone formation is influenced by:
- **Diurnal Cycles**  
- **Synoptic Variability (days–weeks)**  
- **Seasonal/Interannual Trends**

Capturing these requires engineered temporal features capable of representing periodic and long-term behaviors.

### 3. High-Volume Data Processing
With **24+ million hourly observations**, traditional time-series methods become insufficient. Efficient, memory-conscious data pipelines are essential.

### 4. Advanced Feature Engineering
Model success depends on:
- lagged variables  
- rolling statistics  
- differencing  
- cyclical encodings  

A systematic Python-based pipeline is implemented to generate these features.

---

## 1.3 Project Objectives

### 1. Scalable Data Extraction System
Automated, reproducible ingestion of multi-year CASTNET records with unified schema definitions.

### 2. Robust Preprocessing Pipeline
Includes:
- missing timestamp detection and filling  
- QA/QC filtering  
- site-standardized data formats  

### 3. High-Dimensional Feature Generation
Creates:
- cyclical encodings  
- lag features  
- rolling window statistics  
- differencing metrics  

### 4. Model-Ready Dataset Preparation
Includes:
- normalization  
- memory optimization  
- training/validation splits  

---

# CHAPTER 2: DATA PROCESSING

## 2.1 Data Sources
CASTNET data consists of four primary streams:
- **Hourly Ozone Data**
- **Gaseous Pollutants (SO₂, HNO₃)**
- **Dry Deposition Chemistry**
- **Site Metadata** (lat/lon, elevation, land use)

Files are discovered via recursive directory scanning through a custom Python architecture.

---

## 2.2 Data Discovery and Extraction Architecture
A custom `CSVDatabaseInventory` class:

### Workflow
1. recursive scan for `.csv` files  
2. metadata extraction (path, size, timestamps)  
3. inventory catalog creation  

Datasets are loaded using **pandas**, with standardized date formats and column names.

---

## 2.3 Data Cleaning & Initial Quality Control

### Filters Applied
- Negative ozone values removed  
- Invalid QA flags filtered  
- Only essential variables retained  
- Site isolation ensuring clean time-series segmentation  

Advanced techniques like winsorization and Z-scoring were analyzed but deferred in favor of physically grounded filtering.

---

## 2.4 Temporal Reconstruction & Standardization
Because sensor downtime results in missing hours:

1. Minimum & maximum timestamps identified  
2. Continuous hourly index generated  
3. Data reindexed, inserting `NaN` for missing hours  

### Output
- **127 CASTNET sites**
- **24,443,603 hourly records**
- Fully standardized timeline for feature engineering  

---

# CHAPTER 3: FEATURE ENGINEERING

## 3.1 Introduction
A multi-stage pipeline was developed to convert raw time series into a supervised ML-ready feature matrix.

---

## 3.2 Temporal Feature Extraction & Cyclical Encoding
Ozone has strong **diurnal** and **seasonal** periodicity.

Standard time labels (0–23) create discontinuities; therefore sine/cosine transformations were applied:

sin(2πt/T)

cos(2πt/T)


### Features Generated
- diurnal (hour, sin, cos, day/night flag)  
- weekly (day of week, weekend indicator, sin, cos)  
- seasonal (month, sin, cos, week of year, day of year)  

**Total temporal features: 27**

---

## 3.3 Lagged Variable Generation
To capture autocorrelation:

- **Short-term:** 1, 2, 3 hours  
- **Transitional:** 6, 12 hours  
- **Daily recurrence:** 24, 48 hours  
- **Weekly recurrence:** 168 hours  

---

## 3.4 Rolling Window Statistics
Windows: **3, 6, 12, 24, 168 hours**

Statistics computed:
- mean  
- median  
- std deviation  
- min  
- max  

**Total rolling features: 25**

---

## 3.5 Differencing Features
Captures rate of change dynamics:

- Δ1 (hourly)
- Δ24 (daily)
- Δ168 (weekly)

---

## 3.6 Final Feature Matrix Structure

| Feature Type | Count |
|--------------|-------|
| Temporal/Cyclical | 27 |
| Lag Features | 8 |
| Rolling Stats | 25 |
| Differencing | 3 |
| **Total** | **~63 features** |

Dataset indexed by **SITE_ID** and **DATE_TIME**.

---

## 3.7 Handling Feature-Induced Missingness
Max look-back = **168 hours**, so first 168 rows per site removed.

This eliminates NaN values and ensures training stability.

---

## 3.8 Data Normalization & Scaling
Z-score standardization applied:

z = (x - μ) / σ


- Fit on training set only  
- Applied to validation & test  
- Target variable left unscaled  

---

# CHAPTER 4: DISCUSSION AND FUTURE ROADMAP

## 4.1 Evaluation of the Data Infrastructure

### Scalability
- Processes **24M+ records** across **127** sites  
- Enables training of *global models* instead of 127 individual ones  

### Reproducibility
- Fully deterministic pipeline  
- Versioned schema and automated inventory  

---

## 4.2 Suitability of Feature Space for ML Models

### Tree-Based Models (XGBoost, LightGBM)
Lag & rolling stats encode temporal behavior without sequences.

### RNNs (LSTM/GRU)
Differencing and scaling improve stability and convergence.

### Temporal Convolutional Networks (TCN)
Require evenly spaced data—enabled by the reconstruction pipeline.

---

## 4.3 Future Modeling Roadmap

### Phase 1: Baseline Models
- Persistence  
- Seasonal naïve baseline  

### Phase 2: Gradient Boosting Models
- Random Forest  
- XGBoost  
- LightGBM  
- SHAP interpretability  

### Phase 3: Deep Learning
- LSTM  
- GRU  
- TCN  

### Phase 4: Hybrid Ensembles
- Stacked meta-learning combining trees + neural networks  

---
