# NYC 311 Complaint Analysis

**Course:** GU4243/GR5243 Applied Data Science  
**Project 1 — Team 5**  
Ketaki Dabade (kvd2112) · Junye Chen (jc6636) · Rui Lin (rl3445) · Xiao Xiao (xx2492)

---

## Overview

This project analyzes NYC 311 non-emergency complaints from January through June 2024, integrating five heterogeneous data sources into a unified daily-by-borough panel. The analysis spans data acquisition, cleaning, exploratory data analysis (EDA), and feature engineering to produce model-ready datasets.

## Repository Structure

```
├── data_acquisition_cleaning_preprocessing.py   ← Part 1–3: data collection & panel building
├── Project1_Section3_Section4_Final.ipynb        ← Part 3–5: EDA & feature engineering
├── new_york_listings_2024.csv                    ← Airbnb data (you provide)
├── data/
│   ├── raw/
│   │   ├── nyc_311_raw.csv
│   │   ├── weather_raw.csv
│   │   └── census_demographics_raw.csv
│   └── processed/
│       └── Daily Borough Events Panel.csv
├── outputs_task3_task4/
│   ├── figures/                                  ← All EDA figures (PNG + PDF)
│   ├── tables/                                   ← All summary tables (CSV + LaTeX + PNG)
│   ├── Daily_Borough_Events_Panel_processed.csv
│   ├── Daily_Borough_Events_Panel_processed_time_safe.csv
│   ├── Daily_Borough_Events_Panel_model_matrix_A.csv
│   ├── Daily_Borough_Events_Panel_model_matrix_A_standard.csv
│   ├── Daily_Borough_Events_Panel_model_matrix_A_minmax.csv
│   └── Daily_Borough_Events_Panel_model_matrix_A_power.csv
└── README.md
```

## Data Sources

| Source | Method | Records | Granularity |
|--------|--------|---------|-------------|
| NYC 311 Service Requests | Socrata API (paginated) | ~1.5 million | Event-level → daily × borough |
| Weather (Open-Meteo) | REST API | ~4,300 hourly obs | Hourly → daily |
| NYC Events | BeautifulSoup scraping + API + manual | ~300 borough-days | Daily × borough |
| U.S. Census ACS 2019 | Census Bureau API | ~200 ZCTAs | ZIP → borough (static) |
| Airbnb Listings | Kaggle CSV (manual upload) | ~100,000 listings | Listing → borough (static) |

## Prerequisites

**Python 3.8+** with the following packages:

```
pandas
numpy
requests
matplotlib
seaborn
beautifulsoup4
scikit-learn
statsmodels
```

Install all at once:

```bash
pip install pandas numpy requests matplotlib seaborn beautifulsoup4 scikit-learn statsmodels
```

## Setup

1. **Download the Airbnb dataset** from [Kaggle](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata)
2. **Rename** the downloaded file to `new_york_listings_2024.csv`
3. **Place it** in the project root directory

All other datasets are downloaded automatically via APIs.

## How to Run

### Step 1: Data Acquisition, Cleaning & Preprocessing

```bash
python data_acquisition_cleaning_preprocessing.py
```

This script runs the full data pipeline (~30–40 min):

- **Part 1 — Data Collection:** Downloads 311 complaints (Socrata API), weather (Open-Meteo), events (web scraping + API), and census data (ACS API). Reads Airbnb from local file.
- **Part 2 — Panel Construction:** Aggregates 311 to daily × borough counts, merges all five sources, engineers time/lag/rolling features.
- **Part 3 — Data Quality:** Missing value analysis, validation checks, IQR-based outlier detection with winsorization of weather variables, and imputation.

**Output:** `data/processed/Daily Borough Events Panel.csv` (~910 rows × 40+ columns)

### Step 2: EDA & Feature Engineering

Open and run the notebook:

```bash
jupyter notebook Project1_Section3_Section4_Final.ipynb
```

The notebook reads `Daily Borough Events Panel.csv` and performs:

#### Section 3 — Data Validation & Overview
- Dataset structure and panel completeness checks
- Auto-generated data dictionary (types, missingness, summary stats)
- Column grouping into semantic categories (IDs, weather, structural, time, lags, Top-K)

#### Section 4 — Exploratory Data Analysis

| Subsection | Content |
|------------|---------|
| 4.1 Descriptive Overview | Summary statistics, missingness patterns, citywide anomaly detection (robust Z-score using median + MAD) |
| 4.2 Spatial Patterns | Distributions and outliers by borough (histograms, box/violin plots), complaint composition (Top-K stacked bars and heatmap) |
| 4.3 Temporal Dynamics | Daily time series by borough, 7-day rolling trends, monthly aggregation, day-of-week and weekend effects |
| 4.4 Event & Weather Effects | Event-day vs. non-event comparison, temperature and precipitation scatter plots with regression |
| 4.5 Correlation Diagnostics | Correlation heatmap, VIF multicollinearity check, influence diagnostics (studentized residuals, leverage, Cook's distance, DFFITS), PCA visualization |

#### Section 5 — Feature Engineering

| Subsection | Features Created |
|------------|-----------------|
| 5.1 Time & Calendar | Cyclical sin/cos encodings for day-of-week and month, weekend indicator, time index |
| 5.2 Lag & Rolling | 1-day and 7-day lagged complaints, 7-day rolling weather means, short-term momentum (day-over-day change) |
| 5.3 Weather Regime | Log-precipitation, precipitation binary flag, weather regime indicators (hot/cold/rainy), temperature bins |
| 5.4 Complaint Composition | Lagged top-type shares, Herfindahl concentration index, complaint diversity (unique types count) |
| 5.5 Interactions | Weekend × weather, event × precipitation, borough × temperature cross-terms |
| 5.6 Missing Value Handling | Two versions: EDA-friendly (forward + backward fill) and time-safe (forward-fill only, no future leakage) |
| 5.7 Model Matrix Construction | One-hot encoding, zero-variance filtering, four scaling variants (raw, standardized, min-max, Yeo–Johnson power) |

## Output Files

### From `data_acquisition_cleaning_preprocessing.py`

| File | Description |
|------|-------------|
| `data/raw/nyc_311_raw.csv` | Raw 311 complaint records |
| `data/raw/weather_raw.csv` | Raw hourly weather data |
| `data/raw/census_demographics_raw.csv` | Census ZIP-level demographics |
| `Daily Borough Events Panel.csv` | Cleaned daily × borough panel |

### From `Project1_Section3_Section4_Final.ipynb`

| File | Description |
|------|-------------|
| `*_processed.csv` | EDA-friendly imputed dataset |
| `*_processed_time_safe.csv` | Time-safe imputed dataset (no future leakage) |
| `*_model_matrix_A.csv` | Model matrix after variance filtering |
| `*_model_matrix_A_standard.csv` | Standardized (z-score) variant |
| `*_model_matrix_A_minmax.csv` | Min-max scaled variant |
| `*_model_matrix_A_power.csv` | Yeo–Johnson power-transformed variant |
| `outputs_task3_task4/figures/` | All EDA figures in PNG and PDF |
| `outputs_task3_task4/tables/` | All summary tables in CSV, LaTeX, and PNG |

## Key Variables in Final Panel

| Variable | Description |
|----------|-------------|
| `date`, `borough` | Panel index |
| `complaints_total` | Daily complaint count |
| `topk_*_cnt` | Counts for top-8 complaint types |
| `temp_mean`, `temp_max`, `temp_min` | Daily temperature (°C) |
| `precipitation_sum`, `snowfall_sum` | Daily precipitation / snowfall |
| `wind_speed_mean`, `cloud_cover_mean` | Daily wind and cloud cover |
| `census_income_borough_median` | Borough median household income |
| `census_population_borough_sum` | Borough total population |
| `airbnb_listing_count`, `airbnb_price_mean` | Borough Airbnb metrics |
| `airbnb_per_1000_people_borough` | Airbnb density per 1,000 residents |
| `event_count`, `event_has_parade`, `event_has_holiday` | Event indicators |
| `day_of_week`, `is_weekend`, `month` | Calendar features |
| `*_lag1`, `*_ma7` | Lag and rolling-window features |
| `log_complaints_total` | Log-transformed target variable |

## Environment Compatibility

The pipeline auto-detects the runtime environment:
- **Google Colab** → uses `/content/` as base directory
- **Local machine** → uses the script's directory as base

## GitHub Repository

[Applied_Data_Science_Project1_Group5](https://github.com/your-org/Applied_Data_Science_Project1_Group5)
