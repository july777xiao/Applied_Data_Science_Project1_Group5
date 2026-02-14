# Applied_Data_Science_Project1_Group5
NYC 311 complaint analysis using multi-source data integration, web scraping, and time series modeling.
# NYC 311 Complaint Analysis — Data Acquisition, Cleaning & Preprocessing

**Course:** GU4243/GR5243 Applied Data Science  
**Project 1 — Team 5**  
Ketaki Dabade (kvd2112) · Junye Chen (jc6636) · Rui Lin (rl3445) · Xiao Xiao (xx2492)

---

## Overview

This script (`data_acquisition_cleaning_preprocessing.py`) collects data from five sources, merges them into a unified **daily × borough** panel for NYC 311 complaints (January–June 2024), and applies data quality checks, outlier handling, and missing-value imputation. The final output is a single analysis-ready CSV.

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
beautifulsoup4
```

Install all at once:

```bash
pip install pandas numpy requests matplotlib beautifulsoup4
```

## Setup

1. **Download the Airbnb dataset** from [Kaggle](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata)
2. **Rename** the downloaded file to `new_york_listings_2024.csv`
3. **Place it** in the same directory as the script

All other datasets are downloaded automatically via APIs.

## Usage

```bash
python data_acquisition_cleaning_preprocessing.py
```

The script auto-detects the environment:
- **Google Colab** → uses `/content/` as the base directory
- **Local machine** → uses the script's directory as the base

Estimated runtime: **30–40 minutes** (mostly 311 API download).

## Output

```
<project_root>/
├── data/
│   ├── raw/
│   │   ├── nyc_311_raw.csv
│   │   ├── weather_raw.csv
│   │   └── census_demographics_raw.csv
│   └── processed/
│       └── Daily Borough Events Panel.csv    ← Final output
├── web_scraped_nyc_jan_jun_2024_expanded.csv
└── new_york_listings_2024.csv                ← You provide this
```

**Final panel:** ~910 rows (5 boroughs × 182 days) with 40+ columns.

## Pipeline Steps

### Part 1 — Data Collection

1. **311 Complaints:** Paginated download from Socrata API (50k records/request), filtered to Jan–Jun 2024, excluding "Unspecified" boroughs.
2. **Weather:** Hourly observations from Open-Meteo Archive API for NYC coordinates.
3. **Events:** Three sources merged — BeautifulSoup scraping of nyctourism.com, NYC Open Data permitted events API, and manually curated holidays/parades. Citywide events are expanded to all five boroughs.
4. **Census:** ACS 2019 5-year estimates for population and median household income, aggregated from ZIP to borough level.
5. **Airbnb:** Kaggle dataset aggregated to borough-level metrics (listing count, price, rating, reviews, entire-home percentage).

### Part 2 — Panel Construction

- Aggregate 311 records to daily × borough counts (total + top-8 complaint types)
- Aggregate weather from hourly to daily (mean/max/min temperature, precipitation sum, wind, cloud cover, snowfall)
- Merge all sources: 311 + weather (by date), census + Airbnb (by borough), events (by date + borough)
- Engineer features: day-of-week, weekend flag, month, week-of-year, 1-day lags, 7-day rolling means, log-transformed complaint count, Airbnb density per 1,000 residents

### Part 3 — Data Quality & Preprocessing

- **Missing value analysis:** Column-level missingness count and percentage
- **Validation:** Borough consistency, date completeness, numeric range checks, duplicate detection
- **Outlier handling:** IQR method on 6 key variables; winsorize weather outliers (precipitation, wind, snowfall) while preserving complaint spikes
- **Imputation:** Forward-fill + median for lag/rolling features; zero-fill for precipitation; median for remaining numerics

## Key Variables in Final Panel

| Variable | Description |
|----------|-------------|
| `date`, `borough` | Panel index |
| `complaints_total` | Daily complaint count |
| `topk_*_cnt` | Counts for top-8 complaint types |
| `temp_mean`, `temp_max`, `temp_min` | Daily temperature (°C) |
| `precipitation_sum`, `snowfall_sum` | Daily precipitation/snowfall (mm/cm) |
| `wind_speed_mean`, `cloud_cover_mean` | Daily wind and cloud cover |
| `census_income_borough_median` | Borough median household income |
| `census_population_borough_sum` | Borough total population |
| `airbnb_listing_count`, `airbnb_price_mean` | Borough Airbnb metrics |
| `airbnb_per_1000_people_borough` | Airbnb density |
| `event_count`, `event_has_parade`, `event_has_holiday` | Event indicators |
| `day_of_week`, `is_weekend`, `month` | Calendar features |
| `*_lag1`, `*_ma7` | Lag and rolling-window features |
| `log_complaints_total` | Log-transformed target |

## GitHub Repository

[Applied_Data_Science_Project1_Group5](https://github.com/your-org/Applied_Data_Science_Project1_Group5)
