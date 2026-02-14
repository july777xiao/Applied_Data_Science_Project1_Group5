# NYC 311 Complaint Analysis

**Course:** GU4243/GR5243 Applied Data Science  
**Project 1 — Team 5**  
Ketaki Dabade (kvd2112) · Junye Chen (jc6636) · Rui Lin (rl3445) · Xiao Xiao (xx2492)

---

## Overview

This project analyzes NYC 311 non-emergency complaints from January through June 2024, integrating five data sources into a unified daily-by-borough panel. The analysis spans data acquisition, cleaning, exploratory data analysis, and feature engineering.

## Repository Structure

```
├── data_acquisition_cleaning_preprocessing.py   ← Data collection & panel building
├── Project1_Section3_Section4_Final.ipynb        ← EDA & feature engineering
├── new_york_listings_2024.csv                    ← Airbnb data (you provide)
├── data/                                         ← Raw & processed data
├── outputs_task3_task4/                           ← Figures, tables & model matrices
└── README.md
```

## Data Sources

| Source | Method | Scale |
|--------|--------|-------|
| NYC 311 Requests | Socrata API | ~1.5M records |
| Weather | Open-Meteo API | ~4,300 hourly obs |
| NYC Events | BeautifulSoup + API + manual | ~300 borough-days |
| U.S. Census ACS 2019 | Census Bureau API | ~200 ZCTAs |
| Airbnb Listings | Kaggle (manual upload) | ~100K listings |

## Prerequisites

```bash
pip install pandas numpy requests matplotlib seaborn beautifulsoup4 scikit-learn statsmodels
```

## Setup

1. Download the Airbnb dataset from [Kaggle](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata)
2. Rename to `new_york_listings_2024.csv` and place in the project root

All other data is downloaded automatically via APIs.

## How to Run

**Step 1 — Data Pipeline** (~30–40 min)

```bash
python data_acquisition_cleaning_preprocessing.py
```

Collects all data, builds the daily × borough panel, runs quality checks and imputation. Outputs `data/processed/Daily Borough Events Panel.csv`.

**Step 2 — EDA & Feature Engineering**

```bash
jupyter notebook Project1_Section3_Section4_Final.ipynb
```

Reads the panel and performs EDA (summary stats, spatial/temporal patterns, weather/event effects, correlation diagnostics, PCA) and feature engineering (temporal encodings, lags, weather regimes, interactions, model matrix construction with four scaling variants). Outputs go to `outputs_task3_task4/`.

## Environment

Auto-detects runtime: Google Colab uses `/content/`, local machines use the script directory.

## GitHub Repository

[Applied_Data_Science_Project1_Group5](https://github.com/your-org/Applied_Data_Science_Project1_Group5)
