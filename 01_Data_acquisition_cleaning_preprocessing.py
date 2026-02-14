"""
03_census_acquisition.py
获取NYC各邮编区域的人口统计数据（Census API）

数据源：US Census Bureau - American Community Survey (ACS) 5-Year Estimates
API文档：https://www.census.gov/data/developers/data-sets/acs-5year.html

输出：data/raw/census_demographics_raw.csv

Usage:
    python 03_census_acquisition.py --year 2022

Required:
    pip install requests pandas

注意：Census API需要免费的API Key
获取方式：https://api.census.gov/data/key_signup.html
设置环境变量：export CENSUS_API_KEY="your_key_here"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import requests
from time import sleep


# ---------------------------
# Config
# ---------------------------

CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")
if not CENSUS_API_KEY:
    print("⚠️  警告：未设置 CENSUS_API_KEY 环境变量")
    print("   获取免费API Key：https://api.census.gov/data/key_signup.html")
    print("   设置方式：export CENSUS_API_KEY='your_key_here'")
    print("   或者直接在代码中修改 CENSUS_API_KEY 变量")
    # 如果没有API key，使用示例key（限流严重，仅供测试）
    CENSUS_API_KEY = ""

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "raw"

# ACS 5-Year Estimates API
ACS5_BASE_URL = "https://api.census.gov/data/{year}/acs/acs5"

# NYC Counties (FIPS codes)
NYC_COUNTIES = {
    "36061": "Manhattan (New York County)",
    "36047": "Brooklyn (Kings County)", 
    "36081": "Queens (Queens County)",
    "36005": "Bronx (Bronx County)",
    "36085": "Staten Island (Richmond County)",
}

# 变量选择：人口、收入、教育、住房等
CENSUS_VARIABLES = {
    # 人口统计
    "B01003_001E": "total_population",           # 总人口
    "B01002_001E": "median_age",                 # 年龄中位数
    
    # 种族/民族
    "B02001_002E": "white_alone",                # 仅白人
    "B02001_003E": "black_alone",                # 仅黑人
    "B03003_003E": "hispanic_latino",            # 西班牙裔/拉丁裔
    
    # 收入
    "B19013_001E": "median_household_income",    # 家庭收入中位数
    "B19301_001E": "per_capita_income",          # 人均收入
    "B17001_002E": "poverty_count",              # 贫困人口数
    
    # 教育
    "B15003_022E": "bachelors_degree",           # 学士学位
    "B15003_023E": "masters_degree",             # 硕士学位
    "B15003_025E": "doctorate_degree",           # 博士学位
    
    # 住房
    "B25003_002E": "owner_occupied_housing",     # 自有住房
    "B25003_003E": "renter_occupied_housing",    # 租赁住房
    "B25077_001E": "median_home_value",          # 房屋价值中位数
    "B25064_001E": "median_gross_rent",          # 租金中位数
    
    # 就业
    "B23025_005E": "unemployed",                 # 失业人数
    "B23025_002E": "in_labor_force",             # 劳动力人口
    
    # 交通
    "B08301_001E": "total_commuters",            # 通勤总人数
    "B08301_010E": "public_transit_commuters",   # 公共交通通勤
}


# ---------------------------
# Helpers
# ---------------------------

def fetch_census_data(
    year: int,
    variables: Dict[str, str],
    state_fips: str = "36",  # New York State
    county_fips: Optional[str] = None,
) -> pd.DataFrame:
    """
    从Census API获取指定年份和地理级别的数据
    
    Args:
        year: 数据年份（如 2022）
        variables: 变量代码到名称的映射
        state_fips: 州FIPS代码（36 = New York）
        county_fips: 县FIPS代码（可选，用于进一步过滤）
    
    Returns:
        包含人口统计数据的DataFrame
    """
    url = ACS5_BASE_URL.format(year=year)
    
    # 构建变量列表（包括地理字段）
    var_codes = list(variables.keys())
    var_string = ",".join(var_codes)
    
    # 设置地理级别为ZIP Code Tabulation Area (ZCTA)
    # 注意：Census使用ZCTA而不是邮编，但大部分情况下可以对应
    params = {
        "get": f"NAME,{var_string}",
        "for": "zip code tabulation area:*",
        "in": f"state:{state_fips}",
        "key": CENSUS_API_KEY,
    }
    
    print(f"正在请求 {year} 年人口统计数据...")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ API请求失败: {e}")
        if not CENSUS_API_KEY:
            print("💡 提示：可能需要设置有效的 CENSUS_API_KEY")
        return pd.DataFrame()
    
    data = response.json()
    
    if not data or len(data) < 2:
        print("⚠️  未获取到数据")
        return pd.DataFrame()
    
    # 第一行是列名
    headers = data[0]
    rows = data[1:]
    
    df = pd.DataFrame(rows, columns=headers)
    
    # 重命名变量
    rename_map = {code: name for code, name in variables.items()}
    df.rename(columns=rename_map, inplace=True)
    
    # 清理列名
    df.rename(columns={"NAME": "area_name", "zip code tabulation area": "zcta"}, inplace=True)
    
    # 转换数值类型
    numeric_cols = list(variables.values())
    for col in numeric_cols:
        if col in df.columns:
            # Census API中-666666666表示缺失值
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace(-666666666, pd.NA)
    
    return df


def filter_nyc_zipcodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤出NYC范围内的邮编
    
    NYC ZIP code范围（大致）：
    - Manhattan: 100xx, 101xx, 102xx
    - Bronx: 104xx
    - Brooklyn: 112xx
    - Queens: 110xx, 111xx, 113xx, 114xx, 116xx
    - Staten Island: 103xx
    """
    if "zcta" not in df.columns:
        return df
    
    # 转换为整数邮编
    df["zip_int"] = pd.to_numeric(df["zcta"], errors="coerce")
    
    # NYC邮编范围过滤
    nyc_condition = (
        # Manhattan
        ((df["zip_int"] >= 10001) & (df["zip_int"] <= 10292)) |
        # Bronx
        ((df["zip_int"] >= 10400) & (df["zip_int"] <= 10499)) |
        # Brooklyn  
        ((df["zip_int"] >= 11200) & (df["zip_int"] <= 11299)) |
        # Queens
        (
            ((df["zip_int"] >= 11000) & (df["zip_int"] <= 11109)) |
            ((df["zip_int"] >= 11350) & (df["zip_int"] <= 11499)) |
            ((df["zip_int"] >= 11690) & (df["zip_int"] <= 11699))
        ) |
        # Staten Island
        ((df["zip_int"] >= 10300) & (df["zip_int"] <= 10399))
    )
    
    df_nyc = df[nyc_condition].copy()
    
    # 添加行政区字段（基于邮编范围推断）
    def assign_borough(zip_code):
        if pd.isna(zip_code):
            return "UNKNOWN"
        z = int(zip_code)
        if 10001 <= z <= 10292:
            return "MANHATTAN"
        elif 10400 <= z <= 10499:
            return "BRONX"
        elif 11200 <= z <= 11299:
            return "BROOKLYN"
        elif (11000 <= z <= 11109) or (11350 <= z <= 11499) or (11690 <= z <= 11699):
            return "QUEENS"
        elif 10300 <= z <= 10399:
            return "STATEN ISLAND"
        else:
            return "UNKNOWN"
    
    df_nyc["borough_inferred"] = df_nyc["zip_int"].apply(assign_borough)
    
    return df_nyc


def calculate_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """计算衍生指标"""
    
    # 贫困率
    if "poverty_count" in df.columns and "total_population" in df.columns:
        df["poverty_rate"] = (df["poverty_count"] / df["total_population"] * 100).round(2)
    
    # 失业率
    if "unemployed" in df.columns and "in_labor_force" in df.columns:
        df["unemployment_rate"] = (df["unemployed"] / df["in_labor_force"] * 100).round(2)
    
    # 高等教育比例
    if all(col in df.columns for col in ["bachelors_degree", "masters_degree", "doctorate_degree", "total_population"]):
        df["higher_education_count"] = (
            df["bachelors_degree"] + df["masters_degree"] + df["doctorate_degree"]
        )
        df["higher_education_rate"] = (
            df["higher_education_count"] / df["total_population"] * 100
        ).round(2)
    
    # 租房比例
    if "renter_occupied_housing" in df.columns and "owner_occupied_housing" in df.columns:
        total_housing = df["renter_occupied_housing"] + df["owner_occupied_housing"]
        df["renter_rate"] = (df["renter_occupied_housing"] / total_housing * 100).round(2)
    
    # 公共交通使用率
    if "public_transit_commuters" in df.columns and "total_commuters" in df.columns:
        df["public_transit_rate"] = (
            df["public_transit_commuters"] / df["total_commuters"] * 100
        ).round(2)
    
    return df


def qc_report(df: pd.DataFrame) -> None:
    """数据质量报告"""
    print("\n" + "="*70)
    print("📊 Census数据质量报告")
    print("="*70)
    print(f"总记录数: {len(df):,}")
    print(f"总列数: {len(df.columns)}")
    
    if "borough_inferred" in df.columns:
        print("\n各行政区邮编数量:")
        print(df["borough_inferred"].value_counts())
    
    print("\n缺失值情况（Top 10）:")
    missing = df.isnull().sum().sort_values(ascending=False).head(10)
    missing_pct = (missing / len(df) * 100).round(2)
    for col, count in missing.items():
        print(f"  {col}: {count} ({missing_pct[col]}%)")
    
    if "median_household_income" in df.columns:
        income_stats = df["median_household_income"].describe()
        print(f"\n家庭收入中位数统计:")
        print(f"  平均: ${income_stats['mean']:,.0f}")
        print(f"  中位数: ${income_stats['50%']:,.0f}")
        print(f"  范围: ${income_stats['min']:,.0f} - ${income_stats['max']:,.0f}")


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="获取NYC各邮编的Census人口统计数据"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="数据年份（默认2022，ACS 5-Year最新）"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取数据
    df = fetch_census_data(
        year=args.year,
        variables=CENSUS_VARIABLES,
    )
    
    if df.empty:
        print("❌ 未能获取Census数据")
        return 1
    
    # 过滤NYC邮编
    df_nyc = filter_nyc_zipcodes(df)
    
    print(f"✓ 获取到 {len(df)} 条记录")
    print(f"✓ 过滤后NYC区域: {len(df_nyc)} 条记录")
    
    # 计算衍生指标
    df_nyc = calculate_derived_metrics(df_nyc)
    
    # 保存
    output_file = output_dir / "census_demographics_raw.csv"
    df_nyc.to_csv(output_file, index=False)
    print(f"✓ 已保存: {output_file}")
    
    # 质量报告
    qc_report(df_nyc)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
