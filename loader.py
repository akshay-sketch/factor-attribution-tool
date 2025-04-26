# loader.py
from pathlib import Path
import os
import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta

def fetch_invespar_factors(url: str) -> tuple[pd.DataFrame, str]:
    """
    Fetch factor returns from Invespar and return a cleaned DataFrame along with the risk-free rate column name.

    Parameters:
    - url (str): URL to download the CSV file from Invespar.

    Returns:
    - Tuple[pd.DataFrame, str]: Cleaned DataFrame of factor returns and the name of the risk-free rate column.
    """
    cache_dir = Path.cwd() / "factor_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "invespar_factors.csv"
    max_age_days = 30

    # Invalidate cache if too old
    if cache_file.exists():
        modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - modified_time > timedelta(days=max_age_days):
            cache_file.unlink()  

    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=True, index_col=0)
        
    else:
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                "Referer": "https://invespar.com"
            }
        )
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text), skiprows=[0])
        df.columns = df.columns.str.strip()

        date_col = df.columns[0]
        df = df[df[date_col].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}$')]
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(how='all')

        df.to_csv(cache_file)
    
    # Identify risk-free column
    rf_keywords = ['rf', 'risk_free', 'riskfree', 'risk free', 'r_f', 'r_f_rate']
    rf_candidates = [col for col in df.columns if any(col.strip().lower() == key for key in rf_keywords)]
    
    if not rf_candidates:
        raise ValueError("No column found that looks like a risk-free rate (e.g., 'RF').")
    if len(rf_candidates) > 1:
        raise ValueError(f"Multiple RF candidates found: {rf_candidates}. Please pass only one RF column")
        
    return df.iloc[:, :-1],  rf_candidates[0]

def fetch_iima_factors( url: str) -> tuple[pd.DataFrame, str]:
    """
    Fetch factor returns from IIMA and return a cleaned DataFrame along with the risk-free rate column name.

    Parameters:
    - url (str): URL to download the CSV file from IIMA.

    Returns:
    - Tuple[pd.DataFrame, str]: Cleaned DataFrame of factor returns and the name of the risk-free rate column.
    """
    
    cache_dir = Path.cwd() / "factor_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "iima_factors.csv"
    max_age_days = 3

    # Invalidate cache if too old
    if cache_file.exists():
        modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - modified_time > timedelta(days=max_age_days):
            cache_file.unlink()

    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=True, index_col=0)
        
    else:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        df.to_csv(cache_file)
    
    
    # Identify risk-free column
    rf_keywords = ['rf', 'risk_free', 'riskfree', 'risk free', 'r_f', 'r_f_rate']
    rf_candidates = [col for col in df.columns if any(col.strip().lower() == key for key in rf_keywords)]
    
    if not rf_candidates:
        raise ValueError("No column found that looks like a risk-free rate (e.g., 'RF').")
    if len(rf_candidates) > 1:
        raise ValueError(f"Multiple RF candidates found: {rf_candidates}. Please pass only one RF column")
        
    return df,  rf_candidates[0]   