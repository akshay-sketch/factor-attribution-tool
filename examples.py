# test_compatibility.py

import pandas as pd
import numpy as np
from factor_attribution import FactorAttribution
from pathlib import Path
import polars as pl


def generate_synthetic_nav(freq='M', format='dataframe', data_range = 60):
    dates = pd.date_range(start="2020-01-01", periods=data_range, freq=freq)
    prices = np.cumprod(1 + np.random.normal(0.01, 0.03, len(dates)))
    
    if format == 'series':
        return pd.Series(prices, index=dates, name="NAV")
    elif format == 'polars':
        return pl.Series("NAV", prices)
    elif format == 'dataframe_with_index':
        return pd.DataFrame({"NAV": prices}, index=dates)
    elif format == 'dataframe_with_column':
        return pd.DataFrame({"date": dates, "NAV": prices})


def generate_synthetic_factors(freq='M', include_rf=True, data_range=60):
    dates = pd.date_range(start="2020-01-01", periods=data_range, freq=freq)
    df = pd.DataFrame({
        "Market": np.random.normal(0.01, 0.02, len(dates)),
        "Size": np.random.normal(0.005, 0.015, len(dates)),
        "Value": np.random.normal(0.007, 0.01, len(dates)),
    }, index=dates)
    if include_rf:
        df["rf"] = np.random.normal(0.003, 0.002, len(dates))
    return df


def test_case(nav, factors=None, freq='M', date=None):
    try:
        print(f"\nTesting NAV format: {type(nav)}, Factors: {'Provided' if factors is not None else 'Default'}, Frequency: {freq}")
        fa = FactorAttribution(nav_series=nav, factors=factors, frequency=freq) if date is None \
             else FactorAttribution(nav_series=nav, factors=factors, frequency=freq, date_pl=date)

        coefs, rsq = fa.rolling_attribution()
        fa.summary()
        print("✅ Success")
    except Exception as e:
        print(f"❌ Failed with error: {e}")



def run_all_tests():
    nav_formats = ['series', 'polars', 'dataframe_with_index', 'dataframe_with_column']
    frequencies = ['D', 'W', 'M', 'Q']

    for nav_fmt in nav_formats:
        for freq in frequencies:
            if freq =="D" or freq=='W':
                data_range = 300
            else:
                data_range = 120
            nav = generate_synthetic_nav(freq=freq, format=nav_fmt, data_range=data_range)
            factors = generate_synthetic_factors(freq=freq, data_range=data_range)

            print(f"\nTesting format={nav_fmt}, freq={freq}")
            if nav_fmt == 'polars':
                # Special case: pass a second date series for polars.Series
                dates = pl.Series("date", pd.date_range(start="2020-01-01", periods=data_range, freq=freq))
                test_case(nav=nav, factors=None, freq=freq, date=dates)
                test_case(nav=nav, factors=factors, freq=freq, date=dates)
            else:
                test_case(nav=nav, factors=None, freq=freq)
                test_case(nav=nav, factors=factors, freq=freq)

if __name__ == "__main__":
    run_all_tests()
