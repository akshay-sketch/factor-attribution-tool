#sanitizer.py
import pandas as pd
import polars as pl
from typing import Union, Optional
from pathlib import Path

def sanitize_nav_series(nav_series: Union[pd.Series, pl.Series, pd.DataFrame, str, Path], date: Optional[Union[pd.Series, pl.Series]] = None ) -> pd.Series:
        """
        Sanitize NAV series input to ensure it is a pandas Series with a datetime index.

        Parameters:
        - nav_series: The NAV series input, which can be:
            - pandas Series
            - polars Series
            - pandas DataFrame

        Returns:
        - A pandas Series with datetime index representing the NAV series.
        """
        if isinstance(nav_series, pd.Series):
            df = nav_series.to_frame()
        elif isinstance(nav_series, pl.Series):
            if date is None:
                raise ValueError("If passing a Polars Series, a date Series must also be provided")
            df = pd.DataFrame({
                "date": date.to_pandas(),
                "NAV": nav_series.to_pandas()})
        elif isinstance(nav_series, pd.DataFrame):
            df = nav_series.copy()
        elif isinstance(nav_series, pl.DataFrame):
            df = nav_series.to_pandas()
        else:
            raise TypeError("Unsupported type for nav_series")
        
        
        if isinstance(df.index, pd.DatetimeIndex):
            pass

        
        
        else:
            if pd.api.types.is_object_dtype(df.index) or pd.api.types.is_string_dtype(df.index):
                sample = pd.Series(df.index).dropna().astype(str).head(15)
                is_mostly_numeric = sample.apply(lambda x: x.replace('.', '', 1).isdigit()).mean() > 0.8
                if not is_mostly_numeric:
                    parsed_index = pd.to_datetime(df.index, errors='coerce')
                else: 
                    parsed_index = pd.Series([pd.NaT]*len(df))
            elif pd.api.types.is_any_real_numeric_dtype(df.index):
                parsed_index = pd.Series([pd.NaT]*len(df))
            else:
                parsed_index = pd.to_datetime(df.index, errors="coerce")
            
            # If a good number of dates are valid, use them and drop the rest
            if parsed_index.notna().sum() >= 0.9 * len(parsed_index):  
                df.index = parsed_index
                df = df[~df.index.isna()]
            else:
                # Look for column with datetime type or convertible string
                date_col = None
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        date_col = col
                        break
                    elif pd.api.types.is_object_dtype(df[col]):
                        try:
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                            if df[col].notna().sum() >= 0.9 * len(df):  
                                date_col = col
                                break
                        except Exception:
                            continue

                # Use valid column as datetime index
                if date_col:
                    df = df[~df[date_col].isna()]
                    df.set_index(date_col, inplace=True)
                else:
                    raise ValueError("No valid date index or column found or could be parsed.")

        price_col = None
        for col in df.columns:
            # Accept numeric types directly
            if pd.api.types.is_numeric_dtype(df[col]):
                price_col = col
                break
            # Attempt to coerce string to numeric
            elif pd.api.types.is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors="raise")
                    price_col = col
                    break
                except Exception:
                    continue


        if price_col is None:
            raise ValueError("No numeric price column found.")

        price_series = pd.to_numeric(df[price_col], errors='coerce').dropna()
        if not isinstance(price_series.index, pd.DatetimeIndex):
            price_series.index = pd.to_datetime(price_series.index, errors="coerce")

        price_series = price_series.dropna()
        return price_series

def sanitize_factors(df: pd.DataFrame) ->tuple[pd.DataFrame, str]:
    """
    Sanitize factor return data.

    - Parses the index or columns to extract datetime and set as index.
    - Converts all columns to numeric.
    - Identifies and returns the risk-free column.

    Parameters:
    - df (pd.DataFrame): Factor return DataFrame with dates in index or a column.

    Returns:
    - Tuple[pd.DataFrame, str]: Cleaned factor DataFrame with datetime index, and name of the risk-free rate column.

    Raises:
    - ValueError: If no valid datetime index or RF column is found.
    """
    # If index is already datetime, all good
    if isinstance(df.index, pd.DatetimeIndex):
        pass

    # Try parsing index to datetime (coerce)
    
    else:
        if pd.api.types.is_object_dtype(df.index) or pd.api.types.is_string_dtype(df.index):
            sample = pd.Series(df.index).dropna().astype(str).head(15)
            is_mostly_numeric = sample.apply(lambda x: x.replace('.', '', 1).isdigit()).mean() > 0.8
            if not is_mostly_numeric:
                parsed_index = pd.to_datetime(df.index, errors='coerce')
            else: 
                parsed_index = pd.Series([pd.NaT]*len(df))
        elif pd.api.types.is_any_real_numeric_dtype(df.index):
            parsed_index = pd.Series([pd.NaT]*len(df))
        else:
            parsed_index = pd.to_datetime(df.index, errors="coerce")
        
        # If a good number of dates are valid, use them and drop the rest
        if parsed_index.notna().sum() >= 0.9 * len(parsed_index):  
            df.index = parsed_index
            df = df[~df.index.isna()]  
        else:
            # Look for column with datetime type or convertible string
            date_col = None
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_col = col
                    break
                elif pd.api.types.is_object_dtype(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                        if df[col].notna().sum() >= 0.9 * len(df):  
                            date_col = col
                            break
                    except Exception:
                        continue

            # Use valid column as datetime index
            if date_col:
                df = df[~df[date_col].isna()]  
                df.set_index(date_col, inplace=True)
                df.drop(columns= [date_col], inplace= True, errors='ignore' )
            else:
                raise ValueError("No valid date index or column found or could be parsed.")
    df = df.apply(pd.to_numeric, errors = 'coerce')
    df = df.dropna(how='all')
    df = df[~df.index.isnull()]
    rf_keywords = ['rf', 'risk_free', 'riskfree', 'risk free', 'r_f', 'r_f_rate']

    def is_rf_col(colname):
        return any(colname.strip().lower() == key for key in rf_keywords)

    rf_candidates = [col for col in df.columns if is_rf_col(col)]

    if not rf_candidates:
        raise ValueError("No column found that looks like a risk-free rate (e.g., 'RF').")
    if len(rf_candidates)>1:
        raise ValueError(f"Multiple RF candidates found: {rf_candidates}. Please pass only one RF column")
    
    return df, rf_candidates[0]

def infer_frequency( index: pd.DatetimeIndex) -> str:
        """
        Infers frequency from datetime index. Accepts sub-daily data but notes
        that it must be resampled later to daily or lower (D, W, M, etc).
        """
        # 1. Use freqstr if set explicitly
        if index.freqstr:
            return index.freqstr

        # 2. Use pandas built-in inference
        freq = pd.infer_freq(index)
        if freq is not None:
            return freq

        # 3. Fallback: use mode of time differences
        inferred = index.to_series().diff().mode()
        if not inferred.empty:
            delta = inferred.iloc[0]

            if not isinstance(delta, pd.Timedelta) or pd.isna(delta):
                raise ValueError("Unable to determine valid time delta from NAV index.")

            seconds = delta.total_seconds()

            if seconds < 60:
                return "S"   # secondly
            elif seconds < 3600:
                return "T"   # minutely
            elif seconds < 86400:
                return "H"   # hourly
            elif 1 <= seconds / 86400 < 2:
                return "D"
            elif 6 <= seconds / 86400 <= 8:
                return "W"
            elif 28 <= seconds / 86400 <= 31:
                return "M"
            elif 89 <= seconds / 86400 <= 92:
                return "Q"
            elif 364 <= seconds / 86400 <= 366:
                return "A"
            else:
                raise ValueError(f"Unable to map timedelta {delta} to known frequency.")
        
        raise ValueError("Unable to infer frequency from NAV index. Ensure it has a uniform datetime index.")

def is_higher_frequency(freq1: str, freq2: str, frequency_levels: dict) -> bool:
    """
    Compare two frequency strings to determine if freq1 represents a higher frequency than freq2.

    Parameters:
    - freq1 (str): First frequency string (e.g., 'D', 'M', 'W').
    - freq2 (str): Second frequency string to compare against.
    - frequency_levels (dict): Dictionary mapping frequency codes to numeric levels.

    Returns:
    - bool: True if freq1 is higher frequency (more granular) than freq2.
    """
    from pandas.tseries.frequencies import to_offset
    f1 = to_offset(freq1).rule_code
    f2 = to_offset(freq2).rule_code
    f1base = f1.split('-')[0] if '-' in f1 else f1
    f2base = f2.split('-')[0] if '-' in f2 else f2
    return frequency_levels[f1base] > frequency_levels[f2base]

def calculate_returns(series: pd.Series, frequency: str) -> pd.Series:
    """
    Calculate periodic returns from a NAV time series.

    Parameters:
    - series (pd.Series): Time series of NAV values with a datetime index.
    - frequency (str): Resampling frequency (e.g., 'M', 'W').

    Returns:
    - pd.Series: Percentage change returns at the specified frequency.
    """
    return series.resample(frequency).last().pct_change(fill_method=None).dropna()

def align_data(nav_returns: pd.Series, factor_returns: pd.DataFrame, col: str):
    """
    Align NAV returns and factor returns on a common datetime index,
    and subtract the risk-free rate from NAV returns.

    Parameters:
    - nav_returns (pd.Series): Series of NAV returns.
    - factor_returns (pd.DataFrame): DataFrame of factor returns including RF column.
    - col (str): Name of the risk-free rate column.

    Returns:
    - Tuple[pd.Series, pd.DataFrame]: Aligned NAV excess returns, aligned factor returns.
    """
    common_index = nav_returns.index.intersection(factor_returns.index)
    aligned_nav = nav_returns.loc[common_index] - factor_returns.loc[common_index, col]
    aligned_factors = factor_returns.loc[common_index, [f for f in factor_returns.columns if f != col]]
    return aligned_nav, aligned_factors
