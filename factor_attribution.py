# factor_attribution.py

import pandas as pd
import numpy as np
import polars as pl
import statsmodels.api as sm
from typing import Union, Optional
from pathlib import Path
from sanitizer import sanitize_factors, sanitize_nav_series, infer_frequency, is_higher_frequency, align_data, calculate_returns
from loader import fetch_invespar_factors, fetch_iima_factors
from regression import run_rolling_ols, run_rolling_ols_with_stability_check
from plotting import plot_attributions



class FactorAttribution:
    """
    This class attributes portfolio to factor returns
    """
    FREQUENCY_LEVELS = {
        'S': 60*60*7*252*0.5, 'T': 60*60*7*252*0.5, 'MIN': 60*7*252*0.5, 'H': 252*7, 'D': 252, 'B': 252*0.5,
        'W': 26, 'W-SUN': 26, 'W-MON': 26,
        'M': 12, 'BM': 12, 'MS': 12, 'BMS': 12, 'ME':12,
        'Q': 2, 'Q-DEC': 2, 'QS': 2, 'QS-JAN': 2,
        'A': 0.5, 'Y': 0.5, 'A-DEC': 0.5, 'AS': 0.5, 'YS': 0.5, 'AS-JAN': 0.5
    }
    
    invespar = "https://invespar.com/ajax/download/ff6"
    iima = "https://web.iima.ac.in/~iffm/Indian-Fama-French-Momentum/DATA/2023-03_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv"
    
    def __init__(self, nav_series: Union[pd.Series, pd.DataFrame], 
                 date_pl: Optional[Union[pd.Series, pl.Series]] =None, 
                 factors: Optional[pd.DataFrame] = None,
                 frequency: str = 'M',
                 is_decimal: bool = True):
        """
        Initialize the FactorAttribution class.

        :param nav_series: Pandas Series or DataFrame of NAVs with datetime index.
        :param factor_returns: DataFrame of factor returns or path to CSV file.
        :param frequency: Frequency to resample returns ('D', 'W', 'M').
        """
        self.nav_series = sanitize_nav_series(nav_series, date=date_pl)

        if frequency not in self.FREQUENCY_LEVELS:
            raise ValueError(f"Unsupported frequency: '{frequency}'. Please use one of: {list(self.FREQUENCY_LEVELS.keys())}")
        self.frequency = frequency
        
        
       
        self.nav_freq = infer_frequency(self.nav_series.index)        
        if is_higher_frequency(self.frequency, self.nav_freq, self.FREQUENCY_LEVELS):
            raise ValueError(f"Cannot compute attribution at '{self.frequency}' frequency "
                             f"when input NAV is at '{self.nav_freq}' frequency. "
                             "Downsampling is fine, but upsampling is not supported."
                             )
        
        if factors is None:
            if is_higher_frequency(self.frequency, 'M', frequency_levels=self.FREQUENCY_LEVELS):
                self.factor_returns, self.risk_free_col = fetch_iima_factors(self.iima)
                self.factor_returns/=100
            else:
                self.factor_returns, self.risk_free_col = fetch_invespar_factors(self.invespar)
                self.factor_returns/=100
        else:
            self.factor_returns, self.risk_free_col = sanitize_factors(factors)
            if not is_decimal:
                self.factor_returns/=100
        
        self.fact_freq = infer_frequency(self.factor_returns.index)    
        if is_higher_frequency(self.frequency, self.fact_freq, self.FREQUENCY_LEVELS):
            raise ValueError(f"Cannot compute attribution at '{self.frequency}' frequency "
                        f"when input FACTOR DATA is at '{self.fact_freq}' frequency. "
                        "Downsampling is fine, but upsampling is not supported."
                        )    

        
        # Ensure both series are aligned and resampled to the same frequency
        self.nav_returns = calculate_returns(self.nav_series, self.frequency)
        self.factor_returns = self.factor_returns.resample(frequency).apply(lambda x: (1 + x).prod() - 1)
        self.nav_returns, self.factor_returns = align_data(self.nav_returns,self.factor_returns, self.risk_free_col)
        
        if len(self.factor_returns.columns) *2 > len(self.factor_returns):
            self.bounds = (len(self.factor_returns.columns)+1, len(self.factor_returns))
        else:
            self.bounds = (len(self.factor_returns.columns)*2, len(self.factor_returns))
 
    
    def _get_default_window(self, mode: str = 'balanced') -> int:
        mode_multipliers = {'responsive': 0.25, 'balanced': 0.5, 'stable': 1.0}
        if mode not in mode_multipliers:
            raise ValueError(f"Unsupported mode: {mode}")

        base_window = len(self.factor_returns.columns) * 3
        freq_scale = self.FREQUENCY_LEVELS[self.frequency.upper()]
        return max(int(base_window + freq_scale * mode_multipliers[mode]), base_window)
    

    def rolling_attribution(self, window: Optional[int] = None, window_mode: str = 'balanced', adaptive_window: bool = False) -> tuple[pd.DataFrame, pd.Series]:
        self.lower, self.upper = self.bounds
        if window is None:
            window = max(min(self.factor_returns.shape[0], self._get_default_window(window_mode)), self.factor_returns.shape[1]+1)
        elif window < self.lower or window > self.upper:
            raise ValueError(f"Window beyond bounds. Lower: {self.lower}, Upper: {self.upper}")

        self.window = window
        X = sm.add_constant(self.factor_returns)
        method = 'elbow' if self.FREQUENCY_LEVELS[self.frequency.upper()]< self.FREQUENCY_LEVELS['W'] else '90%_SNR'

        coef_df, rsquared, self.window = run_rolling_ols(self.nav_returns, X, window, best_window=adaptive_window, method=method)
        return coef_df, rsquared
    
  
    def rolling_attribution_custom_check(self, window: Optional[int] = None, window_mode: str = 'balanced') -> tuple[pd.DataFrame, pd.Series]:
        self.lower, self.upper = self.bounds
        if window is None:
            window = max(min(self.upper, self._get_default_window(window_mode)), self.lower)
        elif window < self.lower or window > self.upper:
            raise ValueError(f"Window beyond bounds. Lower: {self.lower}, Upper: {self.upper}")

        self.window = window
        X = sm.add_constant(self.factor_returns)
        return run_rolling_ols_with_stability_check(self.nav_returns, X, window)
   

    def plot_attributions(self, coef_df: pd.DataFrame, rsquared:pd.Series):
        return plot_attributions(coef_df, rsquared, self.window, self.frequency)
        

    def summary(self, show_summary: bool = True) -> dict:
        """
        Summarize NAV and factor data.

        Parameters:
        - verbose (bool): If True, prints summary. If False, just returns it.

        Returns:
        - dict: Dictionary containing summary stats for further use.
        """
        nav_summary = {
            "shape": self.nav_series.shape,
            "start": self.nav_series.index.min(),
            "end": self.nav_series.index.max(),
            "stats": self.nav_series.describe()
        }

        factor_summary = {
            "shape": self.factor_returns.shape,
            "start": self.factor_returns.index.min(),
            "end": self.factor_returns.index.max(),
            "columns": list(self.factor_returns.columns),
            "stats": self.factor_returns.describe()
        }

        if show_summary:
            print("NAV Data Summary:")
            print(f"  Range: {nav_summary['start']} to {nav_summary['end']}")
            print(f"  Shape: {nav_summary['shape']}")
            print(nav_summary['stats'])

            print("\nFactor Data Summary:")
            print(f"  Range: {factor_summary['start']} to {factor_summary['end']}")
            print(f"  Shape: {factor_summary['shape']}")
            print(f"  Columns: {factor_summary['columns']}")
            print(factor_summary['stats'])

        return {"NAV": nav_summary, "Factors": factor_summary}
