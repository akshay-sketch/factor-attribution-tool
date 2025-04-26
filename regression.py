# regression.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import warnings
snr_scores = {}
def _score_window(returns, factors, window):
    model = RollingOLS(returns, factors, window=window).fit()
    betas = model.params
    ses = model.bse

    drift = betas.std()
    noise = ses.mean()
    sns = (drift/noise).sum()
    return sns
def run_rolling_ols(y: pd.Series, X: pd.DataFrame, window: int, best_window: bool =False, method: str ='elbow') -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Run Rolling OLS regression over a specified window.

    Parameters:
    - y (pd.Series): Target variable (NAV excess returns).
    - X (pd.DataFrame): Factor exposures with constant column.
    - window (int): Window size for rolling regression.

    Returns:
    - Tuple[pd.DataFrame, pd.Series]: Coefficients DataFrame and Rsquared Series.
    """
    if not best_window:
        best_window_size =window
    elif len(y)<X.shape[1]*1.5:
        warnings.warn("Not enough observations for Adaptive Window. Falling Back to default window")
        best_window_size= window
    else:
        if len(y)>1000:
            step =25
        elif len(y)>600:
            step =20
        elif len(y)>300:
            step = 10
        elif len(y)>100:
            step = 5
        else:
            step=2
        best_window_size = None
        
        for w in range(int(X.shape[1]*2),int(0.8*len(y)), step):
            try:
                snr = _score_window(y,X, window=w)
                snr_scores[w] = snr
                
            except Exception as e:
                
                continue  # if any window fails, skip
        
        if (len(snr_scores)==0):
            raise ValueError("Cannot find Best Window when using Adaptive Window.")
        if method=='elbow':
            first_grad = np.gradient(list(snr_scores.values()))
            second_grad = np.gradient(first_grad)
            elbow_index = np.argmax(second_grad)
            best_window_size = list(snr_scores.keys())[elbow_index]
        else:
            max_snr = max(snr_scores.values())
            best_window_size = min(m for m, v in snr_scores.items() if v>(0.9 * max_snr))

        #Can Plot ElbowPlot
        plt.plot(list(snr_scores.keys()), list(snr_scores.values()))
        plt.title(f"Method: {method}")
        
    if best_window_size>=len(y):
        raise ValueError(f"Window Required :  Not enough data points:{len(y)} for smart window detection.")
    rolling_model = RollingOLS(endog=y, exog=X, window=best_window_size)
    rolling_results = rolling_model.fit()
    coef_df = rolling_results.params.dropna()
    rsquared = rolling_results.rsquared
    return coef_df, rsquared, best_window_size
    

def run_rolling_ols_with_stability_check(y: pd.Series, X: pd.DataFrame, window: int, threshold: float = 1e4) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run Rolling OLS regression with condition number check to skip unstable windows.

    Parameters:
    - y (pd.Series): Target variable (NAV excess returns).
    - X (pd.DataFrame): Factor exposures with constant column.
    - window (int): Rolling window size.
    - threshold (float): Condition number threshold for skipping unstable windows.

    Returns:
    - Tuple[pd.DataFrame, pd.Series]: Coefficient DataFrame and R-squared Series.
    """
    coefs = []
    rsq = []
    dates = []

    for i in range(window, len(y)):
        y_win = y.iloc[i - window:i]
        X_win = X.iloc[i - window:i]

        cond = np.linalg.cond(X_win.to_numpy())
        if cond > threshold:
            coefs.append([np.nan] * X_win.shape[1])
            rsq.append(np.nan)
            dates.append(y.index[i])
            continue

        model = sm.OLS(y_win, X_win)
        results = model.fit()
        coefs.append(results.params.values)
        rsq.append(results.rsquared)
        dates.append(y.index[i])
    
    coef_df = pd.DataFrame(coefs, columns=X.columns, index=dates)
    rsquared_df = pd.Series(rsq, index=dates, name="Rsquared")
    return coef_df, rsquared_df

def run_regularized_regression(y: pd.Series, X: pd.DataFrame, method: str = "ridge", alpha: float = 1.0):
    """
    Placeholder for implementing Ridge or Lasso regression.

    Parameters:
    - y (pd.Series): Target returns.
    - X (pd.DataFrame): Factor exposures.
    - method (str): 'ridge' or 'lasso'.
    - alpha (float): Regularization strength.

    Returns:
    - TODO: Coefficients, R-squared, etc.
    """
    raise NotImplementedError(f"{method} regression not implemented yet.")