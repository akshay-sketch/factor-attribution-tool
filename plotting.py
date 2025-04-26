# plots.py
import pandas as pd
import matplotlib.pyplot as plt

def plot_attributions(coef_df: pd.DataFrame, rsquared: pd.Series, window: int, frequency: str):
    """
    Plot rolling factor coefficients and R-squared values.

    Parameters:
    - coef_df (pd.DataFrame): DataFrame of rolling factor coefficients.
    - rsquared (pd.Series): Series of R-squared values over time.
    - window (int): The rolling window used for attribution (for title context).
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    coef_df.clip(-5, 5).drop(columns='const', errors='ignore').plot(ax=axs[0])
    axs[0].set_title(f"Rolling Factor Attributions, Window: {window}, Freq: {frequency}")
    axs[0].set_xlabel(f"Date")
    axs[0].set_ylabel("Factor Coefficients")
    axs[0].grid(True)

    axs[1].plot(rsquared)
    axs[1].set_title(f"Rolling Rsquared of Model, Window: {window}")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Rsquared")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
    return fig