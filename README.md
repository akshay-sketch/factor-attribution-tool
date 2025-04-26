# Factor Attribution Tool

## Overview

This project implements a **robust and flexible Python-based Factor Attribution tool**, designed to decompose portfolio returns into factor exposures over time using **rolling regression techniques**.

It is modular, production-ready, and capable of handling real-world messy data across different input types and frequencies. It also fetches external factor datasets via **API integration**.

## Features

- Accepts NAV inputs as:

  - Pandas Series
  - Pandas DataFrame
  - Polars Series / DataFrame
  - CSV file paths

- Flexible support for various NAV formats:

  - Indexed by date
  - Separate date columns

- Handles multiple data frequencies:

  - Daily (`D`)
  - Weekly (`W`)
  - Monthly (`M`)
  - Quarterly (`Q`)
  - Yearly (`A`, `Y`)

- Factor Data Sourcing:

  - Default: Fetches from **Invespar** or **IIM Ahmedabad** datasets via API.
  - Custom: Allows user-provided factor returns DataFrame.

- Rolling regression based performance attribution:

  - Standard Rolling OLS
  - Optional Rolling OLS with Condition Number Stability Check

- Risk-free rate handling and NAV excess return computation.

- Dynamic window size calculation for rolling regressions.

- Smart(Adaptive) window size calculation for rolling regressions.

- Caching of downloaded factor datasets for efficiency.

- Plotting of rolling factor loadings and R-squared values.

## Installation

```bash
pip install -r requirements.txt
```

Required Packages:

- pandas
- polars
- numpy
- matplotlib
- statsmodels
- requests

## Usage

### Synthetic Example

```python
from factor_attribution import FactorAttribution

nav_data = generate_synthetic_nav()
factors = generate_synthetic_factors()

fa = FactorAttribution(nav_series=nav_data, factors=factors, frequency='M')
coefs, rsq = fa.rolling_attribution()
fa.summary()
fa.plot_attributions(coefs, rsq)
```

### Real-World Example (Parag Parikh Flexi Cap Fund)

```python
import pandas as pd
from factor_attribution import FactorAttribution

nav_data = pd.read_csv("PPFCF.csv")
fa = FactorAttribution(nav_series=nav_data, frequency='M')
coefs, rsq = fa.rolling_attribution()
fa.summary()
fa.plot_attributions(coefs, rsq)
```

## Assumptions and Design Choices

- **Decimal Handling**:

  - If `is_decimal=True` (default) and custom factors are passed, it is assumed that the factors are already in decimal form (not percentages).

- **Regression Window Bounds, Default Window and Adaptive Window**:

  - Lower Bound: `# of factors + 1` if data points are low, else `# of factors x 2`.
  - Upper Bound: Total available data points.
  - Default Window: Computed dynamically as `(number of factors x 3) + frequency scaling factor`, adjusted based on mode (`responsive`, `balanced`, `stable`).
  - Adaptive Window: Computed using Signal to Noise ratio (in-Sample Metric) and elbow curve.
  - Final window selection is strictly validated within bounds.

- **Data Format Flexibility**:

  - The tool can handle NAVs and Factors provided in various file types (DataFrames, Series, Polars objects) with flexible datetime parsing.

- **Supported Frequencies**:

  - Daily (`D`), Weekly (`W`), Monthly (`M`), Quarterly (`Q`), and Yearly (`A`, `Y`) frequencies are fully supported.

- **Factor Usage**:

  - By default, factors are fetched automatically depending on desired frequency.
  - If the frequency is Monthly, it fetches **Invespar** data.
  - If the frequency is Daily, it fetches **IIM Ahmedabad** data.
  - Risk-free rate is automatically identified and used to compute excess returns.

## API Integration

- Factor datasets are fetched via HTTP requests.
- `requests` library is used with custom headers to ensure successful downloading.
- Data is cached locally to avoid redundant downloads.

## Potential Future Improvements

- Add Ridge and Lasso regularized regression options to better handle multicollinearity when using large factor sets.
- Extend factor libraries to include **Global Fama-French Factors** for multi-country portfolios.
- Build live connectors to online APIs for fully dynamic factor sourcing.
- Improvements aare required for Smart Window calculation i.e other in-sample metrics can be used.

## License

This project is shared for educational and demonstration purposes.


