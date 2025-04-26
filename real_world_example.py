# real_world_example.py

import pandas as pd
from factor_attribution import FactorAttribution


# Load real NAV data (Parag Parikh Flexi Cap Fund NAV)
nav_data = pd.read_csv("PPFCF.csv", parse_dates=['Date'], dayfirst=True)

# Initialize Factor Attribution object
# Frequency chosen as 'M' (Monthly) because mutual funds are typically analyzed monthly
fa = FactorAttribution(nav_series=nav_data, frequency='W')

# Perform rolling attribution
coefs, rsq = fa.rolling_attribution(window_mode='responsive', adaptive_window=True)

# Display summary statistics
fa.summary()

# Plot rolling factor attributions and R-squared evolution
fa.plot_attributions(coefs, rsq)

"""
Notes:
- Ensure that 'PPFCF.csv' is present in the project directory.
- The tool automatically fetches monthly factor data from Invespar(Frequency:'M').
- If factors are passed, by default, factors are assumed to be in decimal form.
- If needed, modify the window size or attribution mode (responsive, balanced, stable) by passing arguments to rolling_attribution().
"""
