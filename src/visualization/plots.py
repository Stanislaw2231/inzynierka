import pandas as pd
import numpy as np
from scipy.stats import shapiro

df = pd.read_csv("data/partially processed/resampled_data.csv")

numeric_cols = df.select_dtypes(include=['float64']).columns

for col in numeric_cols:
    if len(df) > 3:
        stat, p_value = shapiro(df[col])
        if p_value > 0.05:
            print(f"{col} is normally distributed")
        else:
            print(f"{col} is not normally distributed")