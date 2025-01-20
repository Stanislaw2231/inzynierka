import pandas as pd
import numpy as np
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor

df = pd.read_csv("data/partially processed/resampled_data.csv")

numeric_cols = df.select_dtypes(include=['float64']).columns

for col in numeric_cols:
    if len(df) > 3:
        stat, p_value = shapiro(df[col])
        if p_value > 0.05:
            print(f"{col} is normally distributed")
        else:
            print(f"{col} is not normally distributed")
            

def LOF_outliers(df, k=20):
    columns_to_check = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    features = df[columns_to_check]
    
    lof = LocalOutlierFactor(n_neighbors=k)
    outlier_labels = lof.fit_predict(features)
    df['outlier_label'] = outlier_labels
    for col in columns_to_check:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df.index, y=df[col], hue=df['outlier_label'], palette='coolwarm')
        plt.title(f'Outlier Detection in {col}')
        plt.xlabel('Index')
        plt.ylabel(col)
        plt.legend(title='Outlier Label')
        plt.show()

    return df