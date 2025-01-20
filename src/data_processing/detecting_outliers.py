import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns

def chauvenet_outliers(values):
    mean_val = np.mean(values)
    std_val = np.std(values)
    N = len(values)
    z = np.abs((values - mean_val) / (std_val if std_val != 0 else 1e-10))
    prob_two_tailed = 2 * (1 - norm.cdf(z))
    return prob_two_tailed < (1.0 / (2 * N))

def LOF_outliers(df, k=20):
    columns_to_check = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    features = df[columns_to_check]
    lof = LocalOutlierFactor(n_neighbors=k)
    outlier_labels = lof.fit_predict(features)
    lof_scores = lof.negative_outlier_factor_

    df['outlier_label'] = outlier_labels
    
    return df

def detect_outliers_chauvenet(df):
    columns_to_check = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    # Identify Chauvenet outliers
    df["outlier"] = False
    for col in columns_to_check:
        outlier_mask = chauvenet_outliers(df[col])
        df.loc[outlier_mask, "outlier"] = True

def plot_outliers(df, columns_to_check):
    # Plot each feature against the outlier label
    for col in columns_to_check:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df.index, y=df[col], hue=df['outlier_label'], palette='coolwarm')
        plt.title(f'Outlier Detection in {col}')
        plt.xlabel('Index')
        plt.ylabel(col)
        plt.legend(title='Outlier Label')
        plt.show()
        
df = pd.read_csv("data/partially processed/resampled_data.csv")
df_no_outliers = df.copy()
columns_to_check = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
for col in columns_to_check:
    for label in df["label"].unique():
        data = LOF_outliers(df[df["label"] == label], col)
        
        data.loc[data[col + "_outlier"] == -1, col] = np.nan
        
        df_no_outliers.loc[df_no_outliers["label"] == label, col] = data[col]
        