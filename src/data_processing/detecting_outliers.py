import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns


def LOF_outliers(df, k=20):
    columns_to_check = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    features = df[columns_to_check]
    
    lof = LocalOutlierFactor(n_neighbors=k)
    outlier_labels = lof.fit_predict(features)
    df['outlier_label'] = outlier_labels
    
    for col in columns_to_check:
        df.loc[df['outlier_label'] == -1, col] = np.nan

    return df

def chauvenet_outliers(values):
    mean_val = np.mean(values)
    std_val = np.std(values)
    N = len(values)
    z = np.abs((values - mean_val) / (std_val if std_val != 0 else 1e-10))
    prob_two_tailed = 2 * (1 - norm.cdf(z))
    return prob_two_tailed < (1.0 / (2 * N))

def detect_outliers_chauvenet(df):
    columns_to_check = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    df["outlier"] = False
    for col in columns_to_check:
        outlier_mask = chauvenet_outliers(df[col])
        df.loc[outlier_mask, "outlier"] = True