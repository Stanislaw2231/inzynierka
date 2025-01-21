import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# odszumianie sygnału

df = pd.read_csv('data/partially processed/resampled_data_no_outliers.csv')
df.drop(columns=['outlier_label'], inplace=True)
df['epoch (ms)'] = pd.to_datetime(df['epoch (ms)'])
df.set_index('epoch (ms)', inplace=True)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def calculate_set_duration(df):
    for i in df["set"].unique():
        duration = df[df["set"] == i].index[-1] - df[df["set"] == i].index[0]
        df.loc[df["set"] == i, "duration"] = duration.seconds
    
    duration_data = df.groupby("category")["duration"].mean()
    return duration_data

def apply_low_pass_filter(df, duration_data):
    df_filtered = df.copy()
    columns_to_check = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

    for cat, duration in duration_data.items():
        subset = df_filtered[df_filtered["category"] == cat]
        if len(subset) == 0:
            continue
        if cat == "heavy":
            fs = len(subset) / (duration / 5)
        if cat == "medium":
            fs = len(subset) / (duration / 10)

        cutoff = fs / 5.0
        for col in columns_to_check:
            filtered = butter_lowpass_filter(subset[col].values, cutoff, fs)
            df_filtered.loc[subset.index, col] = filtered

    df_filtered.to_csv('data/partially processed/filtered_data.csv')
    return df_filtered

duration_data = calculate_set_duration(df)
df_filtered = apply_low_pass_filter(df, duration_data)
    


