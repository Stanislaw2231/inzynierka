import pandas as pd
import os

def read_data(files):
    acc_data = []
    gyr_data = []

    for f in files:
        df = pd.read_csv(f)
        filename = os.path.basename(f)
        parts = filename.split("-")

        participant = parts[0]
        label = parts[1]
        category = parts[2].rstrip("123")

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in filename:
            df["set"] = len(acc_data) + 1
            acc_data.append(df)
        elif "Gyroscope" in filename:
            df["set"] = len(gyr_data) + 1
            gyr_data.append(df)

    acc_df = pd.concat(acc_data, ignore_index=True)
    gyr_df = pd.concat(gyr_data, ignore_index=True)

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    for col in ["epoch (ms)", "time (01:00)", "elapsed (s)"]:
        if col in acc_df:
            del acc_df[col]
        if col in gyr_df:
            del gyr_df[col]

    return acc_df, gyr_df