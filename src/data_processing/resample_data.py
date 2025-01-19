import pandas as pd

from reading_data import read_data


def resample_data(files):
    acc_df, gyr_df = read_data(files)
    df_mergerd = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)

    df_mergerd.columns = [
        "acc_x",
        "acc_y",
        "acc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "participant",
        "label",
        "category",
        "set"
    ]

    sampling_rules = {
        "acc_x": "mean",
        "acc_y": "mean",
        "acc_z": "mean",
        "gyr_x": "mean",
        "gyr_y": "mean",
        "gyr_z": "mean",
        "participant": "last",
        "label": "last",
        "category": "last",
        "set": "last"
    }

    days = [g for n, g in df_mergerd.groupby(pd.Grouper(freq="D"))]
    df_resampled = pd.concat([df.resample("150ms").apply(sampling_rules).dropna() for df in days])

    df_resampled["set"] = df_resampled["set"].astype("int")
    return df_resampled

def export_data(data, output_path):
    data.to_csv(output_path, index=False)