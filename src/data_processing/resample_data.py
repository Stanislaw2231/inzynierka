import pandas as pd

from reading_data import read_data


def resample_data(files):
    agg_rules = {
        "acc_x": "mean",
        "acc_y": "mean",
        "acc_z": "mean",
        "gyro_x": "mean",
        "gyro_y": "mean",
        "gyro_z": "mean",
        "participant": "last",
        "label": "last",
        "category": "last",
        "set": "last"
    }

    acc_df, gyr_df = read_data(files)
    combined_df = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

    combined_df.columns = [
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "participant", "label",
        "category", "set"
    ]

    grouped_days = [group for _, group in combined_df.groupby(pd.Grouper(freq="D"))]
    resampled_frames = [
        day_df.resample("150ms").apply(agg_rules).dropna() for day_df in grouped_days
    ]

    result_df = pd.concat(resampled_frames)
    result_df["set"] = result_df["set"].astype(int)
    result_df['category'] = result_df['category'].str.replace(r'(_MetaWear_2019|\d+)', '', regex=True)
    return result_df

def export_data(data, output_path):
    data.to_csv(output_path, index=False)