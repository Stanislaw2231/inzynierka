from glob import glob
import pandas as pd


pliki = glob("../../data/RawData/*.csv")

sciezka_pliku = "../../data/RawData/*.csv"
p = pliki[0]

participant = p.split("\\")[1].replace(sciezka_pliku, "").split("-")[0]
label = p.split("-")[1]
category = p.split("-")[2].rstrip("123")

df = pd.read_csv(p)

df["participant"] = participant
df["label"] = label
df["category"] = category

acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in pliki:
    participant = f.split("\\")[1].replace(sciezka_pliku, "").split("-")[0]
    label = p.split("-")[1]
    category = p.split("-")[2].rstrip("123")

    df = pd.read_csv(f)

    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])

    if "Gyroscope" in f:
        df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])

acc_df.info()
pd.to_datetime(df["epoch (ms)"], unit="ms")

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]



data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)

data_merged.head(50)

sampling = {
   "x-axis (g)": "mean",
   "y-axis (g)": "mean",
   "z-axis (g)": "mean",
   "x-axis (deg/s)": "mean",
   "y-axis (deg/s)": "mean",
   "z-axis (deg/s)": "mean",
   "label": "last",
   "category": "last",
   "participant": "last",
   "set": "last",
}

data_merged[:1000].resample(rule="200ms").apply(sampling)

days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

data_resampled["set"] = data_resampled["set"].astype("int")
data_resampled.info()

#dodać po picklez folderu interim zawartość po odpaleniu
data_resampled.to_pickle("../../data/intermediate/data_processed.pkl")

a = pd.read_pickle("../../data/intermediate/data_processed.pkl")

