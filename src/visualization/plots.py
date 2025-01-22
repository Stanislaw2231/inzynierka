import pandas as pd
import numpy as np
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor

df = pd.read_csv("data/partially processed/resampled_data.csv")

# numeric_cols = df.select_dtypes(include=['float64']).columns

# for col in numeric_cols:
#     if len(df) > 3:
#         stat, p_value = shapiro(df[col])
#         if p_value > 0.05:
#             print(f"{col} is normally distributed")
#         else:
#             print(f"{col} is not normally distributed")
            

def LOF_outliers(df, k=20):
    columns_to_check = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    features = df[columns_to_check]
    
    lof = LocalOutlierFactor(n_neighbors=k)
    outlier_labels = lof.fit_predict(features)
    df['outlier_label'] = outlier_labels
    for col in columns_to_check:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df.index, y=df[col], hue=df['outlier_label'], palette={1: 'blue', -1: 'red'})
        plt.title(f'Outlier Detection in {col}')
        plt.xlabel('Index')
        plt.ylabel(col)
        plt.legend(title='Outlier Label', labels=['blue - inlier', 'red - outlier'])
        plt.show()

    return df

# df = pd.read_pickle('data/partially processed/data_with_engineered_features.pkl')

# set_to_plot = 4
# subset_df = df[df['set'] == set_to_plot]
# plt.plot(subset_df['acc_y'], label='acc_y')
# plt.plot(subset_df['acc_y_roll'], label='acc_y_roll')
# plt.legend()
# plt.show()


# plt.figure(figsize=(10, 6))
# plt.plot(df_set1.index, df_set1["acc_y"], label="Original acc_y")
# plt.plot(df_filtered_set1.index, df_filtered_set1["acc_y"], label="Filtered acc_y")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("acc_y")
# plt.title("acc_y before and after filtering")
# plt.show()

numeric_cols = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]

def visualize_numeric_cols(df):
    sns.pairplot(df[numeric_cols + ["label"]], hue="label", diag_kind="kde")
    plt.show()
    

#visualize_numeric_cols(df)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def visualize_3d(df):
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(111, projection='3d')
    classes = df["label"].unique()
    colors = sns.color_palette("Set1", n_colors=len(classes))
    
    for cls, color in zip(classes, colors):
        subset = df[df["label"] == cls]
        ax.scatter(
            subset["acc_x"], 
            subset["acc_y"], 
            subset["acc_z"], 
            color=color, 
            label=cls
        )
    
    ax.set_xlabel("acc_x")
    ax.set_ylabel("acc_y")
    ax.set_zlabel("acc_z")
    ax.legend()
    plt.show()
    
visualize_3d(df)