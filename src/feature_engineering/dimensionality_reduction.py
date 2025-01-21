import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split




def plot_explained_variance(explained_variance_ratio, title, xlabel, ylabel):
    plt.plot(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def perform_pca_lda(df):
    numeric_cols = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]
    
    # PCA
    X = df[numeric_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    print("Explained Variance Ratio (PCA):", pca.explained_variance_ratio_)

    df["pca_1"] = X_pca[:,0]
    df["pca_2"] = X_pca[:,1]
    df["pca_3"] = X_pca[:,2]

    # LDA
    y = df["label"].values
    X_scaled = scaler.fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=3)
    X_lda = lda.fit_transform(X_scaled, y)
    print("Explained Variance Ratio (LDA):", lda.explained_variance_ratio_)

    df["lda_1"] = X_lda[:,0]
    df["lda_2"] = X_lda[:,1]
    df["lda_3"] = X_lda[:,2]

    df.to_pickle("data/partially processed/data_with_lda_pca.pkl")
    return df, pca.explained_variance_ratio_, lda.explained_variance_ratio_


def compare_random_forest(original_df, pca_df, numeric_cols, target_col, method):

    X_orig = original_df[numeric_cols]
    y_orig = original_df[target_col]
    

    pca_columns = [col for col in pca_df.columns if col.startswith(method)]
    X_pca = pca_df[pca_columns]
    y_pca = pca_df[target_col]
    

    Xo_train, Xo_test, yo_train, yo_test = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)

    Xp_train, Xp_test, yp_train, yp_test = train_test_split(X_pca, y_pca, test_size=0.2, random_state=42)
    

    rf_orig = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_orig.fit(Xo_train, yo_train)
    orig_preds = rf_orig.predict(Xo_test)
    orig_accuracy = accuracy_score(yo_test, orig_preds)
    

    rf_pca = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_pca.fit(Xp_train, yp_train)
    pca_preds = rf_pca.predict(Xp_test)
    pca_accuracy = accuracy_score(yp_test, pca_preds)
    
    print("Original data accuracy:", orig_accuracy)
    print("LDA data accuracy:", pca_accuracy)
    

    print("\nClassification report for original data:")
    print(classification_report(yo_test, orig_preds))
    
    print("\nClassification report for LDA data:")
    print(classification_report(yp_test, pca_preds))
    