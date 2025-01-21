import pandas as pd
from dimensionality_reduction import perform_pca_lda, plot_explained_variance, compare_random_forest


if __name__ == "__main__":
    df = pd.read_csv('data/partially processed/filtered_data.csv')
    
    print(df.head())
    
    df, pca_explained_variance_ratio, lda_explained_variance_ratio = perform_pca_lda(df)
    #plot_explained_variance(lda_explained_variance_ratio, 'LDA Scree Plot', 'Linear Discriminant', 'Explained Variance Ratio')
    #plot_explained_variance(pca_explained_variance_ratio, 'PCA Scree Plot', 'Principal Component', 'Explained Variance Ratio')
    compare_random_forest(df, df, ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"], "label", "lda")
    
    df.head(10).to_csv('data/partially processed/sample.csv', index=False)
    
    
    # TODO
    #   temporal abstraction
    #   feature selection with decision tree
    #   confusion matrix do wyników
    #   wykresy wyników
    
    #
    #
    #