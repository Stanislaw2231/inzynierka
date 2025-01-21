import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# filepath: /Users/stanislaw/Docs/inzynierka/inzynierka/data/partially processed/filtered_data_sample.csv
df = pd.read_csv("data/partially processed/filtered_data.csv")

# Select numeric columns for PCA
columns_to_check = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]
X = df[columns_to_check].values

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Print explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Optional: plot explained variance
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Scree Plot')
plt.show()

# PCA-transformed DataFrame
df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(len(columns_to_check))])
print(df_pca.head())


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Prepare the data
X = df[columns_to_check]
y = df['label']

X_filtered = df_filtered[columns_to_check]
y_filtered = df_filtered['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(X_filtered, y_filtered, test_size=0.3, random_state=42)

# Train the Random Forest classifier on the original data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Train the Random Forest classifier on the filtered data
clf_filtered = RandomForestClassifier(n_estimators=100, random_state=42)
clf_filtered.fit(X_train_filtered, y_train_filtered)

# Make predictions
y_pred = clf.predict(X_test)
y_pred_filtered = clf_filtered.predict(X_test_filtered)

# Print classification reports
print("Classification report for original data:")
print(classification_report(y_test, y_pred))

print("Classification report for filtered data:")
print(classification_report(y_test_filtered, y_pred_filtered))