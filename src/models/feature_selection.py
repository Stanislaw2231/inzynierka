import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
import pandas as pd


df = pd.read_pickle('F:\Studia\Inżynierka\inzynierka\data\partially processed\data_with_engineered_features.pkl')
features = [col for col in df.columns if col not in ['epoch (ms)', 'label', 'participant', 'set', 'category', 'duration']]
X = df[features]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
sfs = SequentialFeatureSelector(clf, n_features_to_select=10, direction='forward')
sfs.fit(X_train, y_train)

selected_features = [f for f, s in zip(features, sfs.get_support()) if s]
print("Selected features:", selected_features)

clf.fit(X_train, y_train)
importances = clf.feature_importances_
feature_importances_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print(feature_importances_df.sort_values(by='Importance', ascending=False))

accuracy = clf.score(X_test, y_test)
print("Accuracy score:", accuracy)