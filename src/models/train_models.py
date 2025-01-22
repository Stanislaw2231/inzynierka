import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load data
df = pd.read_pickle("data/processed/data_with_engineered_features.pkl")

# Example target
y = df["label"]

# Example feature sets
fs1 = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]
fs2 = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z","pca_1","pca_2","pca_3"]
fs3 = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z","lda_1","lda_2","lda_3"]
fs4 = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z","acc_x_roll","acc_y_roll", "acc_z_roll","gyro_x_roll","gyro_y_roll","gyro_z_roll"]
fs5 = ["lda_1","lda_2","lda_3","acc_x_roll","acc_y_roll"]
fs6 = ['acc_y', 'pca_2', 'acc_x_roll', 'acc_y_roll', 'acc_z_roll', 'gyro_z_roll']
fs7 = ['acc_x', 'gyro_x', 'gyro_z', 'lda_1', 'lda_2', 'acc_x_roll', 'acc_y_roll', 'acc_z_roll', 'gyro_x_roll', 'gyro_z_roll']


feature_sets = [fs1, fs2, fs3, fs4, fs5, fs6, fs7]

# Models and parameter grids
models = {
    "LogisticRegression": (LogisticRegression(max_iter=1000),
        {"clf__C": [0.01, 0.1, 1, 10]}),
    "RandomForest": (RandomForestClassifier(),
        {"clf__n_estimators": [50, 100], "clf__max_depth": [5, 10, None]}),
    "NaiveBayes": (GaussianNB(),
        {}),
    "NeuralNetwork": (MLPClassifier(max_iter=1000),
        {"clf__hidden_layer_sizes": [(50,), (100,), (50, 50)], "clf__alpha": [0.0001, 0.001, 0.01]}),
    "DecisionTree": (DecisionTreeClassifier(),
        {"clf__max_depth": [5, 10, None], "clf__min_samples_split": [2, 5, 10]})
}

# Run best models on feature sets
results = []

for i, fs in enumerate(feature_sets, start=1):
    X = df[fs].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for m_name, (m_model, m_params) in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", m_model)])
        gsearch = GridSearchCV(pipe, param_grid=m_params, cv=3, n_jobs=-1)
        gsearch.fit(X_train, y_train)
        best_model = gsearch.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({"model": m_name, "feature_set": i,
                        "best_params": gsearch.best_params_, "accuracy": acc})

# Create DataFrame of results
res_df = pd.DataFrame(results)
print(res_df)