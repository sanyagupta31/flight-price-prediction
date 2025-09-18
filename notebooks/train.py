import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("data/IndianFlightdata - Sheet1.csv")

# Example preprocessing (adjust as per dataset)
df = df.dropna()  # simple handling

# Encoding categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest baseline
rf = RandomForestRegressor(random_state=42)

# Hyperparameter search
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt", "log2"],
}

random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist,
    n_iter=10, cv=3, scoring="neg_mean_squared_error",
    n_jobs=-1, verbose=1, random_state=42
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

# Evaluate
# Evaluate
y_pred = best_model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Save model
joblib.dump(best_model, "models/flight_price_model.pkl")
print(" Model saved in models/flight_price_model.pkl")

# Save metrics in a text file
with open("models/metrics.txt", "w") as f:
    f.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}\n")
    f.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}\n")
    f.write(f"R²: {r2_score(y_test, y_pred):.4f}\n")

print(" Metrics saved in models/metrics.txt")

