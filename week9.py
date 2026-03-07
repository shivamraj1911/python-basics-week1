# HOUSE PRICE PREDICTION MODEL (ALL CODE IN ONE FRAME)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# 1 Load Dataset
# -----------------------------
df = pd.read_csv("house_prices (1).csv")

print("\nFirst 5 rows of dataset:\n")
print(df.head())

print("\nDataset Shape:", df.shape)


# -----------------------------
# 2 Features and Target
# -----------------------------
X = df.drop(columns=["Price","Property_ID"])
y = df["Price"]


# -----------------------------
# 3 Define Categorical Columns
# -----------------------------
categorical_columns = ["Location","Property_Type"]
numeric_columns = ["Area","Bedrooms","Bathrooms","Age"]


# -----------------------------
# 4 Encoding + Model Pipeline
# -----------------------------
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), categorical_columns),
    ("num", "passthrough", numeric_columns)
])

model = RandomForestRegressor(n_estimators=100, random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# -----------------------------
# 5 Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 6 Train Model
# -----------------------------
pipeline.fit(X_train, y_train)


# -----------------------------
# 7 Predictions
# -----------------------------
predictions = pipeline.predict(X_test)


# -----------------------------
# 8 Evaluation Metrics
# -----------------------------
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE\n")

print("MAE :", mae)
print("MSE :", mse)
print("R2 Score :", r2)


# -----------------------------
# 9 Visualization
# -----------------------------
plt.scatter(y_test, predictions)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

plt.savefig("predictions_vs_actual.png")

plt.show()


print("\nGraph saved as predictions_vs_actual.png")