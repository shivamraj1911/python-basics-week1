# ============================================
# COMPREHENSIVE DATA SCIENCE CAPSTONE PROJECT
# Customer Churn Prediction
# End-to-End Workflow
# ============================================

# -------------------------------
# 1. Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os

print("\n==============================")
print("CAPSTONE DATA SCIENCE PROJECT")
print("==============================\n")


# -------------------------------
# 2. Data Collection / Loading
# -------------------------------
print("Loading dataset...")

if os.path.exists("customer_churn (3).csv"):
    df = pd.read_csv("customer_churn (3).csv")
else:
    print("Dataset not found. Creating simulated dataset...")
    
    np.random.seed(42)
    df = pd.DataFrame({
        "tenure": np.random.randint(1, 60, 500),
        "monthly_charges": np.random.randint(20, 120, 500),
        "total_charges": np.random.randint(100, 5000, 500),
        "support_calls": np.random.randint(0, 10, 500),
        "churn": np.random.randint(0, 2, 500)
    })

print("\nFirst 5 Rows:")
print(df.head())


# -------------------------------
# 3. Data Validation
# -------------------------------
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# -------------------------------
# 4. Data Dictionary
# -------------------------------
print("\nData Dictionary:")
dictionary = {
    "tenure": "Number of months customer stayed",
    "monthly_charges": "Monthly bill amount",
    "total_charges": "Total money spent",
    "support_calls": "Customer support calls",
    "churn": "Customer left company (1 = Yes, 0 = No)"
}

for k, v in dictionary.items():
    print(k, ":", v)


# -------------------------------
# 5. Exploratory Data Analysis
# -------------------------------
print("\nPerforming Exploratory Data Analysis...")

print("\nStatistical Summary:")
print(df.describe())


# Churn Distribution
plt.figure()
sns.countplot(x="churn", data=df)
plt.title("Customer Churn Distribution")
plt.show()


# Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()


# Feature vs churn
plt.figure()
sns.boxplot(x="churn", y="monthly_charges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()


# -------------------------------
# 6. Feature Engineering
# -------------------------------
print("\nFeature Engineering...")

df["avg_charge_per_month"] = df["total_charges"] / (df["tenure"] + 1)

print("New Feature Added: avg_charge_per_month")


# -------------------------------
# 7. Data Preprocessing
# -------------------------------
print("\nPreprocessing Data...")

X = df.drop("churn", axis=1)
y = df["churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -------------------------------
# 8. Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# -------------------------------
# 9. Model Training
# -------------------------------
print("\nTraining Models...")

log_model = LogisticRegression()
rf_model = RandomForestClassifier()

log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)


# -------------------------------
# 10. Hyperparameter Tuning
# -------------------------------
print("\nHyperparameter Tuning (Random Forest)...")

params = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(), params, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)


# -------------------------------
# 11. Model Evaluation
# -------------------------------
print("\nEvaluating Model...")

pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, pred))


# Confusion Matrix
cm = confusion_matrix(y_test, pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# -------------------------------
# 12. Save Model
# -------------------------------
print("\nSaving Model...")

joblib.dump(best_model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model Saved Successfully")


# -------------------------------
# 13. Prediction Function
# -------------------------------
def predict_churn(tenure, monthly, total, calls):

    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")

    avg = total / (tenure + 1)

    data = np.array([[tenure, monthly, total, calls, avg]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        return "Customer will churn"
    else:
        return "Customer will stay"


# -------------------------------
# 14. Example Prediction
# -------------------------------
print("\nExample Prediction")

result = predict_churn(12, 70, 900, 3)

print("Prediction Result:", result)


# -------------------------------
# 15. Business Insights
# -------------------------------
print("\nBusiness Insights:")
print("""
1. Customers with higher monthly charges tend to churn more.
2. Customers with frequent support calls show higher churn risk.
3. Long tenure customers are less likely to churn.

Recommendations:
• Offer loyalty rewards for long-term customers
• Improve customer support response time
• Provide discounted plans for high monthly charge customers
""")


print("\nPROJECT COMPLETED SUCCESSFULLY")