# Customer Churn Prediction Pipeline
# Run this file in VS Code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from scipy import stats

# -------------------------------
# DAY 1 : LOAD AND EXPLORE DATA
# -------------------------------

df = pd.read_csv("customer_churn (2).csv")

print("\nFirst 5 rows\n")
print(df.head())

print("\nDataset Info\n")
print(df.info())

print("\nStatistics\n")
print(df.describe())

print("\nChurn Distribution\n")
print(df['Churn'].value_counts())

# -------------------------------
# DAY 2 : HANDLE CATEGORICAL DATA
# -------------------------------

# Label Encoding
label_encoder = LabelEncoder()
df['Churn_Label'] = label_encoder.fit_transform(df['Churn'])

# One Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# Ordinal Encoding example
ordinal_encoder = OrdinalEncoder()

# -------------------------------
# DAY 3 : FEATURE SCALING
# -------------------------------

minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

numeric_cols = df.select_dtypes(include=np.number).columns

df_minmax = df.copy()
df_standard = df.copy()

df_minmax[numeric_cols] = minmax_scaler.fit_transform(df[numeric_cols])
df_standard[numeric_cols] = standard_scaler.fit_transform(df[numeric_cols])

print("\nMinMax Scaling Applied")
print(df_minmax.head())

print("\nStandard Scaling Applied")
print(df_standard.head())

# -------------------------------
# DAY 4 : OUTLIER DETECTION
# -------------------------------

# IQR Method
Q1 = df['MonthlyCharges'].quantile(0.25)
Q3 = df['MonthlyCharges'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers_iqr = df[(df['MonthlyCharges'] < lower) | (df['MonthlyCharges'] > upper)]

print("\nOutliers using IQR:", len(outliers_iqr))

# Z-Score Method
z = np.abs(stats.zscore(df[numeric_cols]))

outliers_z = np.where(z > 3)

print("Outliers using Z-score:", len(outliers_z[0]))

# -------------------------------
# DAY 5 : FEATURE ENGINEERING
# -------------------------------

# Example engineered features

df['TotalSpend'] = df['MonthlyCharges'] * df['tenure']

df['AvgMonthlySpend'] = df['TotalSpend'] / (df['tenure'] + 1)

df['ChargePerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

df['TenureGroup'] = pd.cut(df['tenure'], bins=4, labels=[1,2,3,4])

df['HighValueCustomer'] = (df['MonthlyCharges'] > df['MonthlyCharges'].mean()).astype(int)

print("\nFeature Engineering Done")
print(df.head())

# -------------------------------
# DAY 6 : FEATURE SELECTION
# -------------------------------

correlation = df.corr()

plt.figure(figsize=(8,6))
sns.heatmap(correlation, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Drop weak correlated features example
df = df.drop(columns=['Churn'], errors='ignore')

# -------------------------------
# DAY 7 : BUILD PIPELINE
# -------------------------------

X = df.drop("Churn_Label", axis=1)
y = df["Churn_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier())
])

pipeline.fit(X_train, y_train)

# -------------------------------
# MODEL EVALUATION
# -------------------------------

pred = pipeline.predict(X_test)

print("\nModel Accuracy")
print(accuracy_score(y_test, pred))

print("\nClassification Report")
print(classification_report(y_test, pred))

# -------------------------------
# VISUALIZATION
# -------------------------------

plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Churn Distribution")
plt.show()

print("\nPipeline Execution Completed Successfully!")