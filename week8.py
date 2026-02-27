# ================================
# Complete Business Analysis Project
# Customer Churn Analysis
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats

# --------------------------------
# Phase 1: Project Planning
# --------------------------------

print("Business Problem: Customer Churn Prediction")
print("Goal: Identify factors affecting churn and predict customer churn")

# --------------------------------
# Phase 2: Data Collection
# --------------------------------

# Load dataset
df = pd.read_csv("customer_churnn.csv")

print("\nDataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# --------------------------------
# Phase 3: Data Cleaning
# --------------------------------

# Fill missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical variables
label = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label.fit_transform(df[col])

# Save cleaned dataset
df.to_csv("cleaned_data.csv", index=False)

print("\nCleaned Data Saved")

# --------------------------------
# Phase 4: Exploratory Data Analysis
# --------------------------------

print("\nStatistical Summary")
print(df.describe())

# Visualization 1: Distribution
plt.figure()
sns.histplot(df.iloc[:,0], kde=True)
plt.title("Distribution of Feature 1")
plt.show()

# Visualization 2: Boxplot
plt.figure()
sns.boxplot(data=df)
plt.title("Boxplot of Dataset Features")
plt.show()

# Visualization 3: Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Visualization 4: Pairplot
sns.pairplot(df)
plt.show()

# Visualization 5: Countplot (Churn)
plt.figure()
sns.countplot(x=df.iloc[:,-1])
plt.title("Customer Churn Distribution")
plt.show()

# --------------------------------
# Phase 5: Hypothesis Testing
# --------------------------------

print("\nHypothesis Testing")

feature1 = df.iloc[:,0]
feature2 = df.iloc[:,1]

t_stat, p_value = stats.ttest_ind(feature1, feature2)

print("T-test Statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Significant Difference Found")
else:
    print("No Significant Difference")

# --------------------------------
# Phase 6: Machine Learning Model
# --------------------------------

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, pred))

print("\nClassification Report")
print(classification_report(y_test, pred))

# Confusion Matrix
cm = confusion_matrix(y_test, pred)

plt.figure()
sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --------------------------------
# Phase 7: Insights
# --------------------------------

print("\nBusiness Insights")
print("1. Certain features strongly correlate with churn.")
print("2. Logistic Regression predicts churn with reasonable accuracy.")
print("3. Businesses can focus on high-risk customers to reduce churn.")

print("\nRecommended Actions")
print("• Improve customer engagement")
print("• Provide loyalty offers")
print("• Monitor high-risk customers using ML predictions")