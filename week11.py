# ================================
# CUSTOMER SEGMENTATION PROJECT
# ================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# ================================
# 1. LOAD DATA
# ================================

df = pd.read_csv("customer_churn (3).csv")

print("\nDATASET SHAPE:", df.shape)
print(df.head())

# ================================
# 2. DATA PREPROCESSING
# ================================

df = df.drop(columns=["CustomerID"])

# Encode categorical columns
le = LabelEncoder()

categorical_cols = ["Contract","PaymentMethod","PaperlessBilling"]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# convert totalcharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df = df.fillna(df.mean())

print("\nDATA AFTER PREPROCESSING")
print(df.head())

# ================================
# 3. FEATURE SCALING
# ================================

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop("Churn",axis=1))

# ================================
# 4. ELBOW METHOD
# ================================

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()

# ================================
# 5. KMEANS CLUSTERING
# ================================

kmeans = KMeans(n_clusters=3, random_state=42)
df["Segment"] = kmeans.fit_predict(scaled_data)

print("\nKMEANS SEGMENT COUNT")
print(df["Segment"].value_counts())

# ================================
# 6. HIERARCHICAL CLUSTERING
# ================================

hc = AgglomerativeClustering(n_clusters=3)

df["HC_Segment"] = hc.fit_predict(scaled_data)

print("\nHIERARCHICAL SEGMENTS")
print(df["HC_Segment"].value_counts())

# ================================
# 7. SEGMENT ANALYSIS
# ================================

segment_profile = df.groupby("Segment").mean()

print("\nSEGMENT PROFILE")
print(segment_profile)

segment_profile.to_csv("segment_profiles.csv")

# ================================
# 8. MODEL BUILDING FOR EACH SEGMENT
# ================================

results = []

segments = df["Segment"].unique()

for seg in segments:

    print("\n=======================")
    print("SEGMENT:", seg)
    print("=======================")

    seg_df = df[df["Segment"]==seg]

    X = seg_df.drop(["Churn","Segment","HC_Segment"],axis=1)
    y = seg_df["Churn"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    # ================================
    # HYPERPARAMETER TUNING
    # ================================

    param_grid = {

        "n_estimators":[50,100,200],
        "max_depth":[3,5,10],
        "min_samples_split":[2,5]
    }

    rf = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(rf,param_grid,cv=3)

    grid.fit(X_train,y_train)

    best_model = grid.best_estimator_

    # ================================
    # PREDICTION
    # ================================

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test,y_pred)
    prec = precision_score(y_test,y_pred)
    rec = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    roc = roc_auc_score(y_test,y_prob)

    print("Accuracy:",acc)
    print("Precision:",prec)
    print("Recall:",rec)
    print("F1 Score:",f1)
    print("ROC AUC:",roc)

    results.append([seg,acc,prec,rec,f1,roc])

# ================================
# 9. SAVE MODEL RESULTS
# ================================

results_df = pd.DataFrame(results,columns=["Segment","Accuracy","Precision","Recall","F1","ROC_AUC"])

print("\nMODEL EVALUATION RESULTS")
print(results_df)

results_df.to_csv("model_evaluation_results.csv",index=False)

# ================================
# 10. BUSINESS INSIGHTS
# ================================

print("\nBUSINESS RECOMMENDATIONS")

print("""
Segment 0 – Premium Customers
• Offer loyalty rewards
• Provide premium support
• Upsell high value plans

Segment 1 – Budget Customers
• Offer discounts
• Promote affordable packages

Segment 2 – High Risk Customers
• Provide retention offers
• Improve customer support
""")

print("\nPROJECT COMPLETE")