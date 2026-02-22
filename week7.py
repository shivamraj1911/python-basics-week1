# Statistical Business Analysis Project
# Author: Shivam Raj

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# -----------------------------
# Load Dataset
# -----------------------------

try:
    df = pd.read_csv("sales_data.csv")
    print("Dataset Loaded Successfully\n")
except:
    print("Dataset not found. Creating sample dataset...")
    
    data = {
        "Sales": np.random.randint(20000,70000,100),
        "Marketing_Spend": np.random.randint(3000,15000,100),
        "Customers": np.random.randint(50,200,100),
        "Region": np.random.choice(["North","South"],100)
    }
    
    df = pd.DataFrame(data)
    df.to_csv("sales_data.csv",index=False)

print(df.head())


# -----------------------------
# DESCRIPTIVE STATISTICS
# -----------------------------

print("\n----- DESCRIPTIVE STATISTICS -----")

mean_sales = df['Sales'].mean()
median_sales = df['Sales'].median()
mode_sales = df['Sales'].mode()[0]
std_sales = df['Sales'].std()

print("Mean:",mean_sales)
print("Median:",median_sales)
print("Mode:",mode_sales)
print("Standard Deviation:",std_sales)

print("\nFull Statistical Summary")
print(df.describe())


# -----------------------------
# DISTRIBUTION ANALYSIS
# -----------------------------

plt.figure()
sns.histplot(df['Sales'], kde=True)
plt.title("Sales Distribution")
plt.show()

# Normality Test
print("\n----- NORMALITY TEST -----")

stat,p = stats.shapiro(df['Sales'])

print("p-value:",p)

if p > 0.05:
    print("Sales data is normally distributed")
else:
    print("Sales data is NOT normally distributed")


# -----------------------------
# CORRELATION ANALYSIS
# -----------------------------

print("\n----- CORRELATION ANALYSIS -----")

corr = df[['Sales','Marketing_Spend','Customers']].corr()

print(corr)

plt.figure()
sns.heatmap(corr,annot=True)
plt.title("Correlation Heatmap")
plt.show()


# -----------------------------
# HYPOTHESIS TESTING
# -----------------------------

print("\n----- HYPOTHESIS TESTING -----")

# Hypothesis 1 (Marketing Impact)

high_marketing = df[df['Marketing_Spend'] > df['Marketing_Spend'].median()]['Sales']
low_marketing = df[df['Marketing_Spend'] <= df['Marketing_Spend'].median()]['Sales']

t_stat,p_value = stats.ttest_ind(high_marketing,low_marketing)

print("\nT-Test Result")
print("T-statistic:",t_stat)
print("p-value:",p_value)

if p_value < 0.05:
    print("Marketing significantly affects sales")
else:
    print("Marketing does NOT significantly affect sales")


# Hypothesis 2 (Region vs Sales)

north_sales = df[df['Region']=="North"]['Sales']
south_sales = df[df['Region']=="South"]['Sales']

anova = stats.f_oneway(north_sales,south_sales)

print("\nANOVA Test")
print("p-value:",anova.pvalue)


# Hypothesis 3 (Marketing vs Customers correlation)

corr_coef,p_corr = stats.pearsonr(df['Marketing_Spend'],df['Customers'])

print("\nPearson Correlation Test")
print("Correlation:",corr_coef)
print("p-value:",p_corr)


# -----------------------------
# CONFIDENCE INTERVAL
# -----------------------------

print("\n----- CONFIDENCE INTERVAL -----")

mean = df['Sales'].mean()
std = df['Sales'].std()
n = len(df)

ci = stats.t.interval(
    0.95,
    df=n-1,
    loc=mean,
    scale=std/np.sqrt(n)
)

print("95% Confidence Interval:",ci)


# -----------------------------
# REGRESSION ANALYSIS
# -----------------------------

print("\n----- LINEAR REGRESSION -----")

X = df[['Marketing_Spend']]
y = df['Sales']

model = LinearRegression()

model.fit(X,y)

pred = model.predict(X)

print("Coefficient:",model.coef_[0])
print("Intercept:",model.intercept_)
print("R² Score:",model.score(X,y))

# Regression Plot
plt.figure()
sns.regplot(x='Marketing_Spend',y='Sales',data=df)
plt.title("Marketing Spend vs Sales")
plt.show()


# -----------------------------
# SAVE HYPOTHESIS RESULTS
# -----------------------------

with open("hypothesis_tests_results.txt","w") as f:
    f.write("T-test p-value: "+str(p_value)+"\n")
    f.write("ANOVA p-value: "+str(anova.pvalue)+"\n")
    f.write("Marketing-Customer Correlation: "+str(corr_coef))

print("\nResults saved to hypothesis_tests_results.txt")


# -----------------------------
# BUSINESS INSIGHTS REPORT
# -----------------------------

print("\n==============================")
print("STATISTICAL ANALYSIS REPORT")
print("==============================")

print(f"Average Sales: {round(mean,2)}")
print(f"95% Confidence Interval: {ci}")

print("\nCorrelation Sales-Marketing:",corr.loc['Sales','Marketing_Spend'])

if p_value < 0.05:
    print("Marketing significantly affects sales")
else:
    print("Marketing impact is not significant")

print("Regression R²:",model.score(X,y))

print("\nAnalysis Completed Successfully")