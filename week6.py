# ============================================================
# INTERACTIVE SALES DASHBOARD (ADAPTED TO YOUR DATASET)
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("🚀 Starting Interactive Sales Dashboard Project...")

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("sales_data (2).csv")

print("Columns in dataset:")
print(df.columns)

# Convert Date
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M').astype(str)

# ==============================
# SEABORN THEME
# ==============================

sns.set_theme(style="whitegrid", palette="deep")

# ==============================
# 1️⃣ STATISTICAL VISUALIZATIONS
# ==============================

# Boxplot: Sales by Product
plt.figure(figsize=(10,6))
sns.boxplot(x='Product', y='Total_Sales', data=df)
plt.xticks(rotation=45)
plt.title("Sales Distribution by Product")
plt.tight_layout()
plt.savefig("boxplot_product_sales.png")
plt.close()

# Violin Plot: Sales by Region
plt.figure(figsize=(10,6))
sns.violinplot(x='Region', y='Total_Sales', data=df)
plt.title("Sales Distribution by Region")
plt.tight_layout()
plt.savefig("violin_region_sales.png")
plt.close()

# Histogram
plt.figure(figsize=(10,6))
sns.histplot(df['Total_Sales'], kde=True)
plt.title("Total Sales Distribution")
plt.tight_layout()
plt.savefig("hist_total_sales.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(8,6))
numeric_df = df[['Quantity','Price','Total_Sales']]
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

print("✅ Seaborn Visualizations Saved")

# ==============================
# DATA AGGREGATION
# ==============================

monthly_sales = df.groupby('Month')['Total_Sales'].sum().reset_index()
product_sales = df.groupby('Product')['Total_Sales'].sum().reset_index()
region_sales = df.groupby('Region')['Total_Sales'].sum().reset_index()

# KPIs
total_sales = df['Total_Sales'].sum()
avg_sales = df['Total_Sales'].mean()

print("Total Sales:", total_sales)

# ==============================
# 2️⃣ INTERACTIVE DASHBOARD
# ==============================

dashboard = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "xy"}, {"type": "xy"}],
           [{"type": "domain"}, {"type": "xy"}]],
    subplot_titles=("Monthly Sales Trend",
                    "Sales by Product",
                    "Region Share",
                    "Sales Distribution")
)

# Line Chart
dashboard.add_trace(
    go.Scatter(
        x=monthly_sales['Month'],
        y=monthly_sales['Total_Sales'],
        mode='lines+markers',
        name="Monthly Sales"
    ),
    row=1, col=1
)

# Bar Chart
dashboard.add_trace(
    go.Bar(
        x=product_sales['Product'],
        y=product_sales['Total_Sales'],
        name="Product Sales"
    ),
    row=1, col=2
)

# Pie Chart
dashboard.add_trace(
    go.Pie(
        labels=region_sales['Region'],
        values=region_sales['Total_Sales'],
        name="Region Share"
    ),
    row=2, col=1
)

# Histogram
dashboard.add_trace(
    go.Histogram(
        x=df['Total_Sales'],
        name="Sales Distribution"
    ),
    row=2, col=2
)

dashboard.update_layout(
    height=900,
    width=1200,
    title_text="Interactive Sales Performance Dashboard",
    template="plotly_white"
)

dashboard.write_html("Interactive_Sales_Dashboard.html")

print("✅ Interactive Dashboard Created")
print("📂 File Generated: Interactive_Sales_Dashboard.html")
print("🎉 PROJECT COMPLETED SUCCESSFULLY!")