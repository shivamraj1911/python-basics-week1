# ==============================
# SALES DATA ANALYSIS PROJECT
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------------------
# 1. Load Dataset with Error Handling
# ------------------------------
file_path = "sales_data.csv"

try:
    if not os.path.exists(file_path):
        raise FileNotFoundError("CSV file not found. Please check file name or path.")

    df = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully")

except Exception as e:
    print("❌ Error loading file:", e)
    exit()

# ------------------------------
# 2. Data Cleaning & Validation
# ------------------------------
print("\n🔎 Original Columns:", df.columns)

# Standardize column names
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

print("🧹 Cleaned Columns:", df.columns)

# Required columns check
required_cols = ['date', 'product', 'quantity', 'price']
for col in required_cols:
    if col not in df.columns:
        print(f"❌ Missing required column: {col}")
        exit()

# Convert datatypes
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Remove missing values
df = df.dropna()

print("✅ Data cleaned successfully")
print(df.head())

# ------------------------------
# 3. Feature Engineering
# ------------------------------
df['total_sale'] = df['quantity'] * df['price']

# ------------------------------
# 4. Basic Analysis
# ------------------------------

# Total sales
total_revenue = df['total_sale'].sum()
print("\n💰 Total Revenue:", total_revenue)

# Best selling product
sales_by_product = df.groupby('product')['total_sale'].sum().sort_values(ascending=False)
best_product = sales_by_product.index[0]

print("🏆 Best Selling Product:", best_product)

# Daily sales trend
daily_sales = df.groupby('date')['total_sale'].sum()

# ------------------------------
# 5. Visualization (Matplotlib)
# ------------------------------

# Chart 1: Bar Chart - Sales by Product
plt.figure()
sales_by_product.plot(kind='bar')
plt.title("Total Sales by Product")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("sales_by_product.png")
plt.show()

# Chart 2: Line Chart - Daily Sales Trend
plt.figure()
daily_sales.plot(kind='line')
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("daily_sales_trend.png")
plt.show()

# ------------------------------
# 6. Report Generation
# ------------------------------
with open("report.txt", "w") as f:
    f.write("SALES DATA ANALYSIS REPORT\n")
    f.write("==========================\n\n")
    f.write(f"Total Revenue: {total_revenue}\n")
    f.write(f"Best Selling Product: {best_product}\n\n")
    f.write("Sales by Product:\n")
    f.write(sales_by_product.to_string())

print("\n📄 Report generated: report.txt")
print("📊 Charts saved as: sales_by_product.png & daily_sales_trend.png")

# ------------------------------
# 7. Final Message
# ------------------------------
print("\n🎉 Project Completed Successfully!")
