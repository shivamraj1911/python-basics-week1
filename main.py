import pandas as pd

df = pd.read_csv("sales_data.csv")

print(df.head())        # First 5 rows
print(df.shape)         # Rows & columns
print(df.columns)       # Column names
print(df.dtypes)        # Data types
print(df.info())        # Dataset info


# Fill missing Quantity and Price with 0
df['Quantity'] = df['Quantity'].fillna(0)
df['Price'] = df['Price'].fillna(0)

# Remove duplicate rows
df = df.drop_duplicates()


df['Total_Sales'] = df['Quantity'] * df['Price']


# Total revenue
total_revenue = df['Total_Sales'].sum()

# Best-selling product
best_product = df.groupby('Product')['Total_Sales'].sum().idxmax()

# Average order value
average_order = df['Total_Sales'].mean()

# Total quantity sold
total_quantity = df['Quantity'].sum()



import pandas as pd

# Load sales data
df = pd.read_csv("sales_data.csv")

# Data exploration
print("Dataset Shape:", df.shape)
print("Columns:", df.columns)

# Data cleaning
df['Quantity'] = df['Quantity'].fillna(0)
df['Price'] = df['Price'].fillna(0)
df = df.drop_duplicates()

# Feature engineering
df['Total_Sales'] = df['Quantity'] * df['Price']

# Sales analysis
total_revenue = df['Total_Sales'].sum()
best_product = df.groupby('Product')['Total_Sales'].sum().idxmax()
average_order_value = df['Total_Sales'].mean()
total_quantity = df['Quantity'].sum()

# Output results
print("\n📊 SALES ANALYSIS REPORT")
print("-" * 30)
print(f"Total Revenue: ₹{total_revenue:,.2f}")
print(f"Best-Selling Product: {best_product}")
print(f"Average Order Value: ₹{average_order_value:,.2f}")
print(f"Total Quantity Sold: {total_quantity}")
