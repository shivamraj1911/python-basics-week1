import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

# Load datasets
sales_df = pd.read_csv("sales_data.csv")
customers_df = pd.read_csv("customer_data.csv")

# Basic exploration
print(sales_df.head())
print(customers_df.head())

print(sales_df.info())
print(customers_df.info())

# Missing values
print(sales_df.isnull().sum())
print(customers_df.isnull().sum())
