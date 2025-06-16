# eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("E:\House price prediction\data\data.csv")

# Basic Info
print(df.head())
print(df.info())
print(df.describe())

# Missing values
print("Missing values:\n", df.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
