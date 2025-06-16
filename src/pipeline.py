# pipeline.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load data from the correct path
df = pd.read_csv("data/data.csv")
df = df.drop(["date", "street", "city", "statezip", "country"], axis=1)

# Features and target
X = df.drop("price", axis=1)
y = df["price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Save model inside 'models' folder
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/model.pkl")
print("✅ Pipeline trained and saved as models/model.pkl")
