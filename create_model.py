import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

print("Starting model creation...")

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# Load dataset
try:
    df = pd.read_csv("dataset/data.csv")
    print("Dataset loaded successfully!")
except Exception as e:
    print("Error loading dataset:", e)
    exit()

# Split data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "models/model.pkl")

print("✅ Model created successfully at models/model.pkl")