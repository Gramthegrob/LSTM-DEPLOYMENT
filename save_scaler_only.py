"""
Save the scaler from your training data
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

DATA_PATH = os.path.join("data", "Improved_All_Combined_hr_rsp_binary.csv")

# Load dataset
print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Fill missing values
df["HR"].fillna(method="ffill", inplace=True)
df["HR"].fillna(method="bfill", inplace=True)

# Get features
features = ["HR", "respr"]
X = df[features].values

# Fit scaler
print("Fitting scaler...")
scaler = StandardScaler()
scaler.fit(X)

# Save
os.makedirs("models", exist_ok=True)
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

print("✅ Scaler saved to models/scaler.pkl")