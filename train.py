#!/usr/bin/env python3
"""
Wine Quality Classification Training Script
Trains a RandomForest model on sklearn's Wine dataset and saves it for serving.
"""

import os
import json
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Load dataset
print("Loading Wine dataset...")
wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names
target_names = wine.target_names.tolist()

print(f"Dataset shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Classes: {target_names}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
print("\nTraining RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Save model
model_path = "models/wine_classifier.pkl"
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")

# Save metadata (feature names + target names)
metadata = {
    "feature_names": feature_names,
    "target_names": target_names,
    "model_type": "RandomForestClassifier",
    "accuracy": float(accuracy),
    "n_features": len(feature_names)
}
metadata_path = "models/metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {metadata_path}")

print("\n✅ Training complete! Model is ready for deployment.")