import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# === Load dataset ===
df = pd.read_csv("dataset packer - metadata.csv")

# === Prepare features and labels ===
X = df.drop(columns=["file_path", "packer_type"])
y = df["packer_type"]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === Load trained XGBoost model ===
xgb_model = joblib.load("random_forest_packer_model.pkl")  # ← YOUR TRAINED MODEL FILE

# === Make predictions ===
y_pred = xgb_model.predict(X_test)

# === Compute confusion matrix ===
labels = np.arange(25)  # Include all class labels from 0 to 24
cm = confusion_matrix(y_test, y_pred, labels=labels)

# === Plot confusion matrix ===
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Random Forest Model")
plt.tight_layout()
plt.show()
