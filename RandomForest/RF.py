import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# === 1. Load the multiclass dataset ===
df = pd.read_csv("metadata_custom_multilabel.csv")
df["packer_type"] = df["packer_type"].astype(int)

# === 2. Prepare features and labels ===
X = df.drop(columns=["file_path", "packer_type"])
y = df["packer_type"]

# === 3. Stratified train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === 4. Train RandomForest model ===
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# === 5. Predict on test set ===
y_pred = model.predict(X_test)

# === 6. Evaluation (force label 0 to be shown) ===
label_range = list(range(25))  # Labels 0 to 24

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=label_range))

print("\n Classification Report:")
print(classification_report(y_test, y_pred, labels=label_range, zero_division=0))

# === 7. Save the trained model ===
joblib.dump(model, "rf_multiclass_packer_model.pkl")
print("\n Model saved as 'rf_multiclass_packer_model.pkl'")
