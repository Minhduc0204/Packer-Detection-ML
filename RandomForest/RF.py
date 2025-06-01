import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("dataset packer - metadata.csv")
df["packer_type"] = df["packer_type"].astype(int)
X = df.drop(columns=["file_path", "packer_type"])
y = df["packer_type"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save
joblib.dump(model, "random_forest_packer_model.pkl")
