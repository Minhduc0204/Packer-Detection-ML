import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("dataset packer - metadata.csv")
df["packer_type"] = df["packer_type"].astype(int)
X = df.drop(columns=["file_path", "packer_type"])
y = df["packer_type"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("✅ Model trained successfully.")
print("📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

# Save
joblib.dump(model, "decision_tree_packer_model.pkl")
print("\n💾 Model saved to decision_tree_packer_model.pkl")
