import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Load dataset
df = pd.read_csv("dataset packer - metadata.csv")
df["packer_type"] = df["packer_type"].astype(int)
X = df.drop(columns=["file_path", "packer_type"])
y = df["packer_type"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(y.unique()),
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_estimators=50,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("=== XGBoost ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "xgboost_packer_model.pkl")

# === Plot Learning Curve ===
train_sizes, train_scores, test_scores = learning_curve(
    model, X_scaled, y, cv=3, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', label="Training Score")
plt.plot(train_sizes, test_mean, 'o--', label="Validation Score")
plt.title("Learning Curve - XGBoost")
plt.xlabel("Number of Training Samples")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("xgboost_learning_curve.png")
plt.show()
