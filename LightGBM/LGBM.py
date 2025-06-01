import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# === Load dataset ===
df = pd.read_csv("dataset packer - metadata.csv")
df["packer_type"] = df["packer_type"].astype(int)

X = df.drop(columns=["file_path", "packer_type"])
y = df["packer_type"]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Train LightGBM model ===
model = lgb.LGBMClassifier(objective='multiclass', num_class=25, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("\n LightGBM Classification Report:")
print(classification_report(y_test, y_pred, labels=list(range(25)), digits=4, zero_division=0))
print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=list(range(25))))
print("Accuracy:", accuracy_score(y_test, y_pred))
# === Save model ===
joblib.dump(model, "lgb_packer_model.pkl")
print(" LightGBM model saved as 'lgb_packer_model.pkl'")
