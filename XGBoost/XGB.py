import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

# === 1. Load Dataset ===
df = pd.read_csv("metadata_custom_multilabel.csv")
df["packer_type"] = df["packer_type"].astype(int)

# === 2. Prepare Features and Labels ===
X = df.drop(columns=["file_path", "packer_type"])
y = df["packer_type"]

# === 3. Stratified Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === 4. Define and Train XGBoost Model ===
xgb_model = XGBClassifier(
    objective="multi:softmax",           # for multiclass classification
    num_class=25,                        # total number of classes (0–24)
    eval_metric="mlogloss",             # multiclass log loss
    use_label_encoder=False,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# === 5. Predict and Evaluate ===
y_pred = xgb_model.predict(X_test)

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=list(range(25))))

print("\n Classification Report:")
print(classification_report(y_test, y_pred, labels=list(range(25)), zero_division=0))

# === 6. Save the Trained Model ===
joblib.dump(xgb_model, "xgb_multiclass_packer_model.pkl")
print("\n XGBoost model saved as 'xgb_multiclass_packer_model.pkl'")
