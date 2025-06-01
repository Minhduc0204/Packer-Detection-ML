import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import mode

# Load dataset
df = pd.read_csv("dataset packer - metadata.csv")
df["packer_type"] = df["packer_type"].astype(int)

X = df.drop(columns=["file_path", "packer_type"])
y = df["packer_type"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Load models
#dt_model = joblib.load(r"D:\MLMAL\Packer-Detection-ML\DecisionTree\decision_tree_packer_model.pkl")
rf_model = joblib.load(r"D:\MLMAL\Packer-Detection-ML\RandomForest\random_forest_packer_model.pkl")
xgb_model = joblib.load(r"D:\MLMAL\Packer-Detection-ML\XGBoost\xgboost_packer_model.pkl")
lgb_model = joblib.load(r"D:\MLMAL\Packer-Detection-ML\LightGBM\lgb_packer_model.pkl")
# Get predictions
#dt_preds = dt_model.predict(X_test)
rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)
lgb_model = lgb_model.predict(X_test)

# Hard voting (majority vote)
all_preds = np.vstack([lgb_model, rf_preds, xgb_preds]) #,dt_preds ])
y_pred, _ = mode(all_preds, axis=0, keepdims=False)

# Evaluate
print("🗳️ Hard Voting Ensemble Evaluation")
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report (Hard Voting):")
print(classification_report(y_test, y_pred, zero_division=0))
print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
