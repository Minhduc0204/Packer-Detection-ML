import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load dataset
df = pd.read_csv("dataset packer - metadata.csv")

# Features and labels
X = df.drop(columns=["file_path", "packer_type"])
y = df["packer_type"]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Load pre-trained models
dt = joblib.load(r"D:\MLMAL\Packer-Detection-ML\DecisionTree\decision_tree_packer_model.pkl")
rf = joblib.load(r"D:\MLMAL\Packer-Detection-ML\RandomForest\random_forest_packer_model.pkl")
xgb = joblib.load(r"D:\MLMAL\Packer-Detection-ML\XGBoost\xgboost_packer_model.pkl")
lgb = joblib.load(r"D:\MLMAL\Packer-Detection-ML\LightGBM\lgb_packer_model.pkl")

# Predict probabilities
probs_xgb = xgb.predict_proba(X_test)
probs_rf  = rf.predict_proba(X_test)
probs_dt  = dt.predict_proba(X_test)
probs_lgb = lgb.predict_proba(X_test)

# Average (equal weight soft voting)
avg_probs = (probs_xgb + probs_rf + probs_dt + probs_lgb) / 4
y_pred = np.argmax(avg_probs, axis=1)

# Evaluate
print("\n📊 Soft Voting (Equal Weight) with 4 Models")
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n", classification_report(y_test, y_pred))
