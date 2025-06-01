import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load dataset
df = pd.read_csv("dataset packer - metadata.csv")
X = df.drop(columns=["file_path", "packer_type"])
y = df["packer_type"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
     "LightGBM": LGBMClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

colors = {
    "Decision Tree": "tab:blue",
    "Random Forest": "tab:orange",
    "KNN": "tab:green",
    "LightGBM": "tab:purple",
    "XGBoost": "tab:red"
}

# Plot
plt.figure(figsize=(14, 8))

for name, model in models.items():
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_scaled, y, cv=3, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean,  'x--', label=f"{name} - Test", color=colors[name])
    plt.plot(train_sizes, test_mean,'o-', label=f"{name} - Train", color=colors[name])

plt.title("Learning Curves of Models")
plt.xlabel("Number of Training Samples")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curves_all_models.png")
plt.show()
