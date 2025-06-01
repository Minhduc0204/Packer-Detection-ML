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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Load dataset
df = pd.read_csv(r"D:\MLMAL\Packer-Detection-ML\dataset packer - metadata.csv")  # Replace with your actual CSV file
X = df.drop(columns=["file_path", "packer_type"])
y = df["packer_type"]


# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(objective="multi:softprob", num_class=25, eval_metric="mlogloss", use_label_encoder=False, random_state=42),
    "LightGBM": LGBMClassifier(objective="multiclass", num_class=25, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
accuracies = []
# Plot learning curves
plt.figure(figsize=(12, 7))

for name, model in models.items():
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_scaled, y, cv=3, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    test_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, test_mean, label=name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
plt.title("Learning Curves for All Models")
plt.xlabel("Training Set Size")
plt.ylabel("Validation Accuracy")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.bar(models.keys(), accuracies, color='skyblue')
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()