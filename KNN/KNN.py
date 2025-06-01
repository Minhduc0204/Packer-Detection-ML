import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# === Train KNN model ===
model = KNeighborsClassifier(n_neighbors=3, weights='distance')
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("\n KNN Classification Report:")
print(classification_report(y_test, y_pred, labels=list(range(25)),digits=4, zero_division=0))
print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=list(range(25))))
print("Accuracy:", accuracy_score(y_test, y_pred))

# === Save model ===
joblib.dump(model, "knn_packer_model.pkl")
print("✅ KNN model saved as 'knn_multiclass_packer_model.pkl'")
