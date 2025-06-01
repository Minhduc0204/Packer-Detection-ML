import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import pefile
import joblib
import os

# Load models
dt = joblib.load(r"D:\MLMAL\Packer-Detection-ML\DecisionTree\decision_tree_packer_model.pkl")
xgb = joblib.load(r"D:\MLMAL\Packer-Detection-ML\XGBoost\xgboost_packer_model.pkl")
rf = joblib.load(r"D:\MLMAL\Packer-Detection-ML\RandomForest\random_forest_packer_model.pkl")

# Feature extraction function
def extract_features(file_path):
    try:
        pe = pefile.PE(file_path)
        entropy = pe.sections[0].get_entropy()
        virtual_size = pe.sections[0].Misc_VirtualSize
        raw_size = pe.sections[0].SizeOfRawData
        number_of_sections = len(pe.sections)
        file_size = os.path.getsize(file_path)
        # Add more features based on your final dataset...
        
        return pd.DataFrame([{
            "entropy": entropy,
            "virtual_size": virtual_size,
            "raw_size": raw_size,
            "number_of_sections": number_of_sections,
            "file_size": file_size
        }])
    except:
        return None

# Voting function
def hard_vote(models, X):
    preds = [model.predict(X)[0] for model in models]
    return max(set(preds), key=preds.count)

# GUI
def browse_file():
    filepath = filedialog.askopenfilename(filetypes=[("Executable files", "*.exe")])
    if not filepath:
        return
    
    features = extract_features(filepath)
    if features is None:
        messagebox.showerror("Error", "Failed to extract features from the file.")
        return

    prediction = hard_vote([rf, xgb, dt], features)
    entropy = features["entropy"].values[0]
    virtual_size = features["virtual_size"].values[0]
    
    result_label.config(text=f"Packer Type: {prediction}\nEntropy: {entropy:.2f}\nVirtual Size: {virtual_size}")

# App window
app = tk.Tk()
app.title("PE Packer Detector")
app.geometry("400x200")

tk.Button(app, text="Select EXE File", command=browse_file).pack(pady=10)
result_label = tk.Label(app, text="", font=("Arial", 12))
result_label.pack(pady=20)

app.mainloop()
