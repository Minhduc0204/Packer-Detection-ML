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
        file_size = os.path.getsize(file_path)
        num_sections = len(pe.sections)

        suspicious_imports = 0
        suspicious_apis = ["LoadLibraryA", "GetProcAddress", "VirtualAlloc", "CreateProcessA"]
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name and imp.name.decode() in suspicious_apis:
                        suspicious_imports += 1

        features = {
            "entropy": entropy,
            "file_size": file_size,
            "num_sections": num_sections,
            "suspicious_imports": suspicious_imports,
            "VirtualSize": pe.sections[0].Misc_VirtualSize,
            "SizeOfImage": pe.OPTIONAL_HEADER.SizeOfImage,
            "SizeOfHeaders": pe.OPTIONAL_HEADER.SizeOfHeaders,
            "Relocations": len(pe.DIRECTORY_ENTRY_BASERELOC) if hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC') else 0,
            "Imports": len(pe.DIRECTORY_ENTRY_IMPORT) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0,
            "BaseOfCode": pe.OPTIONAL_HEADER.BaseOfCode,
            "DllCharacteristics": pe.OPTIONAL_HEADER.DllCharacteristics,
            "AddressOfEntryPoint": pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            "SizeOfInitializedData": pe.OPTIONAL_HEADER.SizeOfInitializedData,
            "SizeOfUninitializedData": pe.OPTIONAL_HEADER.SizeOfUninitializedData
        }

        return pd.DataFrame([features])

    except Exception as e:
        print(f"[!] Feature extraction failed: {e}")
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
    virtual_size = features["VirtualSize"].values[0]
    print("Raw Features:\n", features)
    
    result_label.config(text=f"Packer Type: {prediction}\nEntropy: {entropy:.2f}\nVirtual Size: {virtual_size}")

# App window
app = tk.Tk()
app.title("PE Packer Detector")
app.geometry("400x200")

tk.Button(app, text="Select EXE File", command=browse_file).pack(pady=10)
result_label = tk.Label(app, text="", font=("Arial", 12))
result_label.pack(pady=20)

app.mainloop()
