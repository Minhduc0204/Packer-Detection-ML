import pefile
import os
import math
import pandas as pd

# === PATHS TO YOUR DATA ===
packed_dir = r"D:\MLMAL\Dataset\dataset-packed-pe\packed"
unpacked_dir = r"D:\MLMAL\Dataset\dataset-packed-pe\not-packed"

# === Function to calculate entropy of a byte sequence ===
def calculate_entropy(data):
    if not data:
        return 0.0
    entropy = 0
    byte_count = [0] * 256
    for byte in data:
        byte_count[byte] += 1
    for count in byte_count:
        if count == 0:
            continue
        p = count / len(data)
        entropy -= p * math.log2(p)
    return entropy

# === Extract features from one PE file ===
def extract_features(file_path):
    try:
    

        with open(file_path, "rb") as f:
            raw_bytes = f.read()
        entropy = calculate_entropy(raw_bytes)

        pe = pefile.PE(data=raw_bytes)
        file_size = os.path.getsize(file_path)
        num_sections = len(pe.sections)

        # Decode section names properly
        section_names = [s.Name.decode(errors='ignore').strip('\x00') for s in pe.sections]

        # Check for suspicious imports
        suspicious_imports = 0
        suspicious_list = ['LoadLibraryA', 'GetProcAddress', 'VirtualAlloc', 'CreateProcessA']

        try:
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                for imp in entry.imports:
                    if imp.name:
                        imp_name = imp.name.decode(errors='ignore') if isinstance(imp.name, bytes) else imp.name
                        if imp_name in suspicious_list:
                            suspicious_imports += 1
        except AttributeError:
            pass  # No import table present

        return {
            "file_path": file_path,
            "entropy": round(entropy, 2),
            "file_size": file_size,
            "num_sections": num_sections,
            "suspicious_imports": suspicious_imports
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# === Loop over all EXE files ===
def scan_folder(folder_path, label):
    data = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".exe"):
                full_path = os.path.join(root, file)
                features = extract_features(full_path)
                if features:
                    features['is_packed'] = label
                    data.append(features)
    return data

# === Run the extraction ===
print("Scanning packed files...")
packed_data = scan_folder(packed_dir, label=1)

print("Scanning unpacked files...")
unpacked_data = scan_folder(unpacked_dir, label=0)

all_data = packed_data + unpacked_data

# === Save to CSV ===
df = pd.DataFrame(all_data)
df.to_csv("metadata.csv", index=False)

print("\n Feature extraction complete. Data saved to metadata.csv")
