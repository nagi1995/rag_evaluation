import os
import urllib.request

# ----------------------------
# CONFIGURATION
# ----------------------------
URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
SAVE_DIR = "data/raw"
SAVE_FILENAME = "hotpot_train.json"

# ----------------------------
# ENSURE SAVE DIRECTORY EXISTS
# ----------------------------
os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, SAVE_FILENAME)

# ----------------------------
# DOWNLOAD FILE
# ----------------------------
if not os.path.exists(save_path):
    print(f"🔄 Downloading dataset from {URL} ...")
    urllib.request.urlretrieve(URL, save_path)
    print(f"✅ Saved dataset to {save_path}")
else:
    print(f"✅ Dataset already exists at {save_path}, skipping download.")
