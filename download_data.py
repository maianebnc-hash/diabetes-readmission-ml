# 1. Downloading the raw dataset from OpenML (run once)
import os
from pathlib import Path

import certifi
from sklearn.datasets import fetch_openml

# 2. Dataset ID and save location
OPENML_ID = 4541
DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "diabetes_130_raw.csv"

# 3. Fixing missing SSL certificates
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# 4. Downloading and saving the CSV
def main() -> None:
    diabetes = fetch_openml(data_id=OPENML_ID, as_frame=True, data_home=DATA_DIR)
    raw = diabetes.data.assign(readmitted=diabetes.target)
    raw.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(raw)} rows and {raw.shape[1]} columns to {CSV_PATH}")

if __name__ == "__main__":
    main()
