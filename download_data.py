"""Download the Diabetes 130-US Hospitals dataset from OpenML and save it as one CSV file.

Run once:  python download_data.py
The CSV is the raw data, exactly as published: nothing is cleaned or removed here.
"""
import os
from pathlib import Path

import certifi
from sklearn.datasets import fetch_openml

OPENML_ID = 4541
DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "diabetes_130_raw.csv"

# Some Python installs ship without root certificates; point them to certifi's bundle.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())


def main() -> None:
    diabetes = fetch_openml(data_id=OPENML_ID, as_frame=True, data_home=DATA_DIR)
    raw = diabetes.data.assign(readmitted=diabetes.target)
    raw.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(raw)} rows and {raw.shape[1]} columns to {CSV_PATH}")


if __name__ == "__main__":
    main()
