# 1. Shared paths, settings and data helpers
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# 2. File locations
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "diabetes_130_raw.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
TABLES_DIR = ROOT / "reports" / "tables"

# 3. Random seed and number of folds
SEED = 42
N_FOLDS = 5

# 4. Key columns: target <30 days, grouped by patient
RAW_TARGET = "readmitted"
POSITIVE_LABEL = "<30"
TARGET = "target"
GROUP_COL = "patient_nbr"
ID_COL = "encounter_id"
MISSING_TOKEN = "?"
DIAGNOSIS_COLUMNS = ["diag_1", "diag_2", "diag_3"]

# 5. Death or hospice discharges: cannot be readmitted
EXCLUDED_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]

# 6. Loading the raw CSV (only "?" is missing)
def load_raw() -> pd.DataFrame:
    return pd.read_csv(
        CSV_PATH,
        na_values=[MISSING_TOKEN],
        keep_default_na=False,
        dtype={col: str for col in DIAGNOSIS_COLUMNS},
        low_memory=False,
    )

# 7. Framing the dataset with a binary target
def load_dataset() -> pd.DataFrame:
    df = load_raw()
    df[TARGET] = (df[RAW_TARGET] == POSITIVE_LABEL).astype(int)
    discharged_alive = ~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)
    valid_gender = df["gender"].isin(["Female", "Male"])
    return df[discharged_alive & valid_gender].drop(columns=RAW_TARGET).reset_index(drop=True)

# 8. Creating the grouped, stratified CV splitter
def make_cv() -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# 9. Splitting into development (80%) and test (20%)
def split_dev_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev_idx, test_idx = next(make_cv().split(df, df[TARGET], df[GROUP_COL]))
    return df.iloc[dev_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

# 10. Printing a step title
def title(text: str) -> None:
    print(text)

# 11. Printing a section header
def header(text: str) -> None:
    print()
    print(text)
