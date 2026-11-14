"""Shared by every step: project constants, data loading, framing and splitting."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "diabetes_130_raw.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
TABLES_DIR = ROOT / "reports" / "tables"

SEED = 42
N_FOLDS = 5                 # 1 fold (20%) is the held-out test set, CV uses 5 folds on the rest

RAW_TARGET = "readmitted"   # NO, >30 or <30
POSITIVE_LABEL = "<30"      # positive class: readmitted within 30 days
TARGET = "target"           # binary column created by load_dataset()
GROUP_COL = "patient_nbr"   # one patient can have several encounters
ID_COL = "encounter_id"
MISSING_TOKEN = "?"
DIAGNOSIS_COLUMNS = ["diag_1", "diag_2", "diag_3"]
# Discharge codes meaning death or hospice: these patients cannot be readmitted.
EXCLUDED_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]


def load_raw() -> pd.DataFrame:
    """Read the raw CSV. Only "?" is treated as missing.

    pandas would otherwise read the lab value "None" (test not performed) as a
    missing value, although it is a real, informative category.
    """
    return pd.read_csv(
        CSV_PATH,
        na_values=[MISSING_TOKEN],
        keep_default_na=False,
        dtype={col: str for col in DIAGNOSIS_COLUMNS},
        low_memory=False,
    )


def load_dataset() -> pd.DataFrame:
    """Return the framed dataset: one row per eligible encounter, binary `target`.

    Only row filtering and re-labelling happen here (no statistic is learned),
    so this is safe to run before the train/test split.
    """
    df = load_raw()
    df[TARGET] = (df[RAW_TARGET] == POSITIVE_LABEL).astype(int)
    discharged_alive = ~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)
    valid_gender = df["gender"].isin(["Female", "Male"])  # 3 encounters have an invalid code
    return df[discharged_alive & valid_gender].drop(columns=RAW_TARGET).reset_index(drop=True)


def make_cv() -> StratifiedGroupKFold:
    """Cross-validation splitter for the development set (pass groups=patient)."""
    return StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def split_dev_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into development (~80%) and held-out test (~20%) sets, grouped by patient.

    The test set is one fold of a stratified group K-fold: it keeps the positive
    rate and shares no patient with the development set.
    """
    dev_idx, test_idx = next(make_cv().split(df, df[TARGET], df[GROUP_COL]))
    return df.iloc[dev_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
