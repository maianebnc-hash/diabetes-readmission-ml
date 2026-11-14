"""Step 1 - EDA and problem framing.

Follows the five framing questions of Session 1:
  1. What is one row, and is the data sound?   -> data quality checks
  2. What is the target, and over what horizon? -> binary 30-day readmission
  3. Could a feature leak the answer?           -> leakage audit
  4. How do we split honestly?                  -> patient-grouped dev/test split + CV folds
  5. What is the no-skill baseline?             -> scores any model must beat

Run:  python step1_eda_framing.py
Outputs go to reports/figures and reports/tables.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, roc_auc_score

from common import (EXCLUDED_DISCHARGE_IDS, FIGURES_DIR, GROUP_COL, ID_COL, POSITIVE_LABEL,
                    RAW_TARGET, SEED, TABLES_DIR, TARGET, load_dataset, load_raw, make_cv,
                    split_dev_test)

BLUE, RED = "#4C72B0", "#C44E52"
COUNT_COLUMNS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
RATE_COLUMNS = ["age", "race", "gender", "number_inpatient", "A1Cresult", "insulin"]

# Prediction-time test (Session 1): could we know this value when the patient is discharged?
LEAKAGE_AUDIT = [
    ("Identifiers", "encounter_id, patient_nbr", "Yes", "High",
     "No clinical meaning; patient_nbr links encounters of one patient -> used only to group the split"),
    ("Demographics", "race, gender, age, weight", "Yes", "Low",
     "Known at admission; race and gender are sensitive attributes (fairness audit)"),
    ("Admission", "admission_type_id, admission_source_id, payer_code, medical_specialty", "Yes", "Low",
     "Recorded when the patient arrives"),
    ("Stay", "time_in_hospital, num_lab_procedures, num_procedures, num_medications", "Yes", "Low",
     "Complete once the stay ends, i.e. at prediction time"),
    ("Discharge", "discharge_disposition_id", "Yes", "Medium",
     "Known at discharge; death/hospice codes make readmission impossible -> those rows are excluded"),
    ("History", "number_outpatient, number_emergency, number_inpatient", "Yes", "Low",
     "Visits in the year BEFORE this encounter"),
    ("Diagnoses", "diag_1, diag_2, diag_3, number_diagnoses", "Yes", "Low",
     "Coded for this stay"),
    ("Lab tests", "max_glu_serum, A1Cresult", "Yes", "Low",
     "Measured during the stay; 'None' means not measured"),
    ("Medication", "23 drug columns, change, diabetesMed", "Yes", "Low",
     "Prescriptions decided during the stay"),
    ("Outcome", "readmitted", "No", "Target",
     "Only known 30 days later: it is the label, never a feature"),
]


def save_table(table: pd.DataFrame, name: str, index: bool = True) -> None:
    table.to_csv(TABLES_DIR / name, index=index)


# --------------------------------------------------------------------------- 1. Data quality
def data_quality(raw: pd.DataFrame) -> None:
    """Rows, duplicates, types, missing values and value ranges of the raw file."""
    print("\n[1] DATA QUALITY (raw file)")
    print(f"Rows: {len(raw)} | columns: {raw.shape[1]}")
    print(f"Duplicated rows: {raw.duplicated().sum()} | duplicated {ID_COL}: {raw[ID_COL].duplicated().sum()}")
    visits = raw[GROUP_COL].value_counts()
    print(f"Patients: {len(visits)} | patients with several encounters: {(visits > 1).sum()} "
          f"(max {visits.max()} encounters)")

    overview = pd.DataFrame({
        "dtype": raw.dtypes.astype(str),
        "n_unique": raw.nunique(),
        "missing_share": raw.isna().mean().round(4),
        "most_frequent": [raw[col].mode(dropna=True).iloc[0] for col in raw],
    })
    save_table(overview, "step1_column_overview.csv")
    missing = overview.loc[overview["missing_share"] > 0, "missing_share"].sort_values(ascending=False)
    print(f"Columns with missing values:\n{missing.to_string()}")
    constant = overview.index[overview["n_unique"] == 1].tolist()
    print(f"Columns with a single value (no information): {constant}")

    ranges = raw[COUNT_COLUMNS].describe().T[["min", "50%", "mean", "max"]].round(2)
    save_table(ranges, "step1_numeric_ranges.csv")
    print(f"Numeric ranges (no negative or impossible value):\n{ranges.to_string()}")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(missing.index[::-1], missing.values[::-1] * 100, color=BLUE)
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Columns with missing values")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step1_missingness.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- 2. Target
def target_framing(raw: pd.DataFrame, framed: pd.DataFrame) -> None:
    """From the 3-class outcome to a binary 30-day target on an eligible population."""
    print("\n[2] TARGET AND POPULATION")
    counts = raw[RAW_TARGET].value_counts()
    print(f"Raw outcome:\n{(counts / len(raw)).round(4).to_string()}")
    excluded = raw["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)
    rate_excluded = (raw.loc[excluded, RAW_TARGET] == POSITIVE_LABEL).mean()
    print(f"Death/hospice discharges removed: {excluded.sum()} "
          f"(their 30-day readmission rate: {rate_excluded:.4f})")
    print(f"Framed dataset: {len(framed)} encounters | positive rate: {framed[TARGET].mean():.4f} "
          f"| imbalance ratio: 1 positive for {(1 - framed[TARGET].mean()) / framed[TARGET].mean():.1f} negatives")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(counts.index, counts.values, color=[RED if c == POSITIVE_LABEL else BLUE for c in counts.index])
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value / len(raw):.1%}", ha="center", va="bottom")
    ax.set_title("Raw outcome (positive class in red)")
    ax.set_ylabel("Encounters")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step1_target_distribution.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- 3. Leakage
def leakage_audit() -> None:
    """Feature-group audit: availability at discharge and leakage risk."""
    print("\n[3] LEAKAGE AUDIT")
    audit = pd.DataFrame(LEAKAGE_AUDIT,
                         columns=["group", "columns", "available_at_discharge", "risk", "comment"])
    save_table(audit, "step1_leakage_audit.csv", index=False)
    print(audit[["group", "available_at_discharge", "risk"]].to_string(index=False))


# --------------------------------------------------------------------------- 4. Split
def split_report(dev: pd.DataFrame, test: pd.DataFrame) -> None:
    """Check that the split is stratified and that no patient is on both sides."""
    print("\n[4] HONEST SPLIT (grouped by patient, stratified on the target)")
    for name, part in [("development", dev), ("test", test)]:
        print(f"{name:12s} n={len(part):6d}  patients={part[GROUP_COL].nunique():6d}  "
              f"positive rate={part[TARGET].mean():.4f}")
    print(f"Patients in both development and test: {len(set(dev[GROUP_COL]) & set(test[GROUP_COL]))}")
    for fold, (train_idx, val_idx) in enumerate(make_cv().split(dev, dev[TARGET], dev[GROUP_COL])):
        shared = set(dev[GROUP_COL].iloc[train_idx]) & set(dev[GROUP_COL].iloc[val_idx])
        print(f"fold {fold}: train={len(train_idx)}  val={len(val_idx)}  "
              f"val positive rate={dev[TARGET].iloc[val_idx].mean():.4f}  shared patients={len(shared)}")


# --------------------------------------------------------------------------- EDA on development data
def rate_table(dev: pd.DataFrame, col: str) -> pd.DataFrame:
    """Encounter count and 30-day readmission rate per level of `col`."""
    values = dev[col].fillna("Missing")
    if col == "number_inpatient":
        values = dev[col].clip(upper=5)  # 5 means "5 or more"
    return dev.groupby(values)[TARGET].agg(n="count", positive_rate="mean").round(4)


def explore_development_set(dev: pd.DataFrame) -> None:
    """Relationship between features and the target. The test set is never looked at."""
    print("\n[EDA] FEATURES VS TARGET (development set only)")
    base = dev[TARGET].mean()
    rates = pd.concat({col: rate_table(dev, col) for col in RATE_COLUMNS}, names=["feature", "level"])
    save_table(rates, "step1_readmission_rates.csv")
    print(rates.loc[["number_inpatient", "race", "gender"]].to_string())

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), RATE_COLUMNS):
        table = rate_table(dev, col)
        ax.bar(table.index.astype(str), table["positive_rate"] * 100, color=BLUE)
        ax.axhline(base * 100, color=RED, linestyle="--", label=f"overall {base:.1%}")
        ax.set_title(f"30-day readmission rate by {col}")
        ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step1_readmission_rates.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for ax, col in zip(axes.ravel(), COUNT_COLUMNS):
        for label, color in [(0, BLUE), (1, RED)]:
            ax.hist(dev.loc[dev[TARGET] == label, col], bins=30, density=True,
                    alpha=0.5, color=color, label=f"target={label}")
        ax.set_title(col)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step1_numeric_distributions.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- 5. No-skill baseline
def no_skill_baselines(dev: pd.DataFrame) -> None:
    """Scores of predictors that learn nothing, averaged over the validation folds."""
    print("\n[5] NO-SKILL BASELINES (mean over the 5 validation folds)")
    rng = np.random.default_rng(SEED)
    rows = []
    for _, val_idx in make_cv().split(dev, dev[TARGET], dev[GROUP_COL]):
        y = dev[TARGET].to_numpy()[val_idx]
        predictors = {
            "majority (always 'not readmitted')": (np.zeros(len(y)), np.zeros(len(y))),
            "random": (rng.random(len(y)), None),
        }
        for name, (scores, labels) in predictors.items():
            labels = (scores >= 0.5).astype(int) if labels is None else labels
            rows.append({
                "baseline": name,
                "accuracy": accuracy_score(y, labels),
                "recall": recall_score(y, labels),
                "average_precision": average_precision_score(y, scores),
                "roc_auc": roc_auc_score(y, scores),
            })
    table = pd.DataFrame(rows).groupby("baseline").mean().round(4)
    save_table(table, "step1_no_skill_baselines.csv")
    print(table.to_string())
    print("Accuracy is misleading here: the majority predictor is right 89% of the time "
          "and finds no readmission. Average precision of a no-skill model = positive rate.")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    framed = load_dataset()
    dev, test = split_dev_test(framed)

    data_quality(raw)
    target_framing(raw, framed)
    leakage_audit()
    split_report(dev, test)
    explore_development_set(dev)
    no_skill_baselines(dev)
    print(f"\nFigures saved in {FIGURES_DIR.relative_to(FIGURES_DIR.parents[1])}, "
          f"tables in {TABLES_DIR.relative_to(TABLES_DIR.parents[1])}")


if __name__ == "__main__":
    main()
