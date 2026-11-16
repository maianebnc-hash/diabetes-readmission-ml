import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, roc_auc_score

from common import (header, title, EXCLUDED_DISCHARGE_IDS, FIGURES_DIR, GROUP_COL, ID_COL, POSITIVE_LABEL,
                    RAW_TARGET, SEED, TABLES_DIR, TARGET, load_dataset, load_raw, make_cv,
                    split_dev_test)

# 1. Plot colours and columns to explore
BLUE, RED = "#4C72B0", "#C44E52"
COUNT_COLUMNS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
RATE_COLUMNS = ["age", "race", "gender", "number_inpatient", "A1Cresult", "insulin"]

# 2. Leakage audit: is each value known at discharge?
LEAKAGE_AUDIT = [
    ("Identifiers", "encounter_id, patient_nbr", "Yes", "High", "just IDs, patient_nbr is only used to group the split"),
    ("Demographics", "race, gender, age, weight", "Yes", "Low", "known at admission, race and gender are sensitive"),
    ("Admission", "admission_type_id, admission_source_id, payer_code, medical_specialty", "Yes", "Low", "recorded on arrival"),
    ("Stay", "time_in_hospital, num_lab_procedures, num_procedures, num_medications", "Yes", "Low", "known once the stay is over"),
    ("Discharge", "discharge_disposition_id", "Yes", "Medium", "we remove patients who died or went to hospice"),
    ("History", "number_outpatient, number_emergency, number_inpatient", "Yes", "Low", "visits in the year before"),
    ("Diagnoses", "diag_1, diag_2, diag_3, number_diagnoses", "Yes", "Low", "coded during the stay"),
    ("Lab tests", "max_glu_serum, A1Cresult", "Yes", "Low", "'None' means the test wasn't done"),
    ("Medication", "23 drug columns, change, diabetesMed", "Yes", "Low", "prescribed during the stay"),
    ("Outcome", "readmitted", "No", "Target", "this is what we predict"),
]

# 3. Saving a table
def save_table(table: pd.DataFrame, name: str, index: bool = True) -> None:
    table.to_csv(TABLES_DIR / name, index=index)

# 4. Understanding the raw data
def data_quality(raw: pd.DataFrame) -> None:
    header("[1] Data quality")
    print(f"Rows: {len(raw)}")
    print(f"Columns: {raw.shape[1]}")
    print(f"Duplicated rows: {raw.duplicated().sum()}")
    print(f"Duplicated {ID_COL}: {raw[ID_COL].duplicated().sum()}")
    visits = raw[GROUP_COL].value_counts()
    print(f"Patients: {len(visits)}")
    print(f"Patients with several encounters: {(visits > 1).sum()} (max {visits.max()})")

    overview = pd.DataFrame({
        "dtype": raw.dtypes.astype(str),
        "n_unique": raw.nunique(),
        "missing_share": raw.isna().mean().round(4),
        "most_frequent": [raw[col].mode(dropna=True).iloc[0] for col in raw],
    })
    save_table(overview, "step1_column_overview.csv")
    missing = overview.loc[overview["missing_share"] > 0, "missing_share"].sort_values(ascending=False)
    print()
    print("Columns with missing values:")
    print(missing.to_string())
    constant = overview.index[overview["n_unique"] == 1].tolist()
    print()
    print(f"Columns with a single value: {', '.join(constant)}")

    ranges = raw[COUNT_COLUMNS].describe().T[["min", "50%", "mean", "max"]].round(2)
    save_table(ranges, "step1_numeric_ranges.csv")
    print()
    print("Numeric ranges:")
    print(ranges.to_string())

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(missing.index[::-1], missing.values[::-1] * 100, color=BLUE)
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Columns with missing values")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step1_missingness.png", dpi=150)
    plt.close(fig)

# 5. Framing the 30-day target
def target_framing(raw: pd.DataFrame, framed: pd.DataFrame) -> None:
    header("[2] Target")
    counts = raw[RAW_TARGET].value_counts()
    print("Raw outcome:")
    print((counts / len(raw)).round(4).rename_axis(None).to_string())
    print()
    excluded = raw["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)
    rate_excluded = (raw.loc[excluded, RAW_TARGET] == POSITIVE_LABEL).mean()
    print(f"Death/hospice discharges removed: {excluded.sum()} (readmission rate {rate_excluded:.4f})")
    rate = framed[TARGET].mean()
    print(f"Framed dataset: {len(framed)} encounters")
    print(f"Positive rate: {rate:.4f}")
    print(f"Imbalance: 1 positive for {(1 - rate) / rate:.1f} negatives")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(counts.index, counts.values, color=[RED if c == POSITIVE_LABEL else BLUE for c in counts.index])
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value / len(raw):.1%}", ha="center", va="bottom")
    ax.set_title("Raw outcome (positive class in red)")
    ax.set_ylabel("Encounters")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step1_target_distribution.png", dpi=150)
    plt.close(fig)

# 6. Showing the leakage audit
def leakage_audit() -> None:
    header("[3] Leakage check")
    audit = pd.DataFrame(LEAKAGE_AUDIT,
                         columns=["group", "columns", "available_at_discharge", "risk", "comment"])
    save_table(audit, "step1_leakage_audit.csv", index=False)
    shown = audit[["group", "available_at_discharge", "risk"]]
    left = {col: (lambda v, w=shown[col].str.len().max(): f"{v:<{w}}") for col in shown}
    print(shown.to_string(index=False, formatters=left, justify="left"))

# 7. Checking the split
def split_report(dev: pd.DataFrame, test: pd.DataFrame) -> None:
    header("[4] Train/test split")
    for name, part in [("development", dev), ("test", test)]:
        patients = part[GROUP_COL].nunique()
        rate = part[TARGET].mean()
        print(f"{name:<12} rows: {len(part):<6}  patients: {patients:<6}  positive rate: {rate:.4f}")
    print(f"Patients in both: {len(set(dev[GROUP_COL]) & set(test[GROUP_COL]))}")
    print()
    for fold, (train_idx, val_idx) in enumerate(make_cv().split(dev, dev[TARGET], dev[GROUP_COL])):
        shared = set(dev[GROUP_COL].iloc[train_idx]) & set(dev[GROUP_COL].iloc[val_idx])
        rate = dev[TARGET].iloc[val_idx].mean()
        print(f"fold {fold}  train: {len(train_idx)}  val: {len(val_idx)}  val positive rate: {rate:.4f}  shared patients: {len(shared)}")

# 8. Readmission rate per value of a column
def rate_table(dev: pd.DataFrame, col: str) -> pd.DataFrame:
    values = dev[col].fillna("Missing")
    if col == "number_inpatient":
        values = dev[col].clip(upper=5)
    return dev.groupby(values)[TARGET].agg(n="count", positive_rate="mean").round(4)

# 9. Exploring the features (development set only)
def explore_development_set(dev: pd.DataFrame) -> None:
    header("[5] Readmission rate by feature")
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

# 10. Setting no-skill baselines
def no_skill_baselines(dev: pd.DataFrame) -> None:
    header("[6] Baselines")
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
    print(table.rename_axis(None).to_string())
    print()
    majority_accuracy = table.loc["majority (always 'not readmitted')", "accuracy"]
    print(f"Accuracy is misleading: always predicting 'no readmission' is {majority_accuracy:.0%} accurate but finds no readmission.")
    print("A no-skill model's average precision equals the positive rate.")

# 11. Running step 1
def main() -> None:
    title("Step 1: EDA and framing")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    framed = load_dataset()
    dev, test = split_dev_test(framed)

    # 1. Understanding the raw data
    data_quality(raw)

    # 2. Framing the 30-day target
    target_framing(raw, framed)

    # 3. Auditing leakage
    leakage_audit()

    # 4. Checking the split
    split_report(dev, test)

    # 5. Exploring the features
    explore_development_set(dev)

    # 6. Setting no-skill baselines
    no_skill_baselines(dev)

    print()
    print("Figures saved in reports/figures")
    print("Tables saved in reports/tables")

if __name__ == "__main__":
    main()
