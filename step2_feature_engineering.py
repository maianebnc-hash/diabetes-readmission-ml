"""Step 2 - Feature engineering and preprocessing.

Two building blocks, both designed to sit inside one scikit-learn Pipeline (Session 6):
  - FeatureEngineer: fixed, row-by-row rules. Nothing is learned from the data.
  - build_preprocessor(): imputation, scaling and one-hot encoding. Everything it
    learns (medians, means, category levels) comes from the training fold only.

Later steps import build_feature_pipeline() and append a model to it.

Run:  python step2_feature_engineering.py
Outputs go to reports/figures and reports/tables.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (DIAGNOSIS_COLUMNS, FIGURES_DIR, GROUP_COL, ID_COL, TABLES_DIR, TARGET,
                    load_dataset, make_cv, split_dev_test)

# Columns removed, with the reason found in Step 1.
DROPPED_COLUMNS = {
    ID_COL: "identifier, no clinical meaning",
    GROUP_COL: "identifier, used only to group the split",
    "weight": "97% missing",
    "examide": "single value in the whole dataset",
    "citoglipton": "single value in the whole dataset",
}
VISIT_COLUMNS = ["number_outpatient", "number_emergency", "number_inpatient"]
DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide.metformin", "glipizide.metformin", "glimepiride.pioglitazone",
    "metformin.rosiglitazone", "metformin.pioglitazone",
]
# Stored as integers, but they are labels: they must not be scaled or ordered.
CODE_COLUMNS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
ENGINEERED_NUMERIC = ["age_mid", "prior_visits_log", "has_prior_inpatient",
                      "n_diabetes_meds", "n_med_changes", "meds_per_day"]
ENGINEERED_CATEGORICAL = [f"{col}_group" for col in DIAGNOSIS_COLUMNS]

NUMERIC_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    *VISIT_COLUMNS, "number_diagnoses", *ENGINEERED_NUMERIC,
]
CATEGORICAL_FEATURES = [
    "race", "gender", "payer_code", "medical_specialty", "max_glu_serum", "A1Cresult",
    "change", "diabetesMed", *CODE_COLUMNS, *ENGINEERED_CATEGORICAL, *DRUG_COLUMNS,
]
RARE_LEVEL_MIN_FREQUENCY = 0.01  # levels seen in <1% of training rows are merged together

# ICD-9 chapters, following the grouping of Strack et al. (2014) for this dataset.
ICD9_RANGES = [
    (140, 239, "Neoplasms"),
    (390, 459, "Circulatory"),
    (460, 519, "Respiratory"),
    (520, 579, "Digestive"),
    (580, 629, "Genitourinary"),
    (710, 739, "Musculoskeletal"),
    (800, 999, "Injury"),
]
ICD9_SYMPTOM_CODES = {785: "Circulatory", 786: "Respiratory", 787: "Digestive", 788: "Genitourinary"}


def icd9_group(code) -> str:
    """Map one ICD-9 code (e.g. "428", "250.01", "V27") to a clinical family."""
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):  # supplementary codes (context, external causes)
        return "Other"
    number = int(float(code))
    if number == 250:
        return "Diabetes"
    if number in ICD9_SYMPTOM_CODES:
        return ICD9_SYMPTOM_CODES[number]
    for low, high, name in ICD9_RANGES:
        if low <= number <= high:
            return name
    return "Other"


def age_midpoint(bracket: str) -> float:
    """Turn an age bracket such as "[70-80)" into its midpoint, 75."""
    low, high = bracket.strip("[)").split("-")
    return (int(low) + int(high)) / 2


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create the engineered features and drop unusable columns.

    Stateless: each output value depends only on its own row, so this
    transformer cannot leak information from validation or test data.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1. Age bracket -> number, so that models can use its order.
        X["age_mid"] = X["age"].map(age_midpoint)

        # 2. ~700 diagnosis codes -> 10 clinical families per diagnosis slot.
        for col in DIAGNOSIS_COLUMNS:
            X[f"{col}_group"] = X[col].map(icd9_group)

        # 3. Use of care in the previous year. The log tames a few extreme counts,
        #    the threshold flag marks patients with at least one prior hospital stay.
        X["prior_visits_log"] = np.log1p(X[VISIT_COLUMNS].sum(axis=1))
        X["has_prior_inpatient"] = (X["number_inpatient"] > 0).astype(int)

        # 4. Treatment complexity and instability across the 21 diabetes drugs.
        drugs = X[DRUG_COLUMNS]
        X["n_diabetes_meds"] = (drugs != "No").sum(axis=1)
        X["n_med_changes"] = drugs.isin(["Up", "Down"]).sum(axis=1)

        # 5. Intensity of care: medications administered per day of stay.
        X["meds_per_day"] = X["num_medications"] / X["time_in_hospital"]

        # Administrative codes are labels, not quantities.
        X[CODE_COLUMNS] = X[CODE_COLUMNS].astype(str)

        replaced = ["age", *DIAGNOSIS_COLUMNS]
        return X.drop(columns=[*DROPPED_COLUMNS, *replaced], errors="ignore")


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Imputation, scaling and encoding, each learned on training data only.

    - numeric: median imputation, then standardisation. Scaling is required by
      regularised linear models; tree models can switch it off.
    - categorical: a missing value becomes its own "Missing" level (in Step 1,
      missing race or specialty went with a different readmission rate), then
      one-hot encoding. Rare levels are merged, unseen levels are handled at predict time.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=RARE_LEVEL_MIN_FREQUENCY, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)


def build_feature_pipeline(scale_numeric: bool = True) -> Pipeline:
    """Raw encounter rows -> model-ready matrix. Later steps append a model to it."""
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocess", build_preprocessor(scale_numeric)),
    ])


# --------------------------------------------------------------------------- checks and report material
def engineered_feature_summary(engineered: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Readmission rate per level (or quartile) of each engineered feature."""
    summaries = {}
    for col in [*ENGINEERED_NUMERIC, "diag_1_group"]:
        values = engineered[col]
        if values.nunique() > 12:
            values = pd.qcut(values, 4, duplicates="drop").astype(str)
        summaries[col] = y.groupby(values.to_numpy()).agg(n="count", positive_rate="mean")
    return pd.concat(summaries, names=["feature", "level"]).round(4)


def plot_engineered_features(summary: pd.DataFrame, base_rate: float) -> None:
    features = summary.index.get_level_values("feature").unique()
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    for ax, feature in zip(axes.ravel(), features):
        table = summary.loc[feature]
        table = table[table["n"] >= 100]  # hide levels too small to read a rate from
        ax.bar(table.index.astype(str), table["positive_rate"] * 100, color="#4C72B0")
        ax.axhline(base_rate * 100, color="#C44E52", linestyle="--", label=f"overall {base_rate:.1%}")
        ax.set_title(feature)
        ax.set_ylabel("30-day readmission rate (%)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step2_engineered_features.png", dpi=150)
    plt.close(fig)


def check_fold_safety(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> None:
    """Fit on one training fold, transform its validation fold: the leakage-safe pattern."""
    train_idx, val_idx = next(make_cv().split(X, y, groups))
    pipeline = build_feature_pipeline().fit(X.iloc[train_idx], y.iloc[train_idx])
    train_matrix = pipeline.transform(X.iloc[train_idx])
    val_matrix = pipeline.transform(X.iloc[val_idx])
    n_numeric = len(NUMERIC_FEATURES)
    print(f"Fitted on the training fold only ({len(train_idx)} rows), applied to the validation fold "
          f"({len(val_idx)} rows): {val_matrix.shape[1]} columns, "
          f"{int(np.isnan(val_matrix).sum())} missing values")
    print(f"Scaled numeric columns, mean on training fold: {train_matrix[:, :n_numeric].mean():.3f} "
          f"(exactly 0 by construction) | on validation fold: {val_matrix[:, :n_numeric].mean():.3f} "
          f"(close to 0, not exactly: its statistics were never used)")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    dev, _ = split_dev_test(load_dataset())  # the test set is not touched
    X, y, groups = dev.drop(columns=TARGET), dev[TARGET], dev[GROUP_COL]

    print("[1] DROPPED COLUMNS")
    for col, reason in DROPPED_COLUMNS.items():
        print(f"  {col:15s} {reason}")

    print("\n[2] ENGINEERED FEATURES (development set)")
    engineered = FeatureEngineer().fit_transform(X)
    summary = engineered_feature_summary(engineered, y)
    summary.to_csv(TABLES_DIR / "step2_engineered_features.csv")
    plot_engineered_features(summary, y.mean())
    print(summary.to_string())

    print("\n[3] MODEL-READY MATRIX")
    pipeline = build_feature_pipeline().fit(X, y)
    matrix = pipeline.transform(X)
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    pd.Series(names, name="feature").to_csv(TABLES_DIR / "step2_model_columns.csv", index=False)
    print(f"{X.shape[1]} raw columns -> {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} "
          f"categorical features -> {matrix.shape[1]} columns after one-hot encoding, "
          f"{int(np.isnan(matrix).sum())} missing values")

    print("\n[4] LEAKAGE-SAFE USE INSIDE CROSS-VALIDATION")
    check_fold_safety(X, y, groups)


if __name__ == "__main__":
    main()
