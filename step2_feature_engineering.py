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

from common import (header, title, DIAGNOSIS_COLUMNS, FIGURES_DIR, GROUP_COL, ID_COL, TABLES_DIR, TARGET,
                    load_dataset, make_cv, split_dev_test)

# 1. Columns we drop and why
DROPPED_COLUMNS = {
    ID_COL: "identifier, no clinical meaning",
    GROUP_COL: "identifier, used only to group the split",
    "weight": "97% missing",
    "examide": "single value in the whole dataset",
    "citoglipton": "single value in the whole dataset",
}

# 2. Visit and drug columns
VISIT_COLUMNS = ["number_outpatient", "number_emergency", "number_inpatient"]
DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide.metformin", "glipizide.metformin", "glimepiride.pioglitazone",
    "metformin.rosiglitazone", "metformin.pioglitazone",
]

# 3. Codes that are labels, not numbers
CODE_COLUMNS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

# 4. New features
ENGINEERED_NUMERIC = ["age_mid", "prior_visits_log", "has_prior_inpatient",
                      "n_diabetes_meds", "n_med_changes", "meds_per_day"]
ENGINEERED_CATEGORICAL = [f"{col}_group" for col in DIAGNOSIS_COLUMNS]

# 5. Numeric and categorical model inputs
NUMERIC_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    *VISIT_COLUMNS, "number_diagnoses", *ENGINEERED_NUMERIC,
]
CATEGORICAL_FEATURES = [
    "race", "gender", "payer_code", "medical_specialty", "max_glu_serum", "A1Cresult",
    "change", "diabetesMed", *CODE_COLUMNS, *ENGINEERED_CATEGORICAL, *DRUG_COLUMNS,
]
RARE_LEVEL_MIN_FREQUENCY = 0.01

# 6. ICD-9 chapters (Strack et al., 2014)
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

# 7. Mapping an ICD-9 code to a clinical family
def icd9_group(code) -> str:
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):
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

# 8. Turning an age bracket into its midpoint
def age_midpoint(bracket: str) -> float:
    low, high = bracket.strip("[)").split("-")
    return (int(low) + int(high)) / 2

# 9. Creating features row by row (no leakage)
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1. Age as a number
        X["age_mid"] = X["age"].map(age_midpoint)

        # 2. Diagnosis codes into clinical families
        for col in DIAGNOSIS_COLUMNS:
            X[f"{col}_group"] = X[col].map(icd9_group)

        # 3. Prior visits
        X["prior_visits_log"] = np.log1p(X[VISIT_COLUMNS].sum(axis=1))
        X["has_prior_inpatient"] = (X["number_inpatient"] > 0).astype(int)

        # 4. Number of drugs and dose changes
        drugs = X[DRUG_COLUMNS]
        X["n_diabetes_meds"] = (drugs != "No").sum(axis=1)
        X["n_med_changes"] = drugs.isin(["Up", "Down"]).sum(axis=1)

        # 5. Medications per day
        X["meds_per_day"] = X["num_medications"] / X["time_in_hospital"]

        # 6. Codes as labels
        X[CODE_COLUMNS] = X[CODE_COLUMNS].astype(str)

        replaced = ["age", *DIAGNOSIS_COLUMNS]
        return X.drop(columns=[*DROPPED_COLUMNS, *replaced], errors="ignore")

# 10. Imputing, scaling and encoding (fit on training data only)
def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
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

# 11. Chaining features and preprocessing
def build_feature_pipeline(scale_numeric: bool = True) -> Pipeline:
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocess", build_preprocessor(scale_numeric)),
    ])

# 12. Readmission rate per new feature
def engineered_feature_summary(engineered: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    summaries = {}
    for col in [*ENGINEERED_NUMERIC, "diag_1_group"]:
        values = engineered[col]
        if values.nunique() > 12:
            values = pd.qcut(values, 4, duplicates="drop").astype(str)
        summary = y.groupby(values.to_numpy()).agg(n="count", positive_rate="mean")
        summaries[col] = summary.rename(index=str)
    return pd.concat(summaries, names=["feature", "level"]).round(4)

# 13. Plotting the new features
def plot_engineered_features(summary: pd.DataFrame, base_rate: float) -> None:
    features = summary.index.get_level_values("feature").unique()
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    for ax, feature in zip(axes.ravel(), features):
        table = summary.loc[feature]
        table = table[table["n"] >= 100]
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

# 14. Checking for leakage across folds
def check_fold_safety(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> None:
    train_idx, val_idx = next(make_cv().split(X, y, groups))
    pipeline = build_feature_pipeline().fit(X.iloc[train_idx], y.iloc[train_idx])
    train_matrix = pipeline.transform(X.iloc[train_idx])
    val_matrix = pipeline.transform(X.iloc[val_idx])
    n_numeric = len(NUMERIC_FEATURES)
    missing = int(np.isnan(val_matrix).sum())
    print(f"Fitted on the training fold only ({len(train_idx)} rows)")
    print(f"Applied to the validation fold ({len(val_idx)} rows): {val_matrix.shape[1]} columns, {missing} missing values")
    train_mean = train_matrix[:, :n_numeric].mean()
    val_mean = val_matrix[:, :n_numeric].mean()
    print(f"Mean of scaled columns on the training fold: {train_mean:.3f} (exactly 0 by construction)")
    print(f"Mean of scaled columns on the validation fold: {val_mean:.3f} (close to 0, as its statistics were never used)")

# 15. Running step 2
def main() -> None:
    title("Step 2: Feature engineering")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    dev, _ = split_dev_test(load_dataset())
    X, y, groups = dev.drop(columns=TARGET), dev[TARGET], dev[GROUP_COL]

    # 1. Dropping columns
    header("[1] Dropped columns")
    for col, reason in DROPPED_COLUMNS.items():
        print(f"  {col:15s} {reason}")

    # 2. Checking the new features
    header("[2] New features")
    engineered = FeatureEngineer().fit_transform(X)
    summary = engineered_feature_summary(engineered, y)
    summary.to_csv(TABLES_DIR / "step2_engineered_features.csv")
    plot_engineered_features(summary, y.mean())
    print(summary.to_string())

    # 3. Building the model-ready matrix
    header("[3] Final feature matrix")
    pipeline = build_feature_pipeline().fit(X, y)
    matrix = pipeline.transform(X)
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    pd.Series(names, name="feature").to_csv(TABLES_DIR / "step2_model_columns.csv", index=False)
    missing = int(np.isnan(matrix).sum())
    print(f"Raw columns: {X.shape[1]}")
    print(f"Features: {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical")
    print(f"Columns after one-hot encoding: {matrix.shape[1]}, missing values: {missing}")

    # 4. Checking for leakage
    header("[4] Leakage check on one fold")
    check_fold_safety(X, y, groups)

if __name__ == "__main__":
    main()
