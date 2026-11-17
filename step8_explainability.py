import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from common import (header, title, FIGURES_DIR, GROUP_COL, TABLES_DIR, TARGET, load_dataset, make_cv,
                    split_dev_test)
from step2_feature_engineering import CATEGORICAL_FEATURES, ENGINEERED_NUMERIC, NUMERIC_FEATURES
from step4_model_comparison import interpret
from step6_tuning import BEST_PARAMS_PATH, tuned_pipeline

# 1. Final model: best validation AP in steps 4 and 6
FINAL_MODEL = "gradient_boosting"

# 2. Sensitive attributes we look at closely
SENSITIVE_FEATURES = ["race", "gender"]

# 3. Number of features and contributions shown
TOP_N = 15
TOP_COLUMNS = 10
TOP_LOCAL = 5

# 4. Mapping a model column back to its feature
def source_feature(column: str) -> str:
    if column in NUMERIC_FEATURES:
        return column
    matches = [feature for feature in CATEGORICAL_FEATURES if column.startswith(feature + "_")]
    return max(matches, key=len)

# 5. Fitting on four folds and explaining the fifth
def explain_validation_fold(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    train_idx, val_idx = next(make_cv().split(X, y, groups))
    params = json.loads(BEST_PARAMS_PATH.read_text())[FINAL_MODEL]
    pipeline = tuned_pipeline(FINAL_MODEL, params).fit(X.iloc[train_idx], y.iloc[train_idx])

    columns = pipeline.named_steps["preprocess"].get_feature_names_out()
    matrix = pd.DataFrame(pipeline[:-1].transform(X.iloc[val_idx]), columns=columns)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    explanation = explainer(matrix)
    risk = pipeline.predict_proba(X.iloc[val_idx])[:, 1]
    raw_score = pipeline.decision_function(X.iloc[val_idx])
    return explanation, matrix, risk, raw_score, y.iloc[val_idx].to_numpy()

# 6. Mean |SHAP| per feature (one-hot columns summed)
def global_importance(explanation, matrix: pd.DataFrame) -> pd.DataFrame:
    values = pd.DataFrame(explanation.values, columns=matrix.columns)
    per_feature = values.T.groupby(source_feature).sum().T
    importance = per_feature.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").to_frame()
    importance["share"] = importance["mean_abs_shap"] / importance["mean_abs_shap"].sum()
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance["engineered"] = [f in ENGINEERED_NUMERIC or f.endswith("_group") for f in importance.index]
    return importance

# 7. Does a high value raise or lower the risk?
def direction_table(explanation, matrix: pd.DataFrame) -> pd.DataFrame:
    values = pd.DataFrame(explanation.values, columns=matrix.columns)
    top = values.abs().mean().sort_values(ascending=False).head(TOP_COLUMNS).index
    rows = []
    for column in top:
        if matrix[column].nunique() > 2:
            high = matrix[column] > matrix[column].median()
        else:
            high = matrix[column] == 1
        when_high, when_low = values.loc[high, column].mean(), values.loc[~high, column].mean()
        importance = values[column].abs().mean()
        if abs(when_high - when_low) < 0.25 * importance:
            effect = "not monotone"
        else:
            effect = "raises risk" if when_high > when_low else "lowers risk"
        rows.append({
            "model_column": column,
            "mean_abs_shap": importance,
            "mean_shap_when_high_or_yes": when_high,
            "mean_shap_when_low_or_no": when_low,
            "effect_of_high_or_yes": effect,
        })
    return pd.DataFrame(rows).set_index("model_column")

# 8. Picking four contrasted patients
def pick_patients(risk: np.ndarray, y_true: np.ndarray) -> dict:
    order = np.argsort(risk)
    readmitted, not_readmitted = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    return {
        "high risk, readmitted": readmitted[np.argmax(risk[readmitted])],
        "high risk, not readmitted": not_readmitted[np.argmax(risk[not_readmitted])],
        "low risk, readmitted": readmitted[np.argmin(risk[readmitted])],
        "median risk": order[len(order) // 2],
    }

# 9. Largest contributions for one patient
def local_table(explanation, matrix: pd.DataFrame, row: int) -> pd.DataFrame:
    contributions = pd.Series(explanation.values[row], index=matrix.columns)
    top = contributions.abs().sort_values(ascending=False).head(TOP_LOCAL).index
    return pd.DataFrame({"value": matrix.loc[row, top].round(2), "shap_log_odds": contributions[top].round(4)})

# 10. Plotting global importance
def plot_global(importance: pd.DataFrame) -> None:
    top = importance.head(TOP_N).iloc[::-1]
    colors = ["#C44E52" if f in SENSITIVE_FEATURES else "#4C72B0" for f in top.index]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(top.index, top["mean_abs_shap"], color=colors)
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title("Global feature importance (validation fold)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step8_global_importance.png", dpi=150)
    plt.close(fig)

# 11. Plotting the SHAP values of the main columns
def plot_beeswarm(explanation) -> None:
    shap.plots.beeswarm(explanation, max_display=TOP_N, show=False)
    plt.title("SHAP values of the main model columns")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "step8_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

# 12. Plotting the explanation of one patient
def plot_local(explanation, label: str, row: int, risk: float) -> None:
    shap.plots.waterfall(explanation[row], max_display=10, show=False)
    plt.title(f"{label} (predicted risk {risk:.0%})")
    name = label.replace(", ", "_").replace(" ", "_")
    plt.savefig(FIGURES_DIR / f"step8_local_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()

# 13. Running step 8
def main() -> None:
    title("Step 8: Explainability")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    dev, _ = split_dev_test(load_dataset())
    X, y, groups = dev.drop(columns=TARGET), dev[TARGET], dev[GROUP_COL]
    explanation, matrix, risk, raw_score, y_true = explain_validation_fold(X, y, groups)

    # 1. Checking the SHAP values
    header("[1] Tree SHAP on the final model")
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    error = np.abs(reconstructed - raw_score).max()
    print(f"Model: {FINAL_MODEL} (tuned)")
    print(f"Patients explained (validation fold): {len(risk)}")
    print(f"Model columns: {matrix.shape[1]}")
    print(f"Base value (log-odds): {explanation.base_values[0]:.4f}")
    print(f"Largest gap between SHAP sum and model score: {error:.6f}")
    interpret(f"SHAP values add up to the model score (largest gap {error:.4f}).",
              "We explain patients the model did not see during training.")

    # 2. Ranking the features
    header("[2] Global importance")
    importance = global_importance(explanation, matrix)
    importance.round(4).to_csv(TABLES_DIR / "step8_global_importance.csv")
    print(importance.head(TOP_N).round(4).to_string())
    share = importance["share"]
    print()
    print(f"Features: {len(importance)}")
    print(f"Share of the top 5: {share.head(5).sum():.4f}")
    interpret(f"{importance.index[0]} and {importance.index[1]} dominate: {share.head(2).sum():.0%} of importance.",
              f"The top five features carry {share.head(5).sum():.0%} of the total.")

    # 3. Reading the direction of the effects
    header("[3] Direction of the main effects")
    directions = direction_table(explanation, matrix)
    directions.round(4).to_csv(TABLES_DIR / "step8_directions.csv")
    print(directions.round(4).rename_axis(None).to_string())
    interpret("More past stays, more diagnoses and older age raise the risk.",
              "Being discharged home lowers it: both make clinical sense.")

    # 4. Looking at sensitive and new features
    header("[4] Sensitive attributes and new features")
    shown = ["mean_abs_shap", "share", "rank"]
    engineered = importance[importance["engineered"]]
    print(importance.loc[SENSITIVE_FEATURES, shown].round(4).to_string())
    print()
    print(engineered[shown].round(4).to_string())
    sensitive = importance.loc[SENSITIVE_FEATURES]
    shares = " and ".join(f"{s:.1%}" for s in sensitive["share"])
    ranks = " and ".join(str(r) for r in sensitive["rank"])
    interpret(f"race and gender weigh {shares} of importance: ranks {ranks}.",
              f"New features in the top seven: {(engineered['rank'] <= 7).sum()}; has_prior_inpatient weighs {share['has_prior_inpatient']:.1%}.")

    # 5. Explaining individual predictions
    header("[5] Local explanations")
    local_rows = []
    for label, row in pick_patients(risk, y_true).items():
        table = local_table(explanation, matrix, row)
        if local_rows:
            print()
        print(f"{label}: predicted risk {risk[row]:.4f}, readmitted = {y_true[row]}")
        print(table.to_string())
        plot_local(explanation, label, row, risk[row])
        local_rows.append(table.assign(patient=label, predicted_risk=round(risk[row], 4)))
    pd.concat(local_rows).to_csv(TABLES_DIR / "step8_local_explanations.csv")
    interpret("The same few features explain each individual prediction.",
              "Missing payer code raises risk: a data artefact, not medicine.")

    plot_global(importance)
    plot_beeswarm(explanation)

    print()
    print("Figures saved in reports/figures")
    print("Tables saved in reports/tables")

if __name__ == "__main__":
    main()
