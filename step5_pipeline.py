import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from common import (header, title, GROUP_COL, ROOT, TABLES_DIR, TARGET, load_dataset, make_cv,
                    split_dev_test)
from step4_model_comparison import MODEL_NAMES, build_pipeline, interpret

# 1. Where the fitted pipelines are saved
MODELS_DIR = ROOT / "models"

# 2. Columns and patients shown as examples
SHOWN_COLUMNS = ["age", "time_in_hospital", "number_inpatient", "diag_1"]
N_EXAMPLES = 5

# 3. What each pipeline step learns during fit
LEARNED_DURING_FIT = {
    "features": "nothing (row-wise rules)",
    "preprocess": "medians, scaling, category levels",
    "model": "model parameters",
}

# 4. Describing the steps of a pipeline
def describe_pipeline(pipeline) -> pd.DataFrame:
    rows = [{"step": name, "object": type(step).__name__, "learned_during_fit": LEARNED_DURING_FIT[name]}
            for name, step in pipeline.steps]
    return pd.DataFrame(rows).set_index("step")

# 5. Simulating new patients with unseen categories and missing values
def corrupt_rows(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["medical_specialty"] = "SpecialtyNeverSeen"
    rows["diag_1"] = "999.99"
    rows["race"] = np.nan
    rows["payer_code"] = np.nan
    return rows

# 6. Running step 5
def main() -> None:
    title("Step 5: Pipeline")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    dev, _ = split_dev_test(load_dataset())
    X, y, groups = dev.drop(columns=TARGET), dev[TARGET], dev[GROUP_COL]

    # 1. Showing the pipeline structure
    header("[1] Pipeline structure")
    for name in MODEL_NAMES:
        if name != MODEL_NAMES[0]:
            print()
        print(name)
        print(describe_pipeline(build_pipeline(name)).to_string())
    n_steps = len(build_pipeline(MODEL_NAMES[0]).steps)
    interpret(f"Each model is a single object with {n_steps} chained steps.",
              f"Only the last step changes between our {len(MODEL_NAMES)} models.")

    # 2. Predicting from raw data
    header("[2] Raw data in, predicted risk out")
    fitted = {}
    for name in MODEL_NAMES:
        print(f"Fitting {name} ...")
        fitted[name] = build_pipeline(name).fit(X, y)
    examples = X.head(N_EXAMPLES)
    predictions = examples[SHOWN_COLUMNS].copy()
    for name in MODEL_NAMES:
        predictions[name] = fitted[name].predict_proba(examples)[:, 1]
    predictions.round(4).to_csv(TABLES_DIR / "step5_example_predictions.csv")
    print()
    print(f"Raw columns per patient: {X.shape[1]}")
    print(predictions.round(4).to_string())
    interpret("One call to fit, one call to predict_proba.",
              "The raw table goes in; a readmission risk comes out.")

    # 3. Checking the pipeline is refitted in each fold
    header("[3] Cross-validation check")
    fold_ap = cross_val_score(build_pipeline("logistic_regression"), X, y, groups=groups,
                              cv=make_cv(), scoring="average_precision")
    step4 = pd.read_csv(TABLES_DIR / "step4_fold_metrics.csv")
    step4_ap = step4.loc[step4["model"] == "logistic_regression", "val_ap"].to_numpy()
    folds = pd.DataFrame({"pipeline_in_cross_val_score": fold_ap, "step4_manual_loop": step4_ap})
    folds.index.name = "fold"
    folds.round(4).to_csv(TABLES_DIR / "step5_fold_check.csv")
    print(folds.round(4).to_string())
    print()
    print(f"Mean AP: {fold_ap.mean():.4f} vs {step4_ap.mean():.4f}")
    largest = np.abs(fold_ap - step4_ap).max()
    interpret(f"Both give the same average precision in every fold (largest difference {largest:.4f}).",
              "Preprocessing is learned on training folds only: no leakage.")

    # 4. Testing unseen categories and missing values
    header("[4] Robustness")
    corrupted = corrupt_rows(examples)
    robustness = pd.DataFrame({
        name: {"rows_scored": len(fitted[name].predict_proba(corrupted)),
               "mean_risk_clean": fitted[name].predict_proba(examples)[:, 1].mean(),
               "mean_risk_corrupted": fitted[name].predict_proba(corrupted)[:, 1].mean()}
        for name in MODEL_NAMES
    }).T
    print(robustness.round(4).to_string())
    shift = (robustness["mean_risk_corrupted"] - robustness["mean_risk_clean"]).abs().max()
    interpret("New categories and missing values do not break the pipeline.",
              f"Mean risk moves by at most {shift:.3f}: the endpoint will not crash.")

    # 5. Saving one file per model
    header("[5] Saved pipelines")
    rows = []
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"step5_{name}.joblib"
        joblib.dump(fitted[name], path, compress=3)
        reloaded = joblib.load(path)
        same = np.allclose(reloaded.predict_proba(examples), fitted[name].predict_proba(examples))
        rows.append({"model": name, "file": str(path.relative_to(ROOT)),
                     "size_mb": path.stat().st_size / 1e6, "same_predictions_after_reload": same})
    artifacts = pd.DataFrame(rows).set_index("model")
    artifacts.round(2).to_csv(TABLES_DIR / "step5_artifacts.csv")
    print(artifacts.round(2).rename_axis(None).to_string())
    interpret("Each saved file contains preprocessing and model together.",
              "Reloaded pipelines give exactly the same predictions.")

    print()
    print("Tables saved in reports/tables")
    print("Models saved in models")

if __name__ == "__main__":
    main()
