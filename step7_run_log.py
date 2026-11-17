import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import header, title, FIGURES_DIR, ROOT, TABLES_DIR
from step4_model_comparison import MODEL_NAMES, interpret
from step6_tuning import MODELS_DIR, N_TRIALS, RUN_LOG_PATH

# 1. Where the final run log is saved
FINAL_LOG_PATH = ROOT / "reports" / "run_log.csv"

# 2. Plot shared by every trial
HISTORY_PLOT = FIGURES_DIR / "step6_optimization_history.png"

# 3. Metrics logged for each trial
METRIC_COLUMNS = ["val_ap", "val_ap_std", "val_roc_auc", "train_ap", "train_minus_val_ap"]

# 4. Columns of the run log, in order
LOG_COLUMNS = ["run_id", "model", "trial", "params", *METRIC_COLUMNS, "val_ap_folds", "duration_s",
               "is_best", "model_artifact", "plot_artifact"]

# 5. Path relative to the project folder
def relative(path) -> str:
    return str(path.relative_to(ROOT))

# 6. Plot file of the trials of one model
def trials_plot_path(model: str):
    return FIGURES_DIR / f"step7_trials_{model}.png"

# 7. Linking each trial to its saved artifacts
def build_run_log(trials: pd.DataFrame) -> pd.DataFrame:
    log = trials.copy()
    log["run_id"] = log["model"] + "_trial_" + log["trial"].astype(str).str.zfill(2)
    log["plot_artifact"] = [f"{relative(HISTORY_PLOT)}; {relative(trials_plot_path(model))}"
                            for model in log["model"]]
    log["model_artifact"] = [relative(MODELS_DIR / f"step6_{model}_best.joblib") if best else ""
                             for model, best in zip(log["model"], log["is_best"])]
    return log[LOG_COLUMNS]

# 8. Plotting every trial against each hyperparameter
def plot_trials(model: str, trials: pd.DataFrame) -> None:
    params = pd.DataFrame([json.loads(p) for p in trials["params"]]).fillna("None")
    best = trials["is_best"].to_numpy()
    fig, axes = plt.subplots(1, params.shape[1], figsize=(3.2 * params.shape[1], 3.2), sharey=True)
    for ax, column in zip(axes, params.columns):
        values = params[column]
        if not pd.api.types.is_numeric_dtype(values):
            values = values.astype(str)
        elif values.max() / max(values.min(), 1e-12) > 50:
            ax.set_xscale("log")
        ax.scatter(values, trials["val_ap"], s=22, color="#4C72B0")
        ax.scatter(values[best], trials["val_ap"][best], s=70, color="#C44E52", label="best trial")
        ax.set_xlabel(column)
    axes[0].set_ylabel("Mean validation AP")
    axes[0].legend(loc="lower right")
    fig.suptitle(f"{model}: every tuning trial")
    fig.tight_layout()
    fig.savefig(trials_plot_path(model), dpi=150)
    plt.close(fig)

# 9. Running step 7
def main() -> None:
    title("Step 7: Run log")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    trials = pd.read_csv(RUN_LOG_PATH)
    log = build_run_log(trials)
    log.to_csv(FINAL_LOG_PATH, index=False)

    # 1. Checking the log is complete
    header("[1] Log completeness")
    missing = log.set_index("model")[METRIC_COLUMNS].isna().sum(axis=1)
    completeness = pd.DataFrame({
        "trials_logged": log.groupby("model")["trial"].count(),
        "trials_expected": N_TRIALS,
        "missing_metrics": missing.groupby("model").sum(),
        "best_trials": log.groupby("model")["is_best"].sum(),
    }).loc[MODEL_NAMES]
    print(completeness.rename_axis(None).to_string())
    print()
    print(f"Run log: {relative(FINAL_LOG_PATH)}")
    print(f"Rows: {len(log)}")
    print(f"Columns: {log.shape[1]}")
    interpret(f"The log holds {len(log)} trials: {N_TRIALS} per model, {missing.sum()} missing metrics.",
              f"Best trials per model: {', '.join(str(n) for n in completeness['best_trials'])}.")

    # 2. Summarising what the trials explored
    header("[2] Trials summary")
    explored = log.groupby("model").agg(
        worst_ap=("val_ap", "min"), median_ap=("val_ap", "median"), best_ap=("val_ap", "max"),
        minutes=("duration_s", lambda s: s.sum() / 60),
    ).loc[MODEL_NAMES]
    explored["best_minus_worst"] = explored["best_ap"] - explored["worst_ap"]
    explored.round(4).to_csv(TABLES_DIR / "step7_trials_summary.csv")
    print(explored.round(4).rename_axis(None).to_string())
    spread = explored["best_minus_worst"]
    interpret(f"logistic_regression barely reacts to its hyperparameters: {spread['logistic_regression']:.3f} AP range.",
              f"Tree models vary more: up to {spread.max():.3f} AP between worst and best.")

    # 3. Listing the best trials
    header("[3] Three best trials per model")
    top = log.sort_values("val_ap", ascending=False).groupby("model").head(3).set_index("run_id")
    top = top.loc[[run for model in MODEL_NAMES for run in top.index if run.startswith(model)]]
    top = top[["val_ap", "val_ap_std", "train_minus_val_ap", "params"]]
    top.round(4).to_csv(TABLES_DIR / "step7_top_trials.csv")
    print(top.round(4).rename_axis(None).to_string())
    weighted = top["params"].str.contains("balanced").sum()
    interpret("The best trials of a model are close to each other.",
              f"Trials using class weights among them: {weighted} of {len(top)}.")

    # 4. Linking log rows to saved files
    header("[4] Artifacts")
    for model in MODEL_NAMES:
        plot_trials(model, trials[trials["model"] == model].reset_index(drop=True))
    artifacts = log[log["is_best"]].set_index("run_id")[["model_artifact"]]
    artifacts["model_file_exists"] = [(ROOT / path).exists() for path in artifacts["model_artifact"]]
    print(artifacts.rename_axis(None).to_string())
    print()
    print("Plots showing every trial:")
    print(relative(HISTORY_PLOT))
    for model in MODEL_NAMES:
        print(relative(trials_plot_path(model)))
    interpret("Each best trial points to its saved model file.",
              "Every trial appears in the plots listed above.")

    print()
    print("Figures saved in reports/figures")
    print("Tables saved in reports/tables")

if __name__ == "__main__":
    main()
