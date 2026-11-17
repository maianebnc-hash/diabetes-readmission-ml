import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from sklearn.model_selection import cross_validate

from common import (header, title, FIGURES_DIR, GROUP_COL, N_FOLDS, ROOT, SEED, TABLES_DIR, TARGET,
                    load_dataset, make_cv, split_dev_test)
from step4_model_comparison import MODEL_NAMES, build_pipeline, interpret

# 1. Trials per model (one trial = full pipeline + 5-fold grouped CV)
N_TRIALS = 20

# 2. Where the trials, best parameters and models are saved
MODELS_DIR = ROOT / "models"
RUN_LOG_PATH = TABLES_DIR / "step6_trials.csv"
BEST_PARAMS_PATH = TABLES_DIR / "step6_best_params.json"

# 3. Metrics computed in each fold
SCORING = {"ap": "average_precision", "roc_auc": "roc_auc"}

# 4. Search space of each model (class weights are tuned too)
def suggest_params(trial: optuna.Trial, name: str) -> dict:
    if name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 10, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            "max_features": trial.suggest_float("max_features", 0.05, 0.4),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if name == "gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 300, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    raise ValueError(f"Unknown model: {name}")

# 5. Putting hyperparameters into the step 5 pipeline
def tuned_pipeline(name: str, params: dict):
    return build_pipeline(name).set_params(**{f"model__{key}": value for key, value in params.items()})

# 6. One trial: the whole pipeline refitted in each fold
def make_objective(name: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    n_jobs = N_FOLDS if name == "logistic_regression" else 1

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, name)
        cv = cross_validate(tuned_pipeline(name, params), X, y, groups=groups, cv=make_cv(),
                            scoring=SCORING, return_train_score=True, n_jobs=n_jobs)
        trial.set_user_attr("val_ap_std", cv["test_ap"].std())
        trial.set_user_attr("val_ap_folds", [round(v, 4) for v in cv["test_ap"]])
        trial.set_user_attr("val_roc_auc", cv["test_roc_auc"].mean())
        trial.set_user_attr("train_ap", cv["train_ap"].mean())
        return cv["test_ap"].mean()

    return objective

# 7. Turning the trials of a study into log rows
def trials_table(name: str, study: optuna.Study) -> pd.DataFrame:
    return pd.DataFrame([{
        "model": name,
        "trial": t.number,
        "params": json.dumps(t.params),
        "val_ap": t.value,
        "val_ap_std": t.user_attrs["val_ap_std"],
        "val_ap_folds": json.dumps(t.user_attrs["val_ap_folds"]),
        "val_roc_auc": t.user_attrs["val_roc_auc"],
        "train_ap": t.user_attrs["train_ap"],
        "train_minus_val_ap": t.user_attrs["train_ap"] - t.value,
        "duration_s": t.duration.total_seconds(),
        "is_best": t.number == study.best_trial.number,
    } for t in study.trials])

# 8. Plotting the optimisation history
def plot_optimization_history(log: pd.DataFrame, untuned: pd.Series) -> None:
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(13, 3.8), sharey=True)
    for ax, name in zip(axes, MODEL_NAMES):
        trials = log[log["model"] == name]
        ax.scatter(trials["trial"], trials["val_ap"], s=18, color="#4C72B0", label="trial")
        ax.plot(trials["trial"], trials["val_ap"].cummax(), color="#C44E52", label="best so far")
        ax.axhline(untuned[name], color="grey", linestyle="--", label="untuned (step 4)")
        ax.set_title(name)
        ax.set_xlabel("Trial")
    axes[0].set_ylabel("Mean validation AP (5 folds)")
    axes[-1].legend(loc="lower right")
    fig.suptitle("Optuna optimisation history")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step6_optimization_history.png", dpi=150)
    plt.close(fig)

# 9. Running step 6
def main() -> None:
    title("Step 6: Hyperparameter tuning")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    dev, _ = split_dev_test(load_dataset())
    X, y, groups = dev.drop(columns=TARGET), dev[TARGET], dev[GROUP_COL]

    # 1. Tuning the three models
    header("[1] Optuna search")
    print(f"Trials per model: {N_TRIALS}")
    print(f"Sampler: TPE, {N_FOLDS}-fold grouped CV per trial")
    print()
    studies, logs = {}, []
    for name in MODEL_NAMES:
        start = time.time()
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(make_objective(name, X, y, groups), n_trials=N_TRIALS)
        studies[name] = study
        logs.append(trials_table(name, study))
        pd.concat(logs).round(5).to_csv(RUN_LOG_PATH, index=False)
        minutes = (time.time() - start) / 60
        print(f"{name:<20} best AP: {study.best_value:.4f}  trial: {study.best_trial.number:<3} minutes: {minutes:.1f}")
    log = pd.concat(logs).reset_index(drop=True)
    interpret(f"Each trial refits the full pipeline on {N_FOLDS} grouped folds.",
              f"{len(log)} trials are logged; the test set stays untouched.")

    # 2. Keeping the best hyperparameters
    header("[2] Best hyperparameters")
    best_params = {name: study.best_params for name, study in studies.items()}
    BEST_PARAMS_PATH.write_text(json.dumps(best_params, indent=2))
    for name, params in best_params.items():
        print(f"{name}: {params}")
    weighted = [name for name, params in best_params.items() if params["class_weight"]]
    interpret(f"Models keeping class weights: {len(weighted)} of {len(best_params)}.",
              "The best trees are more regularised than our step 4 defaults.")

    # 3. Measuring what tuning adds
    header("[3] Tuned vs untuned")
    step4 = pd.read_csv(TABLES_DIR / "step4_ranking_metrics.csv", index_col=0)
    best = log[log["is_best"]].set_index("model")
    gains = pd.DataFrame({
        "untuned_ap": step4["val_ap"],
        "tuned_ap": best["val_ap"],
        "gain": best["val_ap"] - step4["val_ap"],
        "fold_std": best["val_ap_std"],
        "tuned_roc_auc": best["val_roc_auc"],
    }).loc[MODEL_NAMES]
    gains.round(4).to_csv(TABLES_DIR / "step6_tuning_gains.csv")
    print(gains.round(4).to_string())
    same_order = list(gains["tuned_ap"].sort_values().index) == list(gains["untuned_ap"].sort_values().index)
    interpret(f"Tuning gains at most {gains['gain'].max():+.3f} AP; the fold deviation is {gains['fold_std'].mean():.3f}.",
              f"The ranking of the {len(gains)} models {'does not change' if same_order else 'changes'}.")

    # 4. Diagnosing overfitting after tuning
    header("[4] Train vs validation")
    overfit = pd.DataFrame({
        "untuned_gap": step4["train_minus_val_ap"],
        "tuned_gap": best["train_minus_val_ap"],
    }).loc[MODEL_NAMES]
    overfit.round(4).to_csv(TABLES_DIR / "step6_overfitting.csv")
    print(overfit.round(4).to_string())
    forest = overfit.loc["random_forest"]
    interpret(f"Tuning moves the random forest gap from {forest['untuned_gap']:.2f} to {forest['tuned_gap']:.2f}.",
              "Models overfit less, yet validation scores barely move.")

    # 5. Saving the best pipeline of each model
    header("[5] Saved models")
    for name in MODEL_NAMES:
        path = MODELS_DIR / f"step6_{name}_best.joblib"
        joblib.dump(tuned_pipeline(name, best_params[name]).fit(X, y), path, compress=3)
        print(f"{path.relative_to(ROOT)}  ({path.stat().st_size / 1e6:.1f} MB)")
    interpret("The best pipeline of each model is saved for later steps.",
              "The best CV score is slightly optimistic; the test set will decide.")

    plot_optimization_history(log, step4["val_ap"])

    print()
    print("Figures saved in reports/figures")
    print("Tables saved in reports/tables")
    print("Models saved in models")

if __name__ == "__main__":
    main()
