import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import cross_val_predict

from common import (header, title, FIGURES_DIR, GROUP_COL, SEED, TABLES_DIR, TARGET, load_dataset, make_cv,
                    split_dev_test)
from step2_feature_engineering import FeatureEngineer, build_preprocessor

# 1. A missed readmission costs 10x a false alarm
COST_FALSE_NEGATIVE = 10
COST_FALSE_POSITIVE = 1
THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)

# 2. Share of patients the care team can follow up
ALERT_VOLUMES = [0.10, 0.20]

# 3. Building the pipeline for one imbalance strategy
def build_strategy(name: str) -> ImbPipeline:
    class_weight = "balanced" if name == "class_weight" else None
    samplers = {
        "undersampling": RandomUnderSampler(random_state=SEED),
        "smote": SMOTE(random_state=SEED),
    }
    steps = [("features", FeatureEngineer()), ("preprocess", build_preprocessor())]
    if name in samplers:
        steps.append(("sampler", samplers[name]))
    steps.append(("model", LogisticRegression(max_iter=1000, class_weight=class_weight)))
    return ImbPipeline(steps)

# 4. The four imbalance strategies
STRATEGIES = ["none", "class_weight", "undersampling", "smote"]

# 5. Out-of-fold predicted probabilities
def out_of_fold_scores(name: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> np.ndarray:
    return cross_val_predict(build_strategy(name), X, y, groups=groups, cv=make_cv(),
                             method="predict_proba", n_jobs=-1)[:, 1]

# 6. Total cost at a threshold
def total_cost(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> int:
    flagged = scores >= threshold
    false_negatives = int(((y_true == 1) & ~flagged).sum())
    false_positives = int(((y_true == 0) & flagged).sum())
    return COST_FALSE_NEGATIVE * false_negatives + COST_FALSE_POSITIVE * false_positives

# 7. Picking the cheapest threshold
def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    costs = [total_cost(y_true, scores, t) for t in THRESHOLD_GRID]
    return float(THRESHOLD_GRID[int(np.argmin(costs))])

# 8. Threshold-free metrics
def ranking_metrics(y: pd.Series, scores: np.ndarray, groups: pd.Series) -> dict:
    y_arr = y.to_numpy()
    folds = [val for _, val in make_cv().split(scores, y, groups)]
    ap = [average_precision_score(y_arr[val], scores[val]) for val in folds]
    auc = [roc_auc_score(y_arr[val], scores[val]) for val in folds]
    return {
        "average_precision": np.mean(ap),
        "ap_std": np.std(ap),
        "roc_auc": np.mean(auc),
        "brier": brier_score_loss(y_arr, scores),
        "mean_predicted_risk": scores.mean(),
    }

# 9. Metrics at a threshold
def decision_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    flagged = (scores >= threshold).astype(int)
    return {
        "threshold": threshold,
        "flagged_share": flagged.mean(),
        "precision": precision_score(y_true, flagged, zero_division=0),
        "recall": recall_score(y_true, flagged),
        "f1": f1_score(y_true, flagged),
        "cost": total_cost(y_true, scores, threshold),
    }

# 10. Metrics when flagging the top share
def alert_volume_metrics(y_true: np.ndarray, scores: np.ndarray, volume: float) -> dict:
    threshold = float(np.quantile(scores, 1 - volume))
    flagged = scores >= threshold
    return {
        "threshold": threshold,
        "precision": y_true[flagged].mean(),
        "recall": y_true[flagged].sum() / y_true.sum(),
        "lift": y_true[flagged].mean() / y_true.mean(),
    }

# 11. Plotting precision-recall curves
def plot_pr_curves(y_true: np.ndarray, all_scores: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, scores in all_scores.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.plot(recall, precision, label=f"{name} (AP={average_precision_score(y_true, scores):.3f})")
    ax.axhline(y_true.mean(), color="grey", linestyle="--", label=f"no skill ({y_true.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 0.6)
    ax.set_title("Precision-recall curves (out-of-fold, logistic regression)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step3_pr_curves.png", dpi=150)
    plt.close(fig)

# 12. Plotting cost per threshold
def plot_cost_curves(y_true: np.ndarray, all_scores: dict, thresholds: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, scores in all_scores.items():
        costs = [total_cost(y_true, scores, t) for t in THRESHOLD_GRID]
        line, = ax.plot(THRESHOLD_GRID, costs, label=f"{name} (best t={thresholds[name]:.2f})")
        ax.scatter([thresholds[name]], [min(costs)], color=line.get_color(), zorder=3)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(f"Cost = {COST_FALSE_NEGATIVE} x missed readmissions + {COST_FALSE_POSITIVE} x false alarms")
    ax.set_title("Cost as a function of the threshold (out-of-fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step3_cost_curves.png", dpi=150)
    plt.close(fig)

# 13. Running step 3
def main() -> None:
    title("Step 3: Class imbalance")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    dev, _ = split_dev_test(load_dataset())
    X, y, groups = dev.drop(columns=TARGET), dev[TARGET], dev[GROUP_COL]
    y_arr = y.to_numpy()
    print(f"Development set: {len(y)} encounters, {y.sum()} positives ({y.mean():.2%})")
    print(f"Ratio negatives/positives: {(1 - y.mean()) / y.mean():.1f}")

    all_scores = {name: out_of_fold_scores(name, X, y, groups) for name in STRATEGIES}

    # 1. Comparing ranking quality
    header("[1] Ranking metrics")
    ranking = pd.DataFrame({n: ranking_metrics(y, s, groups) for n, s in all_scores.items()}).T
    ranking.round(4).to_csv(TABLES_DIR / "step3_ranking_metrics.csv")
    print(ranking.round(4).to_string())

    # 2. Testing the default threshold
    header("[2] Threshold 0.5")
    default = pd.DataFrame({n: decision_metrics(y_arr, s, 0.5) for n, s in all_scores.items()}).T
    default.round(4).to_csv(TABLES_DIR / "step3_decisions_default_threshold.csv")
    print(default.round(4).to_string())

    # 3. Tuning the threshold
    header("[3] Best threshold by cost")
    print(f"Costs used: {COST_FALSE_NEGATIVE} per missed readmission, {COST_FALSE_POSITIVE} per false alarm")
    print()
    thresholds = {n: choose_threshold(y_arr, s) for n, s in all_scores.items()}
    tuned = pd.DataFrame({n: decision_metrics(y_arr, s, thresholds[n]) for n, s in all_scores.items()}).T
    tuned.round(4).to_csv(TABLES_DIR / "step3_decisions_cost_threshold.csv")
    print(tuned.round(4).to_string())
    flag_all = COST_FALSE_POSITIVE * int((y_arr == 0).sum())
    flag_none = COST_FALSE_NEGATIVE * int(y_arr.sum())
    print()
    print(f"Cost if we flag everyone: {flag_all}")
    print(f"Cost if we flag no one: {flag_none}")

    # 4. Limiting the alert volume
    header("[4] Flagging the top 10% and 20%")
    volume_rows = {(f"top {volume:.0%}", name): alert_volume_metrics(y_arr, scores, volume)
                   for volume in ALERT_VOLUMES for name, scores in all_scores.items()}
    by_volume = pd.DataFrame(volume_rows).T
    by_volume.round(4).to_csv(TABLES_DIR / "step3_decisions_alert_volume.csv")
    print(by_volume.round(4).to_string())

    plot_pr_curves(y_arr, all_scores)
    plot_cost_curves(y_arr, all_scores, thresholds)

if __name__ == "__main__":
    main()
