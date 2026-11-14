"""Step 3 - Class imbalance handling.

Only 11% of encounters are positive. Following Session 7, we go from the least
to the most invasive remedy and compare them with the same model (logistic
regression), the same folds and the same metrics:
  1. no remedy            -> then only the decision threshold is adapted
  2. class weights        -> errors on positives cost more during training
  3. random undersampling -> negatives are removed from the training fold
  4. SMOTE                -> synthetic positives are added to the training fold

Resampling happens inside each training fold only (imblearn Pipeline); validation
folds always keep the real 11% prevalence. The strategy is selected on average
precision, and the decision threshold on out-of-fold scores with a cost function.

Run:  python step3_class_imbalance.py
"""
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

from common import (FIGURES_DIR, GROUP_COL, SEED, TABLES_DIR, TARGET, load_dataset, make_cv,
                    split_dev_test)
from step2_feature_engineering import FeatureEngineer, build_preprocessor

# Business assumption: missing a readmission costs 10 times more than a needless follow-up.
COST_FALSE_NEGATIVE = 10
COST_FALSE_POSITIVE = 1
THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
# Operational feasibility: share of discharged patients the care team could follow up.
ALERT_VOLUMES = [0.10, 0.20]


def build_strategy(name: str) -> ImbPipeline:
    """Full pipeline (features -> preprocessing -> [sampler] -> logistic regression)."""
    class_weight = "balanced" if name == "class_weight" else None
    samplers = {
        "undersampling": RandomUnderSampler(random_state=SEED),
        "smote": SMOTE(random_state=SEED),
    }
    steps = [("features", FeatureEngineer()), ("preprocess", build_preprocessor())]
    if name in samplers:
        steps.append(("sampler", samplers[name]))  # applied when fitting, skipped when predicting
    steps.append(("model", LogisticRegression(max_iter=1000, class_weight=class_weight)))
    return ImbPipeline(steps)


STRATEGIES = ["none", "class_weight", "undersampling", "smote"]


def out_of_fold_scores(name: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> np.ndarray:
    """Predicted probability for every development row, from a model that never saw it."""
    return cross_val_predict(build_strategy(name), X, y, groups=groups, cv=make_cv(),
                             method="predict_proba", n_jobs=-1)[:, 1]


def total_cost(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> int:
    """Cost of flagging every patient whose score is at least `threshold`."""
    flagged = scores >= threshold
    false_negatives = int(((y_true == 1) & ~flagged).sum())
    false_positives = int(((y_true == 0) & flagged).sum())
    return COST_FALSE_NEGATIVE * false_negatives + COST_FALSE_POSITIVE * false_positives


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold of the grid with the lowest total cost (to be chosen on out-of-fold scores)."""
    costs = [total_cost(y_true, scores, t) for t in THRESHOLD_GRID]
    return float(THRESHOLD_GRID[int(np.argmin(costs))])


def ranking_metrics(y: pd.Series, scores: np.ndarray, groups: pd.Series) -> dict:
    """Threshold-free metrics. AP and ROC-AUC are averaged over the validation folds."""
    y_arr = y.to_numpy()
    folds = [val for _, val in make_cv().split(scores, y, groups)]
    ap = [average_precision_score(y_arr[val], scores[val]) for val in folds]
    auc = [roc_auc_score(y_arr[val], scores[val]) for val in folds]
    return {
        "average_precision": np.mean(ap),
        "ap_std": np.std(ap),
        "roc_auc": np.mean(auc),
        "brier": brier_score_loss(y_arr, scores),
        "mean_predicted_risk": scores.mean(),  # should be close to the 11.4% prevalence
    }


def decision_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """What happens when patients above `threshold` are flagged."""
    flagged = (scores >= threshold).astype(int)
    return {
        "threshold": threshold,
        "flagged_share": flagged.mean(),
        "precision": precision_score(y_true, flagged, zero_division=0),
        "recall": recall_score(y_true, flagged),
        "f1": f1_score(y_true, flagged),
        "cost": total_cost(y_true, scores, threshold),
    }


def alert_volume_metrics(y_true: np.ndarray, scores: np.ndarray, volume: float) -> dict:
    """Flag only the `volume` share of patients with the highest scores."""
    threshold = float(np.quantile(scores, 1 - volume))
    flagged = scores >= threshold
    return {
        "threshold": threshold,
        "precision": y_true[flagged].mean(),
        "recall": y_true[flagged].sum() / y_true.sum(),
        "lift": y_true[flagged].mean() / y_true.mean(),
    }


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


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    dev, _ = split_dev_test(load_dataset())  # the test set is not touched
    X, y, groups = dev.drop(columns=TARGET), dev[TARGET], dev[GROUP_COL]
    y_arr = y.to_numpy()
    print(f"Development set: {len(y)} encounters, {y.sum()} positives ({y.mean():.2%}), "
          f"ratio negatives/positives = {(1 - y.mean()) / y.mean():.1f}")

    all_scores = {name: out_of_fold_scores(name, X, y, groups) for name in STRATEGIES}

    # Our reading (development set, out-of-fold scores):
    # Accuracy is not our metric: the accuracy paradox applies here.
    # We select on average precision, as recommended for rare events.
    # The no-skill baseline equals the prevalence, 0.114.
    # All four strategies reach about 0.21.
    # The gaps are smaller than the fold-to-fold standard deviation.
    # So no remedy improves discrimination.
    # SMOTE is the weakest: our features are mostly categorical.
    # Class weights and resampling harm calibration.
    # Their mean predicted risk is 46%, against a prevalence of 11%.
    # Without remedy, the mean predicted risk matches the prevalence.
    print("\n[1] RANKING QUALITY AND PROBABILITIES (threshold-free)")
    ranking = pd.DataFrame({n: ranking_metrics(y, s, groups) for n, s in all_scores.items()}).T
    ranking.round(4).to_csv(TABLES_DIR / "step3_ranking_metrics.csv")
    print(ranking.round(4).to_string())

    # Our reading:
    # At threshold 0.5, the unweighted model flags 0.3% of patients.
    # Its recall is 1%: it misses almost every readmission.
    # The model is not useless, the default threshold is.
    # Training estimates probabilities; thresholding turns them into decisions.
    print("\n[2] DECISIONS AT THE DEFAULT THRESHOLD 0.5")
    default = pd.DataFrame({n: decision_metrics(y_arr, s, 0.5) for n, s in all_scores.items()}).T
    default.round(4).to_csv(TABLES_DIR / "step3_decisions_default_threshold.csv")
    print(default.round(4).to_string())

    # Our reading:
    # We tune the threshold on out-of-fold scores, never on the test set.
    # Our cost function is C(t) = 10 x FN(t) + 1 x FP(t).
    # The ratio of 10 is our business assumption.
    # After tuning, the four strategies have almost the same cost.
    # Threshold tuning alone does the job of the heavier remedies.
    # The optimal threshold, 0.09, matches the theory: 1 / (1 + 10).
    # But it flags 53% of patients.
    # That alert volume is not operationally feasible for a hospital.
    print(f"\n[3] DECISIONS AT THE COST-OPTIMAL THRESHOLD "
          f"(missed readmission = {COST_FALSE_NEGATIVE}, false alarm = {COST_FALSE_POSITIVE})")
    thresholds = {n: choose_threshold(y_arr, s) for n, s in all_scores.items()}
    tuned = pd.DataFrame({n: decision_metrics(y_arr, s, thresholds[n]) for n, s in all_scores.items()}).T
    tuned.round(4).to_csv(TABLES_DIR / "step3_decisions_cost_threshold.csv")
    print(tuned.round(4).to_string())
    flag_all = COST_FALSE_POSITIVE * int((y_arr == 0).sum())
    flag_none = COST_FALSE_NEGATIVE * int(y_arr.sum())
    print(f"Reference costs: flag everyone = {flag_all}, flag no one = {flag_none}")

    # Our reading:
    # So we also fix the alert volume, as a capacity constraint.
    # With the top 10% flagged, we reach 23% of readmissions.
    # Precision is 26%, a lift of 2.3 over the prevalence.
    # With the top 20% flagged, recall rises to 38%.
    # Again, the four strategies are equivalent.
    #
    # Our chosen remedy: the least invasive one, threshold tuning.
    # We keep the original training distribution and calibrated probabilities.
    # Class weights stay a hyperparameter for the tuning step.
    # The threshold is frozen before we touch the test set.
    print("\n[4] DECISIONS UNDER A LIMITED ALERT VOLUME (only the highest-risk patients are flagged)")
    volume_rows = {(f"top {volume:.0%}", name): alert_volume_metrics(y_arr, scores, volume)
                   for volume in ALERT_VOLUMES for name, scores in all_scores.items()}
    by_volume = pd.DataFrame(volume_rows).T
    by_volume.round(4).to_csv(TABLES_DIR / "step3_decisions_alert_volume.csv")
    print(by_volume.round(4).to_string())

    plot_pr_curves(y_arr, all_scores)
    plot_cost_curves(y_arr, all_scores, thresholds)


if __name__ == "__main__":
    main()
