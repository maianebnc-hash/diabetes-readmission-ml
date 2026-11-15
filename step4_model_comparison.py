"""Step 4 - Modelling progression: three models, same folds, same metrics.

The progression follows the course, from simple to flexible:
  1. logistic regression  - linear decision boundary, L2 regularisation (Sessions 2-3)
  2. random forest        - bagging of deep trees, reduces variance (Sessions 4-5)
  3. gradient boosting    - trees fitted sequentially on the errors (Session 5)

Gradient boosting uses scikit-learn's histogram-based implementation, the same
algorithm family as LightGBM, which needs no extra system library.

Hyperparameters are sensible defaults here; they are tuned in the next steps.
No class weights or resampling: Step 3 showed threshold tuning is enough.

Run:  python step4_model_comparison.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline

from common import (FIGURES_DIR, GROUP_COL, SEED, TABLES_DIR, TARGET, load_dataset, make_cv,
                    split_dev_test)
from step2_feature_engineering import FeatureEngineer, build_preprocessor
from step3_class_imbalance import (ALERT_VOLUMES, alert_volume_metrics, choose_threshold,
                                   decision_metrics)

MODEL_NAMES = ["logistic_regression", "random_forest", "gradient_boosting"]


def build_model(name: str):
    """Untuned estimator for each stage of the progression."""
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                      n_jobs=-1, random_state=SEED)
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                              early_stopping=False, random_state=SEED)
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(name: str) -> Pipeline:
    """Features -> preprocessing -> model. Only the linear model needs scaled inputs."""
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocess", build_preprocessor(scale_numeric=(name == "logistic_regression"))),
        ("model", build_model(name)),
    ])


def cross_validate_model(name: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """Fit a fresh pipeline in each fold. Return out-of-fold scores and per-fold metrics."""
    oof = np.zeros(len(y))
    rows = []
    for fold, (train_idx, val_idx) in enumerate(make_cv().split(X, y, groups)):
        pipeline = build_pipeline(name).fit(X.iloc[train_idx], y.iloc[train_idx])
        train_scores = pipeline.predict_proba(X.iloc[train_idx])[:, 1]
        oof[val_idx] = pipeline.predict_proba(X.iloc[val_idx])[:, 1]
        rows.append({
            "fold": fold,
            "train_ap": average_precision_score(y.iloc[train_idx], train_scores),
            "val_ap": average_precision_score(y.iloc[val_idx], oof[val_idx]),
            "val_roc_auc": roc_auc_score(y.iloc[val_idx], oof[val_idx]),
        })
    return oof, pd.DataFrame(rows)


def reading(*sentences: str) -> None:
    """Print our interpretation of the result just shown (two short sentences at most)."""
    print("Reading: " + " ".join(sentences))


def label(name: str) -> str:
    return name.replace("_", " ")


def plot_pr_curves(y_true: np.ndarray, all_scores: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, scores in all_scores.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.plot(recall, precision, label=f"{name} (AP={average_precision_score(y_true, scores):.3f})")
    ax.axhline(y_true.mean(), color="grey", linestyle="--", label=f"no skill ({y_true.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 0.7)
    ax.set_title("Precision-recall curves (out-of-fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step4_pr_curves.png", dpi=150)
    plt.close(fig)


def plot_train_vs_validation(ranking: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(ranking))
    ax.bar(x - 0.2, ranking["train_ap"], width=0.4, label="training folds", color="#C44E52")
    ax.bar(x + 0.2, ranking["val_ap"], width=0.4, yerr=ranking["val_ap_std"], capsize=4,
           label="validation folds", color="#4C72B0")
    ax.set_xticks(x, ranking.index, rotation=15)
    ax.set_ylabel("Average precision")
    ax.set_title("Training vs validation: overfitting diagnosis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step4_train_vs_validation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    dev, _ = split_dev_test(load_dataset())  # the test set is not touched
    X, y, groups = dev.drop(columns=TARGET), dev[TARGET], dev[GROUP_COL]
    y_arr = y.to_numpy()
    prevalence = y.mean()

    all_scores, fold_tables = {}, {}
    for name in MODEL_NAMES:
        print(f"Cross-validating {name} ...")
        all_scores[name], fold_tables[name] = cross_validate_model(name, X, y, groups)
    pd.concat(fold_tables, names=["model"]).round(4).to_csv(TABLES_DIR / "step4_fold_metrics.csv")

    print(f"\n[1] RANKING QUALITY (5 validation folds; no-skill AP = {prevalence:.3f}, ROC-AUC = 0.5)")
    ranking = pd.DataFrame({
        name: {
            "val_ap": folds["val_ap"].mean(),
            "val_ap_std": folds["val_ap"].std(ddof=0),
            "val_roc_auc": folds["val_roc_auc"].mean(),
            "train_ap": folds["train_ap"].mean(),
            "train_minus_val_ap": folds["train_ap"].mean() - folds["val_ap"].mean(),
        } for name, folds in fold_tables.items()
    }).T
    ranking.round(4).to_csv(TABLES_DIR / "step4_ranking_metrics.csv")
    print(ranking[["val_ap", "val_ap_std", "val_roc_auc"]].round(4).to_string())
    best = ranking["val_ap"].idxmax()
    reading(f"{label(best).capitalize()} ranks best: AP {ranking.loc[best, 'val_ap']:.3f}, "
            f"no-skill {prevalence:.3f}.",
            "Gains between models stay close to the fold-to-fold standard deviation.")

    print("\n[2] OVERFITTING DIAGNOSIS (average precision on training vs validation folds)")
    print(ranking[["train_ap", "val_ap", "train_minus_val_ap"]].round(4).to_string())
    most_overfit = ranking["train_minus_val_ap"].idxmax()
    least_overfit = ranking["train_minus_val_ap"].idxmin()
    reading(f"{label(most_overfit).capitalize()} memorises its training folds: "
            f"train AP {ranking.loc[most_overfit, 'train_ap']:.2f}, "
            f"validation {ranking.loc[most_overfit, 'val_ap']:.2f}.",
            f"{label(least_overfit).capitalize()} shows almost no gap: it is limited by bias.")

    print("\n[3] PROBABILITY QUALITY (out-of-fold)")
    probability = pd.DataFrame({
        name: {"brier": brier_score_loss(y_arr, s), "mean_predicted_risk": s.mean(),
               "prevalence": prevalence} for name, s in all_scores.items()
    }).T
    probability.round(4).to_csv(TABLES_DIR / "step4_probability_metrics.csv")
    print(probability.round(4).to_string())
    reading("All three predict a mean risk close to the prevalence.",
            "Brier scores are nearly equal; calibration is examined in a later step.")

    print("\n[4] DECISIONS UNDER A LIMITED ALERT VOLUME")
    by_volume = pd.DataFrame({
        (f"top {volume:.0%}", name): alert_volume_metrics(y_arr, s, volume)
        for volume in ALERT_VOLUMES for name, s in all_scores.items()
    }).T
    by_volume.round(4).to_csv(TABLES_DIR / "step4_decisions_alert_volume.csv")
    print(by_volume.round(4).to_string())
    top10 = by_volume.loc["top 10%", "recall"]
    reading(f"With the top 10% flagged, {label(top10.idxmax())} reaches "
            f"{top10.max():.0%} of readmissions.",
            f"The weakest model reaches {top10.min():.0%}: a real but small gain.")

    print("\n[5] DECISIONS AT THE COST-OPTIMAL THRESHOLD (chosen on out-of-fold scores)")
    by_cost = pd.DataFrame({
        name: decision_metrics(y_arr, s, choose_threshold(y_arr, s)) for name, s in all_scores.items()
    }).T
    by_cost.round(4).to_csv(TABLES_DIR / "step4_decisions_cost_threshold.csv")
    print(by_cost.round(4).to_string())
    cheapest = by_cost["cost"].idxmin()
    low, high = by_cost["threshold"].min(), by_cost["threshold"].max()
    thresholds = f"threshold {low:.2f}" if low == high else f"thresholds from {low:.2f} to {high:.2f}"
    reading(f"The models pick {thresholds}; theory says 0.09.",
            f"{label(cheapest).capitalize()} has the lowest cost but flags "
            f"{by_cost.loc[cheapest, 'flagged_share']:.0%} of patients.")

    print("\n[6] WHAT EACH STEP BOUGHT (gain over the previous model)")
    progression = pd.DataFrame({
        "val_ap": ranking["val_ap"],
        "ap_gain": ranking["val_ap"].diff(),
        "recall_top10": top10,
        "recall_top10_gain": top10.diff(),
        "cost": by_cost["cost"],
        "cost_gain": by_cost["cost"].diff(),
    })
    progression.round(4).to_csv(TABLES_DIR / "step4_progression.csv")
    print(progression.round(4).to_string())
    gains = progression["ap_gain"].dropna()
    reading(f"From linear to {label(gains.index[0])}: {gains.iloc[0]:+.3f} AP; "
            f"then {label(gains.index[1])}: {gains.iloc[1]:+.3f}.",
            "Each step helps a little: the data limit us, not the model.")

    plot_pr_curves(y_arr, all_scores)
    plot_train_vs_validation(ranking)


if __name__ == "__main__":
    main()
