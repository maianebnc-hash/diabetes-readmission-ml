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

from common import (header, title, FIGURES_DIR, GROUP_COL, SEED, TABLES_DIR, TARGET, load_dataset, make_cv,
                    split_dev_test)
from step2_feature_engineering import FeatureEngineer, build_preprocessor
from step3_class_imbalance import (ALERT_VOLUMES, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE,
                                   alert_volume_metrics, choose_threshold, decision_metrics)

# 1. The three models we compare
MODEL_NAMES = ["logistic_regression", "random_forest", "gradient_boosting"]

# 2. Creating an untuned model
def build_model(name: str):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                      n_jobs=-1, random_state=SEED)
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                              early_stopping=False, random_state=SEED)
    raise ValueError(f"Unknown model: {name}")

# 3. Chaining features, preprocessing and model
def build_pipeline(name: str) -> Pipeline:
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocess", build_preprocessor(scale_numeric=(name == "logistic_regression"))),
        ("model", build_model(name)),
    ])

# 4. Cross-validating one model
def cross_validate_model(name: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series):
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

# 5. Printing our interpretation
def interpret(first: str, second: str) -> None:
    print()
    print(f"Interpretation: {first}")
    print(f"                {second}")

# 6. Plotting precision-recall curves
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

# 7. Plotting training vs validation scores
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

# 8. Running step 4
def main() -> None:
    title("Step 4: Model comparison")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    dev, _ = split_dev_test(load_dataset())
    X, y, groups = dev.drop(columns=TARGET), dev[TARGET], dev[GROUP_COL]
    y_arr = y.to_numpy()
    prevalence = y.mean()

    all_scores, fold_tables = {}, {}
    for name in MODEL_NAMES:
        print(f"Cross-validating {name} ...")
        all_scores[name], fold_tables[name] = cross_validate_model(name, X, y, groups)
    pd.concat(fold_tables, names=["model"]).round(4).to_csv(TABLES_DIR / "step4_fold_metrics.csv")

    # 1. Comparing ranking quality
    header("[1] Ranking metrics")
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
    interpret(f"{best} ranks best: AP {ranking.loc[best, 'val_ap']:.3f}, no-skill {prevalence:.3f}.",
              f"Gains between models stay close to the fold-to-fold standard deviation ({ranking['val_ap_std'].mean():.3f}).")

    # 2. Diagnosing overfitting
    header("[2] Train vs validation")
    print(ranking[["train_ap", "val_ap", "train_minus_val_ap"]].round(4).to_string())
    gaps = ranking["train_minus_val_ap"]
    worst, smallest = gaps.idxmax(), gaps.idxmin()
    interpret(f"{worst} overfits most: train AP {ranking.loc[worst, 'train_ap']:.2f}, validation {ranking.loc[worst, 'val_ap']:.2f}.",
              f"{smallest} has the smallest gap ({gaps[smallest]:.3f}): it underfits rather than overfits.")

    # 3. Checking probability quality
    header("[3] Probabilities")
    probability = pd.DataFrame({
        name: {"brier": brier_score_loss(y_arr, s), "mean_predicted_risk": s.mean(),
               "prevalence": prevalence} for name, s in all_scores.items()
    }).T
    probability.round(4).to_csv(TABLES_DIR / "step4_probability_metrics.csv")
    print(probability.round(4).to_string())
    risk, brier = probability["mean_predicted_risk"], probability["brier"]
    interpret(f"Mean predicted risk ranges from {risk.min():.3f} to {risk.max():.3f}; the prevalence is {prevalence:.3f}.",
              f"Brier scores range from {brier.min():.4f} to {brier.max():.4f}; we examine calibration in a later step.")

    # 4. Limiting the alert volume
    header("[4] Flagging the top 10% and 20%")
    by_volume = pd.DataFrame({
        (f"top {volume:.0%}", name): alert_volume_metrics(y_arr, s, volume)
        for volume in ALERT_VOLUMES for name, s in all_scores.items()
    }).T
    by_volume.round(4).to_csv(TABLES_DIR / "step4_decisions_alert_volume.csv")
    print(by_volume.round(4).to_string())
    top10 = by_volume.loc["top 10%", "recall"]
    best = top10.idxmax()
    interpret(f"With the top 10% flagged, {best} catches {top10[best]:.0%} of readmissions.",
              f"logistic_regression catches {top10['logistic_regression']:.0%}.")

    # 5. Tuning the threshold
    header("[5] Best threshold by cost")
    by_cost = pd.DataFrame({
        name: decision_metrics(y_arr, s, choose_threshold(y_arr, s)) for name, s in all_scores.items()
    }).T
    by_cost.round(4).to_csv(TABLES_DIR / "step4_decisions_cost_threshold.csv")
    print(by_cost.round(4).to_string())
    theory = COST_FALSE_POSITIVE / (COST_FALSE_POSITIVE + COST_FALSE_NEGATIVE)
    chosen = ", ".join(f"{t:.2f}" for t in by_cost["threshold"])
    cheapest = by_cost["cost"].idxmin()
    interpret(f"Chosen thresholds: {chosen}; theory says {theory:.2f}.",
              f"{cheapest} has the lowest cost but flags {by_cost.loc[cheapest, 'flagged_share']:.0%} of patients.")

    # 6. Measuring what each model adds
    header("[6] Improvement per model")
    progression = pd.DataFrame({
        "val_ap": ranking["val_ap"],
        "ap_gain": ranking["val_ap"].diff(),
        "recall_top10": top10,
        "recall_top10_gain": top10.diff(),
        "cost": by_cost["cost"],
        "cost_gain": by_cost["cost"].diff(),
    })
    progression.round(4).to_csv(TABLES_DIR / "step4_progression.csv")
    print(progression.round(4).to_string(na_rep="-"))
    gains = progression["ap_gain"]
    interpret(f"From linear to random forest: {gains['random_forest']:+.3f} AP; then gradient boosting: {gains['gradient_boosting']:+.3f}.",
              "Each step helps a little: the data limit us, not the model.")

    plot_pr_curves(y_arr, all_scores)
    plot_train_vs_validation(ranking)

if __name__ == "__main__":
    main()
