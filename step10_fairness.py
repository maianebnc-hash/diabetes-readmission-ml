import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict

from common import (header, title, FIGURES_DIR, GROUP_COL, TABLES_DIR, TARGET, load_dataset, make_cv,
                    split_dev_test)
from step3_class_imbalance import ALERT_VOLUMES
from step4_model_comparison import interpret
from step6_tuning import BEST_PARAMS_PATH, tuned_pipeline

# 1. Final model: best validation AP in steps 4 and 6
FINAL_MODEL = "gradient_boosting"

# 2. Sensitive attribute and its groups (small groups are merged)
SENSITIVE = "race"
MAIN_GROUPS = ["Caucasian", "AfricanAmerican"]
OTHER_GROUP = "Other or unknown"
GROUPS = [*MAIN_GROUPS, OTHER_GROUP]

# 3. Share of patients the care team can follow up
ALERT_VOLUME = ALERT_VOLUMES[0]

# 4. Normal quantile for 95% Wilson intervals
Z_95 = 1.96

# 5. Gaps we report and the fairness definition behind each
GAP_COLUMNS = {
    "flagged_share": "demographic parity",
    "recall": "equal opportunity",
    "false_positive_rate": "equalized odds (with recall)",
    "precision": "predictive parity",
}

# 6. Merging small race groups
def race_group(race: pd.Series) -> pd.Series:
    return race.where(race.isin(MAIN_GROUPS), OTHER_GROUP)

# 7. Wilson interval of a proportion
def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    p = successes / n
    centre = (p + Z_95**2 / (2 * n)) / (1 + Z_95**2 / n)
    half = Z_95 * np.sqrt(p * (1 - p) / n + Z_95**2 / (4 * n**2)) / (1 + Z_95**2 / n)
    return centre - half, centre + half

# 8. Decision metrics for each group
def group_metrics(y_true: np.ndarray, risk: np.ndarray, flagged: np.ndarray, group: pd.Series) -> pd.DataFrame:
    rows = {}
    for name in GROUPS:
        mask = (group == name).to_numpy()
        y, f = y_true[mask], flagged[mask]
        caught, positives = int((f & (y == 1)).sum()), int(y.sum())
        low, high = wilson_interval(caught, positives)
        rows[name] = {
            "patients": int(mask.sum()),
            "readmitted": positives,
            "prevalence": y.mean(),
            "mean_predicted_risk": risk[mask].mean(),
            "flagged_share": f.mean(),
            "recall": caught / positives,
            "recall_low": low,
            "recall_high": high,
            "false_positive_rate": f[y == 0].mean(),
            "precision": y[f].mean(),
        }
    return pd.DataFrame(rows).T

# 9. Largest difference between groups for each metric
def fairness_gaps(table: pd.DataFrame) -> pd.Series:
    return pd.Series({col: table[col].max() - table[col].min() for col in GAP_COLUMNS})

# 10. Overall performance of a set of decisions
def overall_metrics(y_true: np.ndarray, flagged: np.ndarray) -> dict:
    return {
        "flagged_share": flagged.mean(),
        "recall": flagged[y_true == 1].mean(),
        "precision": y_true[flagged].mean(),
    }

# 11. One threshold per group, giving every group the same recall
def group_thresholds(y_true: np.ndarray, risk: np.ndarray, group: pd.Series, target_recall: float) -> dict:
    thresholds = {}
    for name in GROUPS:
        positives = risk[(group == name).to_numpy() & (y_true == 1)]
        thresholds[name] = float(np.quantile(positives, 1 - target_recall))
    return thresholds

# 12. Applying the group thresholds
def flag_by_group(risk: np.ndarray, group: pd.Series, thresholds: dict) -> np.ndarray:
    return risk >= group.map(thresholds).to_numpy()

# 13. Plotting recall per group, before and after
def plot_recall(before: pd.DataFrame, after: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(GROUPS))
    for offset, table, label, color in [(-0.2, before, "before (one threshold)", "#4C72B0"),
                                        (0.2, after, "after (group thresholds)", "#C44E52")]:
        errors = [table["recall"] - table["recall_low"], table["recall_high"] - table["recall"]]
        ax.bar(x + offset, table["recall"], width=0.4, yerr=errors, capsize=4, label=label, color=color)
    ax.set_xticks(x, GROUPS)
    ax.set_ylabel("Recall (share of readmissions flagged)")
    ax.set_title("Equal opportunity on the test set (95% Wilson intervals)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step10_recall_by_group.png", dpi=150)
    plt.close(fig)

# 14. Running step 10
def main() -> None:
    title("Step 10: Fairness audit")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    dev, test = split_dev_test(load_dataset())
    X, y, groups = dev.drop(columns=TARGET), dev[TARGET], dev[GROUP_COL]
    X_test, y_test = test.drop(columns=TARGET), test[TARGET].to_numpy()
    y_dev = y.to_numpy()
    race_dev, race_test = race_group(dev[SENSITIVE]), race_group(test[SENSITIVE])

    params = json.loads(BEST_PARAMS_PATH.read_text())[FINAL_MODEL]
    print(f"Cross-validating {FINAL_MODEL} (tuned) ...")
    risk_dev = cross_val_predict(tuned_pipeline(FINAL_MODEL, params), X, y, groups=groups, cv=make_cv(),
                                 method="predict_proba")[:, 1]
    print("Fitting on the development set, predicting the test set ...")
    risk_test = tuned_pipeline(FINAL_MODEL, params).fit(X, y).predict_proba(X_test)[:, 1]

    # 1. Fixing the decision rule on the development set
    header("[1] Decision rule")
    threshold = float(np.quantile(risk_dev, 1 - ALERT_VOLUME))
    target_recall = (risk_dev >= threshold)[y_dev == 1].mean()
    print(f"Alert volume: top {ALERT_VOLUME:.0%} of patients")
    print(f"Threshold (development, out-of-fold): {threshold:.4f}")
    print(f"Recall at this threshold (development): {target_recall:.4f}")
    interpret(f"We flag patients with a predicted risk above {threshold:.2f}.",
              "The threshold comes from the development set; the test set is new.")

    # 2. Scoring the untouched test set
    header("[2] Test set performance")
    flagged_before = risk_test >= threshold
    step6 = pd.read_csv(TABLES_DIR / "step6_tuning_gains.csv", index_col=0).loc[FINAL_MODEL]
    performance = {
        "average_precision": average_precision_score(y_test, risk_test),
        "roc_auc": roc_auc_score(y_test, risk_test),
        "brier": brier_score_loss(y_test, risk_test),
        "mean_predicted_risk": risk_test.mean(),
        "prevalence": y_test.mean(),
        **overall_metrics(y_test, flagged_before),
    }
    pd.Series(performance, name="test").round(4).to_csv(TABLES_DIR / "step10_test_performance.csv")
    print(f"Patients: {len(y_test)}")
    for name, value in performance.items():
        print(f"{name}: {value:.4f}")
    print()
    print(f"Cross-validated AP (step 6): {step6['tuned_ap']:.4f} (fold deviation {step6['fold_std']:.4f})")
    difference = performance["average_precision"] - step6["tuned_ap"]
    interpret(f"Test AP {performance['average_precision']:.3f}; cross-validation said {step6['tuned_ap']:.3f} ({difference:+.3f}).",
              f"With {performance['flagged_share']:.0%} flagged, we catch {performance['recall']:.0%} of readmissions.")

    # 3. Auditing the groups before mitigation
    header("[3] Metrics per race group (before)")
    before = group_metrics(y_test, risk_test, flagged_before, race_test)
    before.round(4).to_csv(TABLES_DIR / "step10_groups_before.csv")
    sizes = before[["patients", "readmitted", "prevalence", "mean_predicted_risk"]].round(4)
    print(sizes.astype({"patients": int, "readmitted": int}).to_string())
    print()
    print(before[["flagged_share", "recall", "recall_low", "recall_high", "false_positive_rate", "precision"]].round(4).to_string())
    lowest, highest = before["recall"].idxmin(), before["recall"].idxmax()
    interpret(f"Recall is lowest for {lowest} ({before.loc[lowest, 'recall']:.0%}), highest for {highest} ({before.loc[highest, 'recall']:.0%}).",
              "Mean predicted risk stays close to the prevalence in each group.")

    # 4. Measuring the fairness gaps
    header("[4] Fairness gaps (before)")
    gaps_before = fairness_gaps(before)
    gap_table = pd.DataFrame({"fairness_definition": GAP_COLUMNS, "gap_before": gaps_before})
    print(gap_table.round(4).to_string())
    overlap = before["recall_low"].max() <= before["recall_high"].min()
    interpret(f"We choose equal opportunity: the recall gap is {gaps_before['recall'] * 100:.1f} points.",
              f"The 95% intervals of the groups {'overlap: the gap may be noise' if overlap else 'do not all overlap: the gap is real'}.")

    # 5. Mitigating with group thresholds
    header("[5] Mitigation: one threshold per group")
    thresholds = group_thresholds(y_dev, risk_dev, race_dev, target_recall)
    threshold_table = pd.DataFrame({"single_threshold": threshold, "group_threshold": pd.Series(thresholds)})
    threshold_table["change"] = threshold_table["group_threshold"] - threshold_table["single_threshold"]
    threshold_table.round(4).to_csv(TABLES_DIR / "step10_group_thresholds.csv")
    print(f"Target recall for every group (development): {target_recall:.4f}")
    print(threshold_table.round(4).to_string())
    lowered = threshold_table["change"].idxmin()
    interpret("Thresholds are learned on the development set, not on the test set.",
              f"The threshold moves most for {lowered}: {threshold_table.loc[lowered, 'change']:+.3f}.")

    # 6. Comparing before and after on the test set
    header("[6] Before vs after (test set)")
    flagged_after = flag_by_group(risk_test, race_test, thresholds)
    after = group_metrics(y_test, risk_test, flagged_after, race_test)
    after.round(4).to_csv(TABLES_DIR / "step10_groups_after.csv")
    gaps_after = fairness_gaps(after)
    comparison = pd.DataFrame({
        "before": {**{f"gap_{col}": gaps_before[col] for col in GAP_COLUMNS}, **overall_metrics(y_test, flagged_before)},
        "after": {**{f"gap_{col}": gaps_after[col] for col in GAP_COLUMNS}, **overall_metrics(y_test, flagged_after)},
    })
    comparison["change"] = comparison["after"] - comparison["before"]
    comparison.round(4).to_csv(TABLES_DIR / "step10_before_after.csv")
    print(after[["flagged_share", "recall", "recall_low", "recall_high", "false_positive_rate", "precision"]].round(4).to_string())
    print()
    print(comparison.round(4).to_string())
    recall_gap, recall, precision = comparison.loc["gap_recall"], comparison.loc["recall"], comparison.loc["precision"]
    interpret(f"The recall gap goes from {recall_gap['before'] * 100:.1f} to {recall_gap['after'] * 100:.1f} points.",
              f"Overall recall changes by {recall['change'] * 100:+.2f} points, precision by {precision['change'] * 100:+.2f}.")

    plot_recall(before, after)

    print()
    print("Figures saved in reports/figures")
    print("Tables saved in reports/tables")

if __name__ == "__main__":
    main()
