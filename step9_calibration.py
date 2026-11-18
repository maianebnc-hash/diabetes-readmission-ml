import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import cross_val_predict

from common import (header, title, FIGURES_DIR, GROUP_COL, TABLES_DIR, TARGET, load_dataset, make_cv,
                    split_dev_test)
from step3_class_imbalance import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE
from step4_model_comparison import interpret
from step6_tuning import BEST_PARAMS_PATH, tuned_pipeline

# 1. Final model: best validation AP in steps 4 and 6
FINAL_MODEL = "gradient_boosting"

# 2. Equal-size bins (deciles of predicted risk)
N_BINS = 10

# 3. Normal quantile for 95% Wilson intervals
Z_95 = 1.96

# 4. Recalibration methods we try
CALIBRATION_METHODS = ["platt", "isotonic"]

# 5. Turning a probability into log-odds
def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

# 6. Predicted vs observed rate per decile
def reliability_table(y_true: np.ndarray, risk: np.ndarray) -> pd.DataFrame:
    bins = pd.qcut(risk, N_BINS, labels=False, duplicates="drop")
    table = pd.DataFrame({"y": y_true, "risk": risk, "bin": bins}).groupby("bin").agg(
        patients=("y", "size"), mean_predicted=("risk", "mean"), observed_rate=("y", "mean"))
    n, p = table["patients"], table["observed_rate"]
    centre = (p + Z_95**2 / (2 * n)) / (1 + Z_95**2 / n)
    half = Z_95 * np.sqrt(p * (1 - p) / n + Z_95**2 / (4 * n**2)) / (1 + Z_95**2 / n)
    table["wilson_low"], table["wilson_high"] = centre - half, centre + half
    table["gap"] = table["mean_predicted"] - table["observed_rate"]
    return table

# 7. ECE: mean |predicted - observed|, weighted by bin size
def expected_calibration_error(table: pd.DataFrame) -> float:
    return float((table["gap"].abs() * table["patients"]).sum() / table["patients"].sum())

# 8. Calibration slope and intercept (slope < 1: over-confident)
def calibration_slope_intercept(y_true: np.ndarray, risk: np.ndarray) -> tuple[float, float]:
    model = LogisticRegression(C=1e6, max_iter=1000).fit(logit(risk).reshape(-1, 1), y_true)
    return float(model.coef_[0, 0]), float(model.intercept_[0])

# 9. All calibration metrics for one set of predictions
def summary_metrics(y_true: np.ndarray, risk: np.ndarray) -> dict:
    prevalence = y_true.mean()
    brier = brier_score_loss(y_true, risk)
    slope, intercept = calibration_slope_intercept(y_true, risk)
    return {
        "brier": brier,
        "brier_skill_score": 1 - brier / (prevalence * (1 - prevalence)),
        "ece": expected_calibration_error(reliability_table(y_true, risk)),
        "mean_predicted_risk": risk.mean(),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "average_precision": average_precision_score(y_true, risk),
    }

# 10. Recalibrating each fold with the other folds only
def cross_fitted_calibration(y_true: np.ndarray, risk: np.ndarray, groups: pd.Series, method: str) -> np.ndarray:
    calibrated = np.zeros_like(risk)
    for fit_idx, apply_idx in make_cv().split(risk, y_true, groups):
        if method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            calibrated[apply_idx] = calibrator.fit(risk[fit_idx], y_true[fit_idx]).predict(risk[apply_idx])
        else:
            calibrator = LogisticRegression(C=1e6, max_iter=1000)
            calibrator.fit(logit(risk[fit_idx]).reshape(-1, 1), y_true[fit_idx])
            calibrated[apply_idx] = calibrator.predict_proba(logit(risk[apply_idx]).reshape(-1, 1))[:, 1]
    return calibrated

# 11. Plotting the reliability curve
def plot_reliability(table: pd.DataFrame, risk: np.ndarray, prevalence: float) -> None:
    fig, (ax, hist) = plt.subplots(2, 1, figsize=(6, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    limit = table["wilson_high"].max() * 1.1
    ax.plot([0, limit], [0, limit], color="grey", linestyle="--", label="perfect calibration")
    errors = [table["observed_rate"] - table["wilson_low"], table["wilson_high"] - table["observed_rate"]]
    ax.errorbar(table["mean_predicted"], table["observed_rate"], yerr=errors, marker="o", capsize=3,
                color="#4C72B0", label="gradient boosting (deciles, 95% Wilson interval)")
    ax.axhline(prevalence, color="#C44E52", linestyle=":", label=f"prevalence ({prevalence:.3f})")
    ax.set_ylabel("Observed readmission rate")
    ax.set_title("Reliability curve (out-of-fold predictions)")
    ax.legend(loc="upper left")
    hist.hist(risk, bins=60, range=(0, limit), color="#4C72B0")
    hist.set_xlabel("Predicted risk")
    hist.set_ylabel("Patients")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "step9_reliability_curve.png", dpi=150)
    plt.close(fig)

# 12. Running step 9
def main() -> None:
    title("Step 9: Calibration")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    dev, _ = split_dev_test(load_dataset())
    X, y, groups = dev.drop(columns=TARGET), dev[TARGET], dev[GROUP_COL]
    y_true, prevalence = y.to_numpy(), y.mean()

    params = json.loads(BEST_PARAMS_PATH.read_text())[FINAL_MODEL]
    print(f"Cross-validating {FINAL_MODEL} (tuned) ...")
    risk = cross_val_predict(tuned_pipeline(FINAL_MODEL, params), X, y, groups=groups, cv=make_cv(),
                             method="predict_proba")[:, 1]

    # 1. Measuring calibration
    header("[1] Calibration metrics")
    metrics = summary_metrics(y_true, risk)
    print(f"Brier score: {metrics['brier']:.4f}")
    print(f"Brier score if we always predict the prevalence: {prevalence * (1 - prevalence):.4f}")
    print(f"Brier skill score: {metrics['brier_skill_score']:.4f}")
    print(f"ECE ({N_BINS} equal-size bins): {metrics['ece']:.4f}")
    print(f"Mean predicted risk: {metrics['mean_predicted_risk']:.4f}")
    print(f"Prevalence: {prevalence:.4f}")
    interpret(f"Mean predicted risk {metrics['mean_predicted_risk']:.3f}; the prevalence is {prevalence:.3f}.",
              f"ECE is {metrics['ece']:.3f}: predictions are off by {metrics['ece'] * 100:.1f} points on average.")

    # 2. Building the reliability curve
    header("[2] Reliability table")
    table = reliability_table(y_true, risk)
    table.round(4).to_csv(TABLES_DIR / "step9_reliability_table.csv")
    print(table.round(4).to_string())
    inside = table["mean_predicted"].between(table["wilson_low"], table["wilson_high"]).sum()
    print()
    print(f"Bins where the prediction is inside the 95% interval: {inside} of {len(table)}")
    interpret(f"The prediction lies inside the 95% interval in {inside} of {len(table)} deciles.",
              "The curve follows the diagonal.")

    # 3. Reading over- or under-confidence
    header("[3] Over- or under-confident")
    slope = metrics["calibration_slope"]
    low_gap, high_gap = table["gap"].iloc[0], table["gap"].iloc[-1]
    print(f"Calibration slope: {slope:.4f} (1 is ideal, below 1 is over-confident)")
    print(f"Calibration intercept: {metrics['calibration_intercept']:.4f} (0 is ideal)")
    print(f"Gap in the lowest decile: {low_gap:+.4f}")
    print(f"Gap in the highest decile: {high_gap:+.4f}")
    interpret(f"Slope {slope:.2f}: the model is very slightly {'under' if slope > 1 else 'over'}-confident.",
              f"High risks are off by {abs(high_gap) * 100:.1f} points: negligible.")

    # 4. Trying recalibration
    header("[4] Recalibration")
    variants = {"as is": risk}
    for method in CALIBRATION_METHODS:
        variants[method] = cross_fitted_calibration(y_true, risk, groups, method)
    comparison = pd.DataFrame({name: summary_metrics(y_true, scores) for name, scores in variants.items()}).T
    comparison.round(4).to_csv(TABLES_DIR / "step9_recalibration.csv")
    print(comparison[["brier", "ece", "calibration_slope", "average_precision"]].round(4).to_string())
    brier, ap = comparison["brier"], comparison["average_precision"]
    interpret(f"Recalibration barely moves the Brier score: {brier.min():.4f} to {brier.max():.4f}.",
              f"Isotonic changes average precision by {ap['isotonic'] - ap['as is']:+.3f}; we keep the model as is.")

    # 5. Checking the regions where we decide
    header("[5] Probabilities where we decide")
    theory = COST_FALSE_POSITIVE / (COST_FALSE_POSITIVE + COST_FALSE_NEGATIVE)
    regions = {
        f"within 0.01 of threshold {theory:.2f}": np.abs(risk - theory) < 0.01,
        "top 10%": risk >= np.quantile(risk, 0.90),
    }
    decision = pd.DataFrame({
        name: {"patients": mask.sum(), "mean_predicted": risk[mask].mean(), "observed_rate": y_true[mask].mean()}
        for name, mask in regions.items()
    }).T
    decision["gap"] = decision["mean_predicted"] - decision["observed_rate"]
    decision.round(4).to_csv(TABLES_DIR / "step9_decision_regions.csv")
    print(decision.round(4).to_string())
    interpret(f"Largest gap where we decide: {decision['gap'].abs().max() * 100:.1f} points.",
              "We trust these probabilities as is, for this population only.")

    plot_reliability(table, risk, prevalence)

    print()
    print("Figures saved in reports/figures")
    print("Tables saved in reports/tables")

if __name__ == "__main__":
    main()
