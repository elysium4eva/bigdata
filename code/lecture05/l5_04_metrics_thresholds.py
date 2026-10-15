"""l5_04_metrics_thresholds.py -- classification metrics, decision thresholds and calibration.

Every line carries an English comment. The script takes the credit-default sample,
fits one classifier and then answers three practical questions: which metric to
report, where to set the threshold given asymmetric costs, and whether the
predicted probabilities can be trusted.
"""
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # four-panel metrics figure
from sklearn.calibration import CalibratedClassifierCV, calibration_curve   # reliability of probabilities
from sklearn.compose import ColumnTransformer                      # per-type preprocessing
from sklearn.linear_model import LogisticRegression                # base classifier
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,   # ranking and threshold metrics
                             average_precision_score, brier_score_loss,   # summary metrics
                             precision_score, recall_score, f1_score)   # threshold metrics
from sklearn.model_selection import train_test_split               # train/test split
from sklearn.pipeline import Pipeline                              # preprocessing plus model
from sklearn.preprocessing import OneHotEncoder, StandardScaler    # scaling and one-hot encoding
from _common_l5 import DAT, save, dump, style_axes, cost_curve, OKABE   # shared helpers

# ---------------------------------------------------------------- 1) data and a single honest split
credit = pd.read_csv(DAT / "l5_credit_default.csv")                # read the synthetic credit dataset
X = credit.drop(columns=["default"])                               # feature matrix
y = credit["default"].values                                       # binary target
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()   # numeric feature names
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()   # categorical feature names
X_train, X_test, y_train, y_test = train_test_split(               # one stratified split for the whole script
    X, y, test_size=0.3, random_state=42, stratify=y)              # stratify keeps the default rate equal
prep = ColumnTransformer([("num", StandardScaler(), num_cols),     # standardise numeric columns
                          ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])   # encode categories
base = Pipeline([("prep", prep),                                   # preprocessing step
                 ("model", LogisticRegression(max_iter=2000, random_state=42))]).fit(X_train, y_train)   # fit once
proba = base.predict_proba(X_test)[:, 1]                           # predicted probabilities on the test split
print("test rows:", len(y_test), "| positives:", int(y_test.sum()))   # report the evaluation sample

# ---------------------------------------------------------------- 2) ranking metrics: ROC and PR
fpr, tpr, _ = roc_curve(y_test, proba)                             # ROC curve points
roc_auc = auc(fpr, tpr)                                            # area under the ROC curve
prec, rec, thr_pr = precision_recall_curve(y_test, proba)          # precision-recall curve points
ap = average_precision_score(y_test, proba)                        # average precision, the PR summary
print("ROC-AUC %.4f | average precision %.4f" % (roc_auc, ap))      # report the ranking quality

# ---------------------------------------------------------------- 3) threshold metrics and expected cost
thresholds = np.linspace(0.02, 0.98, 60)                           # thresholds scanned for the report
rows = []                                                          # one row per threshold
for t in thresholds:                                               # loop over the threshold grid
    pred = (proba >= t).astype(int)                                # hard labels at this threshold
    rows.append({"threshold": round(float(t), 3),                  # threshold value
                 "precision": round(precision_score(y_test, pred, zero_division=0), 4),   # precision
                 "recall": round(recall_score(y_test, pred, zero_division=0), 4),   # recall
                 "f1": round(f1_score(y_test, pred, zero_division=0), 4),   # harmonic mean
                 "flag_rate": round(float(pred.mean()), 4)})       # share of accounts flagged
table = pd.DataFrame(rows)                                         # assemble the threshold table
costs = cost_curve(y_test, proba, thresholds, cost_fp=1.0, cost_fn=5.0)   # total cost, false negatives cost 5x
best_cost_idx = int(np.argmin(costs))                              # threshold with the lowest total cost
table["cost"] = costs.round(1)                                     # attach the cost column
dump(table, "l5_threshold_table.csv")                              # persist the table
best_thr = float(thresholds[best_cost_idx])                        # optimal threshold under these costs
print("cost-minimising threshold: %.2f (cost %.0f)" % (best_thr, costs[best_cost_idx]))   # report the optimum
at_half = table.iloc[int(np.argmin(np.abs(table["threshold"] - 0.5)))]   # the naive 0.5 threshold
print("compare at 0.5: recall %.3f | cost %.0f" % (at_half["recall"], at_half["cost"]))   # show the trade-off

# ---------------------------------------------------------------- 4) calibration of the probabilities
from sklearn.ensemble import RandomForestClassifier                 # tree ensembles are famously over-confident
rf = Pipeline([("prep", prep),                                      # the same preprocessing
               ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,   # a strong but uncalibrated model
                                                random_state=42))]).fit(X_train, y_train)   # fit on the training split
rf_prob = rf.predict_proba(X_test)[:, 1]                            # over-confident probabilities from the forest
base_prob = proba.copy()                                            # probabilities from the well-behaved linear model
calibrated = CalibratedClassifierCV(                                # wrap the forest with isotonic calibration
    Pipeline([("prep", prep), ("model", RandomForestClassifier(n_estimators=300,   # same base estimator
                                                               min_samples_leaf=2, random_state=42))]),   # same settings
    method="isotonic", cv=5)                                        # isotonic regression fitted with cross-validation
calibrated.fit(X_train, y_train)                                    # fit the calibrated model on the training split
cal_prob = calibrated.predict_proba(X_test)[:, 1]                   # calibrated probabilities from the forest
frac_pos, mean_pred = calibration_curve(y_test, rf_prob, n_bins=10, strategy="quantile")   # reliability before fixing
frac_pos_cal, mean_pred_cal = calibration_curve(y_test, cal_prob, n_bins=10, strategy="quantile")   # reliability after
print("Brier score: forest %.4f | isotonic forest %.4f | logistic %.4f" % (brier_score_loss(y_test, rf_prob),   # over-confident
                                                                          brier_score_loss(y_test, cal_prob),   # calibrated
                                                                          brier_score_loss(y_test, base_prob)))   # reference

# ---------------------------------------------------------------- 5) figure: four questions, four panels
fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.5))                # four panels in one row
axes[0].plot(fpr, tpr, lw=2.0, color=OKABE[5], label=f"ROC (AUC {roc_auc:.3f})")   # ROC curve
axes[0].plot(rec, prec, lw=2.0, color=OKABE[2], label=f"PR (AP {ap:.3f})")   # precision-recall curve
axes[0].plot([0, 1], [0, 1], ls="--", lw=1.0, color="#66717D", label="random")   # random reference
style_axes(axes[0], "Ranking quality: ROC and PR", "False positive rate / recall", "True positive rate / precision")   # style
axes[0].legend(fontsize=7.4, loc="lower left")                     # compact legend
axes[1].plot(table["threshold"], table["precision"], lw=1.8, color=OKABE[0], label="precision")   # precision curve
axes[1].plot(table["threshold"], table["recall"], lw=1.8, color=OKABE[2], label="recall")   # recall curve
axes[1].plot(table["threshold"], table["f1"], lw=1.8, color=OKABE[4], label="F1")   # F1 curve
axes[1].axvline(0.5, color="#66717D", ls="--", lw=1.0)             # naive threshold
axes[1].axvline(best_thr, color=OKABE[5], ls=":", lw=1.6)          # cost-optimal threshold
style_axes(axes[1], "Metrics move with the threshold", "Decision threshold", "Score")   # shared style
axes[1].legend(fontsize=7.4, loc="center right")                   # legend inside the panel
axes[2].plot(table["threshold"], table["cost"], lw=2.0, color=OKABE[5])   # expected cost curve
axes[2].scatter([best_thr], [costs[best_cost_idx]], s=42, color=OKABE[5], zorder=3)   # mark the optimum
axes[2].annotate(f"optimal {best_thr:.2f}\ncost {costs[best_cost_idx]:.0f}",   # annotate the minimum
                 xy=(best_thr, costs[best_cost_idx]), xytext=(0.35, 0.75),   # text placed in axes coordinates
                 textcoords="axes fraction", fontsize=8.2, color="#1E2630",   # small font
                 arrowprops=dict(arrowstyle="->", color="#66717D", lw=1.0))   # thin arrow to the point
style_axes(axes[2], "Asymmetric costs move the threshold", "Decision threshold", "Total cost (FP=1, FN=5)")   # style
axes[3].plot([0, 1], [0, 1], ls="--", lw=1.0, color="#66717D", label="perfect calibration")   # diagonal reference
axes[3].plot(mean_pred, frac_pos, "-o", ms=4, color=OKABE[5], label="forest (over-confident)")   # reliability before
axes[3].plot(mean_pred_cal, frac_pos_cal, "-s", ms=4, color=OKABE[2], label="forest + isotonic")   # reliability after
style_axes(axes[3], "Probabilities must be calibrated", "Mean predicted probability", "Observed default rate")   # style
axes[3].legend(fontsize=7.4, loc="upper left")                     # legend inside the panel
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_metrics_thresholds.png")                            # save the metrics figure
