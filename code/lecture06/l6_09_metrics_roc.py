"""l6_09_metrics_roc.py -- class imbalance, thresholds, ROC/PR curves and probability calibration.

Every line carries an English comment. The script builds an imbalanced credit-like
sample, fits logistic regression, and then shows why accuracy is not enough: the
confusion matrix at 0.5, the metric-versus-threshold curves, the effect of class
weighting, and a reliability diagram with and without isotonic calibration.
"""
import numpy as np                                                 # arrays for the sample and metrics
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # two figures
from sklearn.calibration import CalibratedClassifierCV, calibration_curve   # probability calibration
from sklearn.linear_model import LogisticRegression               # the model under study
from sklearn.metrics import (confusion_matrix, roc_curve, auc,     # evaluation metrics
                             precision_recall_curve, average_precision_score,   # # ranking metrics
                             precision_score, recall_score, f1_score,   # # threshold metrics
                             brier_score_loss, accuracy_score)          # # probability error
from sklearn.model_selection import train_test_split               # hold-out split
from sklearn.pipeline import Pipeline                              # scaling plus model
from sklearn.preprocessing import StandardScaler                   # standardisation helps convergence
from _common_l6 import RNG, save, dump, style_axes, annotated, OKABE   # shared helpers

# ---------------------------------------------------------------- 1) an imbalanced sample
N = 2400                                                           # number of accounts
x1 = RNG.normal(0, 1, N)                                           # first risk driver
x2 = RNG.normal(0, 1, N)                                           # second risk driver
logit = -3.0 + 1.4 * x1 + 0.8 * x2 + 0.6 * x1 * x2                 # latent risk score with an interaction
prob = 1.0 / (1.0 + np.exp(-logit))                                # true probability
y = (RNG.random(N) < prob).astype(int)                             # draw the label
imb = pd.DataFrame({"x1": x1.round(3), "x2": x2.round(3), "y": y})   # tabulate the sample
dump(imb, "l6_imbalanced.csv")                                     # persist it
print("rows:", N, "| positive rate: %.3f" % y.mean())              # report the imbalance
X_train, X_test, y_train, y_test = train_test_split(               # stratified hold-out split
    imb[["x1", "x2"]].values, y, test_size=0.35, random_state=42, stratify=y)   # keep the rate identical
pipe = Pipeline([("scale", StandardScaler()),                      # scaling inside the pipeline
                 ("model", LogisticRegression(max_iter=2000))])     # plain logistic regression
pipe.fit(X_train, y_train)                                         # fit on the training split
proba = pipe.predict_proba(X_test)[:, 1]                           # predicted probabilities
pred05 = (proba >= 0.5).astype(int)                                # naive threshold
print("naive accuracy: %.3f | recall: %.3f" % (accuracy_score(y_test, pred05),   # accuracy hides the problem
                                               recall_score(y_test, pred05)))   # the positive class is missed
cm = confusion_matrix(y_test, pred05)                              # confusion matrix at 0.5
print("confusion matrix:\n", cm)                                    # report TP/FP/FN/TN

# ---------------------------------------------------------------- 2) metrics as a function of the threshold
thresholds = np.linspace(0.02, 0.95, 80)                           # threshold grid
rows = []                                                          # one row per threshold
for t in thresholds:                                               # sweep the grid
    pred = (proba >= t).astype(int)                                # hard labels
    rows.append({"threshold": round(float(t), 3),                  # threshold value
                 "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),   # precision
                 "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),   # recall
                 "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4)})   # F1
thr_table = pd.DataFrame(rows)                                     # assemble the table
dump(thr_table, "l6_threshold_metrics.csv")                        # persist it
best_f1 = thr_table.loc[thr_table["f1"].idxmax()]                  # threshold that maximises F1
print("best F1 %.3f at threshold %.2f" % (best_f1["f1"], best_f1["threshold"]))   # report the trade-off

# ---------------------------------------------------------------- 3) class weighting changes the trade-off
weighted = Pipeline([("scale", StandardScaler()),                  # same preprocessing
                     ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])   # reweighted loss
weighted.fit(X_train, y_train)                                     # fit the reweighted model
proba_w = weighted.predict_proba(X_test)[:, 1]                     # reweighted probabilities
prec_w, rec_w, _ = precision_recall_curve(y_test, proba_w)         # PR curve for the reweighted model
prec_u, rec_u, _ = precision_recall_curve(y_test, proba)           # PR curve for the plain model
fpr_u, tpr_u, _ = roc_curve(y_test, proba)                         # ROC curve for the plain model
fpr_w, tpr_w, _ = roc_curve(y_test, proba_w)                       # ROC curve for the reweighted model

# ---------------------------------------------------------------- 4) calibration
calibrated = CalibratedClassifierCV(                               # isotonic calibration with internal CV
    Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]),   # base model
    method="isotonic", cv=5)                                       # isotonic regression
calibrated.fit(X_train, y_train)                                   # fit the calibrated model
proba_c = calibrated.predict_proba(X_test)[:, 1]                   # calibrated probabilities
frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")   # reliability before
frac_pos_c, mean_pred_c = calibration_curve(y_test, proba_c, n_bins=10, strategy="quantile")   # reliability after
summary = pd.DataFrame({"model": ["plain", "class_weight=balanced", "isotonic"],   # the three variants
                        "auc": [round(float(auc(fpr_u, tpr_u)), 4),   # AUC of the plain model
                                round(float(auc(fpr_w, tpr_w)), 4),   # AUC of the reweighted model
                                round(float(auc(*roc_curve(y_test, proba_c)[:2])), 4)],   # AUC after calibration
                        "average_precision": [round(float(average_precision_score(y_test, proba)), 4),   # AP plain
                                              round(float(average_precision_score(y_test, proba_w)), 4),   # AP weighted
                                              round(float(average_precision_score(y_test, proba_c)), 4)],  # AP calibrated
                        "brier": [round(float(brier_score_loss(y_test, proba)), 4),   # Brier plain
                                  round(float(brier_score_loss(y_test, proba_w)), 4),   # Brier weighted
                                  round(float(brier_score_loss(y_test, proba_c)), 4)]})   # Brier calibrated
dump(summary, "l6_metrics_summary.csv")                            # persist the summary
print(summary.to_string(index=False))                              # report the three variants

# ---------------------------------------------------------------- 5) figure one: imbalance and thresholds
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.6))                # three panels
im = axes[0].imshow(cm, cmap="RdPu")                               # confusion matrix at the naive threshold
axes[0].set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=8.4)      # predicted labels
axes[0].set_yticks([0, 1], ["true 0", "true 1"], fontsize=8.4)      # actual labels
axes[0].grid(False)                                                # a heatmap needs no grid
for i in range(2):                                                 # annotate the four cells
    for j in range(2):                                             # loop over columns
        axes[0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=10,   # the count
                     color="white" if cm[i, j] > cm.max() * 0.6 else "#1E2630")   # contrast against the fill
axes[0].set_title("Confusion matrix at threshold 0.5")             # panel title
axes[1].plot(thr_table["threshold"], thr_table["precision"], lw=1.8, color=OKABE[0], label="precision")   # precision
axes[1].plot(thr_table["threshold"], thr_table["recall"], lw=1.8, color=OKABE[2], label="recall")   # recall
axes[1].plot(thr_table["threshold"], thr_table["f1"], lw=1.8, color=OKABE[4], label="F1")   # F1
axes[1].axvline(0.5, ls="--", lw=1.0, color="#66717D")             # naive threshold
axes[1].axvline(float(best_f1["threshold"]), ls=":", lw=1.4, color=OKABE[5])   # best-F1 threshold
style_axes(axes[1], "Metrics move with the threshold", "threshold", "score")   # shared style
axes[1].legend(fontsize=7.2, loc="center right")                   # compact legend
axes[2].plot(rec_u, prec_u, lw=1.8, color=OKABE[0], label="plain (AP %.3f)" % average_precision_score(y_test, proba))   # PR plain
axes[2].plot(rec_w, prec_w, lw=1.8, color=OKABE[2], ls="--", label="balanced (AP %.3f)" % average_precision_score(y_test, proba_w))   # PR weighted
axes[2].axhline(y_test.mean(), ls=":", lw=1.0, color="#66717D", label="random")   # random baseline
style_axes(axes[2], "PR curves: reweighting lifts recall", "recall", "precision")   # shared style
axes[2].legend(fontsize=7.0, loc="lower left")                     # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_class_imbalance.png")                               # save the imbalance figure

# ---------------------------------------------------------------- 6) figure two: calibration and ranking
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.6))                # three panels
axes[0].plot(mean_pred, frac_pos, "-o", ms=4, color=OKABE[5], label="plain")   # reliability before
axes[0].plot(mean_pred_c, frac_pos_c, "-s", ms=4, color=OKABE[2], label="isotonic")   # reliability after
axes[0].plot([0, 1], [0, 1], ls="--", lw=1.0, color="#66717D", label="perfect")   # diagonal reference
style_axes(axes[0], "Reliability diagram", "mean predicted probability", "observed rate")   # shared style
axes[0].legend(fontsize=7.2, loc="upper left")                     # compact legend
axes[1].plot(fpr_u, tpr_u, lw=1.8, color=OKABE[0], label="plain (AUC %.3f)" % auc(fpr_u, tpr_u))   # ROC plain
axes[1].plot(fpr_w, tpr_w, lw=1.8, color=OKABE[2], ls="--", label="balanced (AUC %.3f)" % auc(fpr_w, tpr_w))   # ROC weighted
axes[1].plot([0, 1], [0, 1], ls="--", lw=1.0, color="#66717D")      # random reference
style_axes(axes[1], "ROC curves are similar, PR is not", "false positive rate", "true positive rate")   # shared style
axes[1].legend(fontsize=7.2, loc="lower right")                    # compact legend
axes[2].bar(["plain", "balanced", "isotonic"], summary["brier"],   # Brier scores
            color=[OKABE[0], OKABE[2], OKABE[4]], alpha=0.9)        # one colour per variant
axes[2].set_ylim(0, float(summary["brier"].max()) * 1.25)           # leave room for labels
style_axes(axes[2], "Brier score after calibration", "", "Brier score")   # shared style
for i, v in enumerate(summary["brier"]):                            # annotate each bar
    axes[2].text(i, v, "%.4f" % v, ha="center", va="bottom", fontsize=8.4)   # the value
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_calibration.png")                                   # save the calibration figure
print("done: imbalance, thresholds, ROC/PR and calibration")        # closing log
