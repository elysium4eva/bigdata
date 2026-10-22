"""l7_10_imbalance_cost.py -- class imbalance, cost-sensitive learning and the threshold.

Every line carries an English comment. The script builds an imbalanced sample, shows
how accuracy hides the error, prices the two error types, and compares plain,
reweighted and threshold-tuned models so the deck can show the cost curve.
"""
import numpy as np                                                 # arrays for the sample and costs
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # three panels
from sklearn.linear_model import LogisticRegression               # keep the model simple and readable
from sklearn.ensemble import RandomForestClassifier                # a second, tree-based baseline
from sklearn.model_selection import train_test_split               # hold-out split
from sklearn.metrics import (confusion_matrix, precision_recall_curve,   # evaluation tools
                             average_precision_score, precision_score,   # precision-type metrics
                             recall_score, f1_score, accuracy_score)    # threshold metrics
from _common_l7 import (RNG, save, dump, style_axes, annotated, OKABE)   # shared helpers

# ---------------------------------------------------------------- 1) an imbalanced sample
n = 2600                                                           # number of rows
x1 = RNG.normal(0, 1, n)                                           # first signal
x2 = RNG.normal(0, 1, n)                                           # second signal
logit = -3.4 + 1.5 * x1 + 0.9 * x2 + 0.5 * x1 * x2                 # the latent score
probability = 1.0 / (1.0 + np.exp(-logit))                         # true probability
y = (RNG.random(n) < probability).astype(int)                      # draw the label
frame = pd.DataFrame({"x1": x1.round(3), "x2": x2.round(3), "y": y})   # tabulate the sample
dump(frame, "l7_imbalanced.csv")                                   # persist it
print("rows:", n, "| positive rate: %.3f" % y.mean())               # report the imbalance
X_train, X_test, y_train, y_test = train_test_split(               # stratified hold-out
    frame[["x1", "x2"]].values, y, test_size=0.35, random_state=42, stratify=y)   # keep the rate equal
plain = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1).fit(X_train, y_train)   # plain forest
proba = plain.predict_proba(X_test)[:, 1]                          # predicted probabilities
pred05 = (proba >= 0.5).astype(int)                                # naive threshold
cm = confusion_matrix(y_test, pred05)                              # the confusion matrix at 0.5
print("plain: accuracy %.3f | recall %.3f | precision %.3f" % (         # report the plain forest at threshold 0.5
    accuracy_score(y_test, pred05), recall_score(y_test, pred05), precision_score(y_test, pred05, zero_division=0)))   # report

# ---------------------------------------------------------------- 2) price the two errors
c_fn, c_fp = 12.0, 1.0                                             # missing a positive costs twelve times a false alarm
thresholds = np.linspace(0.02, 0.95, 120)                          # threshold grid
rows = []                                                          # per-threshold rows
for t in thresholds:                                               # sweep the grid
    pred = (proba >= t).astype(int)                                # hard labels at this threshold
    tp = int(((pred == 1) & (y_test == 1)).sum())                  # true positives
    fp = int(((pred == 1) & (y_test == 0)).sum())                  # false positives
    fn = int(((pred == 0) & (y_test == 1)).sum())                  # false negatives
    cost = c_fn * fn + c_fp * fp                                   # total business cost
    rows.append({"threshold": round(float(t), 3), "tp": tp, "fp": fp, "fn": fn,   # counts
                 "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),   # precision
                 "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),   # recall
                 "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),   # F1
                 "cost": round(float(cost), 2)})                   # the priced cost
cost_table = pd.DataFrame(rows)                                    # assemble the table
dump(cost_table, "l7_cost_thresholds.csv")                         # persist it
best = cost_table.loc[cost_table["cost"].idxmin()]                 # the cheapest threshold
best_f1 = cost_table.loc[cost_table["f1"].idxmax()]                 # the best-F1 threshold
print("min cost at threshold %.2f (cost %.0f); best F1 at %.2f" % (     # report the two answers differ
    best["threshold"], best["cost"], best_f1["threshold"]))         # report the two answers differ

# ---------------------------------------------------------------- 3) reweighting
weighted = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42,   # same forest
                                  class_weight="balanced", n_jobs=-1).fit(X_train, y_train)   # balanced weights
proba_w = weighted.predict_proba(X_test)[:, 1]                     # reweighted probabilities
pred_w = (proba_w >= 0.5).astype(int)                              # at the naive threshold again
print("balanced: accuracy %.3f | recall %.3f | precision %.3f" % (      # report the reweighted forest
    accuracy_score(y_test, pred_w), recall_score(y_test, pred_w), precision_score(y_test, pred_w, zero_division=0)))   # report

# ---------------------------------------------------------------- 4) figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.7))                # three panels
im = axes[0].imshow(cm, cmap="RdPu")                               # confusion matrix at 0.5
axes[0].set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=8.4)      # predicted labels
axes[0].set_yticks([0, 1], ["true 0", "true 1"], fontsize=8.4)      # actual labels
axes[0].grid(False)                                                # a heatmap needs no grid
for i in range(2):                                                 # annotate the cells
    for j in range(2):                                             # loop over columns
        axes[0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=10,   # the count
                     color="white" if cm[i, j] > cm.max() * 0.6 else "#1E2630")   # contrast
axes[0].set_title("Confusion matrix at threshold 0.5")             # panel title
axes[1].plot(cost_table["threshold"], cost_table["cost"], lw=1.9, color=OKABE[5], label="business cost")   # cost curve
axes[1].axvline(float(best["threshold"]), ls="--", lw=1.2, color="#66717D", label="min cost")   # cheapest threshold
axes[1].axvline(0.5, ls=":", lw=1.0, color=OKABE[1], label="naive 0.5")   # naive threshold
style_axes(axes[1], "Priced cost versus threshold", "threshold", "total cost")   # shared style
axes[1].legend(fontsize=7.0, loc="upper right")                    # compact legend
prec_u, rec_u, _ = precision_recall_curve(y_test, proba)           # plain PR curve
prec_w, rec_w, _ = precision_recall_curve(y_test, proba_w)         # reweighted PR curve
axes[2].plot(rec_u, prec_u, lw=1.8, color=OKABE[1], label="plain (AP %.3f)" % average_precision_score(y_test, proba))   # plain
axes[2].plot(rec_w, prec_w, lw=1.8, ls="--", color=OKABE[2], label="balanced (AP %.3f)" % average_precision_score(y_test, proba_w))   # weighted
axes[2].axhline(y_test.mean(), ls=":", lw=1.0, color="#66717D", label="random")   # random baseline
style_axes(axes[2], "Reweighting changes the PR curve", "recall", "precision")   # shared style
axes[2].legend(fontsize=6.8, loc="lower left")                     # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_imbalance_cost.png")                                # save the figure
print("done: imbalance, cost curve and reweighting")               # closing log
