"""l15_08_churn.py -- churn prediction with an expected-loss threshold.

Every line carries an English comment. The script trains logistic regression and gradient
boosting on 3,000 customers, compares AUC, and chooses an intervention threshold from expected
cost rather than from accuracy, which is what the deck argues for.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from sklearn.model_selection import train_test_split               # split
from sklearn.linear_model import LogisticRegression                 # the interpretable model
from sklearn.ensemble import GradientBoostingClassifier            # the flexible model
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix   # metrics
from sklearn.preprocessing import OneHotEncoder                     # encode the plan column
from _common_l15 import (make_churn, save, dump, style_axes,       # shared helpers
                         annotated, OKABE)                              # shared plotting helpers

# ---------------------------------------------------------------- 1) the customer base
churn = make_churn(3000)                                           # 3,000 customers
dump(churn, "l15_churn.csv")                                       # persist
print("customers:", len(churn), "| churn rate: %.3f" % churn["churn"].mean())   # report

# ---------------------------------------------------------------- 2) features
enc = OneHotEncoder(sparse_output=False, drop="first")              # drop the first plan level
plan_mat = enc.fit_transform(churn[["plan"]])                       # one-hot plan columns
plan_cols = ["plan_%s" % c for c in enc.categories_[0][1:]]          # the kept level names
base = churn[["tenure", "usage", "complaints"]].values               # numeric features
X = np.column_stack([base, plan_mat])                               # the design matrix
y = churn["churn"].values                                          # the label
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)   # stratified split
print("train/test:", len(X_tr), len(X_te))                          # report

# ---------------------------------------------------------------- 3) two models
logit = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)            # logistic regression
gbm = GradientBoostingClassifier(random_state=42, n_estimators=250, max_depth=3).fit(X_tr, y_tr)   # gradient boosting
p_logit = logit.predict_proba(X_te)[:, 1]                           # predicted probabilities
p_gbm = gbm.predict_proba(X_te)[:, 1]                               # predicted probabilities
rows = []                                                          # one row per model
for name, prob in [("Logistic", p_logit), ("GBDT", p_gbm)]:          # loop over models
    rows.append({"model": name, "auc": round(float(roc_auc_score(y_te, prob)), 4)})   # AUC
    print("%-9s AUC %.3f" % (name, roc_auc_score(y_te, prob)))       # report
metrics = pd.DataFrame(rows)                                       # assemble
dump(metrics, "l15_churn_metrics.csv")                             # persist

# ---------------------------------------------------------------- 4) expected-loss threshold
value_at_risk = 1800.0                                             # revenue lost when a customer churns
cost_contact = 120.0                                               # cost of one retention contact
success_rate = 0.30                                                # the intervention saves 30% of churners
thresholds = np.linspace(0.01, 0.99, 99)                            # candidate thresholds
best = None                                                        # the best threshold so far
curve = []                                                         # expected cost per threshold
for t in thresholds:                                               # loop over thresholds
    flagged = p_gbm >= t                                           # who gets contacted
    n_flagged = int(flagged.sum())                                 # how many
    caught = int(((y_te == 1) & flagged).sum())                     # churners who are flagged
    missed = int(((y_te == 1) & ~flagged).sum())                    # churners who are missed
    cost = (n_flagged * cost_contact                                 # contact cost
            + missed * value_at_risk                                 # lost revenue from misses
            + caught * (1 - success_rate) * value_at_risk)           # non-saved customers still leave
    curve.append({"threshold": round(float(t), 2), "cost": round(float(cost), 0),   # the row
                  "flagged": n_flagged, "caught": caught, "missed": missed})   # details
    if best is None or cost < best[1]:                               # keep the cheapest
        best = (float(t), float(cost))                               # remember it
curve = pd.DataFrame(curve)                                        # assemble
dump(curve, "l15_churn_threshold.csv")                             # persist
print("best threshold %.2f | expected cost %.0f | flagged %d of %d" % (   # report the threshold
    best[0], best[1], int((p_gbm >= best[0]).sum()), len(y_te)))     # report
print("cost at 0.50 threshold: %.0f" % float(curve.iloc[48]["cost"]))   # compare with the naive cut-off

# ---------------------------------------------------------------- 5) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                 # three panels
for name, prob, colour in [("Logistic", p_logit, OKABE[1]), ("GBDT", p_gbm, OKABE[5])]:   # two curves
    fpr, tpr, _ = roc_curve(y_te, prob)                             # ROC points
    axes[0].plot(fpr, tpr, lw=1.8, color=colour, label="%s (AUC %.3f)" % (name, roc_auc_score(y_te, prob)))   # the curve
axes[0].plot([0, 1], [0, 1], ls="--", lw=1.0, color="#66717D")      # the diagonal
style_axes(axes[0], "ROC curves", "false positive rate", "true positive rate")   # shared style
axes[0].legend(fontsize=7.4, loc="lower right")                     # compact legend
axes[1].hist(p_gbm[y_te == 0], bins=30, alpha=0.7, color=OKABE[2], label="未流失")   # stayers
axes[1].hist(p_gbm[y_te == 1], bins=30, alpha=0.7, color=OKABE[5], label="流失")     # churners
style_axes(axes[1], "Predicted churn probability", "probability", "customers")   # shared style
axes[1].legend(fontsize=7.4)                                        # compact legend
axes[2].plot(curve["threshold"], curve["cost"] / 1e4, lw=2.0, color=OKABE[4])   # cost curve
axes[2].axvline(best[0], ls="--", lw=1.2, color=OKABE[5])            # the optimum
style_axes(axes[2], "Expected cost versus threshold", "threshold", "成本 (万元)")   # shared style
annotated(axes[2], "最优阈值 %.2f" % best[0], "upper right")           # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_churn_threshold.png")                               # save the figure
print("done: churn models and the expected-loss threshold")          # closing log
