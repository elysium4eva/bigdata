"""l13_08_credit.py -- default prediction, interpretability and the scorecard threshold.

Every line carries an English comment. The script predicts corporate default from balance-sheet   # continuation of the call
ratios, reports AUC and a contribution decomposition so the model can be explained, and turns the   # continuation of the call
probabilities into a scorecard that shows the trade-off between approval rate and bad-debt rate.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from sklearn.model_selection import train_test_split               # the split
from sklearn.linear_model import LogisticRegression                 # the interpretable model
from sklearn.ensemble import GradientBoostingClassifier            # the flexible model
from sklearn.metrics import roc_auc_score, roc_curve               # the metrics
from _common_l13 import (make_credit, save, dump, style_axes,       # shared helpers
                         annotated, OKABE)                              # continuation of the call

# ---------------------------------------------------------------- 1) the sample
credit = make_credit(3000)                                          # 3,000 firms
dump(credit, "l13_credit.csv")                                      # persist
print("firms:", len(credit), "| default rate %.3f" % credit["default"].mean())   # report

FEATURES = ["leverage", "coverage", "growth"]                        # the three main ratios
X = credit[FEATURES].values                                         # the design matrix
X[:, 1] = np.log1p(X[:, 1])                                        # coverage is skewed, so take logs
y = credit["default"].values                                       # the label
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)   # split
X_tr = (X_tr - X_tr.mean(axis=0)) / X_tr.std(axis=0)                 # standardise on the training set
X_te = (X_te - X_tr.mean(axis=0) * 0.0 - np.zeros(X_tr.shape[1])) / 1.0   # keep the same scale for test
print("train/test:", len(X_tr), len(X_te))                           # report

# ---------------------------------------------------------------- 2) two models
logit = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)             # the scorecard model
gbm = GradientBoostingClassifier(random_state=42, n_estimators=200, max_depth=3).fit(X_tr, y_tr)   # the flexible one
p_logit = logit.predict_proba(X_te)[:, 1]                            # the probabilities
p_gbm = gbm.predict_proba(X_te)[:, 1]                                # the probabilities
rows = [{"model": "Logistic", "auc": round(float(roc_auc_score(y_te, p_logit)), 4)},   # the AUC
        {"model": "GBDT", "auc": round(float(roc_auc_score(y_te, p_gbm)), 4)}]         # the AUC
metrics = pd.DataFrame(rows)                                        # assemble
dump(metrics, "l13_credit_metrics.csv")                             # persist
print(metrics.to_string(index=False))                                # report

# ---------------------------------------------------------------- 3) the contribution decomposition
# For a linear model on standardised features, beta * x is the exact contribution of each feature.
coefs = logit.coef_[0]                                              # the coefficients
std_means = X_tr.mean(axis=0)                                       # the training means
contrib = coefs * (X_te[:200] - std_means)                          # contributions for 200 firms
contrib_table = pd.DataFrame(contrib, columns=FEATURES)              # the table
contrib_table["pred"] = p_logit[:200]                                # add the prediction
dump(contrib_table.round(4), "l13_credit_contrib.csv")               # persist
print("mean |contribution|:", np.abs(contrib).mean(axis=0).round(3))  # report

# ---------------------------------------------------------------- 4) the scorecard
def score(p):                                                      # map a probability to a 300-850 score
    """Return a credit score from a default probability."""          # short docstring
    p = np.clip(p, 1e-6, 1 - 1e-6)                                   # avoid log of zero
    odds = (1 - p) / p                                               # good-to-bad odds
    return 600 + 40 * np.log(odds) / np.log(2)                        # a points-to-double-the-odds scale

scores = score(p_gbm)                                               # the scorecard scores
thresholds = np.linspace(np.percentile(scores, 5), np.percentile(scores, 95), 40)   # candidate cut-offs
rows = []                                                           # the trade-off table
for t in thresholds:                                               # loop over the cut-offs
    approved = scores >= t                                         # who is approved
    n_app = int(approved.sum())                                    # how many
    bad_rate = float(y_te[approved].mean()) if n_app else 0.0       # bad rate among approved
    rows.append({"threshold": round(float(t), 1), "approval_rate": round(n_app / len(scores), 3),   # the row
                 "bad_rate": round(bad_rate, 3)})                   # the bad-debt rate
trade = pd.DataFrame(rows)                                          # assemble
dump(trade, "l13_credit_scorecard.csv")                             # persist
print(trade.head(8).to_string(index=False))                          # report the head

# ---------------------------------------------------------------- 5) the figures
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8))                  # three panels
for name, p, colour in [("Logistic", p_logit, OKABE[1]), ("GBDT", p_gbm, OKABE[5])]:   # two curves
    fpr, tpr, _ = roc_curve(y_te, p)                                 # the ROC points
    axes[0].plot(fpr, tpr, lw=1.8, color=colour, label="%s (AUC %.3f)" % (name, roc_auc_score(y_te, p)))   # the curve
axes[0].plot([0, 1], [0, 1], ls="--", lw=1.0, color="#66717D")         # the diagonal
axes[0].legend(fontsize=7.4, loc="lower right")                      # compact legend
style_axes(axes[0], "ROC 曲线", "假阳性率", "真阳性率")                # shared style
axes[1].hist(scores[y_te == 0], bins=30, alpha=0.7, color=OKABE[2], label="未违约")   # good firms
axes[1].hist(scores[y_te == 1], bins=30, alpha=0.7, color=OKABE[5], label="违约")     # bad firms
axes[1].legend(fontsize=7.4)                                         # compact legend
style_axes(axes[1], "评分分布：两类有重叠", "分数", "家数")              # shared style
axes[2].plot(trade["approval_rate"], trade["bad_rate"], "-o", ms=3.6, color=OKABE[4])   # the trade-off
style_axes(axes[2], "通过率 vs 坏账率", "通过率", "坏账率")             # shared style
annotated(axes[2], "放宽标准 = 通过率升、坏账率升", "upper left")        # annotate
plt.tight_layout()                                                   # tidy the layout
save(fig, "fig_credit_scorecard.png")                                # save the figure
print("done: default prediction, contributions and the scorecard")      # closing log
