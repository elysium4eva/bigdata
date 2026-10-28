"""l12_04_psm.py -- propensity scores, nearest-neighbour matching and the balance table.

Every line carries an English comment. The script estimates the propensity score on a confounded   # continuation of the call
sample, matches each treated unit to its nearest control, reports the standardised differences   # continuation of the call
before and after matching, and draws the balance plot and the overlap check.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from sklearn.linear_model import LogisticRegression                 # the propensity model
from sklearn.neighbors import NearestNeighbors                      # the matching engine
from _common_l12 import (make_potential, save, dump, style_axes,    # shared helpers
                         annotated, OKABE)                              # continuation of the call

# ---------------------------------------------------------------- 1) the sample
df = make_potential(2000)                                          # the confounded sample
print("treated share: %.3f" % df["z"].mean())                        # report
X = df[["x"]].values                                               # the observed covariate(s)
z = df["z"].values                                                 # the treatment
y = df["y"].values                                                 # the observed outcome

# ---------------------------------------------------------------- 2) the propensity score
ps_model = LogisticRegression(max_iter=1000).fit(X, z)               # logistic regression on the covariates
ps = ps_model.predict_proba(X)[:, 1]                                # the fitted propensity scores
df["ps"] = ps.round(4)                                              # attach them
print("propensity range: %.3f-%.3f" % (ps.min(), ps.max()))          # report
# overlap: keep only units inside the common support
support = (ps > np.percentile(ps, 2)) & (ps < np.percentile(ps, 98))   # drop the extreme 2% tails
print("common support keeps %.3f of the sample" % support.mean())     # report

# ---------------------------------------------------------------- 3) nearest-neighbour matching
treated_idx = np.where((z == 1) & support)[0]                       # the treated units on support
control_idx = np.where((z == 0) & support)[0]                       # the controls on support
nn = NearestNeighbors(n_neighbors=1).fit(ps[control_idx].reshape(-1, 1))   # fit on the controls
dist, pos = nn.kneighbors(ps[treated_idx].reshape(-1, 1))             # find the nearest control
matched_control = control_idx[pos.ravel()]                          # the matched control indices
att = float(y[treated_idx].mean() - y[matched_control].mean())        # the matched ATT
print("matched ATT: %.3f | naive difference: %.3f" % (                  # report the result
    att, float(y[z == 1].mean() - y[z == 0].mean())))                 # report

# ---------------------------------------------------------------- 4) the balance table
def std_diff(a, b):                                                 # standardised difference
    """Return the standardised difference between two vectors."""    # short docstring
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)            # pooled standard deviation
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0   # the difference in sd units

before = std_diff(df.loc[z == 1, "x"].values, df.loc[z == 0, "x"].values)   # before matching
after = std_diff(df.loc[treated_idx, "x"].values, df.loc[matched_control, "x"].values)   # after matching
balance = pd.DataFrame([{"covariate": "x", "before": round(before, 3), "after": round(after, 3),   # assemble the frame
                         "reduction": round(1 - abs(after) / max(abs(before), 1e-9), 3)}])   # the table
dump(balance, "l12_balance.csv")                                    # persist
print(balance.to_string(index=False))                               # report
ps_balance = pd.DataFrame([{"covariate": "ps", "before": round(std_diff(ps[z == 1], ps[z == 0]), 3),   # assemble the frame
                            "after": round(std_diff(ps[treated_idx], ps[matched_control]), 3)}])   # the PS balance
dump(ps_balance, "l12_balance_ps.csv")                              # persist

# ---------------------------------------------------------------- 5) the figures
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                  # two panels
axes[0].hist(ps[z == 1], bins=30, alpha=0.6, color=OKABE[5], label="处理组")   # treated PS distribution
axes[0].hist(ps[z == 0], bins=30, alpha=0.6, color=OKABE[1], label="对照组")   # control PS distribution
axes[0].legend(fontsize=7.4)                                        # compact legend
style_axes(axes[0], "倾向得分的分布与共同支撑域", "倾向得分", "单位数")      # shared style
labels = ["协变量 x", "倾向得分"]                                     # the two rows
bx = np.arange(len(labels))                                          # positions
axes[1].barh(bx - 0.18, [abs(before), abs(ps_balance["before"][0])], height=0.34,   # draw the panel
             color=OKABE[1], alpha=0.92, label="匹配前")             # before
axes[1].barh(bx + 0.18, [abs(after), abs(ps_balance["after"][0])], height=0.34,   # draw the panel
             color=OKABE[2], alpha=0.92, label="匹配后")             # after
axes[1].axvline(0.1, ls="--", lw=1.0, color="#66717D")                # the 0.1 rule of thumb
axes[1].set_yticks(bx, labels)                                       # the labels
axes[1].legend(fontsize=7.4)                                        # compact legend
style_axes(axes[1], "标准化差异：匹配后应小于 0.1", "|标准化差异|", "")      # shared style
annotated(axes[1], "x: %.2f → %.2f" % (abs(before), abs(after)), "lower right")   # annotate
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_balance.png")                                        # save the balance figure
print("done: propensity scores, matching and balance")                 # closing log
