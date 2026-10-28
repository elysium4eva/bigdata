"""l12_10_causal_forest.py -- a simplified causal forest for heterogeneous effects.

Every line carries an English comment. The script builds a sample whose treatment effect depends on   # continuation of the call
a covariate, estimates the conditional average treatment effect with a "honest" two-half forest   # continuation of the call
(a lightweight stand-in for the real algorithm), and draws the estimated CATE against the truth   # continuation of the call
plus the sorted CATE curve used for targeting.                          # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from sklearn.ensemble import RandomForestRegressor                  # the splitting engine
from sklearn.model_selection import KFold                           # the folds
from _common_l12 import (make_cate, save, dump, style_axes,          # shared helpers
                         annotated, OKABE)                              # continuation of the call

# ---------------------------------------------------------------- 1) the sample
df = make_cate(4000)                                                # 4,000 units, tau = 1 + 2*x1
X = df[["x1", "x2"]].values                                        # the covariates
z = df["z"].values.astype(float)                                    # the treatment
y = df["y"].values                                                 # the outcome
tau_true = df["tau_true"].values                                    # the true individual effect
print("n:", len(df), "| true ATE %.3f | true CATE range %.1f to %.1f" % (   # report the result
    tau_true.mean(), tau_true.min(), tau_true.max()))                # report

# ---------------------------------------------------------------- 2) the honest two-half forest
def honest_cate(Xv, zv, yv, folds=2, seed=7):                       # a lightweight honest forest
    """Return an estimate of tau(x) built from two independent halves."""   # short docstring
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)       # the halves
    tau_hat = np.zeros(len(yv))                                     # the estimate per unit
    for train, test in kf.split(Xv):                                # loop over the halves
        # fit the two outcome surfaces on the training half (T-learner inside the fold)
        m1 = RandomForestRegressor(random_state=seed, n_estimators=120, min_samples_leaf=40).fit(   # estimate the model
            Xv[train][zv[train] == 1], yv[train][zv[train] == 1])     # E[Y | X, Z=1]
        m0 = RandomForestRegressor(random_state=seed, n_estimators=120, min_samples_leaf=40).fit(   # estimate the model
            Xv[train][zv[train] == 0], yv[train][zv[train] == 0])     # E[Y | X, Z=0]
        tau_hat[test] = m1.predict(Xv[test]) - m0.predict(Xv[test])   # the effect on the held-out half
    return tau_hat                                                  # the CATE estimate

tau_hat = honest_cate(X, z, y)                                      # run it
print("estimated ATE %.3f (true %.3f)" % (tau_hat.mean(), tau_true.mean()))   # report
corr = float(np.corrcoef(tau_hat, tau_true)[0, 1])                   # how well the ranking matches
print("corr(estimated, true CATE): %.3f" % corr)                     # report
# a targeting exercise: treat the top 25% by estimated CATE
cut = np.quantile(tau_hat, 0.75)                                    # the 75th percentile
top = tau_hat >= cut                                                # the target group
gain_targeted = float(tau_true[top].mean())                          # the true gain if we target
gain_all = float(tau_true.mean())                                    # the gain if we treat everyone
print("targeting top 25%%: true gain %.2f vs %.2f for everyone" % (gain_targeted, gain_all))   # report
out = pd.DataFrame({"x1": X[:, 0].round(3), "x2": X[:, 1].round(3), "z": z.astype(int),   # assemble the frame
                    "tau_true": tau_true.round(3), "tau_hat": tau_hat.round(3)})   # the frame
dump(out, "l12_cate.csv")                                           # persist
summary = pd.DataFrame([{"targeting": "top 25%", "true_gain": round(gain_targeted, 3)},   # assemble the frame
                        {"targeting": "everyone", "true_gain": round(gain_all, 3)},   # continuation of the call
                        {"corr_tau": round(corr, 3)}])              # the summary
dump(summary, "l12_cate_summary.csv")                               # persist

# ---------------------------------------------------------------- 3) the figure
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                  # two panels
axes[0].scatter(tau_true, tau_hat, s=6, alpha=0.3, color=OKABE[4])    # estimate vs truth
lims = [float(min(tau_true.min(), tau_hat.min())), float(max(tau_true.max(), tau_hat.max()))]   # the axes range
axes[0].plot(lims, lims, ls="--", lw=1.0, color="#66717D")            # the 45 degree line
style_axes(axes[0], "估计的 CATE vs 真实 CATE（相关 %.2f）" % corr, "真实效应", "估计效应")   # shared style
order = np.argsort(tau_hat)[::-1]                                    # sort by the estimate
axes[1].plot(np.arange(len(order)) / len(order), tau_hat[order], lw=1.8, color=OKABE[5], label="估计")   # the estimated curve
axes[1].plot(np.arange(len(order)) / len(order), tau_true[order], lw=1.2, ls="--", color=OKABE[1], label="真实")   # the true curve
axes[1].axvline(0.25, ls=":", lw=1.2, color="#66717D")                # the targeting cut
axes[1].legend(fontsize=7.4)                                        # compact legend
style_axes(axes[1], "按估计效应排序：谁最该被处理", "人口比例（从高到低）", "效应")   # shared style
annotated(axes[1], "前 25%% 的真实增益 %.2f" % gain_targeted, "upper right")   # annotate
plt.tight_layout()                                                   # tidy the layout
save(fig, "fig_cate.png")                                           # save the figure
print("done: a simplified causal forest and CATE-based targeting")     # closing log
