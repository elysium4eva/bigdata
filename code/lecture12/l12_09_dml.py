"""l12_09_dml.py -- double machine learning with cross-fitting.

Every line carries an English comment. The script shows why a naive ML plug-in is biased in the   # continuation of the call
presence of a confounder that the ML model can fit too well, then implements the partially linear   # continuation of the call
DML estimator with cross-fitting and compares the three estimates with the truth.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from sklearn.ensemble import GradientBoostingRegressor              # the nuisance learner
from sklearn.model_selection import KFold                          # the cross-fitting folds
from _common_l12 import (make_cate, RNG, save, dump, style_axes,     # shared helpers
                         OKABE, annotated)                              # continuation of the call

# ---------------------------------------------------------------- 1) a sample with a strong confounder
n = 4000                                                            # sample size
x1 = RNG.normal(0, 1, n)                                           # an effect modifier (and confounder)
x2 = RNG.normal(0, 1, n)                                           # a strong confounder
p_z = 1 / (1 + np.exp(-(2.0 * x2)))                                 # assignment depends strongly on x2
z = (RNG.random(n) < p_z).astype(int)                               # the treatment
theta = 2.0                                                        # the true treatment effect
y = 3.0 + 1.5 * x2 + theta * z + 0.5 * x1 + RNG.normal(0, 1.0, n)    # the outcome
print("treated share %.3f | true theta %.1f" % (z.mean(), theta))    # report

# ---------------------------------------------------------------- 2) the three estimators
# (a) naive difference in means
naive = float(y[z == 1].mean() - y[z == 0].mean())                  # ignores x2 entirely
# (b) logistic/linear regression adjustment
import statsmodels.api as sm                                       # the OLS machinery
ols = sm.OLS(y, sm.add_constant(np.column_stack([z, x2]))).fit()     # control for the confounder
adjusted = float(ols.params[1])                                     # the adjusted estimate
# (c) a naive ML plug-in: predict y with z as a feature
X = np.column_stack([z, x2, x1])                                    # features including the treatment
ml = GradientBoostingRegressor(random_state=42, n_estimators=200, max_depth=3).fit(X, y)   # the plug-in
X1 = np.column_stack([np.ones(n), x2, x1])                          # everyone treated
X0 = np.column_stack([np.zeros(n), x2, x1])                         # nobody treated
plugin = float(np.mean(ml.predict(X1) - ml.predict(X0)))             # the plug-in estimate
# (d) DML with cross-fitting
def dml_partial_linear(yv, dv, Xconf, folds=5, seed=42):            # the DML estimator
    """Return the partially linear DML estimate of the treatment effect."""   # short docstring
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)      # the cross-fitting folds
    res_y = np.zeros(len(yv))                                       # the residualised outcome
    res_d = np.zeros(len(dv))                                       # the residualised treatment
    for train, test in kf.split(Xconf):                             # loop over the folds
        my = GradientBoostingRegressor(random_state=seed, n_estimators=150, max_depth=3).fit(Xconf[train], yv[train])   # E[Y|X]
        md = GradientBoostingRegressor(random_state=seed, n_estimators=150, max_depth=3).fit(Xconf[train], dv[train])   # E[D|X]
        res_y[test] = yv[test] - my.predict(Xconf[test])             # the outcome residual
        res_d[test] = dv[test] - md.predict(Xconf[test])             # the treatment residual
    theta_hat = float((res_d @ res_y) / (res_d @ res_d))             # the residual-on-residual regression
    return theta_hat, res_y, res_d                                   # the estimate and the residuals

dml, res_y, res_d = dml_partial_linear(y, z.astype(float), np.column_stack([x1, x2]))   # run DML
print("naive %.3f | OLS adjusted %.3f | ML plug-in %.3f | DML %.3f (true %.1f)" % (   # report the result
    naive, adjusted, plugin, dml, theta))                            # report
table = pd.DataFrame([{"estimator": "朴素差值", "estimate": round(naive, 3)},   # naive
                      {"estimator": "OLS 调整", "estimate": round(adjusted, 3)},   # OLS
                      {"estimator": "ML plug-in", "estimate": round(plugin, 3)},   # plug-in
                      {"estimator": "DML (cross-fitting)", "estimate": round(dml, 3)},   # DML
                      {"estimator": "真值", "estimate": theta}])    # the truth
dump(table, "l12_dml.csv")                                          # persist

# ---------------------------------------------------------------- 3) the figure
rows = []                                                           # one row per estimator
for name, val in [("朴素", naive), ("OLS", adjusted), ("ML plug-in", plugin), ("DML", dml)]:   # loop
    rows.append({"estimator": name, "estimate": round(val, 3), "bias": round(val - theta, 3)})   # the bias
plot = pd.DataFrame(rows)                                           # assemble
fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.8))                  # two panels
axes[0].bar(plot["estimator"], plot["estimate"], color=[OKABE[1], OKABE[3], OKABE[5], OKABE[2]], alpha=0.92)   # estimates
axes[0].axhline(theta, ls="--", lw=1.2, color="#66717D", label="真值 %.1f" % theta)   # the truth
axes[0].legend(fontsize=7.4)                                        # compact legend
style_axes(axes[0], "四种估计与真值的距离", "", "估计值")                # shared style
for i, v in enumerate(plot["estimate"]):                            # annotate
    axes[0].text(i, v, "%.2f" % v, ha="center", va="bottom", fontsize=8.6)   # the value
axes[1].scatter(res_d, res_y, s=6, alpha=0.3, color=OKABE[4])        # the residual scatter
grid = np.linspace(res_d.min(), res_d.max(), 40)                     # a grid for the line
axes[1].plot(grid, dml * grid, lw=2.0, color=OKABE[5])                # the residual regression
style_axes(axes[1], "DML 的核心：残差对残差，斜率就是效应", "D 的残差", "Y 的残差")   # shared style
annotated(axes[1], "斜率 %.2f" % dml, "upper left")                   # annotate
plt.tight_layout()                                                   # tidy the layout
save(fig, "fig_dml.png")                                            # save the figure
print("done: double machine learning with cross-fitting")              # closing log
