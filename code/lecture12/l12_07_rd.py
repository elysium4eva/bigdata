"""l12_07_rd.py -- a sharp regression-discontinuity design and bandwidth sensitivity.

Every line carries an English comment. The script estimates the jump at the cutoff with local   # continuation of the call
linear regressions on several bandwidths, draws the binned means and the two fitted lines, and   # continuation of the call
shows how the estimate and its precision change with the bandwidth.     # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import statsmodels.formula.api as smf                              # OLS formulas
from _common_l12 import (make_rd, save, dump, style_axes,          # shared helpers
                         annotated, OKABE)                              # continuation of the call

# ---------------------------------------------------------------- 1) the sample
df = make_rd(4000, 0.0, 4.0)                                       # 4,000 units, cutoff at zero
dump(df, "l12_rd.csv")                                              # persist
print("n:", len(df), "| treated share %.3f" % df["treated"].mean())   # report

# ---------------------------------------------------------------- 2) local linear estimates
def rd_estimate(bandwidth):                                        # one bandwidth
    """Return the RD estimate and its standard error within a bandwidth."""   # short docstring
    sub = df[df["running"].abs() <= bandwidth].copy()                # keep units near the cutoff
    sub["above"] = (sub["running"] >= 0).astype(int)                 # the post-cutoff side
    sub["r_centered"] = sub["running"] - 0.0                         # centre the running variable
    model = smf.ols("y ~ above + r_centered + above:r_centered", data=sub).fit()   # allow different slopes
    return float(model.params["above"]), float(model.bse["above"]), len(sub)   # the jump and its se

rows = []                                                           # one row per bandwidth
for bw in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:                          # from very local to global
    est, se, n_sub = rd_estimate(bw)                                # estimate within the bandwidth
    rows.append({"bandwidth": bw, "jump": round(est, 3), "se": round(se, 3), "n": n_sub})   # the row
bandwidths = pd.DataFrame(rows)                                     # assemble
dump(bandwidths, "l12_rd_bandwidth.csv")                            # persist
print(bandwidths.to_string(index=False))                            # report

# ---------------------------------------------------------------- 3) the figure
bins = np.linspace(-1, 1, 41)                                       # bins of the running variable
df["bin"] = pd.cut(df["running"], bins, labels=False)                # assign each unit to a bin
binned = df.groupby("bin").agg(running=("running", "mean"), y=("y", "mean"), n=("y", "size")).dropna()   # binned means
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                  # two panels
for side, colour in ((-1, OKABE[1]), (1, OKABE[5])):                 # the two sides of the cutoff
    sub = binned[np.sign(binned["running"]) == side]                 # the bins on this side
    axes[0].scatter(sub["running"], sub["y"], s=18, color=colour)     # the binned means
    fit = smf.ols("y ~ running", data=df[(np.sign(df["running"]) == side) &   # estimate the model
                                         (df["running"].abs() <= 0.5)]).fit()   # local linear fit
    grid = np.linspace(-0.5, 0.5, 40)                                # a grid near the cutoff
    axes[0].plot(grid, fit.params["Intercept"] + fit.params["running"] * grid, lw=1.8, color=colour)   # the fit
axes[0].axvline(0, ls="--", lw=1.0, color="#66717D")                  # the cutoff
style_axes(axes[0], "断点两侧的跳跃就是因果效应（带宽 0.5）", "运行变量（距门槛）", "y")   # shared style
est_bw, se_bw, _ = rd_estimate(0.5)                                  # the headline estimate
annotated(axes[0], "跳跃 %.2f (se %.2f)" % (est_bw, se_bw), "upper left")   # annotate
axes[1].errorbar(bandwidths["bandwidth"], bandwidths["jump"], yerr=1.96 * bandwidths["se"],   # draw the panel
                 fmt="-o", ms=5, color=OKABE[4], ecolor="#2B2B2B", capsize=3)   # sensitivity
axes[1].axhline(4.0, ls="--", lw=1.0, color="#66717D", label="真值 4.0")   # the truth
axes[1].legend(fontsize=7.4)                                         # compact legend
style_axes(axes[1], "带宽敏感性：越宽越偏、越窄越不稳", "带宽", "断点估计")   # shared style
plt.tight_layout()                                                   # tidy the layout
save(fig, "fig_rd.png")                                             # save the figure
print("done: sharp RD and bandwidth sensitivity")                      # closing log
