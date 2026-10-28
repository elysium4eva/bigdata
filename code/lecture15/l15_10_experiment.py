"""l15_10_experiment.py -- two-proportion A/B test, power curves and CUPED.

Every line carries an English comment. The script runs a two-proportion z-test on the experiment
data, computes a confidence interval and the relative lift, draws the power curve against the
minimum detectable effect, and shows how a pre-period covariate (CUPED) shrinks the variance.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from scipy import stats                                            # normal distribution and tests
from _common_l15 import (make_ab, RNG, save, dump, style_axes,     # shared helpers
                         annotated, OKABE)                              # shared plotting helpers

# ---------------------------------------------------------------- 1) the experiment
ab = make_ab(5000, 0.050, 0.006)                                   # 5,000 users per arm
dump(ab, "l15_ab.csv")                                             # persist
agg = ab.groupby("arm")["converted"].agg(["sum", "count"])          # conversions and users
agg["rate"] = agg["sum"] / agg["count"]                            # conversion rates
print(agg.round(4).to_string())                                    # report

# ---------------------------------------------------------------- 2) two-proportion z test
x_c, n_c = int(agg.loc["control", "sum"]), int(agg.loc["control", "count"])   # control counts
x_t, n_t = int(agg.loc["treatment", "sum"]), int(agg.loc["treatment", "count"])   # treatment counts
p_c, p_t = x_c / n_c, x_t / n_t                                    # the two rates
p_pool = (x_c + x_t) / (n_c + n_t)                                 # pooled rate under H0
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))      # standard error under H0
z = (p_t - p_c) / se_pool                                          # the z statistic
p_value = 2 * (1 - stats.norm.cdf(abs(z)))                         # two-sided p value
lift = (p_t - p_c) / p_c                                           # relative lift
se_diff = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)    # unpooled standard error
ci = (p_t - p_c - 1.96 * se_diff, p_t - p_c + 1.96 * se_diff)       # 95% interval on the difference
print("control %.4f | treatment %.4f | lift %.2f%%" % (p_c, p_t, lift * 100))   # report
print("z = %.2f | p = %.4f | 95%% CI on the difference: [%.4f, %.4f]" % (z, p_value, ci[0], ci[1]))   # report
result = pd.DataFrame([{"control_rate": round(p_c, 4), "treatment_rate": round(p_t, 4),   # the summary
                        "lift_pct": round(lift * 100, 2), "z": round(z, 2),   # more columns
                        "p_value": round(p_value, 4), "ci_low": round(ci[0], 4),   # the interval
                        "ci_high": round(ci[1], 4)}])              # the upper bound
dump(result, "l15_ab_result.csv")                                  # persist

# ---------------------------------------------------------------- 3) power and the minimum detectable effect
def power_two_prop(p1, p2, n):                                     # analytic power of a two-proportion test
    """Return the power of a two-sided 5% test with n users per arm."""   # short docstring
    p_bar = (p1 + p2) / 2                                          # average rate
    se0 = np.sqrt(2 * p_bar * (1 - p_bar) / n)                      # standard error under H0
    se1 = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)             # standard error under H1
    crit = stats.norm.ppf(0.975)                                    # the 5% critical value
    return float(stats.norm.cdf((abs(p2 - p1) - crit * se0) / se1))   # power

p0 = 0.05                                                          # baseline conversion
rows = []                                                          # one row per sample size
for n in [500, 1000, 2000, 5000, 10000, 20000, 50000]:              # candidate sample sizes
    mde = 0.006                                                    # a fixed relative-lift scenario
    rows.append({"n_per_arm": n, "power": round(power_two_prop(p0, p0 + mde, n), 4)})   # power
power_curve = pd.DataFrame(rows)                                   # assemble
dump(power_curve, "l15_power_curve.csv")                            # persist
print(power_curve.to_string(index=False))                          # report
def mde_for_power(target=0.8, p1=0.05, n=5000):                    # invert the power formula
    """Return the minimum detectable absolute effect for a target power."""   # short docstring
    lo, hi = 1e-5, 0.2                                             # search bounds
    for _ in range(60):                                            # bisection
        mid = (lo + hi) / 2                                        # midpoint
        if power_two_prop(p1, p1 + mid, n) < target:                # too small an effect
            lo = mid                                               # raise the lower bound
        else:                                                      # large enough
            hi = mid                                               # lower the upper bound
    return hi                                                      # the boundary
print("MDE at n=5000, power 80%%: %.4f (relative %.1f%%)" % (           # report the MDE
    mde_for_power(0.8, 0.05, 5000), 100 * mde_for_power(0.8, 0.05, 5000) / 0.05))   # report

# ---------------------------------------------------------------- 4) CUPED: variance reduction with a pre-period covariate
n_users = len(ab)                                                  # total users in the experiment
pre = RNG.normal(0, 1.0, n_users)                                  # a pre-period covariate
post = ab["converted"].values + 0.35 * pre + RNG.normal(0, 0.9, n_users)   # outcome with correlation
theta = float(np.cov(post, pre)[0, 1] / np.var(pre))               # the CUPED coefficient
post_adj = post - theta * (pre - pre.mean())                        # the adjusted outcome
var_before = float(np.var(post))                                   # variance before
var_after = float(np.var(post_adj))                                # variance after
print("CUPED theta %.3f | variance reduction %.1f%%" % (theta, 100 * (1 - var_after / var_before)))   # report
cuped = pd.DataFrame([{"theta": round(theta, 3), "var_before": round(var_before, 4),   # the summary
                       "var_after": round(var_after, 4),           # the adjusted variance
                       "reduction_pct": round(100 * (1 - var_after / var_before), 1)}])   # the gain
dump(cuped, "l15_cuped.csv")                                       # persist

# ---------------------------------------------------------------- 5) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                 # three panels
axes[0].bar(["对照组", "实验组"], [p_c * 100, p_t * 100], color=[OKABE[1], OKABE[2]], alpha=0.9)   # the two rates
axes[0].errorbar([0, 1], [p_c * 100, p_t * 100],                        # the two point estimates
                 yerr=[1.96 * np.sqrt(p_c * (1 - p_c) / n_c) * 100,     # their 95% intervals
                       1.96 * np.sqrt(p_t * (1 - p_t) / n_t) * 100], fmt="none", ecolor="#2B2B2B")   # intervals
style_axes(axes[0], "Conversion rate with 95% intervals", "", "转化率 (%)")   # shared style
for i, v in enumerate([p_c * 100, p_t * 100]):                      # annotate
    axes[0].text(i, v, "%.2f%%" % v, ha="center", va="bottom", fontsize=9)   # value
axes[1].plot(power_curve["n_per_arm"], power_curve["power"], "-o", ms=5, color=OKABE[5])   # power curve
axes[1].axhline(0.8, ls="--", lw=1.0, color="#66717D")               # the conventional 80%
axes[1].set_xscale("log")                                          # log scale for sample size
style_axes(axes[1], "Power versus sample size (MDE 0.6pp)", "n per arm (log)", "power")   # shared style
annotated(axes[1], "80% 需要约 1 万人/臂", "lower right")             # annotate
axes[2].hist(post, bins=40, alpha=0.6, color=OKABE[1], label="原始 (var %.2f)" % var_before)   # before
axes[2].hist(post_adj, bins=40, alpha=0.6, color=OKABE[2], label="CUPED (var %.2f)" % var_after)   # after
style_axes(axes[2], "CUPED shrinks the noise", "outcome", "count")   # shared style
axes[2].legend(fontsize=7.2)                                        # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_power_curve.png")                                   # save the figure
print("done: A/B test, power curve and CUPED")                      # closing log
