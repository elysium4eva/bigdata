"""l12_03_power.py -- power, the minimum detectable effect and multiple testing.

Every line carries an English comment. The script computes analytic power for a two-proportion   # continuation of the call
test, verifies it by simulation, draws the power curve against sample size, and shows how many   # continuation of the call
false positives appear when twenty null metrics are tested at the 5% level.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from scipy import stats                                            # the normal distribution
from _common_l12 import (make_experiment, RNG, save, dump,          # shared helpers
                         style_axes, annotated, OKABE)                  # continuation of the call

# ---------------------------------------------------------------- 1) the experiment
exp = make_experiment(5000, 0.10, 0.012)                            # two arms of 5,000
dump(exp, "l12_experiment.csv")                                    # persist
agg = exp.groupby("arm")["converted"].agg(["sum", "count"])          # conversions and users
p_c = float(agg.loc["control", "sum"] / agg.loc["control", "count"])   # control rate
p_t = float(agg.loc["treatment", "sum"] / agg.loc["treatment", "count"])   # treatment rate
n_arm = int(agg["count"].iloc[0])                                   # users per arm
se = np.sqrt(p_c * (1 - p_c) / n_arm + p_t * (1 - p_t) / n_arm)       # standard error of the difference
z = (p_t - p_c) / se                                                # the test statistic
p_value = 2 * (1 - stats.norm.cdf(abs(z)))                          # the two-sided p value
print("control %.4f | treatment %.4f | diff %.4f | z %.2f | p %.4f" % (   # report the result
    p_c, p_t, p_t - p_c, z, p_value))                                # report

# ---------------------------------------------------------------- 2) analytic power
def power_two_prop(p1, p2, n, alpha=0.05):                          # analytic power
    """Return the power of a two-sided two-proportion test."""       # short docstring
    p_bar = (p1 + p2) / 2                                          # the average rate
    se0 = np.sqrt(2 * p_bar * (1 - p_bar) / n)                       # standard error under H0
    se1 = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)             # standard error under H1
    crit = stats.norm.ppf(1 - alpha / 2)                            # the critical value
    return float(stats.norm.cdf((abs(p2 - p1) - crit * se0) / se1))   # power

rows = []                                                           # one row per sample size
for n in [500, 1000, 2000, 5000, 10000, 20000, 50000]:               # candidate sample sizes
    rows.append({"n_per_arm": n, "power": round(power_two_prop(0.10, 0.112, n), 4)})   # power at 1.2pp lift
curve = pd.DataFrame(rows)                                          # assemble
dump(curve, "l12_power_curve.csv")                                  # persist
print(curve.to_string(index=False))                                 # report

# ---------------------------------------------------------------- 3) verify by simulation
def simulate_power(p1, p2, n, reps=400):                            # simulation check
    """Return the rejection rate when the alternative is true."""    # short docstring
    rejections = 0                                                  # the counter
    for _ in range(reps):                                           # loop over the replications
        a = RNG.random(n) < p1                                       # the control arm
        b = RNG.random(n) < p2                                       # the treatment arm
        pa, pb = a.mean(), b.mean()                                  # the two rates
        pooled = (a.sum() + b.sum()) / (2 * n)                        # pooled rate
        se_p = np.sqrt(pooled * (1 - pooled) * (2 / n))              # standard error
        if se_p > 0 and abs(pb - pa) / se_p > 1.96:                  # the 5% two-sided rule
            rejections += 1                                         # count the rejection
    return rejections / reps                                        # the empirical power

sim_power = simulate_power(0.10, 0.112, 5000)                        # check the 5,000 case
print("analytic power at n=5000: %.3f | simulated %.3f" % (             # report the result
    power_two_prop(0.10, 0.112, 5000), sim_power))                    # report

# ---------------------------------------------------------------- 4) multiple testing
n_metrics = 20                                                       # twenty null metrics
false_positives = []                                                # one trial of the whole family
for _ in range(2000):                                               # two thousand "experiments"
    pvals = RNG.random(n_metrics)                                   # all null: uniform p values
    false_positives.append(int((pvals < 0.05).sum() > 0))            # did any of them look significant
fw_any = float(np.mean(false_positives))                             # family-wise false positive rate
print("P(at least one false positive among %d null metrics): %.3f" % (n_metrics, fw_any))   # report
bonf_any = []                                                       # after Bonferroni
for _ in range(2000):                                               # repeat the simulation
    pvals = RNG.random(n_metrics)                                   # null p values
    bonf_any.append(int((pvals < 0.05 / n_metrics).sum() > 0))        # the adjusted threshold
print("after Bonferroni: %.3f" % float(np.mean(bonf_any)))            # report
dump(pd.DataFrame([{"family_size": n_metrics, "fw_false_positive": round(fw_any, 3),   # assemble the frame
                    "bonferroni_false_positive": round(float(np.mean(bonf_any)), 3)}]),   # continuation of the call
     "l12_multiple_testing.csv")                                    # persist

# ---------------------------------------------------------------- 5) the figures
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                  # two panels
axes[0].plot(curve["n_per_arm"], curve["power"], "-o", ms=5, color=OKABE[5])   # the power curve
axes[0].axhline(0.8, ls="--", lw=1.0, color="#66717D")               # the 80% convention
axes[0].set_xscale("log")                                          # log scale reads better
style_axes(axes[0], "功效曲线：1.2pp 的提升需要多少样本", "每臂样本量（对数）", "功效")   # shared style
annotated(axes[0], "80% 需要约 1 万人/臂", "lower right")              # annotate
axes[1].bar(["不校正", "Bonferroni"], [fw_any, float(np.mean(bonf_any))],   # draw the panel
            color=[OKABE[1], OKABE[2]], alpha=0.92)                  # the two rates
axes[1].axhline(0.05, ls="--", lw=1.0, color="#66717D")              # the nominal 5%
style_axes(axes[1], "20 个无效指标里出现假阳性的概率", "", "概率")        # shared style
for i, v in enumerate([fw_any, float(np.mean(bonf_any))]):           # annotate
    axes[1].text(i, v, "%.2f" % v, ha="center", va="bottom", fontsize=9)   # value
plt.tight_layout()                                                   # tidy the layout
save(fig, "fig_power_curve.png")                                    # save the power figure

fig, ax = plt.subplots(figsize=(8.6, 3.4))                           # one panel
counts = np.zeros(n_metrics)                                        # the count of "significant" metrics
for _ in range(2000):                                               # repeat the simulation
    counts += (RNG.random(n_metrics) < 0.05)                         # count false positives per metric
ax.bar(range(1, n_metrics + 1), counts / 2000, color=OKABE[4], alpha=0.9)   # the per-metric rate
ax.axhline(0.05, ls="--", lw=1.2, color=OKABE[5])                    # the nominal level
style_axes(ax, "每个无效指标的假阳性率都接近 5%：所以 20 个指标几乎必然有假阳性", "指标编号", "假阳性率")   # shared style
annotated(ax, "至少一个假阳性：%.0f%%" % (100 * fw_any), "upper right")   # annotate
plt.tight_layout()                                                   # tidy the layout
save(fig, "fig_multiple_testing.png")                               # save the multiple-testing figure
print("done: power, MDE and multiple testing")                         # closing log
