"""l12_08_late.py -- non-compliance, the four compliance types and LATE.

Every line carries an English comment. The script builds an experiment with one-sided   # continuation of the call
non-compliance, tabulates the four compliance types, estimates LATE by the Wald ratio (IV with   # continuation of the call
assignment as the instrument), and draws the collider DAG that motivates the folate example.   # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import statsmodels.api as sm                                       # OLS machinery
from _common_l12 import (make_compliance, save, dump, style_axes,   # shared helpers
                         annotated, OKABE)                              # continuation of the call

# ---------------------------------------------------------------- 1) the sample
df = make_compliance(4000)                                         # 4,000 units with one-sided non-compliance
dump(df, "l12_compliance.csv")                                     # persist
types = df["type"].value_counts()                                   # how many of each type
share = (types / len(df)).round(3)                                  # their shares
print(share.to_string())                                           # report

# ---------------------------------------------------------------- 2) intention-to-treat and LATE
itt = float(df.loc[df["z"] == 1, "y"].mean() - df.loc[df["z"] == 0, "y"].mean())   # ITT
first = float(df.loc[df["z"] == 1, "d"].mean() - df.loc[df["z"] == 0, "d"].mean())   # compliance rate
late = itt / first if first != 0 else np.nan                        # the Wald ratio
naive = float(df.loc[df["d"] == 1, "y"].mean() - df.loc[df["d"] == 0, "y"].mean())   # per-protocol (biased)
print("ITT %.3f | compliance rate %.3f | LATE %.3f | naive %.3f (true 2.0)" % (   # report the result
    itt, first, late, naive))                                       # report
table = pd.DataFrame([{"quantity": "ITT", "value": round(itt, 3)},   # the four quantities
                      {"quantity": "取用率 take-up", "value": round(first, 3)},   # continuation of the call
                      {"quantity": "LATE (Wald)", "value": round(late, 3)},   # continuation of the call
                      {"quantity": "朴素 per-protocol", "value": round(naive, 3)},   # continuation of the call
                      {"quantity": "真值", "value": 2.0}])           # the truth
dump(table, "l12_late.csv")                                        # persist
# the same thing with 2SLS, as a cross-check
z = sm.add_constant(df["z"].values)                                # instrument
d_hat = sm.OLS(df["d"].values, z).fit().fittedvalues                # first stage
second = sm.OLS(df["y"].values, sm.add_constant(d_hat)).fit()        # second stage
print("2SLS check: %.3f" % float(second.params[1]))                  # report

# ---------------------------------------------------------------- 3) the figures
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                  # two panels
labels = list(share.index)                                          # the three observed types
axes[0].bar(labels, share.values, color=[OKABE[1], OKABE[5], OKABE[2]], alpha=0.92)   # the shares
style_axes(axes[0], "三类人：从不接受 / 始终接受 / 顺从者", "", "占比")    # shared style
for i, v in enumerate(share.values):                                # annotate
    axes[0].text(i, v, "%.2f" % v, ha="center", va="bottom", fontsize=9)   # the value
axes[1].bar(["ITT", "LATE", "朴素", "真值"], [itt, late, naive, 2.0],       # draw the panel
            color=[OKABE[1], OKABE[2], OKABE[5], OKABE[7]], alpha=0.92)   # the four quantities
style_axes(axes[1], "分母不同，结论就不同", "", "效应")                  # shared style
for i, v in enumerate([itt, late, naive, 2.0]):                      # annotate
    axes[1].text(i, v, "%.2f" % v, ha="center", va="bottom", fontsize=9)   # the value
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_compliance.png")                                     # save the compliance figure

# ---------------------------------------------------------------- 4) the collider DAG
fig, ax = plt.subplots(figsize=(8.2, 3.6))                          # one panel
ax.axis("off")                                                     # sketch only
ax.annotate("", xy=(0.30, 0.5), xytext=(0.10, 0.5), arrowprops=dict(arrowstyle="->", lw=1.6, color=OKABE[5]))   # Z -> C
ax.annotate("", xy=(0.70, 0.5), xytext=(0.50, 0.5), arrowprops=dict(arrowstyle="->", lw=1.6, color=OKABE[1]))   # Y -> C
ax.text(0.06, 0.55, "处理 Z", fontsize=10)                           # the treatment
ax.text(0.56, 0.55, "结果 Y", fontsize=10)                           # the outcome
ax.text(0.44, 0.42, "共同结果 C（存活）", fontsize=10, ha="center")    # the collider
ax.text(0.10, 0.18, "条件化在 C 上（只看活产）会造出 Z 与 Y 的假关联", fontsize=9.5, color="#8a0c3c")   # the warning
ax.set_title("对撞点：C 是 Z 与 Y 的共同结果")                        # the title
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_collider.png")                                       # save the collider figure
print("done: non-compliance, LATE and the collider")                   # closing log
