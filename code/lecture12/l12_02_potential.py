"""l12_02_potential.py -- potential outcomes, selection bias and the perfect doctor.

Every line carries an English comment. The script builds a confounded sample with two potential   # continuation of the call
outcomes, shows how the naive difference in means decomposes into the true ATE plus a selection   # continuation of the call
bias, draws the assignment mechanism, and reconstructs the "perfect doctor" table that breaks the   # continuation of the call
naive estimator.                                                        # continuation of the call
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import statsmodels.formula.api as smf                              # OLS formulas
from _common_l12 import (make_potential, save, dump, style_axes,   # shared helpers
                         annotated, OKABE)                              # continuation of the call

# ---------------------------------------------------------------- 1) the sample
df = make_potential(2000)                                          # 2,000 units with two potential outcomes
dump(df, "l12_potential.csv")                                      # persist
ate_true = float((df["y1"] - df["y0"]).mean())                      # the true average treatment effect
print("true ATE: %.3f" % ate_true)                                   # report

# ---------------------------------------------------------------- 2) the naive estimator and its bias
naive = float(df.loc[df["z"] == 1, "y"].mean() - df.loc[df["z"] == 0, "y"].mean())   # difference in means
treated_y0 = float(df.loc[df["z"] == 1, "y0"].mean())                # the treated group's counterfactual
control_y0 = float(df.loc[df["z"] == 0, "y0"].mean())                # the control group's baseline
selection_bias = treated_y0 - control_y0                            # the selection bias
print("naive estimate: %.3f | selection bias: %.3f | ATE + bias: %.3f" % (   # report the result
    naive, selection_bias, ate_true + selection_bias))               # report
check = smf.ols("y ~ z", data=df).fit()                             # the regression version
print("regression on z alone: %.3f" % float(check.params["z"]))       # report
controlled = smf.ols("y ~ z + x + u", data=df).fit()                 # controlling for the confounders
print("with x and u controlled: %.3f" % float(controlled.params["z"]))   # report
decomp = pd.DataFrame([{"estimate": "真实 ATE", "value": round(ate_true, 3)},   # the truth
                       {"estimate": "朴素差值", "value": round(naive, 3)},        # naive
                       {"estimate": "选择偏差", "value": round(selection_bias, 3)},   # the bias
                       {"estimate": "控制混杂后", "value": round(float(controlled.params["z"]), 3)}])   # controlled
dump(decomp, "l12_bias_decomp.csv")                                 # persist
print(decomp.to_string(index=False))                                # report

# ---------------------------------------------------------------- 3) the perfect doctor table
# A small 8-patient table where the doctor assigns treatment on the GAIN, so the naive estimator is wrong.
patients = []                                                       # one row per patient
gains = [1, 0, 1, 1, 1, 1, 2, 1]                                    # the individual gains (Y1 - Y0)
baselines = [13, 6, 4, 5, 6, 6, 8, 9]                               # Y(0) for each patient
for i, (y0, gain) in enumerate(zip(baselines, gains)):               # loop over the patients
    y1 = y0 + gain                                                  # Y(1)
    # the doctor treats patients whose gain is high (this is the "perfect doctor" assignment)
    z = int(gain >= 1 and i not in (1, 2))                          # treated unless the gain is small
    observed = y1 if z == 1 else y0                                  # the observed outcome
    patients.append({"patient": i + 1, "y0": y0, "y1": y1, "gain": gain,   # the record
                     "z": z, "observed": observed})                 # the observed data
doctor = pd.DataFrame(patients)                                     # assemble
ate_doctor = float(doctor["gain"].mean())                           # the true ATE in this table
naive_doctor = float(doctor.loc[doctor["z"] == 1, "observed"].mean()    # continuation of the call
                     - doctor.loc[doctor["z"] == 0, "observed"].mean())   # the naive estimate
print("perfect doctor: true ATE %.2f | naive %.2f" % (ate_doctor, naive_doctor))   # report
dump(doctor, "l12_perfect_doctor.csv")                              # persist

# ---------------------------------------------------------------- 4) the figures
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                  # two panels
for z, colour, label in ((1, OKABE[5], "处理组"), (0, OKABE[1], "对照组")):   # loop over the arms
    sub = df[df["z"] == z]                                         # the arm's rows
    axes[0].scatter(sub["x"], sub["y"], s=7, alpha=0.35, color=colour, label=label)   # the points
    grid = np.linspace(df["x"].min(), df["x"].max(), 40)             # a grid for the fit
    fit = smf.ols("y ~ x", data=sub).fit()                          # a simple fit within the arm
    axes[0].plot(grid, fit.params["Intercept"] + fit.params["x"] * grid, lw=1.8, color=colour)   # the fit
axes[0].legend(fontsize=7.4)                                        # compact legend
style_axes(axes[0], "混杂：处理组在高 x 处更密集", "x（混杂变量）", "y")   # shared style
axes[1].bar(["真实 ATE", "朴素差值", "选择偏差", "控制后"], [ate_true, naive, selection_bias,   # draw the panel
             float(controlled.params["z"])], color=[OKABE[2], OKABE[5], OKABE[1], OKABE[4]], alpha=0.92)   # the decomposition
axes[1].axhline(0, color="#2B2B2B", lw=0.9)                         # zero line
style_axes(axes[1], "朴素差值 = ATE + 选择偏差", "", "数值")            # shared style
for i, v in enumerate([ate_true, naive, selection_bias, float(controlled.params["z"])]):   # annotate
    axes[1].text(i, v, "%.2f" % v, ha="center", va="bottom" if v >= 0 else "top", fontsize=8.4)   # value
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_bias_decomp.png")                                   # save the decomposition figure

fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                  # two panels
axes[0].axis("off")                                                # the table needs no axes
cell = [[str(int(r.y0)), str(int(r.y1)), str(int(r.gain)), str(int(r.z)), str(int(r.observed))]   # continuation of the call
        for r in doctor.itertuples()]                                # the table body
table = axes[0].table(cellText=cell, colLabels=["Y(0)", "Y(1)", "增益", "Z", "观测值"],   # draw it
                      loc="upper center", cellLoc="center")          # centred
table.auto_set_font_size(False)                                     # fixed font size
table.set_fontsize(8.4)                                            # readable type
table.scale(1.0, 1.5)                                              # roomier rows
axes[0].set_title("完美医生：8 位病人的潜在结果")                       # panel title
axes[1].bar([0, 1], [float(doctor.loc[doctor["z"] == 1, "observed"].mean()),   # draw the panel
                     float(doctor.loc[doctor["z"] == 0, "observed"].mean())],   # continuation of the call
            color=[OKABE[5], OKABE[1]], alpha=0.92)                  # the two means
axes[1].axhline(doctor["y0"].mean(), ls=":", lw=1.0, color="#66717D")   # the "no effect" level
style_axes(axes[1], "差值 %.1f，但真 ATE 只有 %.1f" % (naive_doctor, ate_doctor), "组别", "平均观测值")   # shared style
axes[1].set_xticks([0, 1], ["处理组", "对照组"])                       # labels
annotated(axes[1], "分配依赖潜在结果 → 差值失效", "upper right")          # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_perfect_doctor.png")                                # save the doctor figure

# ---------------------------------------------------------------- 5) the assignment-mechanism figure
fig, ax = plt.subplots(figsize=(10.2, 3.4))                         # one wide panel
ax.axis("off")                                                     # sketch only
cases = [("完全随机", "Z ⟂ (Y(0), Y(1))", "有效", 0.18),             # random assignment
         ("依可观测选择", "Z ⟂ Y(z) | X", "可用 X 调整", 0.40),         # selection on observables
         ("依潜在结果选择", "Z 依赖 (Y(0), Y(1))", "无效（完美医生）", 0.62),   # selection on potential outcomes
         ("依增益选择", "Z 依赖 Y(1) − Y(0)", "无效", 0.84)]              # selection on gains
for name, cond, verdict, ypos in cases:                            # draw the four cases
    ax.text(0.02, ypos, name, fontsize=10, fontweight="bold")       # the case name
    ax.text(0.26, ypos, cond, fontsize=9.5, color=OKABE[4])          # the assumption
    ax.text(0.62, ypos, verdict, fontsize=9.5, color=OKABE[5])       # the verdict
ax.set_title("分配机制决定识别：四种情形")                            # the title
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_assignment.png")                                    # save the assignment figure
print("done: potential outcomes, selection bias and the perfect doctor")   # closing log
