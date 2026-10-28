"""l15_02_conjoint.py -- conjoint design, part-worth estimation and attribute importance.

Every line carries an English comment. The script builds a balanced profile design for four
new-energy-vehicle attributes, simulates 300 respondents' ratings, estimates part-worths with
effect coding, and draws two slides: the design table and the utility/importance chart.
"""
import itertools                                                   # enumerate the full factorial
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
import statsmodels.formula.api as smf                              # OLS with categorical predictors
from _common_l15 import (ATTRS, TRUE_UTIL, make_conjoint_profiles,   # shared helpers
                         make_conjoint_responses, save, dump,           # shared generators
                         style_axes, annotated, OKABE)                  # shared plotting helpers

# ---------------------------------------------------------------- 1) the design
design = make_conjoint_profiles(12)                                 # twelve balanced profiles
dump(design, "l15_conjoint_design.csv")                             # persist the design
full = np.prod([len(v) for v in ATTRS.values()])                     # the full factorial size
print("full factorial:", full, "| balanced design:", len(design))    # report the reduction
responses = make_conjoint_responses(design, 300)                    # 300 respondents x 12 profiles
dump(responses, "l15_conjoint_responses.csv")                       # persist the responses
print("responses:", len(responses))                                  # report

# ---------------------------------------------------------------- 2) estimate part-worths
merged = responses.merge(design, on="profile")                       # attach the attribute levels
# effect coding: one dummy per level, dropping the last level of each attribute
for attr in ATTRS:                                                  # loop over the attributes
    levels = ATTRS[attr]                                            # its levels
    for lv in levels[:-1]:                                          # all but the last level
        col = "d_%s_%s" % (attr, lv)                                # dummy column name
        merged[col] = (merged[attr] == lv).astype(float)            # 1 for this level
        mask_last = (merged[attr] == levels[-1]).astype(float)      # the reference level
        merged[col] = merged[col] - mask_last                        # effect coding: -1 on the reference
formula = "rating ~ " + " + ".join([c for c in merged.columns if c.startswith("d_")])   # the model formula
model = smf.ols(formula, data=merged).fit()                          # OLS on the stacked data
print("R2: %.3f | adj R2: %.3f" % (model.rsquared, model.rsquared_adj))   # report

# ---------------------------------------------------------------- 3) part-worths and importance
rows = []                                                          # one row per level
for attr in ATTRS:                                                  # loop over attributes
    levels = ATTRS[attr]                                            # its levels
    estimated = {}                                                 # level -> utility
    for lv in levels[:-1]:                                          # the coded levels
        estimated[lv] = float(model.params["d_%s_%s" % (attr, lv)])   # their coefficients
    estimated[levels[-1]] = -float(sum(estimated.values()))          # the reference level closes the sum to zero
    spread = max(estimated.values()) - min(estimated.values())       # the utility range
    for lv, u in estimated.items():                                 # write one row per level
        rows.append({"attribute": attr, "level": lv, "part_worth": round(u, 3),   # the record
                     "range": round(spread, 3)})                    # attribute range
utilities = pd.DataFrame(rows)                                      # assemble the table
ranges = utilities.groupby("attribute")["range"].first()             # the range per attribute
importance = (ranges / ranges.sum() * 100).round(1).sort_values(ascending=False)   # importance in percent
dump(utilities, "l15_conjoint_utilities.csv")                       # persist
dump(importance.reset_index().rename(columns={"range": "importance_pct"}), "l15_conjoint_importance.csv")   # persist
print(importance.to_string())                                       # report the ranking
recovered = utilities.pivot_table(index="level", columns="attribute", values="part_worth", aggfunc="first")   # convenience view
print(utilities.to_string(index=False))                             # report the part-worths

# ---------------------------------------------------------------- 4) individual-level spread
per_resp = []                                                       # one row per respondent
for resp, group in merged.groupby("resp"):                          # loop over respondents
    m = smf.ols(formula, data=group).fit()                          # a within-respondent regression
    per_resp.append({"resp": resp, "price_18": float(m.params.get("d_价格_18万", np.nan)),   # key utility
                     "range_800": float(m.params.get("d_续航_800km", np.nan))})   # key utility
individual = pd.DataFrame(per_resp)                                 # assemble the frame
dump(individual.describe().round(3).reset_index(), "l15_conjoint_individual.csv")   # persist a summary
print("individual price(18万) mean %.2f sd %.2f" % (individual["price_18"].mean(), individual["price_18"].std()))   # report

# ---------------------------------------------------------------- 5) the design figure
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
axes[0].axis("off")                                                # the design table needs no axes
axes[0].text(0.0, 1.02, "平衡设计（12 个轮廓，全因子 %d 个）" % full, fontsize=10, fontweight="bold")   # caption
col_labels = list(design.columns)                                   # the columns to show
cell = [[str(design.iloc[r][c]) for c in col_labels] for r in range(min(8, len(design)))]   # the first eight rows
table = axes[0].table(cellText=cell, colLabels=col_labels, loc="upper left", cellLoc="center")   # draw the table
table.auto_set_font_size(False)                                     # fixed font size
table.set_fontsize(7.4)                                            # small type
table.scale(1.0, 1.28)                                             # roomier rows
axes[0].set_title("Profile design")                                # panel title
counts = pd.Series({lv: (design[attr] == lv).sum() for attr in ATTRS for lv in ATTRS[attr]})   # level balance
axes[1].bar(range(len(counts)), counts.values, color=OKABE[4], alpha=0.9)   # the balance check
axes[1].set_xticks(range(len(counts)), [l for l in counts.index], rotation=70, fontsize=6.8)   # labels
style_axes(axes[1], "Each level appears equally often", "", "profiles")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_conjoint_design.png")                               # save the design figure

# ---------------------------------------------------------------- 6) the utility figure
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))                 # two panels
order = list(importance.index)                                      # attributes by importance
width = 0.26                                                       # bar width
for k, attr in enumerate(order):                                   # one attribute per group
    sub = utilities[utilities["attribute"] == attr].reset_index(drop=True)   # its levels
    axes[0].bar(np.arange(len(sub)) + (k - 1) * width, sub["part_worth"], width=width,   # the bars
                label=attr, color=OKABE[k], alpha=0.92)             # one colour per attribute
axes[0].axhline(0, color="#2B2B2B", lw=0.9)                         # zero line
axes[0].set_xticks([0, 1, 2], ["低/基础", "中/增强", "高/旗舰"], fontsize=8.4)   # generic level labels
style_axes(axes[0], "Part-worths by level (effect coding)", "", "utility")   # shared style
axes[0].legend(fontsize=7.4, ncol=2, loc="upper left")              # compact legend
axes[1].barh(order[::-1], importance.values[::-1], color=OKABE[5], alpha=0.9)   # importance bars
style_axes(axes[1], "Attribute importance (%)", "percent", "")      # shared style
for i, v in enumerate(importance.values[::-1]):                     # annotate the bars
    axes[1].text(v, i, "%.1f%%" % v, va="center", fontsize=8.4)      # the value
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_conjoint_utility.png")                              # save the utility figure
print("done: conjoint design, estimation and importance")            # closing log
