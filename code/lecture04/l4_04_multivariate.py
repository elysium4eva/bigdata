"""l4_04_multivariate.py -- multivariate: correlation heatmap, grouped scatter, small multiples, conditional distributions.

A firm panel (60 firms x 8 years) answers "what should we look at with many variables":
pairwise correlation first (heatmap), then whether grouping changes the relation (scatter and conditional distributions),
then small multiples spread the trend by industry -- ten colors in one chart is the classic mistake.
"""
import numpy as np                                                 # numeric computation
import pandas as pd                                                # panel data handling
import matplotlib.pyplot as plt                                    # four-panel figure
from _common_l4 import RNG, save, dump                             # reuse shared helpers

N_FIRM, N_YEAR = 60, 8                                             # 60 firms over 8 years
INDUSTRIES = ["电子", "医药", "机械", "消费"]                       # four industries
firms = pd.DataFrame({                                             # firm-level static attributes
    "firm_id": [f"F{i:03d}" for i in range(1, N_FIRM + 1)],        # firm id
    "industry": RNG.choice(INDUSTRIES, N_FIRM),                    # industry
    "size": RNG.choice(["小", "中", "大"], N_FIRM, p=[0.45, 0.35, 0.2]),   # size band
    "base_rev": RNG.lognormal(mean=2.2, sigma=0.6, size=N_FIRM),   # initial revenue (100M CNY)
    "growth": RNG.normal(0.09, 0.05, N_FIRM),                      # firm growth rate
    "leverage": RNG.uniform(0.15, 0.75, N_FIRM).round(3),          # debt-to-asset ratio
})                                                                # end of the firm attribute table
rows = []                                                          # rows of the long panel table
for _, f in firms.iterrows():                                      # expand years for every firm
    for year in range(2018, 2018 + N_YEAR):                        # the eight years
        rev = f["base_rev"] * (1 + f["growth"]) ** (year - 2018) * np.exp(RNG.normal(0, 0.06))   # revenue evolution
        roa = (0.075 - 0.05 * (f["leverage"] - 0.45) + RNG.normal(0, 0.012))   # return on assets: lower when leverage is high
        rows.append({"firm_id": f["firm_id"], "industry": f["industry"],   # ids and industry
                     "size": f["size"], "year": year,              # size and year
                     "revenue": round(float(rev), 3),              # revenue
                     "roa": round(float(roa), 4),                  # return on assets
                     "leverage": f["leverage"]})                   # leverage (firm level, constant over years)
panel = pd.DataFrame(rows)                                         # collect into a panel
dump(panel, "l4_firm_panel.csv")                                   # write the sample

numeric_cols = ["revenue", "roa", "leverage"]                      # numeric columns used for correlation
corr = panel[numeric_cols].corr().round(3)                         # three-variable correlation matrix
print("相关矩阵:\n", corr.to_string())                              # print to the terminal
cond = (panel.groupby(["industry", "size"])["roa"].agg(["mean", "count"]).round(4).reset_index())   # conditional mean by industry x size
dump(cond, "l4_conditional_roa.csv")                               # write the conditional means
print("条件均值（行业 × 规模，前 6 行）:\n", cond.head(6).to_string(index=False))   # print a slice

# ---------------------------------------------------------------- figure: four multivariate views
fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.7))                # heatmap / grouped scatter / small multiples / conditional distribution
im = axes[0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)   # first panel: correlation heatmap (diverging palette)
axes[0].set_xticks(range(len(numeric_cols)), ["营收", "ROA", "杠杆率"], fontsize=9)   # x axis labels
axes[0].set_yticks(range(len(numeric_cols)), ["营收", "ROA", "杠杆率"], fontsize=9)   # y axis labels
axes[0].grid(False)                                                # no grid needed on a heatmap
for i in range(len(numeric_cols)):                                 # write a value in every cell
    for j in range(len(numeric_cols)):                             # double loop over the matrix
        axes[0].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8.6,   # write each correlation in its cell
                     color="white" if abs(corr.values[i, j]) > 0.6 else "#1E2630")   # white text on dark cells
axes[0].set_title("相关矩阵（发散色板）")                            # subplot title
plt.colorbar(im, ax=axes[0], fraction=0.046)                       # color bar
for ind, c in zip(INDUSTRIES, ["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"]):   # color by industry
    sub = panel[panel["industry"] == ind]                          # take the industry subsample
    axes[1].scatter(sub["leverage"], sub["roa"], s=12, color=c, alpha=0.5, label=ind)   # second panel: grouped scatter
axes[1].axhline(panel["roa"].mean(), color="#1E2630", lw=1.0, ls="--")   # overall mean reference line
axes[1].set_title("关系是否被分组改变")                             # subplot title
axes[1].set_xlabel("资产负债率"); axes[1].set_ylabel("ROA")         # axis labels
axes[1].legend(fontsize=8.2, ncol=2)                               # two-column legend
for k, ind in enumerate(INDUSTRIES):                               # small multiples: one panel per industry
    sub = panel[panel["industry"] == ind].groupby("year")["revenue"].mean()   # industry yearly mean
    axes[2].plot(sub.index, sub.values, "-o", ms=3, lw=1.5,        # draw the industry trend line
                 color=["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"][k], label=ind)   # color and legend
axes[2].set_title("小多图：行业营收趋势")                            # subplot title
axes[2].set_xlabel("年份"); axes[2].set_ylabel("平均营收（亿元）")    # axis labels
axes[2].legend(fontsize=8.2)                                       # legend
data = [panel.loc[panel["industry"] == ind, "roa"].values for ind in INDUSTRIES]   # ROA by industry
bp = axes[3].boxplot(data, patch_artist=True)                      # fourth panel: conditional distribution (box plot)
axes[3].set_xticklabels(INDUSTRIES)                                # group labels (set_xticklabels since 3.9)
for patch, c in zip(bp["boxes"], ["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"]):   # color the boxes
    patch.set_facecolor(c); patch.set_alpha(0.55)                  # set fill color and alpha
axes[3].set_title("条件分布：各行业 ROA")                           # subplot title
axes[3].set_ylabel("ROA")                                          # axis label
plt.tight_layout()                                                 # adjust the four subplots
save(fig, "fig_multivariate.png")                                  # save: one-page multivariate figure
