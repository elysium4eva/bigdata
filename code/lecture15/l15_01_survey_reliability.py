"""l15_01_survey_reliability.py -- scale reliability, validity and the measurement error chain.

Every line carries an English comment. The script builds a 600-respondent Likert survey,
computes Cronbach's alpha for the three constructs, runs an exploratory factor analysis,
checks convergent and discriminant validity, and draws the error chain that the deck uses to
argue that measurement caps everything downstream.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from sklearn.decomposition import PCA                              # exploratory factor analysis
from sklearn.preprocessing import StandardScaler                   # standardise items first
from _common_l15 import (make_survey, SURVEY_ITEMS, save, dump,    # shared helpers
                         style_axes, annotated, OKABE)                  # shared plotting helpers

# ---------------------------------------------------------------- 1) the survey sample
survey = make_survey(600)                                          # 600 respondents, 12 items
dump(survey, "l15_survey.csv")                                     # persist the raw sample
print("respondents:", len(survey))                                  # report
items = [c for c in survey.columns if c.startswith("q")]            # the twelve item columns
blocks = {"品牌信任": items[0:4], "价格价值": items[4:8], "服务体验": items[8:12]}   # construct -> items

# ---------------------------------------------------------------- 2) Cronbach's alpha
def cronbach_alpha(frame):                                         # reliability of a block of items
    """Return Cronbach's alpha for a dataframe of items."""         # short docstring
    k = frame.shape[1]                                             # number of items
    var_items = frame.var(axis=0, ddof=1).sum()                     # sum of item variances
    var_total = frame.sum(axis=1).var(ddof=1)                       # variance of the total score
    return float(k / (k - 1) * (1 - var_items / var_total))          # the standard formula

rows = []                                                          # one row per construct
for name, cols in blocks.items():                                  # loop over the constructs
    alpha = cronbach_alpha(survey[cols])                           # reliability
    mean = float(survey[cols].mean(axis=1).mean())                 # average score
    rows.append({"construct": name, "items": len(cols), "alpha": round(alpha, 3),   # the row
                 "mean": round(mean, 2)})                          # more columns
    print("%s: alpha = %.3f (k=%d)" % (name, alpha, len(cols)))     # report
alphas = pd.DataFrame(rows)                                        # assemble the table
all_alpha = cronbach_alpha(survey[items])                          # the whole scale
print("whole scale alpha: %.3f" % all_alpha)                        # report

# ---------------------------------------------------------------- 3) exploratory factor analysis
X = StandardScaler().fit_transform(survey[items])                  # standardise the twelve items
pca = PCA(n_components=4).fit(X)                                   # four components
loadings = pd.DataFrame(pca.components_.T, index=items,            # item loadings
                        columns=["F1", "F2", "F3", "F4"])          # four factor columns
loadings["construct"] = [c for c, cols in blocks.items() for _ in cols]   # attach the construct label
dump(loadings.round(3).reset_index().rename(columns={"index": "item"}), "l15_efa_loadings.csv")   # persist
print("explained variance (first four):", np.round(pca.explained_variance_ratio_, 3))   # report
eigenvalues = np.linalg.eigvalsh(np.corrcoef(X.T))[::-1]            # eigenvalues of the correlation matrix
print("eigenvalues > 1:", int((eigenvalues > 1).sum()))             # Kaiser criterion count

# ---------------------------------------------------------------- 4) convergent and discriminant validity
def construct_score(cols):                                         # mean score of a construct
    """Return the respondent-level mean score of a block of items."""   # short docstring
    return survey[cols].mean(axis=1)                               # the mean across items

scores = pd.DataFrame({name: construct_score(cols) for name, cols in blocks.items()})   # three scores
corr = scores.corr()                                               # construct correlations
dump(corr.round(3).reset_index().rename(columns={"index": "construct"}), "l15_construct_corr.csv")   # persist
avg_var = float(np.mean([cronbach_alpha(survey[cols]) for cols in blocks.values()]))   # average alpha
sqrt_ave = float(np.sqrt(avg_var))                                 # square root of the average variance
print("construct correlations:\n", corr.round(3))                  # report
print("AVE (approx) sqrt:", round(sqrt_ave, 3))                    # report the discriminant benchmark

# ---------------------------------------------------------------- 5) the figures
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                # three panels
axes[0].bar(alphas["construct"], alphas["alpha"], color=[OKABE[4], OKABE[2], OKABE[5]], alpha=0.9)   # alphas
axes[0].axhline(0.7, ls="--", lw=1.0, color="#66717D")              # the common threshold
style_axes(axes[0], "Cronbach's alpha by construct", "", "alpha")   # shared style
for i, row in alphas.iterrows():                                   # annotate the bars
    axes[0].text(i, row["alpha"], "%.2f" % row["alpha"], ha="center", va="bottom", fontsize=9)   # value
axes[0].set_ylim(0, 1.0)                                          # full scale
for j, f in enumerate(["F1", "F2", "F3"]):                         # plot three factor loadings
    axes[1].scatter(range(len(items)), loadings[f], s=26, label=f, color=OKABE[j], alpha=0.9)   # the points
axes[1].set_xticks(range(len(items)), items, rotation=70, fontsize=7)   # item labels
style_axes(axes[1], "Item loadings on the first three factors", "", "loading")   # shared style
axes[1].legend(fontsize=7.4, ncol=3)                                # compact legend
im = axes[2].imshow(corr.values, cmap="RdPu", vmin=0, vmax=1)        # construct correlation matrix
axes[2].set_xticks(range(3), corr.columns, rotation=20, fontsize=8)  # labels
axes[2].set_yticks(range(3), corr.index, fontsize=8)                # labels
axes[2].grid(False)                                                # a heatmap needs no grid
for (i, j), v in np.ndenumerate(corr.values):                      # annotate the cells
    axes[2].text(j, i, "%.2f" % v, ha="center", va="center", fontsize=8,   # the value
                 color="white" if v > 0.6 else "#2B2B2B")           # readable colour
axes[2].set_title("Construct correlations")                        # panel title
plt.colorbar(im, ax=axes[2], fraction=0.046)                       # colour bar
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_survey_structure.png")                              # save the structure figure

# ---------------------------------------------------------------- 6) the measurement error chain
fig, ax = plt.subplots(figsize=(12.4, 3.0))                        # one wide panel
ax.axis("off")                                                    # sketch only
steps = [("抽样\nsampling", 0.16), ("覆盖\ncoverage", 0.26), ("无回答\nnon-response", 0.36),   # the first three links
         ("测量\nmeasurement", 0.46), ("录入与处理\nprocessing", 0.56)]                        # the last two links
for label, xpos in steps:                                          # draw the boxes
    ax.text(xpos, 0.62, label, ha="center", va="center", fontsize=9.5,   # the label
            bbox=dict(boxstyle="round,pad=0.45", fc="#F7F3F2", ec=OKABE[4], lw=1.2))   # the box
for xpos in [0.21, 0.31, 0.41, 0.51]:                               # arrows between the boxes
    ax.annotate("", xy=(xpos + 0.045, 0.62), xytext=(xpos - 0.045, 0.62),   # an arrow
                arrowprops=dict(arrowstyle="->", color=OKABE[5], lw=1.4))   # arrow style
ax.text(0.62, 0.62, "→  总误差  total survey error", fontsize=10, fontweight="bold", color="#8a0c3c")   # the summary
ax.text(0.06, 0.24, "本讲只改得了最后两个环节；前三个决定样本能不能代表总体。", fontsize=9, color="#5A5A5A")   # the takeaway
ax.set_title("测量误差链 the error chain of a survey")               # one title
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_error_chain.png")                                   # save the error figure
summary = {"alpha_whole": round(all_alpha, 3), "alphas": alphas.to_dict("records"),   # the run summary
           "explained_variance": [round(float(v), 3) for v in pca.explained_variance_ratio_]}   # more fields
dump(pd.DataFrame([{"scale": "whole", "alpha": round(all_alpha, 3)}]), "l15_scale_alpha.csv")   # persist
print("done: reliability, validity and the error chain")             # closing log
