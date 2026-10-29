"""tools_figs_l5.py -- extra explanatory figures for the reworked Lecture 5.

Every line carries an English comment. The script draws four figures that the deck uses to make
the core machine-learning ideas visible: the bias-variance target, the curse of dimensionality,
the generalisation gap, and a confusion matrix with derived metrics.
"""
import numpy as np                                                 # arrays
import matplotlib                                                  # backend before pyplot
matplotlib.use("Agg")                                              # headless backend
import matplotlib.pyplot as plt                                    # figures
from matplotlib.patches import Circle, FancyArrowPatch              # simple shapes
from pathlib import Path                                           # paths

FIG = Path(__file__).resolve().parents[2] / "figures" / "lecture05"   # website/figures/lecture05
FIG.mkdir(parents=True, exist_ok=True)                             # make sure it exists
RNG = np.random.default_rng(20261005)                              # a fixed seed

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]   # CJK stack
plt.rcParams["axes.unicode_minus"] = False                         # real minus signs
plt.rcParams["savefig.dpi"] = 200                                  # slide resolution
plt.rcParams["savefig.bbox"] = "tight"                             # trim white space
plt.rcParams["axes.grid"] = True                                   # light grid
plt.rcParams["grid.alpha"] = 0.25                                  # subtle grid
plt.rcParams["axes.spines.top"] = False                            # drop the top spine
plt.rcParams["axes.spines.right"] = False                          # drop the right spine
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",               # colour-blind safe palette
         "#0072B2", "#D55E00", "#CC79A7", "#000000"]               # eight colours


def save(fig, name):                                               # persist a figure
    """Save into website/figures/lecture05 and close it."""        # short docstring
    path = FIG / name                                              # the target path
    fig.savefig(path, facecolor="white")                           # white background
    plt.close(fig)                                                 # release memory
    print("[figure]", path)                                        # log it


# ---------------------------------------------------------------- 1) the bias-variance target
fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5))                 # four dartboards
cases = [("高偏差 · 低方差\n欠拟合", 1.5, 0.18, OKABE[1]),           # underfit
         ("低偏差 · 高方差\n过拟合", 0.15, 0.95, OKABE[5]),          # overfit
         ("高偏差 · 高方差", 1.2, 0.85, OKABE[3]),                  # both
         ("低偏差 · 低方差\n理想", 0.12, 0.16, OKABE[2])]           # good
for ax, (title, bias, spread, colour) in zip(axes, cases):          # loop over the cases
    for r, alpha in ((1.0, 0.25), (0.66, 0.18), (0.33, 0.12)):      # three rings
        ax.add_patch(Circle((0, 0), r, fill=False, lw=1.0, color="#9AA5B1", alpha=alpha))   # draw a ring
    ax.plot([0], [0], marker="+", ms=12, color="#2B2B2B", mew=1.6)  # the bullseye
    xs = bias + RNG.normal(0, spread, 26)                           # the x coordinates
    ys = bias + RNG.normal(0, spread, 26)                           # the y coordinates
    ax.scatter(xs, ys, s=22, color=colour, alpha=0.85)               # the throws
    ax.set_xlim(-2.1, 2.1); ax.set_ylim(-2.1, 2.1)                    # equal bounds
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])      # no ticks
    ax.grid(False)                                                  # the target needs no grid
    ax.set_title(title, fontsize=10.5)                              # the case title
fig.suptitle("偏差-方差靶心图：瞄准点与散布是两件事", fontsize=13.5)   # the figure title
save(fig, "fig_bias_variance_target.png")                           # save it

# ---------------------------------------------------------------- 2) the curse of dimensionality
dims = np.arange(1, 51)                                            # dimensions from 1 to 50
# fraction of a unit cube's volume within 10% of the surface, and the distance concentration
shell = 1 - 0.9 ** dims                                            # volume near the boundary
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.8))                 # two panels
axes[0].plot(dims, shell, lw=2.0, color=OKABE[5])                   # the shell fraction
axes[0].axhline(0.99, ls="--", lw=1.0, color="#66717D")             # the 99% line
axes[0].set_xlabel("维度 d"); axes[0].set_ylabel("落在边界薄层内的体积占比")   # labels
axes[0].set_title("维度越高，数据越贴边：$1-0.9^d$")                    # the panel title
annotated_at = 20                                                   # annotate at d = 20
axes[0].annotate("d=20 时已约 88%", xy=(annotated_at, 1 - 0.9 ** annotated_at),
                 xytext=(22, 0.62), fontsize=8.6,
                 arrowprops=dict(arrowstyle="->", lw=1.0, color="#2B2B2B"))   # the note
# nearest and farthest neighbour distances in a unit cube
n = 500                                                            # sample size
for colour, d in ((OKABE[1], 2), (OKABE[2], 10), (OKABE[5], 50)):   # three dimensions
    pts = RNG.random((n, d))                                        # uniform points
    q = RNG.random((50, d))                                         # queries
    d2 = ((q[:, None, :] - pts[None, :, :]) ** 2).sum(-1)           # squared distances
    dist = np.sqrt(d2)                                              # distances
    ratios = dist.max(axis=1) / np.maximum(dist.min(axis=1), 1e-9)   # far over near
    axes[1].hist(ratios, bins=18, alpha=0.55, color=colour, label="d=%d" % d)   # the histogram
axes[1].legend(fontsize=8.0)                                       # compact legend
axes[1].set_xlabel("最远/最近邻距离之比"); axes[1].set_ylabel("查询点个数")   # labels
axes[1].set_title("维度越高，远近邻居越来越分不开")                     # the panel title
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_curse_dimensionality.png")                           # save it

# ---------------------------------------------------------------- 3) the generalisation gap
complexity = np.linspace(1, 12, 120)                                # model complexity
train = 0.30 + 0.62 * np.exp(-complexity / 2.2)                      # training error falls
test = 0.34 + 0.16 * np.exp(-complexity / 1.6) + 0.030 * complexity   # test error is U-shaped
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.8))                  # two panels
axes[0].plot(complexity, train, lw=2.0, color=OKABE[2], label="训练误差")   # the training curve
axes[0].plot(complexity, test, lw=2.0, color=OKABE[5], label="测试误差")    # the test curve
best = complexity[int(np.argmin(test))]                              # the optimum
axes[0].axvline(best, ls="--", lw=1.0, color="#66717D")               # mark it
axes[0].fill_between(complexity, train, test, color=OKABE[1], alpha=0.18, label="泛化差距")   # the gap
axes[0].legend(fontsize=8.0)                                        # compact legend
axes[0].set_xlabel("模型复杂度"); axes[0].set_ylabel("误差")             # labels
axes[0].set_title("训练跟得上，测试会先降后升")                          # the panel title
# the gap itself
axes[1].plot(complexity, test - train, lw=2.2, color=OKABE[4])        # the gap curve
axes[1].set_xlabel("模型复杂度"); axes[1].set_ylabel("测试 − 训练")        # labels
axes[1].set_title("泛化差距随复杂度单调变大：这就是过拟合的定义")            # the panel title
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_generalization_gap.png")                             # save it

# ---------------------------------------------------------------- 4) a confusion matrix with metrics
tn, fp, fn, tp = 820, 60, 90, 130                                   # a realistic credit default table
grid = np.array([[tn, fp], [fn, tp]])                                # the matrix
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.8))                  # two panels
im = axes[0].imshow(grid, cmap="Blues")                              # colour the cells
axes[0].set_xticks([0, 1], ["预测为负", "预测为正"])                    # x labels
axes[0].set_yticks([0, 1], ["实际为负", "实际为正"])                    # y labels
axes[0].grid(False)                                                 # no grid inside
for i in range(2):                                                  # loop over rows
    for j in range(2):                                              # loop over columns
        axes[0].text(j, i, str(grid[i, j]), ha="center", va="center",
                     fontsize=15, color="white" if grid[i, j] > 400 else "#2B2B2B")   # the counts
axes[0].set_title("混淆矩阵（信用违约，共 1,100 单）")                   # the panel title
metrics = [("准确率 Accuracy", (tp + tn) / grid.sum(), OKABE[4]),      # accuracy
           ("精确率 Precision", tp / (tp + fp), OKABE[2]),            # precision
           ("召回率 Recall", tp / (tp + fn), OKABE[5]),               # recall
           ("F1", 2 * tp / (2 * tp + fp + fn), OKABE[1]),             # F1
           ("特异度 Specificity", tn / (tn + fp), OKABE[3])]          # specificity
names = [m[0] for m in metrics]                                     # the names
vals = [m[1] for m in metrics]                                      # the values
axes[1].barh(names, vals, color=[m[2] for m in metrics], alpha=0.9)   # the bars
for i, v in enumerate(vals):                                        # annotate each bar
    axes[1].text(v + 0.01, i, "%.3f" % v, va="center", fontsize=9.0)   # the value
axes[1].set_xlim(0, 1.08)                                           # headroom for the labels
axes[1].set_title("同一张表，五个指标：别只看准确率")                    # the panel title
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_confusion_metrics.png")                              # save it
print("done: four extra figures for Lecture 5")                      # closing log
