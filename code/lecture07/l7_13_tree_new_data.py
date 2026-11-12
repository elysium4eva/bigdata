"""l7_13_tree_new_data.py -- a fresh simulated churn dataset for trees and random forests.

Every line carries an English comment. The script simulates 1,200 business customers with a known
generating process, then uses the same dataset for two lessons: the anatomy of a decision tree
(which splits it chooses and why) and the gap between one tree and a random forest (accuracy by
depth, feature importance, out-of-bag score).
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib                                                  # backend before pyplot
matplotlib.use("Agg")                                              # headless backend
import matplotlib.pyplot as plt                                    # figures
from sklearn.tree import DecisionTreeClassifier, plot_tree          # a single tree and its picture
from sklearn.ensemble import RandomForestClassifier                 # the forest
from sklearn.model_selection import train_test_split                # the split
from sklearn.metrics import accuracy_score                          # the metric
from pathlib import Path                                           # paths

RNG = np.random.default_rng(20261120)                              # a fixed seed
ROOT = Path(__file__).resolve().parents[2]                         # website/
FIG = ROOT / "figures" / "lecture07"                               # figure folder
DAT = ROOT / "data" / "lecture07"                                  # data folder
for d in (FIG, DAT):                                               # ensure both exist
    d.mkdir(parents=True, exist_ok=True)                           # create when missing

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]   # CJK stack
plt.rcParams["axes.unicode_minus"] = False                         # real minus signs
plt.rcParams["savefig.dpi"] = 200                                  # slide resolution
plt.rcParams["savefig.bbox"] = "tight"                             # trim white space
plt.rcParams["axes.grid"] = True                                   # light grid
plt.rcParams["grid.alpha"] = 0.25                                  # subtle grid
plt.rcParams["axes.spines.top"] = False                            # drop the top spine
plt.rcParams["axes.spines.right"] = False                          # drop the right spine
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]

# ---------------------------------------------------------------- 1) a simulated churn dataset
n = 1200                                                            # 1,200 enterprise customers
tenure = RNG.integers(1, 73, n)                                     # months with us
monthly = np.clip(RNG.normal(1200, 420, n), 200, 3600)               # monthly fee (yuan)
usage = np.clip(RNG.normal(0.62, 0.18, n), 0.02, 1.0)                # share of seats actually used
complaints = RNG.poisson(0.9, n)                                     # support tickets last quarter
seats = RNG.integers(5, 400, n)                                      # contracted seats
sector = RNG.choice(["制造", "零售", "金融", "科技"], n, p=[0.35, 0.3, 0.2, 0.15])   # industry
# the true process: short tenure, low usage, many complaints and high fee raise churn
logit = (-1.4 - 0.045 * tenure + 3.1 * usage - 0.62 * complaints
         + 0.00042 * monthly - 0.0022 * seats
         + np.where(sector == "零售", 0.55, 0.0) + np.where(sector == "制造", 0.25, 0.0))
prob = 1 / (1 + np.exp(-logit))                                     # the churn probability
churn = (RNG.random(n) < prob).astype(int)                          # the realised label
df = pd.DataFrame({"tenure": tenure, "monthly_fee": monthly.round(1), "usage_ratio": usage.round(3),
                   "complaints": complaints, "seats": seats, "sector": sector, "churn": churn})   # the table
df.to_csv(DAT / "l7_churn_sim.csv", index=False, encoding="utf-8-sig")   # persist
print("rows: %d | churn rate %.3f | mean tenure %.1f" % (len(df), df["churn"].mean(), df["tenure"].mean()))

# ---------------------------------------------------------------- 2) the single tree
FEATURES = ["tenure", "monthly_fee", "usage_ratio", "complaints", "seats"]   # numeric features
X = pd.get_dummies(df[FEATURES + ["sector"]], columns=["sector"], drop_first=True).astype(float)   # one-hot
y = df["churn"].to_numpy()                                          # the label
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)   # a hold-out
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, random_state=42).fit(Xtr, ytr)   # depth 3
acc_tree = accuracy_score(yte, tree.predict(Xte))                   # test accuracy
print("single tree (depth 3): test accuracy %.3f | leaves %d" % (acc_tree, tree.get_n_leaves()))
fig, ax = plt.subplots(figsize=(12.4, 4.6))                         # room for the diagram
plot_tree(tree, feature_names=list(X.columns), class_names=["留存", "流失"], filled=True,
          rounded=True, fontsize=8, impurity=True, proportion=False, ax=ax)   # draw the tree
ax.set_title("新数据集上的一棵深度 3 的决策树：每个节点显示分裂条件、Gini 与样本数", fontsize=11)   # title
fig.savefig(FIG / "fig_tree_new_data.png", facecolor="white")         # save
plt.close(fig)                                                     # release memory
print("[figure]", FIG / "fig_tree_new_data.png")                     # log it

# ---------------------------------------------------------------- 3) one tree versus a forest
depths = list(range(1, 16))                                        # depth from 1 to 15
rows = []                                                          # one row per depth
for d in depths:                                                   # loop over the depths
    t = DecisionTreeClassifier(max_depth=d, random_state=42).fit(Xtr, ytr)   # one tree
    f = RandomForestClassifier(n_estimators=200, max_depth=d, random_state=42, n_jobs=-1).fit(Xtr, ytr)   # a forest
    rows.append({"depth": d,                                  # the running record
                 "tree_train": round(float(t.score(Xtr, ytr)), 3),        # tree, training
                 "tree_test": round(float(t.score(Xte, yte)), 3),         # tree, test
                 "rf_train": round(float(f.score(Xtr, ytr)), 3),          # forest, training
                 "rf_test": round(float(f.score(Xte, yte)), 3)})          # forest, test
curve = pd.DataFrame(rows)                                         # assemble
curve.to_csv(DAT / "l7_tree_vs_forest.csv", index=False, encoding="utf-8-sig")   # persist
print(curve.to_string(index=False))                                # report

# ---------------------------------------------------------------- 4) importance and OOB
rf = RandomForestClassifier(n_estimators=400, max_depth=8, random_state=42, oob_score=True, n_jobs=-1).fit(Xtr, ytr)   # a tuned forest
imp_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)   # forest importance
imp_tree = pd.Series(tree.feature_importances_, index=X.columns).reindex(imp_rf.index)       # tree importance
print("forest test accuracy %.3f | OOB %.3f" % (float(rf.score(Xte, yte)), float(rf.oob_score_)))
print("top importances:", imp_rf.head(4).round(3).to_dict())        # report
pd.DataFrame({"feature": imp_rf.index, "forest": imp_rf.values.round(4),
              "single_tree": imp_tree.values.round(4)}).to_csv(
    DAT / "l7_importance_new_data.csv", index=False, encoding="utf-8-sig")   # persist

# ---------------------------------------------------------------- 5) the comparison figure
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.8))                 # three panels
axes[0].plot(curve["depth"], curve["tree_test"], "-o", ms=4, color=OKABE[5], label="单树（测试）")
axes[0].plot(curve["depth"], curve["rf_test"], "-s", ms=4, color=OKABE[2], label="随机森林（测试）")
axes[0].plot(curve["depth"], curve["tree_train"], "--", lw=1.0, color=OKABE[5], alpha=0.6, label="单树（训练）")
axes[0].legend(fontsize=7.0)                                       # compact legend
axes[0].set_xlabel("树深 max_depth"); axes[0].set_ylabel("准确率")    # labels
axes[0].set_title("同一份数据：森林一直比单树高，且更早饱和")           # panel title
top = imp_rf.head(6)                                               # the six most important features
axes[1].barh(top.index[::-1], top.values[::-1], color=OKABE[4], alpha=0.9)   # forest importance
axes[1].set_xlabel("不纯度下降（归一化）")                            # label
axes[1].set_title("随机森林的特征重要性")                              # panel title
n_trees = [10, 25, 50, 100, 200, 400]                               # forest sizes
oob = []                                                           # OOB scores
for t in n_trees:                                                  # loop over the sizes
    m = RandomForestClassifier(n_estimators=t, max_depth=8, random_state=42, oob_score=True, n_jobs=-1).fit(Xtr, ytr)   # fit
    oob.append(float(m.oob_score_))                                 # record the OOB score
axes[2].plot(n_trees, oob, "-o", ms=5, color=OKABE[3])              # the OOB curve
axes[2].set_xscale("log")                                          # log x
axes[2].set_xlabel("树的数量（对数）"); axes[2].set_ylabel("OOB 准确率")   # labels
axes[2].set_title("树越多越稳，200 棵之后基本饱和")                     # panel title
for x, v in zip(n_trees, oob):                                     # annotate the last points
    axes[2].annotate("%.3f" % v, (x, v), textcoords="offset points", xytext=(0, 6), fontsize=7.4)
plt.tight_layout()                                                 # tidy the layout
fig.savefig(FIG / "fig_tree_vs_forest.png", facecolor="white")        # save
plt.close(fig)                                                     # release memory
print("[figure]", FIG / "fig_tree_vs_forest.png")                    # log it
print("done: new churn dataset for trees and forests")               # closing log
