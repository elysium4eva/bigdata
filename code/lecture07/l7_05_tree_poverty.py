"""l7_05_tree_poverty.py -- a CART regression tree on the poverty-targeting sample.

Every line carries an English comment. The script builds a regression tree on a
synthetic version of the poverty-targeting example, compares it with ordinary least
squares inside and outside the sample, and records the cost-complexity pruning path
so the deck can show how a tree is regularised.
"""
import numpy as np                                                 # numeric helpers for the residuals
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # three panels
from sklearn.tree import DecisionTreeRegressor, plot_tree          # the CART model and its drawing
from sklearn.linear_model import LinearRegression                  # the OLS reference point
from sklearn.model_selection import train_test_split               # hold-out split
from sklearn.metrics import r2_score, mean_squared_error           # evaluation metrics
from _common_l7 import (make_poverty, save, dump, style_axes,      # shared helpers
                        annotated, OKABE)                               # helper names imported from the lecture module

# ---------------------------------------------------------------- 1) the sample
data = make_poverty(600)                                           # synthetic household sample
dump(data, "l7_poverty.csv")                                       # persist it for the deck
features = ["log_income", "assets", "size", "educ", "rural", "migration"]   # predictor names
X = data[features].values                                          # feature matrix
y = data["consumption"].values                                     # target: log consumption
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   # split
print("rows:", len(data), "| features:", len(features), "| target sd: %.3f" % y.std())   # report

# ---------------------------------------------------------------- 2) OLS reference point
ols = LinearRegression().fit(X_train, y_train)                     # ordinary least squares on the training split
ols_train = ols.predict(X_train)                                   # in-sample predictions
ols_test = ols.predict(X_test)                                     # out-of-sample predictions
print("OLS  | train R2 %.3f | test R2 %.3f | test RMSE %.3f" % (        # report the linear baseline
    r2_score(y_train, ols_train), r2_score(y_test, ols_test),           # its two R2 values and the error
    float(np.sqrt(mean_squared_error(y_test, ols_test)))))         # report the linear baseline

# ---------------------------------------------------------------- 3) regression trees of several depths
depth_rows = []                                                    # results per depth
for depth in [1, 2, 3, 4, 6, 10, None]:                            # shallow to fully grown
    tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=5, random_state=42)   # a tree of this depth
    tree.fit(X_train, y_train)                                     # fit on the training split
    tr = r2_score(y_train, tree.predict(X_train))                  # in-sample R2
    te = r2_score(y_test, tree.predict(X_test))                    # out-of-sample R2
    depth_rows.append({"max_depth": depth if depth is not None else "None",   # label the depth
                       "leaves": int(tree.get_n_leaves()),          # number of leaves
                       "train_r2": round(float(tr), 4),             # in-sample fit
                       "test_r2": round(float(te), 4),              # out-of-sample fit
                       "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, tree.predict(X_test)))), 4)})   # error
    print("depth=%-4s | leaves=%3d | train R2 %.3f | test R2 %.3f" % (   # report the row for this depth
        depth, tree.get_n_leaves(), tr, te))                       # report the row
depth_table = pd.DataFrame(depth_rows)                             # assemble the table
dump(depth_table, "l7_tree_depth.csv")                             # persist it
best_depth = depth_table.loc[depth_table["test_r2"].idxmax(), "max_depth"]   # best depth by test R2
print("best depth:", best_depth)                                   # report the choice

# ---------------------------------------------------------------- 4) cost-complexity pruning path
full = DecisionTreeRegressor(min_samples_leaf=5, random_state=42).fit(X_train, y_train)   # fully grown tree
path = full.cost_complexity_pruning_path(X_train, y_train)         # alpha values and impurities
alphas = path.ccp_alphas[:-1]                                      # drop the last (single node) alpha
prune_rows = []                                                    # results per alpha
for alpha in alphas[:: max(1, len(alphas) // 12)]:                 # sample a dozen alphas for speed
    small = DecisionTreeRegressor(random_state=42, ccp_alpha=float(alpha))   # tree pruned to this alpha
    small.fit(X_train, y_train)                                    # fit it
    prune_rows.append({"alpha": round(float(alpha), 6),            # the pruning strength
                       "leaves": int(small.get_n_leaves()),        # leaves after pruning
                       "test_r2": round(float(r2_score(y_test, small.predict(X_test))), 4)})   # test fit
prune_table = pd.DataFrame(prune_rows)                             # assemble the table
dump(prune_table, "l7_tree_pruning.csv")                           # persist it
print("pruning path: %d alphas, %d leaves down to %d" % (               # report the pruning path
    len(alphas), int(full.get_n_leaves()), int(prune_table["leaves"].min())))   # report the path

# ---------------------------------------------------------------- 5) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8))                # three panels
plot_tree(DecisionTreeRegressor(max_depth=2, min_samples_leaf=20, random_state=42)   # draw a readable two-level tree
          .fit(X_train, y_train), feature_names=features, filled=True,   # fit it on the training split first
          rounded=True, fontsize=5, ax=axes[0], impurity=False)  # a readable two-level tree
axes[0].set_title("A depth-2 tree (stump)")                        # panel title
axes[1].plot([str(d) for d in depth_table["max_depth"]], depth_table["train_r2"],   # in-sample R2
             "-o", ms=4, color=OKABE[2], label="in-sample $R^2$")   # legend label
axes[1].plot([str(d) for d in depth_table["max_depth"]], depth_table["test_r2"],    # out-of-sample R2
             "-s", ms=4, color=OKABE[5], label="out-of-sample $R^2$")   # legend label
axes[1].axhline(r2_score(y_test, ols_test), ls="--", lw=1.0, color="#66717D", label="OLS out-of-sample")   # OLS
style_axes(axes[1], "Tree depth: fit versus generalisation", "max depth", "$R^2$")   # shared style
axes[1].legend(fontsize=6.8, loc="lower right")                    # compact legend
axes[2].plot(prune_table["leaves"], prune_table["test_r2"], "-o", ms=5, color=OKABE[3])   # pruning path
style_axes(axes[2], "Cost-complexity pruning path", "leaves after pruning", "out-of-sample $R^2$")   # style
annotated(axes[2], "fewer leaves, less variance", "lower left")    # annotate the idea
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_tree_poverty.png")                                  # save the tree figure

# ---------------------------------------------------------------- 6) OLS versus tree: in-sample versus out-of-sample
tree_best = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42).fit(X_train, y_train)   # chosen tree
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7))                # three panels
axes[0].scatter(ols_train, y_train, s=12, color=OKABE[1], alpha=0.7, label="in-sample")   # OLS in-sample
axes[0].scatter(ols_test, y_test, s=16, color=OKABE[5], alpha=0.85, label="out-of-sample")   # OLS out-of-sample
lims = [y.min(), y.max()]                                          # common axis limits
axes[0].plot(lims, lims, ls="--", lw=1.0, color="#66717D")          # perfect prediction line
style_axes(axes[0], "OLS: fitted versus actual", "fitted", "actual")   # shared style
axes[0].legend(fontsize=7.2, loc="upper left")                     # compact legend
axes[1].scatter(y_train, tree_best.predict(X_train), s=12, color=OKABE[1], alpha=0.7, label="in-sample")   # tree in-sample
axes[1].scatter(y_test, tree_best.predict(X_test), s=16, color=OKABE[5], alpha=0.85, label="out-of-sample")   # tree out
axes[1].plot(lims, lims, ls="--", lw=1.0, color="#66717D")          # perfect prediction line
style_axes(axes[1], "Depth-4 tree: fitted versus actual", "fitted", "actual")   # shared style
axes[1].legend(fontsize=7.2, loc="upper left")                     # compact legend
models = ["OLS", "Tree (d=2)", "Tree (d=4)"]                       # the three models compared
tr_scores = [r2_score(y_train, ols_train)]                         # start with OLS in-sample
te_scores = [r2_score(y_test, ols_test)]                           # start with OLS out-of-sample
for depth in (2, 4):                                               # add the two trees
    t = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=5, random_state=42).fit(X_train, y_train)   # fit
    tr_scores.append(r2_score(y_train, t.predict(X_train)))        # in-sample R2
    te_scores.append(r2_score(y_test, t.predict(X_test)))          # out-of-sample R2
xpos = np.arange(len(models))                                      # bar positions
axes[2].bar(xpos - 0.19, tr_scores, width=0.36, color=OKABE[2], alpha=0.9, label="in-sample")   # in-sample bars
axes[2].bar(xpos + 0.19, te_scores, width=0.36, color=OKABE[5], alpha=0.9, label="out-of-sample")   # out bars
axes[2].set_xticks(xpos, models, fontsize=8.0)                     # model labels
style_axes(axes[2], "Linear versus step functions", "", "$R^2$")   # shared style
axes[2].legend(fontsize=7.2, loc="lower right")                    # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_ols_vs_tree.png")                                   # save the comparison
print("done: regression tree, pruning path and OLS comparison")     # closing log
