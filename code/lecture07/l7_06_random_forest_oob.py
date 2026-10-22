"""l7_06_random_forest_oob.py -- bagging, the random forest and the out-of-bag error.

Every line carries an English comment. The script compares ordinary least squares, a
single tree and a random forest on the poverty sample, then sweeps the number of
trees so the out-of-bag curve can be shown next to the test curve.
"""
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # two figures
from sklearn.tree import DecisionTreeRegressor                     # the single tree baseline
from sklearn.ensemble import RandomForestRegressor                 # bagging with feature subsampling
from sklearn.linear_model import LinearRegression                  # the OLS baseline
from sklearn.model_selection import train_test_split               # hold-out split
from sklearn.metrics import r2_score, mean_squared_error           # evaluation metrics
from _common_l7 import (DAT, save, dump, style_axes, annotated, OKABE)   # shared helpers

# ---------------------------------------------------------------- 1) the sample
data = pd.read_csv(DAT / "l7_poverty.csv")                         # the sample from lesson 7.5
features = ["log_income", "assets", "size", "educ", "rural", "migration"]   # predictor names
X = data[features].values                                          # feature matrix
y = data["consumption"].values                                     # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   # split
print("rows:", len(data), "| train/test:", X_train.shape[0], X_test.shape[0])   # report

# ---------------------------------------------------------------- 2) the three models
models = {"OLS": LinearRegression(),                               # linear baseline
          "Single tree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42),   # one tree
          "Random forest": RandomForestRegressor(n_estimators=300, max_features="sqrt",   # 300 trees
                                                 min_samples_leaf=3, random_state=42, oob_score=True)}   # OOB on
rows = []                                                          # summary table
fitted = {}                                                        # keep the fitted models
for name, model in models.items():                                 # loop over the three models
    model.fit(X_train, y_train)                                    # fit on the training split
    fitted[name] = model                                           # remember it
    pred = model.predict(X_test)                                   # test predictions
    oob = getattr(model, "oob_score_", None)                       # out-of-bag score when available
    rows.append({"model": name,                                    # model name
                 "train_r2": round(float(r2_score(y_train, model.predict(X_train))), 4),   # in-sample
                 "test_r2": round(float(r2_score(y_test, pred)), 4),   # out-of-sample
                 "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),   # error
                 "oob_r2": round(float(oob), 4) if oob is not None else None})   # OOB estimate
    print("%-14s train %.3f | test %.3f | OOB %s" % (                   # report the row for this model
        name, r2_score(y_train, model.predict(X_train)), r2_score(y_test, pred),   # its accuracy, error and OOB score
        "%.3f" % oob if oob is not None else "-"))                 # report the row
score_table = pd.DataFrame(rows)                                   # assemble the table
dump(score_table, "l7_ensemble_scores.csv")                        # persist it

# ---------------------------------------------------------------- 3) sweep the number of trees
tree_counts = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 400]   # increasing forest size
curve = []                                                         # results per forest size
for n in tree_counts:                                              # loop over the sizes
    rf = RandomForestRegressor(n_estimators=n, max_features="sqrt", min_samples_leaf=3,   # forest of n trees
                               random_state=42, oob_score=True, n_jobs=-1)   # use all the cores
    rf.fit(X_train, y_train)                                       # fit
    curve.append({"n_estimators": n,                               # forest size
                  "oob_r2": round(float(rf.oob_score_), 4),        # out-of-bag estimate
                  "test_r2": round(float(r2_score(y_test, rf.predict(X_test))), 4)})   # test score
    print("trees=%3d | OOB %.3f | test %.3f" % (n, rf.oob_score_, r2_score(y_test, rf.predict(X_test))))   # report
tree_curve = pd.DataFrame(curve)                                   # assemble the curve table
dump(tree_curve, "l7_rf_curve.csv")                                # persist it

# ---------------------------------------------------------------- 4) figure: the three models
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7))                # three panels
for ax, (name, model) in zip(axes, fitted.items()):                # one panel per model
    ax.scatter(y_test, model.predict(X_test), s=16, color=OKABE[5], alpha=0.8, label="out-of-sample")   # test points
    ax.scatter(y_train, model.predict(X_train), s=8, color=OKABE[1], alpha=0.35, label="in-sample")     # train points
    lims = [y.min(), y.max()]                                      # common limits
    ax.plot(lims, lims, ls="--", lw=1.0, color="#66717D")           # perfect prediction line
    style_axes(ax, "%s | test $R^2$ %.3f" % (name, r2_score(y_test, model.predict(X_test))),   # title
               "actual", "predicted")                              # axis labels
    ax.legend(fontsize=6.8, loc="upper left")                      # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_ols_tree_forest.png")                               # save the model comparison

# ---------------------------------------------------------------- 5) figure: the OOB curve
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7))                 # two panels
axes[0].plot(tree_curve["n_estimators"], tree_curve["oob_r2"], "-o", ms=4, color=OKABE[2], label="OOB $R^2$")   # OOB
axes[0].plot(tree_curve["n_estimators"], tree_curve["test_r2"], "-s", ms=4, color=OKABE[5], label="test $R^2$")   # test
axes[0].set_xscale("log")                                          # forest size on a log axis
style_axes(axes[0], "More trees: diminishing returns", "number of trees (log scale)", "$R^2$")   # shared style
axes[0].legend(fontsize=7.4, loc="lower right")                    # compact legend
axes[1].bar(["in-sample", "test", "OOB"], [score_table.loc[score_table["model"] == "Random forest", "train_r2"].iloc[0],   # in-sample
                                            score_table.loc[score_table["model"] == "Random forest", "test_r2"].iloc[0],   # test
                                            score_table.loc[score_table["model"] == "Random forest", "oob_r2"].iloc[0]],   # OOB
            color=[OKABE[1], OKABE[5], OKABE[2]], alpha=0.9)        # three colours
style_axes(axes[1], "OOB tracks the test score", "", "$R^2$")       # shared style
annotated(axes[1], "free validation set", "upper right")            # annotate the idea
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_rf_oob.png")                                        # save the OOB figure
print("done: bagging, random forest and the OOB error")            # closing log
