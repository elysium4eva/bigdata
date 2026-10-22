"""l7_07_gbdt_compare.py -- XGBoost, LightGBM and CatBoost on the same tabular sample.

Every line carries an English comment. The script persists the mixed-feature tabular
sample, fits the three gradient-boosting libraries with comparable settings, records
accuracy and training time, and sweeps the learning rate against the number of trees.
"""
import time                                                        # measure training time
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # the tuning curves
from sklearn.model_selection import train_test_split               # hold-out split
from sklearn.metrics import r2_score, mean_squared_error           # evaluation metrics
from xgboost import XGBRegressor                                   # XGBoost
from lightgbm import LGBMRegressor                                 # LightGBM
from catboost import CatBoostRegressor                             # CatBoost
from _common_l7 import (make_tabular, encode_nominal, save, dump,   # shared helpers
                        style_axes,                                     # shared plotting helpers
                        annotated, OKABE)                               # annotations and colours

# ---------------------------------------------------------------- 1) the sample
data = make_tabular(1400)                                          # mixed-feature tabular sample
dump(data, "l7_tabular.csv")                                       # persist it for later scripts
features = ["f1", "f2", "f3", "f4", "f5", "f6"]                    # predictor names
X_raw = data[features]                                             # keep the categorical column intact
y = data["target"].values                                          # the target
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=42)   # split
X_train, X_test, nom_map = encode_nominal(X_train_raw, X_test_raw)   # encode only the nominal column
print("nominal map:", nom_map)                                     # report the level codes
print("rows:", len(data), "| features:", len(features), "| target sd: %.3f" % y.std())   # report

# ---------------------------------------------------------------- 2) the three libraries
models = {                                                              # the three libraries, each behind a factory
    "XGBoost": lambda: XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,   # XGBoost settings
                                    subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1),   # more
    "LightGBM": lambda: LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,   # LightGBM settings
                                      min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1),   # more
    "CatBoost": lambda: CatBoostRegressor(iterations=300, learning_rate=0.05, depth=4,   # CatBoost settings
                                          random_seed=42, verbose=False),                  # silent training
}                                                                       # end of the factory mapping
rows = []                                                          # summary table
for name, factory in models.items():                               # loop over the three libraries
    model = factory()                                              # build the model
    started = time.perf_counter()                                  # start the timer
    model.fit(X_train, y_train)                                    # fit on the training split
    seconds = time.perf_counter() - started                        # elapsed training time
    pred = model.predict(X_test)                                   # test predictions
    rows.append({"library": name,                                  # library name
                 "seconds": round(float(seconds), 2),              # training time in seconds
                 "train_r2": round(float(r2_score(y_train, model.predict(X_train))), 4),   # in-sample
                 "test_r2": round(float(r2_score(y_test, pred)), 4),   # out-of-sample
                 "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4)})   # error
    print("%-9s | %.2fs | train R2 %.3f | test R2 %.3f" % (name, seconds, r2_score(y_train, model.predict(X_train)), r2_score(y_test, pred)))   # report
summary = pd.DataFrame(rows)                                       # assemble the table
dump(summary, "l7_gbdt_compare.csv")                               # persist it

# ---------------------------------------------------------------- 3) learning rate versus number of trees
lrs = [0.01, 0.03, 0.1, 0.3]                                       # learning rates to compare
counts = [25, 50, 100, 200, 300, 500, 800]                         # numbers of trees
curve_rows = []                                                    # one row per combination
for lr in lrs:                                                     # loop over the learning rates
    for n in counts:                                               # loop over the tree counts
        model = LGBMRegressor(n_estimators=n, learning_rate=lr, num_leaves=31,   # same model, new settings
                              min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1)   # silent
        model.fit(X_train, y_train)                                # fit
        curve_rows.append({"learning_rate": lr, "n_estimators": n,   # the two knobs
                           "test_r2": round(float(r2_score(y_test, model.predict(X_test))), 4)})   # test score
        print("lr=%.2f | trees=%3d | test R2 %.4f" % (lr, n, r2_score(y_test, model.predict(X_test))))   # report
curve = pd.DataFrame(curve_rows)                                   # assemble the curve table
dump(curve, "l7_gbdt_curve.csv")                                   # persist it
best = curve.loc[curve["test_r2"].idxmax()]                        # best combination
print("best: lr %.2f with %d trees -> test R2 %.4f" % (best["learning_rate"], best["n_estimators"], best["test_r2"]))   # report

# ---------------------------------------------------------------- 4) figure: comparison and the tuning surface
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.7))                # three panels
axes[0].bar(summary["library"], summary["test_r2"], color=[OKABE[1], OKABE[2], OKABE[5]], alpha=0.9)   # test R2
axes[0].set_ylim(float(summary["test_r2"].min()) - 0.02, float(summary["test_r2"].max()) + 0.01)   # zoom the bars
style_axes(axes[0], "Test $R^2$ of the three libraries", "", "$R^2$")   # shared style
for i, row in summary.iterrows():                                  # annotate the bars
    axes[0].text(i, row["test_r2"], "%.3f" % row["test_r2"], ha="center", va="bottom", fontsize=8)   # value label
axes[1].bar(summary["library"], summary["seconds"], color=[OKABE[1], OKABE[2], OKABE[5]], alpha=0.9)   # training time
style_axes(axes[1], "Training time (seconds)", "", "seconds")      # shared style
for lr, colour in zip(lrs, [OKABE[1], OKABE[3], OKABE[2], OKABE[5]]):   # one line per learning rate
    sub = curve[curve["learning_rate"] == lr]                      # rows for this learning rate
    axes[2].plot(sub["n_estimators"], sub["test_r2"], "-o", ms=3.5, color=colour, label="lr %.2f" % lr)   # curve
axes[2].set_xscale("log")                                          # tree count on a log axis
style_axes(axes[2], "Learning rate versus number of trees", "number of trees (log scale)", "test $R^2$")   # style
axes[2].legend(fontsize=7.0, loc="lower right")                     # compact legend
annotated(axes[2], "best: lr %.2f, %d trees" % (best["learning_rate"], best["n_estimators"]), "upper left")   # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_gbdt_tuning.png")                                   # save the figure
print("done: three libraries and the tuning surface")              # closing log
