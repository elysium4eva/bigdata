"""l5_07_trees_ensembles.py -- decision trees, random forests and gradient boosting.

Every line carries an English comment. The script grows a single tree to show the
bias-variance trade-off in one picture, then compares bagging, random forests and
three gradient-boosting implementations on the credit-default sample, including
their training cost and the features they rely on.
"""
import time                                                        # measure training cost per model
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # figure with four comparison panels
from sklearn.compose import ColumnTransformer                      # per-type preprocessing
from sklearn.ensemble import (RandomForestClassifier,              # bagging of decorrelated trees
                              HistGradientBoostingClassifier,      # scikit-learn's fast boosting implementation
                              GradientBoostingClassifier)          # classic boosting implementation
from sklearn.metrics import roc_curve, auc                         # ROC curve and its area
from sklearn.model_selection import train_test_split               # hold-out split
from sklearn.pipeline import Pipeline                              # preprocessing plus estimator
from sklearn.preprocessing import OneHotEncoder                    # encode the categorical column
from sklearn.tree import DecisionTreeClassifier                    # a single classification tree
from sklearn.inspection import permutation_importance              # model-agnostic importance scores
from _common_l5 import DAT, save, dump, style_axes, OKABE          # shared helpers

# ---------------------------------------------------------------- 1) data and split
credit = pd.read_csv(DAT / "l5_credit_default.csv")                # read the credit-default sample
X = credit.drop(columns=["default"])                               # feature matrix
y = credit["default"].values                                       # binary target
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()   # categorical column names
X_train, X_test, y_train, y_test = train_test_split(               # stratified hold-out split
    X, y, test_size=0.3, random_state=42, stratify=y)              # keeps the default rate equal
trees_prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],   # trees only need the categories encoded
                               remainder="passthrough")            # numeric columns pass through unchanged
print("train/test rows:", len(y_train), len(y_test), "| positives in test:", int(y_test.sum()))   # report sizes

# ---------------------------------------------------------------- 2) a single tree: depth against error
depths = list(range(1, 16))                                        # depths from a stump to a deep tree
depth_rows = []                                                    # one row per depth
for depth in depths:                                               # grow one tree per depth
    tree = Pipeline([("prep", trees_prep),                         # encoding step
                     ("model", DecisionTreeClassifier(max_depth=depth, random_state=42))]).fit(X_train, y_train)   # fit
    depth_rows.append({"depth": depth,                             # tree depth
                       "train_auc": round(float(tree.score(X_train, y_train)), 4),   # training accuracy
                       "test_auc": round(float(tree.score(X_test, y_test)), 4)})   # test accuracy
depth_table = pd.DataFrame(depth_rows)                             # assemble the table
print(depth_table.to_string(index=False))                          # print the depth sweep

# ---------------------------------------------------------------- 3) the ensemble family
def fit_model(name, model):                                        # helper that fits one model and scores it
    """Fit a pipeline, return a row of metrics plus the fitted object."""   # short docstring
    pipe = Pipeline([("prep", trees_prep), ("model", model)])      # wrap the estimator with preprocessing
    t0 = time.time()                                               # start the clock
    pipe.fit(X_train, y_train)                                     # fit on the training split
    seconds = time.time() - t0                                     # record the cost
    proba = pipe.predict_proba(X_test)[:, 1]                       # predicted probabilities on the test split
    fpr, tpr, _ = roc_curve(y_test, proba)                         # ROC points for the panel
    row = {"model": name,                                          # model identifier
           "test_auc": round(float(auc(fpr, tpr)), 4),             # area under the ROC curve
           "test_accuracy": round(float(pipe.score(X_test, y_test)), 4),   # accuracy at the default threshold
           "train_seconds": round(seconds, 3)}                     # training cost
    return row, pipe, (fpr, tpr)                                   # return everything the figure needs

results, fitted, curves = [], {}, {}                               # collectors for rows, models and ROC curves
for name, model in [("single tree (depth 4)", DecisionTreeClassifier(max_depth=4, random_state=42)),   # interpretable baseline
                    ("random forest", RandomForestClassifier(n_estimators=400, min_samples_leaf=3,   # bagged trees
                                                             random_state=42, n_jobs=1)),   # single job keeps timing fair
                    ("gradient boosting", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,   # classic boosting
                                                                     max_depth=3, random_state=42)),   # shallow trees
                    ("hist gradient boosting", HistGradientBoostingClassifier(max_iter=300,   # scikit-learn's fast version
                                                                              learning_rate=0.08, random_state=42)),   # tuned lightly
                    ]:                                                 # end of the model list
    row, pipe, curve = fit_model(name, model)                      # fit and score the model
    results.append(row)                                            # keep the metrics row
    fitted[name] = pipe                                            # keep the fitted pipeline
    curves[name] = curve                                           # keep the ROC points

# ---------------------------------------------------------------- 4) optional XGBoost and LightGBM back ends
try:                                                               # XGBoost is installed in this environment
    from xgboost import XGBClassifier                              # gradient boosting with regularisation
    row, pipe, curve = fit_model("xgboost", XGBClassifier(n_estimators=400, learning_rate=0.06,   # boosting rounds
                                                          max_depth=4, subsample=0.9, colsample_bytree=0.9,   # row and column sampling
                                                          eval_metric="logloss", random_state=42, n_jobs=1))   # keep timing fair
    results.append(row); fitted["xgboost"] = pipe; curves["xgboost"] = curve   # store the results
    print("[xgboost] test AUC %.4f in %.2fs" % (row["test_auc"], row["train_seconds"]))   # report the score
except Exception as exc:                                           # if the package is missing, skip it gracefully
    print("[xgboost] skipped:", exc)                               # explain why it was skipped
try:                                                               # LightGBM is installed as well
    from lightgbm import LGBMClassifier                            # histogram based boosting
    row, pipe, curve = fit_model("lightgbm", LGBMClassifier(n_estimators=400, learning_rate=0.06,   # boosting rounds
                                                            num_leaves=31, random_state=42, n_jobs=1,   # leaf-wise growth
                                                            verbose=-1))   # keep the console quiet
    results.append(row); fitted["lightgbm"] = pipe; curves["lightgbm"] = curve   # store the results
    print("[lightgbm] test AUC %.4f in %.2fs" % (row["test_auc"], row["train_seconds"]))   # report the score
except Exception as exc:                                           # if the package is missing, skip it gracefully
    print("[lightgbm] skipped:", exc)                              # explain why it was skipped

scores = pd.DataFrame(results).sort_values("test_auc", ascending=False)   # rank the models by test AUC
dump(scores, "l5_ensemble_scores.csv")                             # persist the score table
print(scores.to_string(index=False))                               # print the ranking

# ---------------------------------------------------------------- 5) which features do the ensembles rely on?
best_name = str(scores.iloc[0]["model"])                           # the best model by test AUC
perm = permutation_importance(fitted[best_name], X_test, y_test,    # model-agnostic importance on the test split
                              n_repeats=8, random_state=42, scoring="roc_auc")   # eight repeats for a spread
importance = pd.DataFrame({"feature": X.columns,                    # feature names
                           "importance": perm.importances_mean.round(4),   # mean drop in ROC-AUC
                           "std": perm.importances_std.round(4)})   # spread across repeats
importance = importance.sort_values("importance", ascending=False)  # rank by importance
dump(importance, "l5_feature_importance.csv")                      # persist the importance table
print("top features for %s:\n%s" % (best_name, importance.head(4).to_string(index=False)))   # report the leaders

# ---------------------------------------------------------------- 6) figure: four views of the ensemble story
fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.5))                # four panels in one row
axes[0].plot(depth_table["depth"], depth_table["train_auc"], "-o", ms=4, color=OKABE[2], label="training accuracy")   # in-sample
axes[0].plot(depth_table["depth"], depth_table["test_auc"], "-s", ms=4, color=OKABE[5], label="test accuracy")   # out-of-sample
style_axes(axes[0], "One tree: depth buys variance", "max_depth", "Accuracy")   # shared style
axes[0].legend(fontsize=7.6, loc="lower right")                    # compact legend
for (name, (fpr, tpr)), colour in zip(curves.items(), OKABE):      # draw one ROC curve per model
    axes[1].plot(fpr, tpr, lw=1.7, color=colour, label=f"{name} ({auc(fpr, tpr):.3f})")   # curve with its AUC
axes[1].plot([0, 1], [0, 1], ls="--", lw=1.0, color="#66717D")      # random reference line
style_axes(axes[1], "ROC curves: ensembles win", "False positive rate", "True positive rate")   # shared style
axes[1].legend(fontsize=6.6, loc="lower right")                    # compact legend for six models
axes[2].barh(importance["feature"][::-1][:8],                       # top eight features, largest on top
             importance["importance"][::-1][:8],                    # importance values
             xerr=importance["std"][::-1][:8], capsize=3,           # error bars from the repeats
             color=OKABE[3], alpha=0.9)                             # single colour keeps it readable
style_axes(axes[2], f"Permutation importance ({best_name})", "Drop in ROC-AUC", "")   # shared style
axes[2].tick_params(axis="y", labelsize=7.4)                        # feature names in a small font
axes[3].scatter(scores["train_seconds"], scores["test_auc"], s=55,  # cost against quality
                color=[OKABE[i % len(OKABE)] for i in range(len(scores))],   # one colour per model
                edgecolors="white", lw=0.8)                         # white border for contrast
for _, r in scores.iterrows():                                      # annotate every point with its name
    axes[3].annotate(str(r["model"]), xy=(r["train_seconds"], r["test_auc"]),   # position at the point
                     xytext=(4, 4), textcoords="offset points", fontsize=6.8, color="#1E2630")   # small offset label
style_axes(axes[3], "Cost versus quality", "Training seconds (log scale)", "Test ROC-AUC")   # shared style
axes[3].set_xscale("log")                                           # training times span several orders of magnitude
plt.tight_layout()                                                  # tidy the layout
save(fig, "fig_ensembles.png")                                      # save the ensemble figure
