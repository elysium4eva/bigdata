"""l7_08_optuna_search.py -- hyper-parameter search with Optuna.

Every line carries an English comment. The script tunes a LightGBM model with the
tree-structured Parzen estimator, records every trial, and draws the optimisation
history together with the parameter importance so the search itself becomes visible.
"""
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # two panels
import optuna                                                      # the hyper-parameter search library
from sklearn.model_selection import train_test_split, KFold        # split and folds
from sklearn.metrics import r2_score                               # the objective
from lightgbm import LGBMRegressor                                 # the model being tuned
from _common_l7 import (DAT, encode_nominal, save, dump, write_json,   # shared helpers
                        style_axes,                                     # shared plotting helpers
                        annotated, OKABE)                               # annotations and colours

optuna.logging.set_verbosity(optuna.logging.WARNING)               # keep the console readable

# ---------------------------------------------------------------- 1) the sample
data = pd.read_csv(DAT / "l7_tabular.csv")                         # the tabular sample from lesson 7.7
features = ["f1", "f2", "f3", "f4", "f5", "f6"]                    # predictor names
y = data["target"].values                                          # the target
X_train_raw, X_test_raw, y_train, y_test = train_test_split(data[features], y, test_size=0.3, random_state=42)   # split
X_train, X_test, nom_map = encode_nominal(X_train_raw, X_test_raw)   # encode only the nominal column
print("rows:", len(data), "| features:", len(features))            # report

# ---------------------------------------------------------------- 2) the objective
folds = KFold(n_splits=4, shuffle=True, random_state=42)           # four folds for a cheap estimate

def objective(trial):                                              # one trial of the search
    """Return the mean cross-validated R^2 for a candidate configuration."""   # short docstring
    params = {"n_estimators": trial.suggest_int("n_estimators", 60, 600, step=20),   # number of trees
              "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),   # step size
              "num_leaves": trial.suggest_int("num_leaves", 8, 96, step=4),   # tree width
              "min_child_samples": trial.suggest_int("min_child_samples", 5, 60, step=5),   # leaf size
              "subsample": trial.suggest_float("subsample", 0.6, 1.0),   # row sampling
              "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),   # column sampling
              "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)}   # L2 regulariser
    scores = []                                                    # fold scores
    for train_idx, valid_idx in folds.split(X_train):              # loop over the folds
        model = LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)   # the candidate model
        model.fit(X_train[train_idx], y_train[train_idx])           # fit on the training folds
        scores.append(r2_score(y_train[valid_idx], model.predict(X_train[valid_idx])))   # score the held-out fold
    return float(np.mean(scores))                                  # the objective value

# ---------------------------------------------------------------- 3) run the search
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))   # maximise R^2
study.optimize(objective, n_trials=40, show_progress_bar=False)    # forty trials
print("best value %.4f" % study.best_value)                        # report the best score
print("best params:", study.best_params)                           # report the winning configuration
trials = study.trials_dataframe()                                  # every trial as a table
dump(trials, "l7_optuna_trials.csv")                               # persist the trial log

# ---------------------------------------------------------------- 4) refit the best model and score the test split
best_model = LGBMRegressor(**study.best_params, random_state=42, n_jobs=-1, verbose=-1)   # winning model
best_model.fit(X_train, y_train)                                   # fit on the full training split
test_r2 = float(r2_score(y_test, best_model.predict(X_test)))      # test score of the tuned model
default_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,   # the default configuration for the comparison
                              min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1).fit(X_train, y_train)   # defaults
default_r2 = float(r2_score(y_test, default_model.predict(X_test)))   # test score of the defaults
print("tuned test R2 %.4f | default test R2 %.4f | gain %+.4f" % (test_r2, default_r2, test_r2 - default_r2))   # report
write_json({"best_value": round(float(study.best_value), 4),       # run log: best CV score
            "test_r2": round(test_r2, 4),                          # test score of the tuned model
            "default_test_r2": round(default_r2, 4),               # test score of the defaults
            "n_trials": len(study.trials),                         # number of trials
            "best_params": study.best_params}, "l7_run_log_08.json")   # write it

# ---------------------------------------------------------------- 5) figure: history and importance
fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.8))                # two panels
values = trials["value"].values                                    # objective values in trial order
best_so_far = np.maximum.accumulate(values)                        # running best
axes[0].scatter(np.arange(len(values)), values, s=16, color=OKABE[1], alpha=0.7, label="trial")   # all trials
axes[0].plot(np.arange(len(values)), best_so_far, lw=1.8, color=OKABE[5], label="best so far")     # running best
style_axes(axes[0], "Optimisation history (%d trials)" % len(values), "trial", "CV $R^2$")   # shared style
axes[0].legend(fontsize=7.4, loc="lower right")                     # compact legend
try:                                                               # importance needs scikit-learn to be available
    importance = optuna.importance.get_param_importances(study)     # parameter importance
    names = list(importance.keys())[::-1]                          # reverse for a readable bar chart
    vals = [importance[k] for k in names]                          # matching values
    axes[1].barh(names, vals, color=OKABE[3], alpha=0.9)            # horizontal bars
    style_axes(axes[1], "Which knobs matter", "", "importance")     # shared style
except Exception as exc:                                           # fall back to a note when unavailable
    axes[1].text(0.5, 0.5, "importance unavailable: %s" % str(exc)[:40], ha="center", va="center")   # note
    axes[1].set_axis_off                                            # hide the axes
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_optuna.png")                                        # save the figure
print("done: Optuna search and the refit")                         # closing log
