"""l5_12_interpretability_modelcard.py -- explaining a model and documenting it.

Every line carries an English comment. The script fits a gradient boosting model
on the credit sample, explains it with permutation importance, partial dependence
and ICE curves, then writes a one-page model card plus a JSON run log so that the
experiment can be reproduced and audited later.
"""
import json                                                        # build the run log structure
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # four-panel explanation figure
from sklearn.compose import ColumnTransformer                      # per-type preprocessing
from sklearn.ensemble import HistGradientBoostingClassifier        # strong tabular baseline
from sklearn.inspection import (permutation_importance,            # model-agnostic importance
                                partial_dependence)                # one and two dimensional dependence
from sklearn.metrics import roc_auc_score, brier_score_loss        # ranking quality and probability error
from sklearn.model_selection import train_test_split               # hold-out split
from sklearn.pipeline import Pipeline                              # preprocessing plus estimator
from sklearn.preprocessing import OneHotEncoder                    # encode the categorical column
from _common_l5 import DAT, save, dump, dump_artifact, write_json, style_axes, OKABE   # shared helpers

# ---------------------------------------------------------------- 1) fit the model that will be explained
credit = pd.read_csv(DAT / "l5_credit_default.csv")                # read the credit sample
X = credit.drop(columns=["default"])                               # feature matrix
y = credit["default"].values                                       # binary target
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()   # categorical column names
prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],   # encode categories
                         remainder="passthrough")                  # numeric columns pass through
X_train, X_test, y_train, y_test = train_test_split(               # stratified hold-out split
    X, y, test_size=0.3, random_state=42, stratify=y)              # keep the default rate equal
model = Pipeline([("prep", prep),                                  # preprocessing step
                  ("model", HistGradientBoostingClassifier(max_iter=320, learning_rate=0.07,   # boosting settings
                                                           random_state=42))]).fit(X_train, y_train)   # fit
proba = model.predict_proba(X_test)[:, 1]                          # predicted probabilities
print("model: test ROC-AUC %.4f | Brier %.4f" % (roc_auc_score(y_test, proba),   # ranking quality
                                                 brier_score_loss(y_test, proba)))   # probability error

# ---------------------------------------------------------------- 2) permutation importance on the test split
perm = permutation_importance(model, X_test, y_test, n_repeats=10,  # shuffle one feature at a time
                              random_state=42, scoring="roc_auc")   # measure the drop in ROC-AUC
importance = pd.DataFrame({"feature": X.columns,                    # feature names
                           "importance": perm.importances_mean.round(4),   # mean drop in ROC-AUC
                           "std": perm.importances_std.round(4)})   # spread across the ten repeats
importance = importance.sort_values("importance", ascending=False)  # rank by importance
dump(importance, "l5_model_importance.csv")                        # persist the importance table
top2 = importance["feature"].head(2).tolist()                      # the two most important features
print("top features:", top2)                                       # report them in the console

# ---------------------------------------------------------------- 3) partial dependence and ICE curves
pd_one = partial_dependence(model, X_test, [top2[0]], grid_resolution=40, kind="average")   # one dimensional PDP
pd_two = partial_dependence(model, X_test, top2, grid_resolution=28, kind="average")   # two dimensional PDP for interactions
grid_values = pd_one["grid_values"][0]                             # grid along the first feature
pdp_values = pd_one["average"][0]                                  # average predicted probability on that grid
ice_curves = []                                                    # individual conditional expectation curves
sample_rows = X_test.sample(50, random_state=42)                   # fifty random customers for the ICE bundle
for _, row in sample_rows.iterrows():                              # loop over the sampled customers
    tmp = pd.DataFrame([row] * len(grid_values), columns=X.columns)   # repeat the row along the grid
    tmp[top2[0]] = grid_values                                     # vary the feature of interest
    ice_curves.append(model.predict_proba(tmp)[:, 1])              # store the individual curve
ice = np.array(ice_curves)                                         # shape: customers by grid points
print("ICE curves:", ice.shape, "| PDP range %.3f-%.3f" % (pdp_values.min(), pdp_values.max()))   # report the shape

# ---------------------------------------------------------------- 4) one page model card and a JSON run log
card = f"""# Model card -- Lecture 5 credit default demo

## Intended use
Teaching example for an introduction to machine learning. **Not** a production
credit decision system: the data are synthetic and the model has not been
validated on any real portfolio.

## Model
Gradient boosting classifier (`HistGradientBoostingClassifier`, 320 iterations,
learning rate 0.07) inside a pipeline with one-hot encoding for `education`.

## Data
`l5_credit_default.csv`, {len(credit):,} synthetic accounts, positive rate
{y.mean():.3f}. Split 70/30 stratified (seed 42). No protected attributes are
included; `education` acts as a proxy and is therefore a fairness risk.

## Metrics (held-out test split)
- ROC-AUC: {roc_auc_score(y_test, proba):.4f}
- Brier score: {brier_score_loss(y_test, proba):.4f}
- Most important features (permutation, 10 repeats): {', '.join(top2)}

## Limitations and ethical considerations
- Decision threshold must be chosen from business costs, not from 0.5.
- Feature importance describes the model, **not** causal effects.
- Monitor for distribution shift before any reuse; retrain and re-document.
"""
dump_artifact(card, "model_card_l5_credit.md")                     # write the model card
write_json({"lecture": "L5", "script": "l5_12_interpretability_modelcard.py",   # run log metadata
            "seed": 20260506,                                      # seed used for the run
            "model": "HistGradientBoostingClassifier(max_iter=320, lr=0.07)",   # model description
            "metrics": {"roc_auc": round(float(roc_auc_score(y_test, proba)), 4),   # test ROC-AUC
                        "brier": round(float(brier_score_loss(y_test, proba)), 4)},   # test Brier score
            "top_features": top2,                                  # two leading features
            "artifacts": ["model_card_l5_credit.md", "l5_model_importance.csv"]},   # related files
           "l5_run_log_12.json")                                   # write the run log

# ---------------------------------------------------------------- 5) figure: importance, PDP, ICE, interactions
fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.5))                # four panels in one row
axes[0].barh(importance["feature"][::-1][:8],                       # eight leading features, largest on top
             importance["importance"][::-1][:8],                    # importance values
             xerr=importance["std"][::-1][:8], capsize=3,           # error bars from the repeats
             color=OKABE[5], alpha=0.9)                             # single colour keeps the panel calm
style_axes(axes[0], "Permutation importance (test split)", "Drop in ROC-AUC", "")   # shared style
axes[0].tick_params(axis="y", labelsize=7.4)                        # feature names in a small font
axes[1].plot(grid_values, pdp_values, lw=2.2, color=OKABE[0])       # average partial dependence curve
axes[1].plot(grid_values, ice[:12].T, lw=0.7, color="#B9B3B6", alpha=0.55)   # a few individual curves behind
style_axes(axes[1], f"PDP and ICE for {top2[0]}", f"{top2[0]}", "Predicted default probability")   # shared style
axes[2].plot(grid_values, ice.T, lw=0.7, color=OKABE[2], alpha=0.35)   # full ICE bundle
axes[2].plot(grid_values, ice.mean(axis=0), lw=2.2, color=OKABE[5])   # the mean equals the PDP
style_axes(axes[2], f"ICE bundle ({ice.shape[0]} customers)", f"{top2[0]}", "Individual prediction")   # shared style
x_grid = pd_two["grid_values"][0]                                  # grid along the first feature
y_grid = pd_two["grid_values"][1]                                  # grid along the second feature
zz = pd_two["average"][0].T                                        # average predictions on the two dimensional grid
im = axes[3].imshow(zz, origin="lower", aspect="auto", cmap="RdPu",   # interaction surface as a heatmap
                    extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()])   # align the axes with the data
axes[3].scatter(X_test[top2[0]], X_test[top2[1]], s=4, color="#1E2630", alpha=0.25)   # overlay the actual data
style_axes(axes[3], f"2D PDP: {top2[0]} x {top2[1]}", top2[0], top2[1])   # shared style
axes[3].grid(False)                                                # the heatmap replaces the grid
plt.colorbar(im, ax=axes[3], fraction=0.046, label="Predicted probability")   # colour bar for the surface
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_interpretability.png")                              # save the explanation figure
