"""l6_08_regularization_cv.py -- LASSO, Ridge and Elastic Net with cross-validated strength.

Every line carries an English comment. The script standardises the prostate sample,
draws the coefficient paths of the three penalties, and selects the penalty strength
by cross-validation so that the slides can show shrinkage, sparsity and the CV curve.
"""
import numpy as np                                                 # coefficient paths
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # path and CV figures
from sklearn.linear_model import (RidgeCV, LassoCV, ElasticNetCV,   # CV-fitted penalties
                                  Ridge, Lasso)                    # single-strength penalties for the paths
from sklearn.metrics import mean_squared_error                     # test error
from sklearn.model_selection import train_test_split, KFold        # split and folds
from sklearn.pipeline import Pipeline                              # scaling plus model
from sklearn.preprocessing import StandardScaler                   # standardisation is mandatory before penalties
from _common_l6 import RNG, DAT, save, dump, style_axes, annotated, OKABE   # shared helpers

# ---------------------------------------------------------------- 1) data
prostate = pd.read_csv(DAT / "l6_prostate.csv")                    # the sample from lesson 6.1
cols = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]   # predictors
noise_cols = RNG.normal(0, 1, (len(prostate), 6))                   # six pure-noise predictors for the demo
X = np.column_stack([prostate[cols].values, noise_cols])           # design matrix: 8 real + 6 noise columns
cols = cols + ["noise%d" % (i + 1) for i in range(6)]               # extend the names for the paths
y = prostate["lpsa"].values                                        # target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   # hold-out split
print("train/test:", X_train.shape, X_test.shape)                  # report the split

# ---------------------------------------------------------------- 2) fit the three penalties with CV
alphas = np.logspace(-3, 2.2, 80)                                  # penalty grid
cv = KFold(n_splits=8, shuffle=True, random_state=42)              # eight folds for a stable CV curve
models = {"Ridge": RidgeCV(alphas=alphas, cv=cv),                  # L2 penalty
          "Lasso": LassoCV(alphas=alphas, cv=cv, max_iter=50000, random_state=42),   # L1 penalty
          "ElasticNet": ElasticNetCV(alphas=alphas, l1_ratio=[0.2, 0.5, 0.8], cv=cv,   # mixed penalty
                                     max_iter=50000, random_state=42)}   # generous solver budget
rows = []                                                          # score table
fitted = {}                                                        # keep the fitted pipelines
for name, model in models.items():                                 # loop over the three penalties
    pipe = Pipeline([("scale", StandardScaler()), ("model", model)])   # scaling inside the pipeline
    pipe.fit(X_train, y_train)                                     # fit on the training split
    fitted[name] = pipe                                            # remember the model
    pred = pipe.predict(X_test)                                    # predictions on the hold-out split
    mdl = pipe.named_steps["model"]                                # the fitted penalty object
    coefs = mdl.coef_ if mdl.coef_.ndim == 1 else mdl.coef_[0]     # coefficients (flatten for ElasticNet)
    nonzero = int(np.sum(np.abs(coefs) > 1e-8))                    # count surviving features
    rows.append({"model": name,                                    # model name
                 "alpha": round(float(mdl.alpha_), 5),             # selected penalty strength
                 "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),   # test error
                 "nonzero_coefs": nonzero,                         # sparsity of the solution
                 "max_abs_coef": round(float(np.max(np.abs(coefs))), 4)})   # largest coefficient
scores = pd.DataFrame(rows)                                        # assemble the table
dump(scores.round(6), "l6_regularization_scores.csv")              # persist it
print(scores.to_string(index=False))                               # report the comparison

# ---------------------------------------------------------------- 3) coefficient paths on standardised data
scaler = StandardScaler().fit(X_train)                             # scaling fitted on the training split
Xtr = scaler.transform(X_train)                                    # standardised training matrix
ridge_paths, lasso_paths = [], []                                  # collect coefficients per alpha
for a in alphas:                                                   # walk the penalty grid
    ridge_paths.append(Ridge(alpha=a).fit(Xtr, y_train).coef_)     # ridge coefficients
    lasso_paths.append(Lasso(alpha=a, max_iter=20000).fit(Xtr, y_train).coef_)   # lasso coefficients
ridge_paths = np.array(ridge_paths)                                # shape: alphas by features
lasso_paths = np.array(lasso_paths)                                # shape: alphas by features

# ---------------------------------------------------------------- 4) figure: paths and the CV curve
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.6))                # three panels
for j, col in enumerate(cols):                                     # one thin line per predictor
    axes[0].plot(alphas, ridge_paths[:, j], lw=1.0, color=OKABE[j % len(OKABE)], alpha=0.85, label=col)   # ridge path
axes[0].set_xscale("log")                                          # penalty on a log axis
style_axes(axes[0], "Ridge: coefficients shrink smoothly", "alpha (log scale)", "coefficient")   # shared style
axes[0].legend(fontsize=6.0, ncol=2, loc="lower left")             # compact two-column legend
for j, col in enumerate(cols):                                     # one thin line per predictor
    axes[1].plot(alphas, lasso_paths[:, j], lw=1.0, color=OKABE[j % len(OKABE)], alpha=0.85)   # lasso path
axes[1].set_xscale("log")                                          # penalty on a log axis
axes[1].axhline(0, color="#66717D", lw=0.8, ls="--")               # zero reference
style_axes(axes[1], "LASSO: coefficients reach zero", "alpha (log scale)", "coefficient")   # shared style
annotated(axes[1], "sparse solutions", "upper right")               # annotate the sparsity
lasso_mdl = fitted["Lasso"].named_steps["model"]                   # the CV-fitted lasso
mse_path = lasso_mdl.mse_path_.mean(axis=1)                        # mean CV error per alpha
axes[2].plot(lasso_mdl.alphas_, np.sqrt(mse_path), lw=1.8, color=OKABE[5], label="CV RMSE (LASSO)")   # CV curve
axes[2].axvline(lasso_mdl.alpha_, ls="--", lw=1.2, color="#66717D", label="chosen $\\alpha$")   # chosen strength
axes[2].set_xscale("log")                                          # penalty on a log axis
style_axes(axes[2], "Cross-validation picks the penalty", "alpha (log scale)", "CV RMSE")   # shared style
annotated(axes[2], "$\\alpha$ = %.4f" % lasso_mdl.alpha_, "upper right")   # annotate the choice
axes[2].legend(fontsize=7.2, loc="lower right")                    # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_regularization_paths.png")                          # save the figure
print("done: regularization paths and cross-validated strength")    # closing log
