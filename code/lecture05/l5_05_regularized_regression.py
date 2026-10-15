"""l5_05_regularized_regression.py -- OLS versus Ridge, Lasso and Elastic Net.

Every line carries an English comment. The script shows what the penalty does to
the coefficients (shrinkage versus sparsity), how the cross-validated error picks
the strength, and how the families compare out of sample.
"""
import time                                                        # measure the fitting cost of each model
import numpy as np                                                 # numeric helpers for the coefficient paths
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # figures for the paths and the comparison
from sklearn.linear_model import (LinearRegression, RidgeCV,       # ordinary least squares and ridge with CV
                                  LassoCV, ElasticNetCV, Lasso, Ridge)   # lasso, elastic net and single-alpha models
from sklearn.metrics import mean_squared_error                     # error measure on the test split
from sklearn.model_selection import train_test_split               # hold-out split
from sklearn.pipeline import Pipeline                              # polynomial expansion plus model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler   # expansion and scaling
from _common_l5 import DAT, save, dump, style_axes, OKABE          # shared helpers

# ---------------------------------------------------------------- 1) non-linear housing sample from lesson 5.3
housing = pd.read_csv(DAT / "l5_housing.csv")                      # read the housing sample
X = pd.get_dummies(housing.drop(columns=["price_k"]), drop_first=True)   # one-hot encode the district column
y = housing["price_k"].values                                      # target in thousands of CNY
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)   # 75/25 split
print("rows:", len(housing), "| base features:", X.shape[1])        # report the data size

# ---------------------------------------------------------------- 2) expand to a deliberately wide design matrix
poly = PolynomialFeatures(degree=3, include_bias=False)            # cubic expansion creates many columns
Xtr_poly = poly.fit_transform(X_train)                             # expand the training features
Xte_poly = poly.transform(X_test)                                  # apply the same expansion to the test split
scaler = StandardScaler().fit(Xtr_poly)                            # scaling is fitted on the training split only
Xtr = scaler.transform(Xtr_poly)                                   # scaled training matrix
Xte = scaler.transform(Xte_poly)                                   # scaled test matrix
feature_names = poly.get_feature_names_out(X.columns)              # human readable names of the expanded features
print("expanded features:", Xtr.shape[1], "| training rows:", Xtr.shape[0])   # wide means p is close to n

# ---------------------------------------------------------------- 3) fit the regression family
alphas = np.logspace(-3, 2, 60)                                    # grid of penalty strengths, log spaced
models = {                                                         # one entry per family
    "OLS": LinearRegression(),                                     # no penalty at all
    "Ridge (CV)": RidgeCV(alphas=alphas, cv=5),                    # L2 penalty, strength chosen by CV
    "Lasso (CV)": LassoCV(alphas=alphas, cv=5, max_iter=20000, random_state=42),   # L1 penalty, sparse solutions
    "ElasticNet (CV)": ElasticNetCV(alphas=alphas, l1_ratio=[0.2, 0.5, 0.8], cv=5,   # mixed L1 and L2 penalty
                                    max_iter=20000, random_state=42),   # keep the solver budget generous
}                                                                  # end of the model dictionary
rows = []                                                          # one row of scores per model
coefs = {}                                                         # keep the coefficients for the comparison panel
for name, model in models.items():                                 # loop over the four families
    t0 = time.time()                                               # start the clock
    model.fit(Xtr, y_train)                                        # fit on the training matrix
    seconds = time.time() - t0                                     # record the fitting cost
    pred = model.predict(Xte)                                      # predict on the held-out test rows
    nz = int(np.sum(np.abs(model.coef_) > 1e-8))                   # count coefficients that stayed non-zero
    alpha = float(getattr(model, "alpha_", 0.0))                   # selected penalty strength when available
    rows.append({"model": name, "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),   # test error
                 "non_zero_coefs": nz, "n_features": int(Xtr.shape[1]),   # sparsity of the solution
                 "alpha": round(alpha, 4), "fit_seconds": round(seconds, 3)})   # penalty and cost
    coefs[name] = model.coef_.copy()                               # store the coefficient vector
scores = pd.DataFrame(rows)                                        # assemble the score table
scores["sparsity_pct"] = (100 * (1 - scores["non_zero_coefs"] / scores["n_features"])).round(1)   # share of zeros
dump(scores, "l5_regression_scores.csv")                           # persist the comparison table
print(scores.to_string(index=False))                               # print it in the console

tidy = pd.DataFrame({"feature": feature_names,                     # one row per expanded feature
                     "OLS": coefs["OLS"],                          # unpenalised coefficients
                     "Ridge": coefs["Ridge (CV)"],                 # shrunken coefficients
                     "Lasso": coefs["Lasso (CV)"]})                # sparse coefficients
dump(tidy.round(4), "l5_coefficients.csv")                         # persist the coefficient table

# ---------------------------------------------------------------- 4) coefficient paths along the penalty grid
paths_ridge, paths_lasso = [], []                                  # collect the paths of both penalties
for a in alphas:                                                   # walk the penalty grid from weak to strong
    r = Ridge(alpha=a).fit(Xtr, y_train)                           # ridge at this strength
    l = Lasso(alpha=a, max_iter=20000).fit(Xtr, y_train)           # lasso at this strength
    paths_ridge.append(r.coef_)                                    # store the ridge coefficients
    paths_lasso.append(l.coef_)                                    # store the lasso coefficients
paths_ridge = np.array(paths_ridge)                                # shape: alphas by features
paths_lasso = np.array(paths_lasso)                                # shape: alphas by features

# ---------------------------------------------------------------- 5) figure one: what the penalties do
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6))                # three panels for the geometry of the penalties
for j in range(paths_ridge.shape[1]):                              # draw one thin line per feature
    axes[0].plot(alphas, paths_ridge[:, j], lw=0.9, color=OKABE[j % len(OKABE)], alpha=0.75)   # ridge path
axes[0].set_xscale("log")                                          # the penalty is easier to read on a log axis
style_axes(axes[0], "Ridge: shrinkage, never exactly zero", "alpha (log scale)", "Coefficient")   # shared style
for j in range(paths_lasso.shape[1]):                              # draw one thin line per feature
    axes[1].plot(alphas, paths_lasso[:, j], lw=0.9, color=OKABE[j % len(OKABE)], alpha=0.75)   # lasso path
axes[1].set_xscale("log")                                          # matching log axis
axes[1].axhline(0, color="#66717D", lw=0.8, ls="--")                # zero reference line
style_axes(axes[1], "Lasso: coefficients hit exactly zero", "alpha (log scale)", "Coefficient")   # shared style
axes[2].bar(scores["model"], scores["test_rmse"], color=[OKABE[3], OKABE[0], OKABE[2], OKABE[1]],   # test error per family
            alpha=0.88)                                            # slightly transparent bars
for i, v in enumerate(scores["test_rmse"]):                        # annotate each bar with its value
    axes[2].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9.0)   # RMSE label above the bar
axes[2].tick_params(axis="x", labelsize=8.0)                       # keep the long model names readable
style_axes(axes[2], "Out-of-sample error is the referee", "", "Test RMSE (thousand CNY)")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_regularization_paths.png")                          # save the coefficient-path figure

# ---------------------------------------------------------------- 6) figure two: sparsity and coefficient agreement
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))                # three panels comparing the families
axes[0].barh(scores["model"], scores["non_zero_coefs"], color=[OKABE[3], OKABE[0], OKABE[2], OKABE[1]],   # sparsity
             alpha=0.88)                                           # transparent bars
axes[0].axvline(int(Xtr.shape[1]), color="#66717D", ls="--", lw=1.0)   # total number of expanded features
style_axes(axes[0], "How many coefficients survive?", "Non-zero coefficients", "")   # shared style
order = np.argsort(-np.abs(coefs["OLS"]))[:12]                     # twelve largest OLS coefficients
xpos = np.arange(len(order))                                       # x positions for the grouped bars
axes[1].bar(xpos - 0.21, coefs["OLS"][order], width=0.42, color=OKABE[3], alpha=0.9, label="OLS")   # unpenalised
axes[1].bar(xpos + 0.21, coefs["Lasso (CV)"][order], width=0.42, color=OKABE[2], alpha=0.9, label="Lasso")   # sparse
axes[1].set_xticks(xpos, [feature_names[i] for i in order], rotation=70, ha="right", fontsize=6.2)   # feature labels
style_axes(axes[1], "Same features, smaller weights", "", "Coefficient")   # shared style
axes[1].legend(fontsize=7.6)                                       # legend inside the panel
test_preds = {name: model.predict(Xte) for name, model in models.items()}   # predictions of every family
resid = {name: (y_test - p) for name, p in test_preds.items()}      # residuals of every family
for name, colour in zip(models.keys(), [OKABE[3], OKABE[0], OKABE[2], OKABE[1]]):   # one histogram per family
    axes[2].hist(resid[name], bins=32, histtype="step", lw=1.6, color=colour, label=name)   # residual distribution
axes[2].axvline(0, color="#66717D", lw=0.9, ls="--")                # zero residual reference
style_axes(axes[2], "Residual spread is similar, tails differ", "Residual (thousand CNY)", "Count")   # shared style
axes[2].legend(fontsize=7.2, loc="upper right")                    # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_regression_family.png")                             # save the family comparison figure
