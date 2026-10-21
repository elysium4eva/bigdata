"""l6_01_linreg_normal_equation.py -- least squares, the normal equation and the projection view.

Every line carries an English comment. The script builds a prostate-cancer-like
sample, solves least squares three ways (NumPy normal equation, scikit-learn,
statsmodels), checks that the coefficients agree, and draws the geometric view of
the normal equation next to the usual fit and residual diagnostics.
"""
import numpy as np                                                 # linear algebra for the closed form
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # figures for the fit and the geometry
from sklearn.linear_model import LinearRegression                  # reference implementation
import statsmodels.api as sm                                       # inference-oriented implementation
from _common_l6 import (RNG, save, dump, add_intercept,            # shared helpers from the lecture module
                        normal_equation, rmse, r2_score, style_axes, annotated, OKABE)   # # helper names imported from the lecture module

# ---------------------------------------------------------------- 1) build a prostate-like sample
N = 97                                                             # ESL prostate data has 97 rows
lcavol = RNG.normal(1.35, 1.15, N)                                 # log cancer volume
lweight = RNG.normal(3.6, 0.43, N)                                 # log prostate weight
age = RNG.normal(66, 9, N)                                         # age in years
lbph = RNG.normal(0.1, 1.15, N)                                    # log benign prostatic hyperplasia
svi = (RNG.random(N) < 0.23).astype(int)                           # seminal vesicle invasion (0/1)
lcp = RNG.normal(-0.2, 1.2, N)                                     # log capsular penetration
gleason = RNG.choice([6, 7, 8, 9], N, p=[0.5, 0.35, 0.1, 0.05])    # Gleason score
pgg45 = np.clip(RNG.normal(25, 25, N), 0, 100)                     # percentage Gleason 4 or 5
lpsa = (0.55 * lcavol + 0.5 * lweight + 0.1 * lbph + 0.3 * lcp     # the true linear signal
        + 0.4 * svi + 0.02 * pgg45 + 0.01 * age                    # remaining true effects
        + 0.35 * gleason - 2.3                                     # Gleason is on a different scale
        + RNG.normal(0, 0.55, N))                                  # irreducible noise
prostate = pd.DataFrame({"lcavol": lcavol.round(3), "lweight": lweight.round(3),   # assemble the predictors
                         "age": age.round(1), "lbph": lbph.round(3),              # continue with the features
                         "svi": svi, "lcp": lcp.round(3),                         # binary and continuous features
                         "gleason": gleason, "pgg45": pgg45.round(1),             # grade variables
                         "lpsa": lpsa.round(3)})                                 # target: log PSA
dump(prostate, "l6_prostate.csv")                                  # persist the sample
cols = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]   # predictor order
X = prostate[cols].values                                          # design matrix without intercept
y = prostate["lpsa"].values                                        # target vector
Xa = add_intercept(X)                                              # augmented design matrix X' = [1, X]
print("rows:", X.shape[0], "| predictors:", X.shape[1], "| target sd: %.3f" % y.std())   # report the sample

# ---------------------------------------------------------------- 2) three ways to solve least squares
w_np = normal_equation(Xa, y)                                      # w = (X'^T X')^{-1} X'^T y by our helper
w_sk = LinearRegression(fit_intercept=False).fit(Xa, y).coef_      # scikit-learn on the augmented matrix
w_sm = sm.OLS(y, Xa).fit().params                                  # statsmodels ordinary least squares
print("numpy   :", np.round(w_np, 4))                              # coefficients from the normal equation
print("sklearn :", np.round(w_sk, 4))                              # coefficients from scikit-learn
print("statsmodels:", np.round(w_sm, 4))                           # coefficients from statsmodels
max_gap = float(np.max(np.abs(w_np - w_sk)))                       # largest gap between two implementations
assert max_gap < 1e-8, "implementations disagree"                  # they must agree to numerical precision

# ---------------------------------------------------------------- 3) fit quality
y_hat = Xa @ w_np                                                  # fitted values
resid = y - y_hat                                                  # residuals
gram = Xa.T @ Xa                                                   # gram matrix X'^T X'
cond = float(np.linalg.cond(gram))                                 # condition number of the normal equations
print("RMSE %.4f | R^2 %.4f | cond(X'^TX') %.1f" % (rmse(y, y_hat), r2_score(y, y_hat), cond))   # report fit
orthogonality = Xa.T @ resid                                       # dot products of residuals with the columns
print("max |column . residual| = %.2e" % float(np.max(np.abs(orthogonality))))   # should be ~0 by construction

coef_table = pd.DataFrame({"term": ["intercept"] + cols,           # term names in the augmented order
                           "numpy": np.round(w_np, 4),             # normal-equation solution
                           "sklearn": np.round(w_sk, 4),           # scikit-learn solution
                           "statsmodels": np.round(w_sm, 4)})      # statsmodels solution
dump(coef_table, "l6_coef_compare.csv")                            # persist the comparison table

# ---------------------------------------------------------------- 4) figure one: the projection geometry
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6))                # three panels: projection, orthogonality, spectrum
# Panel A draws the projection of y onto a one-dimensional subspace plus the intercept.
x1 = X[:, 0]                                                       # use lcavol as the single predictor
beta = np.polyfit(x1, y, 1)                                        # least squares line for the illustration
axes[0].scatter(x1, y, s=18, color=OKABE[0], alpha=0.75, label="observations")   # raw data
xs = np.linspace(x1.min(), x1.max(), 50)                           # grid for the fitted line
axes[0].plot(xs, np.polyval(beta, xs), color=OKABE[5], lw=2.0, label="projection of y")   # fitted line
ix = int(np.argmax(np.abs(resid)))                                 # pick the largest residual for the diagram
axes[0].vlines(x1[ix], y[ix], np.polyval(beta, x1[ix]), color="#66717D", lw=1.4)   # draw the residual segment
axes[0].annotate("residual $y-\\hat y$", xy=(x1[ix], (y[ix] + np.polyval(beta, x1[ix])) / 2),   # label the residual
                 xytext=(8, 0), textcoords="offset points", fontsize=8.0, color="#1E2630")   # small offset label
style_axes(axes[0], "Least squares = projection", "lcavol (log cancer volume)", "lpsa (log PSA)")   # shared style
axes[0].legend(fontsize=7.6, loc="upper left")                     # legend inside the panel
axes[1].bar(["1"] + cols, orthogonality, color=OKABE[2], alpha=0.9)   # residual is orthogonal to every column
axes[1].axhline(0, color="#66717D", lw=0.9)                        # zero reference
style_axes(axes[1], "Normal equation: $X'^\\top r = 0$", "", "column $\\cdot$ residual")   # shared style
axes[1].tick_params(axis="x", rotation=60, labelsize=8.0)          # rotate the column names
sv = np.linalg.svd(Xa, compute_uv=False)                           # singular values of the design matrix
axes[2].semilogy(range(1, len(sv) + 1), sv, "-o", ms=4, color=OKABE[3])   # spectrum on a log scale
axes[2].axhline(sv.max() / cond ** 0.5, ls="--", lw=1.0, color="#66717D")   # reference line for the ratio
style_axes(axes[2], "Spectrum sets the condition number", "index of the singular value", "singular value")   # style
annotated(axes[2], "cond(X'^TX') = %.0f" % cond, "upper right")     # annotate the condition number
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_projection.png")                                    # save the geometry figure

# ---------------------------------------------------------------- 5) figure two: fit, residuals, coefficients
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6))                # three panels: fit, residual, coefficients
axes[0].scatter(y_hat, y, s=18, color=OKABE[1], alpha=0.8)         # predicted versus observed
lims = [min(y.min(), y_hat.min()), max(y.max(), y_hat.max())]       # common limits for the diagonal
axes[0].plot(lims, lims, ls="--", lw=1.0, color="#66717D")          # perfect prediction diagonal
style_axes(axes[0], "Predicted vs observed (fitted values)", "fitted $\\hat y$", "observed $y$")   # shared style
annotated(axes[0], "RMSE %.2f | $R^2$ %.2f" % (rmse(y, y_hat), r2_score(y, y_hat)))   # annotate the fit quality
axes[1].scatter(y_hat, resid, s=18, color=OKABE[5], alpha=0.8)     # residual versus fitted
axes[1].axhline(0, color="#66717D", lw=0.9)                        # zero reference
style_axes(axes[1], "Residuals carry no remaining trend", "fitted $\\hat y$", "residual $y-\\hat y$")   # shared style
terms = ["intercept"] + cols                                       # coefficient labels
axes[2].barh(terms[::-1], w_np[::-1], color=OKABE[3], alpha=0.9)   # coefficients, largest on top
axes[2].axvline(0, color="#66717D", lw=0.9)                        # zero reference
style_axes(axes[2], "Normal-equation coefficients", "coefficient", "")   # shared style
axes[2].tick_params(axis="y", labelsize=7.6)                        # small labels for eight predictors
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_prostate_fit.png")                                  # save the fit figure
print("done: normal equation, fit and geometry figures")            # closing log
