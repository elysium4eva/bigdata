"""l6_11_overfit_curves.py -- the approximation-overfit trade-off, measured.

Every line carries an English comment. The script fits polynomials of increasing
degree to a noisy curve, plots training versus test error, draws three fits side by
side, and estimates the squared bias and the variance by simulation so the slides
can show the decomposition rather than only describing it.
"""
import numpy as np                                                 # arrays for the simulation
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # three panels
from sklearn.linear_model import Ridge                             # ridge keeps high degrees stable
from sklearn.metrics import mean_squared_error                     # error measure
from sklearn.pipeline import Pipeline                              # polynomial expansion plus ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler   # expansion and scaling
from _common_l6 import RNG, save, dump, style_axes, annotated, OKABE   # shared helpers

# ---------------------------------------------------------------- 1) a noisy one-dimensional sample
def true_function(x):                                              # the target curve
    """Return the noise-free target, a smooth non-linear function."""   # short docstring
    return np.sin(1.6 * x) + 0.35 * x                                  # a wave plus a linear drift

N = 40                                                             # small sample to make overfitting easy
x = RNG.uniform(-3, 3, N)                                          # feature values
y = true_function(x) + RNG.normal(0, 0.35, N)                      # noisy observations
sample = pd.DataFrame({"x": x.round(4), "y": y.round(4)})          # tabulate the sample
dump(sample, "l6_poly_sample.csv")                                 # persist it
grid = np.linspace(-3, 3, 400)                                     # grid for drawing the fits
print("rows:", N, "| noise sd: 0.35")                              # report the setup

# ---------------------------------------------------------------- 2) training and test error by degree
degrees = list(range(1, 16))                                       # degrees from linear to wild
rows = []                                                          # one row per degree
for d in degrees:                                                  # loop over the degrees
    pipe = Pipeline([("poly", PolynomialFeatures(degree=d, include_bias=False)),   # expansion
                     ("scale", StandardScaler()),                  # scale the expanded features
                     ("model", Ridge(alpha=1e-6))]).fit(x.reshape(-1, 1), y)   # almost unpenalised ridge
    pred_in = pipe.predict(x.reshape(-1, 1))                       # in-sample predictions
    pred_grid = pipe.predict(grid.reshape(-1, 1))                  # predictions on the dense grid
    rows.append({"degree": d,                                      # degree
                 "train_rmse": round(float(np.sqrt(mean_squared_error(y, pred_in))), 4),   # in-sample error
                 "test_rmse": round(float(np.sqrt(mean_squared_error(true_function(grid), pred_grid))), 4)})   # true error
curve = pd.DataFrame(rows)                                         # assemble the table
dump(curve, "l6_overfit_curve.csv")                                # persist it
best_degree = int(curve.loc[curve["test_rmse"].idxmin(), "degree"])   # degree with the lowest true error
print(curve.to_string(index=False))                                # report the sweep
print("best degree:", best_degree)                                 # report the choice

# ---------------------------------------------------------------- 3) squared bias and variance by simulation
reps = 80                                                          # repeated datasets
grid_small = np.linspace(-2.6, 2.6, 60)                            # evaluation grid for the decomposition
bias_var = {}                                                      # results per degree
for d in [1, 3, 5, 9, 15]:                                         # a few informative degrees
    preds = np.zeros((reps, len(grid_small)))                      # predictions across repeats
    for r in range(reps):                                          # each repeat draws a fresh sample
        xr = RNG.uniform(-3, 3, N)                                 # new features
        yr = true_function(xr) + RNG.normal(0, 0.35, N)            # new noisy targets
        pipe = Pipeline([("poly", PolynomialFeatures(degree=d, include_bias=False)),   # same model
                         ("scale", StandardScaler()),              # same preprocessing
                         ("model", Ridge(alpha=1e-6))]).fit(xr.reshape(-1, 1), yr)   # fit on the repeat
        preds[r] = pipe.predict(grid_small.reshape(-1, 1))         # store the predictions
    mean_pred = preds.mean(axis=0)                                 # average prediction at each grid point
    bias2 = float(np.mean((mean_pred - true_function(grid_small)) ** 2))   # squared bias
    variance = float(np.mean(preds.var(axis=0)))                   # variance across repeats
    bias_var[d] = {"degree": d, "bias_sq": round(bias2, 5),        # record the decomposition
                   "variance": round(variance, 5),                 # variance term
                   "total": round(bias2 + variance, 5)}            # total expected squared error
decomp = pd.DataFrame(bias_var.values())                           # assemble the table
dump(decomp, "l6_bias_variance.csv")                               # persist it
print(decomp.to_string(index=False))                               # report it

# ---------------------------------------------------------------- 4) figure: sweep, fits and decomposition
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.6))                # three panels
axes[0].plot(curve["degree"], curve["train_rmse"], "-o", ms=4, color=OKABE[2], label="training error")   # in-sample
axes[0].plot(curve["degree"], curve["test_rmse"], "-s", ms=4, color=OKABE[5], label="true error")   # out-of-sample
axes[0].axvline(best_degree, ls="--", lw=1.0, color="#66717D")     # mark the best degree
style_axes(axes[0], "Approximation improves, overfitting starts", "polynomial degree", "RMSE")   # shared style
axes[0].legend(fontsize=7.6)                                       # compact legend
axes[1].scatter(x, y, s=18, color="#B9B3B6", alpha=0.85, label="noisy sample")   # the observations
axes[1].plot(grid, true_function(grid), lw=2.2, color="#1E2630", label="true function")   # ground truth
for d, colour, style in [(1, OKABE[0], "--"), (5, OKABE[2], "-"), (15, OKABE[5], ":")]:   # three fits
    pipe = Pipeline([("poly", PolynomialFeatures(degree=d, include_bias=False)),   # expansion
                     ("scale", StandardScaler()),                  # scaling
                     ("model", Ridge(alpha=1e-6))]).fit(x.reshape(-1, 1), y)   # fit
    axes[1].plot(grid, pipe.predict(grid.reshape(-1, 1)), ls=style, lw=2.0, color=colour, label="degree %d" % d)   # curve
axes[1].set_ylim(-2.2, 2.6)                                        # crop the wild degree-15 tail
style_axes(axes[1], "Same sample, three hypotheses", "x", "y")     # shared style
axes[1].legend(fontsize=7.0, loc="lower right")                    # compact legend
width = 0.55                                                       # bar width
axes[2].bar(decomp["degree"].astype(str), decomp["bias_sq"], width=width,   # squared bias
            color=OKABE[0], alpha=0.9, label="squared bias")        # legend label
axes[2].bar(decomp["degree"].astype(str), decomp["variance"], width=width,   # variance stacked on top
            bottom=decomp["bias_sq"], color=OKABE[2], alpha=0.9, label="variance")   # legend label
style_axes(axes[2], "Bias falls, variance explodes", "degree", "expected squared error")   # shared style
axes[2].legend(fontsize=7.4, loc="upper left")                     # compact legend
annotated(axes[2], "simulated with %d repeats" % reps, "upper right")   # annotate the method
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_approx_overfit.png")                                # save the figure
print("done: overfit curves and bias-variance decomposition")       # closing log
