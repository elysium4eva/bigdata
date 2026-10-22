"""l7_02_svm_kernels.py -- the four standard kernels and their decision boundaries.

Every line carries an English comment. The script fits an SVM with a linear,
polynomial, RBF and sigmoid kernel on the same sample, draws the four decision
regions, and records cross-validated accuracy and the best (C, gamma) pair.
"""
import numpy as np                                                 # numeric helpers and the grid
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # four decision-region panels
from matplotlib.colors import ListedColormap                       # discrete colour map for the regions
from sklearn.svm import SVC                                        # the kernel SVM
from sklearn.model_selection import GridSearchCV, cross_val_score   # tuning and scoring
from _common_l7 import (DAT, save, dump, style_axes, annotated, OKABE)   # shared helpers

# ---------------------------------------------------------------- 1) the sample
data = pd.read_csv(DAT / "l7_two_class.csv")                       # the sample from lesson 7.1
X = data[["x1", "x2"]].values                                      # feature matrix
y = data["y"].values                                               # binary labels
print("rows:", len(data), "| positive rate: %.3f" % y.mean())      # report the sample

# ---------------------------------------------------------------- 2) fit the four kernels
kernels = {"linear": {"C": 1.0}, "poly": {"C": 1.0, "degree": 3},   # linear and cubic polynomial
           "rbf": {"C": 1.0, "gamma": 1.0}, "sigmoid": {"C": 1.0, "gamma": 0.5}}   # RBF and sigmoid
fitted = {}                                                        # keep the models for plotting
rows = []                                                          # per-kernel summary
for name, params in kernels.items():                               # loop over the four kernels
    model = SVC(kernel=name, **params)                             # build the SVM
    model.fit(X, y)                                                # fit on the whole sample
    fitted[name] = model                                           # remember the model
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")   # five-fold accuracy
    rows.append({"kernel": name, "cv_mean": round(float(scores.mean()), 4),   # mean accuracy
                 "cv_std": round(float(scores.std()), 4),          # spread across the folds
                 "support_vectors": len(model.support_)})          # support vector count
    print("%-8s cv=%.3f +/- %.3f | SV=%d" % (name, scores.mean(), scores.std(), len(model.support_)))   # report
summary = pd.DataFrame(rows)                                       # assemble the table

# ---------------------------------------------------------------- 3) grid search for C and gamma on RBF
param_grid = {"C": [0.1, 1.0, 10.0, 100.0], "gamma": [0.05, 0.2, 1.0, 5.0]}   # a small two-dimensional grid
search = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5, scoring="accuracy")   # five-fold search
search.fit(X, y)                                                   # run the search
print("best RBF: C=%.2f gamma=%.2f | cv=%.4f" % (                       # report the winning configuration
    search.best_params_["C"], search.best_params_["gamma"], search.best_score_))   # report the winner
summary = pd.concat([summary, pd.DataFrame([{"kernel": "rbf (tuned)",   # add the tuned row
                                             "cv_mean": round(float(search.best_score_), 4),   # tuned score
                                             "cv_std": 0.0,        # not applicable
                                             "support_vectors": len(search.best_estimator_.support_)}])],   # support vectors of the tuned model
                    ignore_index=True)                             # append to the summary
dump(summary, "l7_svm_kernels.csv")                                # persist the table

# ---------------------------------------------------------------- 4) figure: four decision regions
xs = np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 240)     # horizontal grid
ys = np.linspace(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3, 240)     # vertical grid
XX, YY = np.meshgrid(xs, ys)                                       # full grid
grid = np.c_[XX.ravel(), YY.ravel()]                               # flatten it for prediction
cmap2 = ListedColormap([OKABE[1], OKABE[5]])                       # one colour per class
fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.4))                # four panels
for ax, (name, model) in zip(axes, fitted.items()):                # one panel per kernel
    Z = model.predict(grid).reshape(XX.shape)                      # predicted class on the grid
    ax.contourf(XX, YY, Z, levels=[-0.5, 0.5, 1.5], alpha=0.22, cmap=cmap2)   # colour the regions
    ax.contour(XX, YY, Z, levels=[0.5], colors="#1E2630", linewidths=1.0)      # draw the boundary
    m = y == 1                                                     # mask for the positive class
    ax.scatter(X[~m, 0], X[~m, 1], s=10, color=OKABE[1], alpha=0.8)            # negatives
    ax.scatter(X[m, 0], X[m, 1], s=10, color=OKABE[5], alpha=0.8)              # positives
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=34,   # support vectors
               facecolors="none", edgecolors="#1E2630", linewidths=0.8)          # hollow markers
    score = float(summary.loc[summary["kernel"] == name, "cv_mean"].iloc[0])     # cross-validated score
    style_axes(ax, "%s kernel | cv %.3f" % (name, score), "x1", "")   # shared style
    ax.set_yticks([])                                              # hide the y ticks to save space
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_svm_kernels.png")                                   # save the kernel comparison
print("done: four kernels and a tuned RBF")                        # closing log
