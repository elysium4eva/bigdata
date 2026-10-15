"""l5_06_knn_svm_tuning.py -- KNN and SVM: decision boundaries and honest tuning.

Every line carries an English comment. The script builds a two-moons sample, shows
how the KNN boundary changes with k, compares SVM kernels, and runs a grid search
inside cross-validation so that tuning never touches the test split.
"""
import time                                                        # measure the search cost
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # figures for boundaries and tuning curves
from sklearn.datasets import make_moons                            # two interleaving half circles
from sklearn.model_selection import (train_test_split,             # hold-out split
                                     GridSearchCV,                 # exhaustive hyper-parameter search
                                     cross_val_score)              # cross-validated scoring
from sklearn.neighbors import KNeighborsClassifier                 # k-nearest neighbours
from sklearn.pipeline import Pipeline                              # scaling plus model
from sklearn.preprocessing import StandardScaler                   # scaling matters for both of these models
from sklearn.svm import SVC                                        # support vector classifier with kernels
from _common_l5 import RNG, save, dump, style_axes, OKABE          # shared helpers

# ---------------------------------------------------------------- 1) a non-linear classification sample
X, y = make_moons(n_samples=600, noise=0.28, random_state=42)      # two interleaving crescent shapes
X = X + RNG.normal(0, 0.15, X.shape)                               # add a little extra jitter for realism
moons = pd.DataFrame({"x1": X[:, 0].round(3), "x2": X[:, 1].round(3), "label": y})   # tabulate the sample
dump(moons, "l5_two_moons.csv")                                    # persist the sample
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)   # split
print("moons:", X.shape, "| train/test:", len(y_train), len(y_test))   # report the sizes

# ---------------------------------------------------------------- 2) grid for drawing decision boundaries
x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6            # horizontal extent of the plot
y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6            # vertical extent of the plot
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 260),               # dense grid along the first axis
                     np.linspace(y_min, y_max, 260))               # dense grid along the second axis
grid = np.c_[xx.ravel(), yy.ravel()]                               # flatten into a matrix of query points

def boundary(ax, model, title):                                    # helper that draws one decision boundary
    """Fit a model, draw its boundary on the grid and scatter the training points."""   # short docstring
    model.fit(X_train, y_train)                                    # fit on the training split only
    zz = model.predict(grid).reshape(xx.shape)                     # predict the class of every grid point
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], alpha=0.18,   # shade the two decision regions
                colors=[OKABE[1], OKABE[5]])                       # blue for class 0, orange for class 1
    ax.contour(xx, yy, zz, levels=[0.5], colors="#1E2630", linewidths=1.2)   # draw the boundary line itself
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], s=13,   # scatter class 0 training points
               color=OKABE[1], edgecolors="white", lw=0.3, label="class 0")   # light border keeps points visible
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], s=13,   # scatter class 1 training points
               color=OKABE[5], edgecolors="white", lw=0.3, label="class 1")   # second class
    acc = model.score(X_test, y_test)                              # test accuracy of the fitted model
    style_axes(ax, f"{title} | test acc {acc:.3f}", "", "")         # title carries the out-of-sample score
    return acc                                                     # return the accuracy for the summary table

# ---------------------------------------------------------------- 3) KNN: the effect of k
fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.6))                # four panels: three KNN values and one SVM
knn_scores = {}                                                    # collect the accuracy per k
for ax, k in zip(axes[:3], [1, 15, 99]):                           # k = 1 (overfit), 15 (good), 99 (too smooth)
    pipe = Pipeline([("scale", StandardScaler()),                  # distance based models need scaling
                     ("model", KNeighborsClassifier(n_neighbors=k))])   # KNN with this neighbourhood size
    knn_scores[k] = boundary(ax, pipe, f"KNN k={k}")               # draw the boundary and record the score
print("KNN test accuracy:", {k: round(v, 3) for k, v in knn_scores.items()})   # report the k sensitivity

# ---------------------------------------------------------------- 4) SVM: kernel and penalty strength
svm_rows = []                                                      # one row per configuration
for kernel in ["linear", "rbf"]:                                   # compare a linear and a radial kernel
    for C in [0.1, 1.0, 10.0, 100.0]:                              # sweep the penalty strength
        pipe = Pipeline([("scale", StandardScaler()),              # scaling keeps the kernel scale meaningful
                         ("model", SVC(kernel=kernel, C=C, gamma="scale", probability=False))])   # classifier
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy").mean()   # honest CV score
        pipe.fit(X_train, y_train)                                 # fit on the full training split
        svm_rows.append({"kernel": kernel, "C": C,                 # configuration
                         "cv_accuracy": round(float(cv), 4),       # cross-validated accuracy
                         "test_accuracy": round(float(pipe.score(X_test, y_test)), 4)})   # held-out accuracy
svm_table = pd.DataFrame(svm_rows)                                 # assemble the tuning table
print(svm_table.to_string(index=False))                            # print the sweep
best_row = svm_table.loc[svm_table["cv_accuracy"].idxmax()]        # the configuration CV likes best
best_svm = Pipeline([("scale", StandardScaler()),                  # rebuild the winning pipeline
                     ("model", SVC(kernel=best_row["kernel"], C=best_row["C"]))])   # with the selected hyper-parameters
boundary(axes[3], best_svm, f"SVM {best_row['kernel']} C={best_row['C']:g}")   # draw its boundary

# ---------------------------------------------------------------- 5) grid search inside cross-validation
param_grid = {"model__n_neighbors": [3, 5, 9, 15, 25, 41],         # neighbourhood sizes to try
              "model__weights": ["uniform", "distance"]}           # weighting schemes to try
grid_pipe = Pipeline([("scale", StandardScaler()),                 # the same preprocessing
                      ("model", KNeighborsClassifier())])          # the estimator with placeholders
t0 = time.time()                                                   # start the clock
search = GridSearchCV(grid_pipe, param_grid, cv=5, scoring="accuracy", n_jobs=1)   # exhaustive search with 5 folds
search.fit(X_train, y_train)                                       # tuning uses the training split only
search_seconds = time.time() - t0                                  # record the search cost
print("grid search: best %s | CV %.4f | test %.4f | %.2fs" % (search.best_params_,   # best configuration
      search.best_score_, search.score(X_test, y_test), search_seconds))   # best CV score, test score and cost
tuning = pd.DataFrame(search.cv_results_)[["param_model__n_neighbors", "param_model__weights",   # readable columns
                                           "mean_test_score", "std_test_score"]]   # mean and spread per candidate
dump(tuning.sort_values("mean_test_score", ascending=False), "l5_tuning_results.csv")   # persist the tuning table
dump(svm_table, "l5_svm_tuning.csv")                               # persist the SVM sweep as well

# ---------------------------------------------------------------- 6) annotate the final panel with the search result
axes[3].text(0.02, 0.03, f"grid search best k={search.best_params_['model__n_neighbors']}\n"   # report the winning k
             f"CV {search.best_score_:.3f} | test {search.score(X_test, y_test):.3f}",   # and the two scores
             transform=axes[3].transAxes, fontsize=7.6, color="#1E2630",   # small note in the corner
             bbox=dict(boxstyle="round,pad=0.25", fc="#F7F3F2", ec="#D6CDD0", lw=0.6))   # pale background box
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_knn_svm.png")                                       # save the boundaries figure
