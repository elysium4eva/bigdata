"""l7_03_knn_k_curve.py -- the k-nearest-neighbour classifier: 1-NN vs 5-NN maps and the k curve.

Every line carries an English comment. The script draws the classic classification
maps for k = 1 and k = 5, then sweeps k from 1 to 40 with cross-validation so the
bias-variance trade-off of k becomes a measured curve rather than a slogan.
"""
import numpy as np                                                 # numeric helpers and the grid
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # three panels
from matplotlib.colors import ListedColormap                       # discrete colour map for the maps
from sklearn.neighbors import KNeighborsClassifier                 # the k-NN classifier
from sklearn.model_selection import cross_val_score, train_test_split   # scoring and split
from sklearn.metrics import accuracy_score                         # accuracy on the hold-out split
from _common_l7 import (make_knn_sample, save, dump,               # shared helpers
                        style_axes, annotated, OKABE)                   # helper names imported from the lecture module

# ---------------------------------------------------------------- 1) the sample
data = make_knn_sample(240)                                        # three-cluster two-dimensional sample
dump(data, "l7_knn_sample.csv")                                    # persist it
X = data[["x1", "x2"]].values                                      # feature matrix
y = data["label"].values                                           # three-class labels
print("rows:", len(data), "| classes:", np.bincount(y).tolist())     # report the class balance

# ---------------------------------------------------------------- 2) the two classic maps
xs = np.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 260)     # horizontal grid
ys = np.linspace(X[:, 1].min() - 0.4, X[:, 1].max() + 0.4, 260)     # vertical grid
XX, YY = np.meshgrid(xs, ys)                                       # full grid
grid = np.c_[XX.ravel(), YY.ravel()]                               # flatten for prediction
cmap3 = ListedColormap([OKABE[1], OKABE[5], OKABE[2]])             # one colour per class
points = [OKABE[1], OKABE[5], OKABE[2]]                            # matching scatter colours

# ---------------------------------------------------------------- 3) sweep k with cross-validation
ks = list(range(1, 41))                                            # candidate neighbourhood sizes
cv_rows = []                                                       # cross-validation results per k
for k in ks:                                                       # loop over the neighbourhood sizes
    model = KNeighborsClassifier(n_neighbors=k)                    # the classifier for this k
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")   # five-fold accuracy
    model.fit(X, y)                                                # fit on everything for the error curve
    train_acc = accuracy_score(y, model.predict(X))                # in-sample accuracy
    cv_rows.append({"k": k, "cv_mean": round(float(scores.mean()), 4),   # cross-validated accuracy
                    "cv_std": round(float(scores.std()), 4),       # fold spread
                    "train_acc": round(float(train_acc), 4)})      # in-sample accuracy
    print("k=%2d | cv=%.3f | train=%.3f" % (k, scores.mean(), train_acc))   # report each k
cv_table = pd.DataFrame(cv_rows)                                   # assemble the table
dump(cv_table, "l7_knn_k_curve.csv")                               # persist it
best_k = int(cv_table.loc[cv_table["cv_mean"].idxmax(), "k"])       # best k by cross-validation
print("best k:", best_k, "| cv accuracy %.3f" % cv_table["cv_mean"].max())   # report the choice

# ---------------------------------------------------------------- 4) hold-out check at k = 1 and best k
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)   # split
holdout = {}                                                       # hold-out accuracy per k
for k in (1, 5, best_k):                                           # the three values we discuss
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)   # fit on the training split
    holdout[k] = round(float(accuracy_score(y_test, model.predict(X_test))), 4)   # test accuracy
    print("k=%2d | hold-out accuracy %.4f" % (k, holdout[k]))       # report it

# ---------------------------------------------------------------- 5) figure: maps and the k curve
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7))                # three panels
for ax, k in zip(axes[:2], (1, 5)):                                # the 1-NN and 5-NN maps
    model = KNeighborsClassifier(n_neighbors=k).fit(X, y)          # fit with this k
    Z = model.predict(grid).reshape(XX.shape)                      # predicted class on the grid
    ax.contourf(XX, YY, Z, levels=[-0.5, 0.5, 1.5, 2.5], alpha=0.26, cmap=cmap3)   # colour the regions
    for c in range(3):                                             # scatter the three classes
        m = y == c                                                 # mask for this class
        ax.scatter(X[m, 0], X[m, 1], s=12, color=points[c], label="class %d" % c)   # points
    style_axes(ax, "%d-NN classification map" % k, "x1", "x2")      # shared style
    ax.legend(fontsize=6.6, loc="lower right")                     # compact legend
    annotated(ax, "hold-out %.3f" % holdout[k], "upper left")      # annotate the hold-out accuracy
axes[2].plot(cv_table["k"], cv_table["cv_mean"], "-o", ms=3.5, color=OKABE[2], label="5-fold CV")   # CV curve
axes[2].plot(cv_table["k"], cv_table["train_acc"], "-", lw=1.6, color=OKABE[5], label="in-sample")   # train curve
axes[2].axvline(best_k, ls="--", lw=1.0, color="#66717D")          # mark the best k
style_axes(axes[2], "k versus accuracy (best k = %d)" % best_k, "k", "accuracy")   # shared style
axes[2].legend(fontsize=7.2, loc="lower right")                    # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_knn_k_curve.png")                                   # save the k sweep

# ---------------------------------------------------------------- 6) figure: the two maps side by side
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))                 # two panels only
for ax, k in zip(axes, (1, 5)):                                    # the two neighbourhood sizes
    model = KNeighborsClassifier(n_neighbors=k).fit(X, y)          # fit
    Z = model.predict(grid).reshape(XX.shape)                      # predictions on the grid
    ax.contourf(XX, YY, Z, levels=[-0.5, 0.5, 1.5, 2.5], alpha=0.3, cmap=cmap3)   # regions
    ax.contour(XX, YY, Z, levels=[0.5, 1.5], colors="#1E2630", linewidths=0.8)    # boundaries
    for c in range(3):                                             # scatter the classes
        m = y == c                                                 # mask for this class
        ax.scatter(X[m, 0], X[m, 1], s=16, color=points[c], label="class %d" % c)   # points
    style_axes(ax, "%d-NN: %s boundaries" % (k, "fragmented" if k == 1 else "smoother"), "x1", "x2")   # style
    ax.legend(fontsize=7.0, loc="lower right")                     # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_knn_maps.png")                                      # save the maps
print("done: k-NN maps and the k curve")                           # closing log
