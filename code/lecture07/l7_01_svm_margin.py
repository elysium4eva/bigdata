"""l7_01_svm_margin.py -- maximum margin, the cost parameter C and the support vectors.

Every line carries an English comment. The script fits linear SVMs with three values
of C on the same two-dimensional sample, draws the margin band and the support
vectors, and records the numbers that quantify the trade-off.
"""
import numpy as np                                                 # numeric helpers for the margins
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # figures for the margins
from sklearn.svm import SVC, LinearSVC                             # the two SVM interfaces
from sklearn.model_selection import train_test_split               # hold-out split for the score
from sklearn.metrics import accuracy_score                         # accuracy on the test split
from _common_l7 import (make_two_class_linear, save, dump,          # shared helpers
                        style_axes, annotated, OKABE)                   # helper names imported from the lecture module

# ---------------------------------------------------------------- 1) the sample
data = make_two_class_linear(200)                                  # nearly linear two-class sample
dump(data, "l7_two_class.csv")                                     # persist it for the deck
X = data[["x1", "x2"]].values                                      # feature matrix
y = data["y"].values                                               # binary labels (0/1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   # hold-out split
print("rows:", len(data), "| positive rate: %.3f" % y.mean())      # report the sample

# ---------------------------------------------------------------- 2) the margin geometry helpers
def draw_boundary(ax, model, xs, colour, label):                   # draw boundary and margins
    """Plot the decision boundary plus the two margin lines of a linear SVM."""   # short docstring
    w = model.coef_[0]                                             # normal vector of the hyperplane
    b = model.intercept_[0]                                        # offset of the hyperplane
    ax.plot(xs, -(w[0] * xs + b) / w[1], color=colour, lw=1.8, label=label)   # boundary w^T x + b = 0
    for offset in (-1.0, 1.0):                                     # the two margin lines at distance 1/|w|
        ax.plot(xs, -(w[0] * xs + b - offset) / w[1], color=colour, lw=0.9, ls="--", alpha=0.75)   # margin line

# ---------------------------------------------------------------- 3) sweep C and collect the numbers
Cs = [0.01, 0.1, 10.0]                                             # weak, moderate and strong penalty
rows = []                                                          # per-C summary table
models = {}                                                        # keep the fitted models for plotting
for C in Cs:                                                       # loop over the penalty values
    svm = SVC(kernel="linear", C=C)                                # linear SVM with this C
    svm.fit(X_train, y_train)                                      # fit on the training split
    models[C] = svm                                                # remember the model
    n_sv = len(svm.support_)                                       # number of support vectors
    w_norm = float(np.linalg.norm(svm.coef_[0]))                   # norm of the normal vector
    margin_width = 2.0 / w_norm                                    # width of the margin band
    acc = accuracy_score(y_test, svm.predict(X_test))               # test accuracy
    rows.append({"C": C, "support_vectors": n_sv,                  # penalty and support vector count
                 "w_norm": round(w_norm, 3),                       # normal vector length
                 "margin_width": round(margin_width, 3),           # margin width in feature units
                 "train_acc": round(accuracy_score(y_train, svm.predict(X_train)), 4),   # in-sample
                 "test_acc": round(acc, 4)})                       # out-of-sample
    print("C=%5.2f | SV=%3d | |w|=%.2f | margin=%.2f | test acc=%.3f" % (   # report the trade-off for this C
        C, n_sv, w_norm, margin_width, acc))                       # report the trade-off
table = pd.DataFrame(rows)                                         # assemble the summary
dump(table, "l7_svm_margin.csv")                                   # persist it

# ---------------------------------------------------------------- 4) figure: margins for each C
xs = np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 200)     # grid for drawing the boundaries
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.7))                # one panel per C
for ax, C in zip(axes, Cs):                                        # draw the three fits
    m = y == 1                                                     # mask for the positive class
    ax.scatter(X[~m, 0], X[~m, 1], s=16, color=OKABE[1], alpha=0.75, label="class 0")   # negatives
    ax.scatter(X[m, 0], X[m, 1], s=16, color=OKABE[5], alpha=0.75, label="class 1")     # positives
    draw_boundary(ax, models[C], xs, "#1E2630", "boundary and margins")   # boundary and margins
    sv = models[C].support_vectors_                               # support vectors of the fit
    ax.scatter(sv[:, 0], sv[:, 1], s=70, facecolors="none", edgecolors=OKABE[2],   # circle the support vectors
               linewidths=1.2, label="support vectors")           # legend label
    style_axes(ax, "C = %g | %d support vectors" % (C, len(sv)), "x1", "x2")   # shared style
    ax.legend(fontsize=6.8, loc="lower right")                     # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_svm_margin.png")                                    # save the C comparison

# ---------------------------------------------------------------- 5) figure: one fitted margin in detail
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.7))                # three panels
best = models[0.1]                                                 # the moderate C for the detail panel
m = y == 1                                                         # mask for the positive class
axes[0].scatter(X[~m, 0], X[~m, 1], s=16, color=OKABE[1], alpha=0.7, label="class 0")   # negatives
axes[0].scatter(X[m, 0], X[m, 1], s=16, color=OKABE[5], alpha=0.7, label="class 1")     # positives
draw_boundary(axes[0], best, xs, "#1E2630", "boundary")            # boundary and margins
axes[0].scatter(best.support_vectors_[:, 0], best.support_vectors_[:, 1], s=80,   # support vectors
                facecolors="none", edgecolors=OKABE[2], linewidths=1.4, label="support vectors")   # hollow circles
style_axes(axes[0], "The margin band", "x1", "x2")                 # shared style
axes[0].legend(fontsize=7.0, loc="lower right")                    # compact legend
axes[1].plot(table["C"], table["support_vectors"], "-o", ms=5, color=OKABE[5])   # support vector count
axes[1].set_xscale("log")                                          # C on a log axis
style_axes(axes[1], "Support vector count falls as C grows", "C (log scale)", "support vectors")   # style
axes[2].plot(table["C"], table["margin_width"], "-s", ms=5, color=OKABE[3])   # margin width
axes[2].plot(table["C"], table["test_acc"], "-^", ms=5, color=OKABE[2])       # test accuracy
axes[2].set_xscale("log")                                          # C on a log axis
style_axes(axes[2], "Margin width vs accuracy", "C (log scale)", "value")   # shared style
axes[2].legend(["margin width", "test accuracy"], fontsize=7.2, loc="center right")   # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_svm_support.png")                                   # save the detail figure
print("done: margins, support vectors and the C trade-off")         # closing log
