"""l7_04_curse_dimensionality.py -- why nearest neighbours stop being near in high dimensions.

Every line carries an English comment. The script draws samples in increasing
dimensions and measures how the nearest and the farthest neighbour distances
converge, which is the standard illustration of the curse of dimensionality.
"""
import numpy as np                                                 # arrays for the simulation
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # three panels
from scipy.spatial.distance import cdist                           # pairwise distances
from _common_l7 import (RNG, save, dump, style_axes, annotated, OKABE)   # shared helpers

# ---------------------------------------------------------------- 1) simulate distances by dimension
dims = [1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 100, 150, 200]      # dimensions to sweep
n_points = 300                                                     # points per simulated sample
rows = []                                                          # one row per dimension
for d in dims:                                                     # loop over the dimensions
    X = RNG.uniform(0, 1, (n_points, d))                           # points in the unit hypercube
    D = cdist(X, X)                                                # pairwise distance matrix
    D_nn = D.copy()                                                # copy for the nearest neighbour search
    np.fill_diagonal(D_nn, np.inf)                                 # ignore the distance to itself
    D_fn = D.copy()                                                # copy for the farthest neighbour search
    np.fill_diagonal(D_fn, -np.inf)                                # ignore the self distance here too
    nn = D_nn.min(axis=1)                                          # nearest neighbour distance per point
    fn = D_fn.max(axis=1)                                          # farthest neighbour distance per point
    rows.append({"dimension": d,                                   # the dimension
                 "mean_nn": round(float(nn.mean()), 4),            # average nearest neighbour distance
                 "mean_fn": round(float(fn.mean()), 4),            # average farthest neighbour distance
                 "ratio": round(float((nn / fn).mean()), 4),       # the concentration ratio
                 "cv_nn": round(float(nn.std() / nn.mean()), 4)})  # spread of the nearest distance
    print("d=%3d | nn=%.3f | fn=%.3f | nn/fn=%.3f | cv(nn)=%.3f" % (    # report the row for this dimension
        d, nn.mean(), fn.mean(), (nn / fn).mean(), nn.std() / nn.mean()))   # report the row
table = pd.DataFrame(rows)                                         # assemble the table
dump(table, "l7_dim_curse.csv")                                    # persist it
print("concentration: nn/fn grows from %.3f (d=1) to %.3f (d=200)" % (   # summarise the concentration effect
    table["ratio"].iloc[0], table["ratio"].iloc[-1]))              # summarise the effect

# ---------------------------------------------------------------- 2) a low- versus high-dimensional figure
def show_pairs(ax, d, colour):                                     # draw a point and its nearest neighbour
    """Scatter a sample projected on two axes and link each point to its nearest neighbour."""   # short docstring
    X = RNG.uniform(0, 1, (120, d))                                # the high-dimensional sample
    D = cdist(X, X)                                                # pairwise distances
    np.fill_diagonal(D, np.inf)                                    # ignore self distance
    j = D.argmin(axis=1)                                           # index of the nearest neighbour
    scale = np.sqrt(d)                                             # distances shrink with the dimension
    ax.scatter(X[:, 0], X[:, 1], s=12, color=colour, alpha=0.7)     # points on the first two axes
    for i in range(60):                                            # link a subset to keep the figure readable
        ax.plot([X[i, 0], X[j[i], 0]], [X[i, 1], X[j[i], 1]], color=colour, lw=0.5, alpha=0.5)   # link line
    style_axes(ax, "d = %d projected on two axes" % d, "axis 1", "axis 2")   # shared style
    ax.set_xlim(-0.02, 1.02)                                       # keep the unit square visible
    ax.set_ylim(-0.02, 1.02)                                       # keep the unit square visible

# ---------------------------------------------------------------- 3) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7))                # three panels
show_pairs(axes[0], 2, OKABE[1])                                   # two dimensions
show_pairs(axes[1], 50, OKABE[5])                                  # fifty dimensions
axes[2].plot(table["dimension"], table["mean_nn"], "-o", ms=4, color=OKABE[2], label="nearest")   # nn distance
axes[2].plot(table["dimension"], table["mean_fn"], "-s", ms=4, color=OKABE[5], label="farthest")   # fn distance
axes[2].set_xscale("log")                                          # dimension on a log axis
style_axes(axes[2], "Nearest and farthest distances converge", "dimension (log scale)", "mean distance")   # style
axes[2].legend(fontsize=7.4, loc="center right")                   # compact legend
annotated(axes[2], "nn/fn -> 1", "upper left")                     # annotate the concentration
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_dim_curse.png")                                     # save the figure
print("done: the curse of dimensionality")                         # closing log
