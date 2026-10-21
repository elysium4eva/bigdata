"""l6_07_softmax_regression.py -- multi-class logistic regression with softmax.

Every line carries an English comment. The script builds a three-class sample, fits
softmax regression with scikit-learn, then re-implements it with NumPy and gradient
descent using the stable softmax. It draws the decision regions and the predicted
probability surface for the slides.
"""
import numpy as np                                                 # arrays and the softmax implementation
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # decision-region figure
from matplotlib.colors import ListedColormap                       # discrete colour map for the regions
from sklearn.linear_model import LogisticRegression               # reference softmax regression
from sklearn.metrics import accuracy_score                        # accuracy for the comparison
from _common_l6 import (RNG, save, dump, add_intercept, sigmoid,   # shared helpers
                        style_axes, annotated, OKABE)                   # # helper names imported from the lecture module

# ---------------------------------------------------------------- 1) a three-class sample in two dimensions
N = 240                                                            # observations
centres = np.array([[0.0, 2.1], [2.0, -1.1], [-2.0, -1.1]])        # three cluster centres
labels = RNG.integers(0, 3, N)                                     # balanced random assignment
X2 = centres[labels] + RNG.normal(0, 0.92, (N, 2))                 # features around the centres
data = pd.DataFrame({"x1": X2[:, 0].round(3), "x2": X2[:, 1].round(3), "label": labels})   # tabulate
dump(data, "l6_three_class.csv")                                   # persist the sample
print("rows:", N, "| classes:", np.bincount(labels).tolist())      # report the class balance

# ---------------------------------------------------------------- 2) scikit-learn softmax regression
sk = LogisticRegression(multi_class="multinomial", max_iter=2000)   # multinomial = softmax regression
sk.fit(X2, labels)                                                 # fit on the raw features
print("sklearn accuracy: %.3f" % accuracy_score(labels, sk.predict(X2)))   # in-sample accuracy
print("sklearn coefficients shape:", sk.coef_.shape)                # one row of weights per class

# ---------------------------------------------------------------- 3) softmax and its gradient from scratch
def softmax_matrix(Z):                                             # row-wise stable softmax
    """Return the softmax of every row of Z, shifted by the row maximum."""   # short docstring
    Z = np.asarray(Z, dtype=float)                                 # work with a float array
    Z = Z - Z.max(axis=1, keepdims=True)                           # remove the overflow risk
    E = np.exp(Z)                                                  # safe exponentials
    return E / E.sum(axis=1, keepdims=True)                        # normalise every row

def onehot(yv, k):                                                 # one-hot encoding for the gradient
    """Return a one-hot matrix for labels yv with k classes."""     # short docstring
    M = np.zeros((len(yv), k))                                     # allocate the matrix
    M[np.arange(len(yv)), yv] = 1.0                                # set the matching entries
    return M                                                       # return the matrix

def softmax_loss_grad(W, Xm, Y_oh):                                # cross-entropy and gradient
    """Return the mean cross-entropy and its gradient for weight matrix W."""   # short docstring
    P = softmax_matrix(Xm @ W)                                     # class probabilities for every row
    eps = 1e-12                                                    # avoid log(0)
    loss = -np.mean(np.sum(Y_oh * np.log(P + eps), axis=1))        # mean cross-entropy over the rows
    grad = Xm.T @ (P - Y_oh) / len(Xm)                             # gradient with respect to W
    return float(loss), grad                                       # return both

Xa = add_intercept(X2)                                             # augmented design matrix
Y_oh = onehot(labels, 3)                                           # one-hot target matrix
W = np.zeros((Xa.shape[1], 3))                                     # weight matrix (parameters per class)
lr, epochs = 0.5, 3000                                             # learning rate and number of epochs
hist = []                                                          # loss history
for _ in range(epochs):                                            # full-batch gradient descent
    loss, grad = softmax_loss_grad(W, Xa, Y_oh)                    # evaluate loss and gradient
    hist.append(loss)                                              # record the loss
    W = W - lr * grad                                              # gradient step
pred = softmax_matrix(Xa @ W).argmax(axis=1)                       # predicted class per row
print("numpy softmax accuracy: %.3f | final loss %.4f" % (accuracy_score(labels, pred), hist[-1]))   # report
dump(pd.DataFrame({"epoch": np.arange(1, epochs + 1), "loss": np.round(hist, 6)}),   # history table
     "l6_softmax_history.csv")                                     # persist it

# ---------------------------------------------------------------- 4) figure: regions and probabilities
xs = np.linspace(X2[:, 0].min() - 0.8, X2[:, 0].max() + 0.8, 260)   # horizontal grid
ys = np.linspace(X2[:, 1].min() - 0.8, X2[:, 1].max() + 0.8, 260)   # vertical grid
XX, YY = np.meshgrid(xs, ys)                                       # full grid
grid = np.c_[XX.ravel(), YY.ravel()]                               # flatten the grid
grid_aug = add_intercept(grid)                                     # augmented grid matrix
prob_grid = softmax_matrix(grid_aug @ W)                           # predicted probabilities on the grid
class_grid = prob_grid.argmax(axis=1).reshape(XX.shape)            # hard class on the grid
map3 = ListedColormap([OKABE[1], OKABE[5], OKABE[2]])              # one colour per class

fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6))                # three panels
axes[0].contourf(XX, YY, class_grid, levels=[-0.5, 0.5, 1.5, 2.5], alpha=0.28, cmap=map3)   # decision regions
for k, colour in enumerate([OKABE[1], OKABE[5], OKABE[2]]):        # scatter the three classes
    m = labels == k                                                # mask for this class
    axes[0].scatter(X2[m, 0], X2[m, 1], s=14, color=colour, label="class %d" % k)   # points of the class
axes[0].contour(XX, YY, class_grid, levels=[0.5, 1.5], colors="#1E2630", linewidths=1.1)   # boundary lines
style_axes(axes[0], "Softmax regression: linear boundaries", "x1", "x2")   # shared style
axes[0].legend(fontsize=7.2, loc="lower right")                    # compact legend
cs = axes[1].contourf(XX, YY, prob_grid[:, 2].reshape(XX.shape), levels=16, cmap="RdPu")   # probability of class 2
axes[1].contour(XX, YY, class_grid, levels=[0.5, 1.5], colors="white", linewidths=0.8)   # boundaries on top
style_axes(axes[1], "Probability surface for class 2", "x1", "x2")   # shared style
plt.colorbar(cs, ax=axes[1], fraction=0.046, label="$P(y=2\\mid x)$")   # colour bar
axes[2].plot(hist, lw=1.8, color=OKABE[5])                         # gradient descent loss curve
axes[2].set_yscale("log")                                          # losses shrink by orders of magnitude
style_axes(axes[2], "Softmax gradient descent converges", "epoch", "cross-entropy (log scale)")   # shared style
annotated(axes[2], "lr = %.1f | acc %.3f" % (lr, accuracy_score(labels, pred)), "upper right")   # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_softmax_regions.png")                               # save the figure
print("done: softmax regression, regions and loss curve")           # closing log
