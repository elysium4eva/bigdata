"""l9_02_linear_softmax.py -- the linear classifier, softmax and the cross-entropy loss.

Every line carries an English comment. The script builds a linear classifier from scratch
on the shape images, computes the softmax probabilities and the cross-entropy, checks the
analytic gradient against finite differences, and draws a slice of the loss surface.
"""
import numpy as np                                                 # arrays and the gradient maths
import pandas as pd                                                # tabular histories
import matplotlib.pyplot as plt                                    # the loss surface figure
from _common_l9 import (DAT, save, dump, style_axes, annotated, OKABE)   # shared helpers

# ---------------------------------------------------------------- 1) the data
data = np.load(DAT / "l9_shapes.npz")                              # the images saved in lesson 9.1
X = data["images"].reshape(len(data["images"]), -1)                # flatten to vectors
y = data["labels"].astype(int)                                     # integer labels
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)                  # standardise per pixel
n, d = X.shape                                                     # rows and features
k = 4                                                              # number of classes
print("rows:", n, "| features:", d, "| classes:", k)               # report

# ---------------------------------------------------------------- 2) softmax and cross-entropy
def softmax(z):                                                    # numerically stable softmax
    """Return the row-wise softmax of a score matrix."""            # short docstring
    z = np.asarray(z, dtype=float)                                 # work with a float array
    z = z - z.max(axis=1, keepdims=True)                           # subtract the row maximum
    e = np.exp(z)                                                  # safe exponentials
    return e / e.sum(axis=1, keepdims=True)                        # normalise the rows

def loss_and_grad(W, Xm, yv, reg=1e-3):                            # cross-entropy and its gradient
    """Return the mean cross-entropy plus L2 and the gradient of W."""   # short docstring
    scores = Xm @ W                                                # linear scores
    probs = softmax(scores)                                        # class probabilities
    correct = probs[np.arange(len(yv)), yv]                        # probability of the true class
    loss = -np.mean(np.log(np.clip(correct, 1e-12, None)))         # mean negative log-likelihood
    loss += 0.5 * reg * np.sum(W * W)                              # weight decay term
    grad = probs.copy()                                            # start from the probabilities
    grad[np.arange(len(yv)), yv] -= 1.0                            # subtract one at the true class
    grad = Xm.T @ grad / len(yv) + reg * W                         # X^T (p - y) / n plus the penalty
    return float(loss), grad                                       # return both

W = np.zeros((d, k))                                               # start from zero weights
loss, grad = loss_and_grad(W, X, y)                                # loss at the start
print("initial loss: %.4f (log 4 = %.4f)" % (loss, np.log(k)))      # should be near log(4)

# ---------------------------------------------------------------- 3) gradient check
Wr = np.random.default_rng(0).normal(0, 0.01, (d, k))              # a random point for the check
_, analytic = loss_and_grad(Wr, X[:40], y[:40])                    # analytic gradient
numerical = np.zeros_like(Wr)                                      # finite-difference gradient
eps = 1e-6                                                         # small step
for i in range(d):                                                 # loop over i, j (subsample for speed)
    for j in range(k):                                             # only a few entries are needed
        if (i * k + j) % 37 != 0:                                  # check roughly every 37th entry
            continue                                               # skip the rest
        Wp = Wr.copy()                                             # perturb upwards
        Wp[i, j] += eps                                            # plus epsilon
        Wm = Wr.copy()                                             # perturb downwards
        Wm[i, j] -= eps                                            # minus epsilon
        numerical[i, j] = (loss_and_grad(Wp, X[:40], y[:40])[0]         # central difference of the loss
                           - loss_and_grad(Wm, X[:40], y[:40])[0]) / (2 * eps)   # central difference
mask = numerical != 0                                              # the entries we checked
gap = float(np.max(np.abs(numerical[mask] - analytic[mask])))      # largest disagreement
print("gradient check: max |analytic - numerical| = %.2e" % gap)    # should be tiny
assert gap < 1e-6, "the gradient is wrong"                         # hard check

# ---------------------------------------------------------------- 4) train with plain gradient descent
lr, epochs = 0.5, 300                                              # learning rate and epochs
history = []                                                       # loss per epoch
W = np.zeros((d, k))                                               # restart from zeros
for _ in range(epochs):                                            # full-batch gradient descent
    loss, grad = loss_and_grad(W, X, y)                            # evaluate
    history.append(loss)                                           # record
    W = W - lr * grad                                              # step
train_acc = float(np.mean((X @ W).argmax(axis=1) == y))            # training accuracy
print("final loss %.4f | train accuracy %.3f" % (history[-1], train_acc))   # report
dump(pd.DataFrame({"epoch": np.arange(1, epochs + 1), "loss": np.round(history, 5)}),   # persist
     "l9_linear_history.csv")                                      # the history table

# ---------------------------------------------------------------- 5) the loss surface in two directions
dir1 = RNG_dir1 = (W[:, 0] / np.linalg.norm(W[:, 0]))              # a direction in weight space
dir2 = (W[:, 1] / np.linalg.norm(W[:, 1]))                         # a second direction
alphas = np.linspace(-2.0, 2.0, 60)                                # grid over the first direction
betas = np.linspace(-2.0, 2.0, 60)                                 # grid over the second direction
surface = np.zeros((len(alphas), len(betas)))                      # loss values on the grid
for i, a in enumerate(alphas):                                     # loop over the grid
    for j, b in enumerate(betas):                                  # second loop
        Wp = W + a * dir1[:, None] + b * dir2[:, None]              # perturbed weights
        surface[i, j] = loss_and_grad(Wp, X, y)[0]                 # the loss there
print("loss surface: min %.3f | max %.3f" % (surface.min(), surface.max()))   # report

# ---------------------------------------------------------------- 6) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8))                # three panels
axes[0].plot(history, lw=1.9, color=OKABE[5])                      # loss curve
style_axes(axes[0], "Cross-entropy falls during training", "epoch", "loss")   # shared style
annotated(axes[0], "start: log 4 = 1.386", "upper right")          # annotate the start
cs = axes[1].contourf(alphas, betas, surface.T, levels=20, cmap="RdPu")   # loss surface
axes[1].plot(0, 0, marker="*", ms=14, color="#1E2630", label="fitted W")   # the fitted point
style_axes(axes[1], "Loss surface slice around W", "direction 1", "direction 2")   # shared style
plt.colorbar(cs, ax=axes[1], fraction=0.046, label="loss")         # colour bar
axes[1].legend(fontsize=7.4, loc="upper left")                     # compact legend
temps = [0.25, 1.0, 4.0]                                           # three softmax temperatures
sample = X[:40] @ W                                                # scores for forty images
for temp, colour in zip(temps, [OKABE[5], OKABE[2], OKABE[1]]):    # one curve per temperature
    p = softmax(sample / temp)                                     # temperature-scaled probabilities
    axes[2].plot(np.arange(40), p.max(axis=1), "-o", ms=3, color=colour, label="T = %.2f" % temp)   # max probability
style_axes(axes[2], "Softmax temperature changes sharpness", "image index", "max probability")   # shared style
axes[2].legend(fontsize=7.4, loc="lower right")                     # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_loss_surface.png")                                  # save the figure

# ---------------------------------------------------------------- 7) softmax temperature figure
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))                # two panels
logits = np.array([2.0, 1.0, 0.1, -1.0])                           # one score vector
bar_temps = [0.25, 0.5, 1.0, 2.0, 4.0]                          # temperatures for the bar chart
for idx_bar, (temp, colour) in enumerate(zip(bar_temps, OKABE[:5])):   # five temperatures
    axes[0].bar(np.arange(4) + (idx_bar - 2) * 0.16,   # offset bars
                softmax(logits[None, :] / temp)[0], width=0.14, color=colour, alpha=0.9, label="T = %.2f" % temp)   # the bars
style_axes(axes[0], "Same logits, different temperatures", "class", "probability")   # shared style
axes[0].legend(fontsize=6.8, loc="upper right")                     # compact legend
tt = np.linspace(0.1, 5, 100)                                      # temperature grid
for cls in range(4):                                               # one curve per class
    probs = [softmax(logits[None, :] / t)[0, cls] for t in tt]     # probability versus temperature
    axes[1].plot(tt, probs, lw=1.6, color=OKABE[cls], label="class %d" % cls)   # the curve
style_axes(axes[1], "Probability versus temperature", "temperature", "probability")   # shared style
axes[1].legend(fontsize=7.0, loc="center right")                    # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_softmax_temperature.png")                           # save the temperature figure
print("done: linear classifier, softmax and the loss surface")      # closing log
