"""l9_03_mlp_numpy.py -- a two-layer network written from scratch, with a gradient check.

Every line carries an English comment. The script implements forward propagation, the
cross-entropy loss, and manual backpropagation for a two-layer network on the two-moons
sample, verifies the gradients numerically, and draws the decision boundary and the
activation functions side by side.
"""
import numpy as np                                                 # arrays and the manual gradients
import pandas as pd                                                # tabular histories
import matplotlib.pyplot as plt                                    # the figures
from _common_l9 import (make_moons, save, dump, style_axes,        # shared helpers
                        annotated, OKABE)                               # shared plotting helpers

# ---------------------------------------------------------------- 1) the sample
X, y = make_moons(400, 0.18)                                       # two intertwined crescents
dump(pd.DataFrame({"x1": X[:, 0].round(3), "x2": X[:, 1].round(3), "y": y}), "l9_moons.csv")   # persist
print("rows:", X.shape[0], "| classes:", np.bincount(y).tolist())   # report

# ---------------------------------------------------------------- 2) activations and their derivatives
def relu(z):                                                       # rectified linear unit
    """Return the element-wise ReLU of z."""                        # short docstring
    return np.maximum(z, 0.0)                                      # clip the negatives

def relu_grad(z):                                                  # derivative of ReLU
    """Return the element-wise derivative of ReLU."""               # short docstring
    return (z > 0).astype(float)                                   # one where the input is positive

def sigmoid(z):                                                    # logistic activation
    """Return the numerically stable sigmoid of z."""               # short docstring
    out = np.empty_like(z)                                         # allocate the output
    pos = z >= 0                                                   # mask for non-negative inputs
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))                       # stable form
    e = np.exp(z[~pos])                                            # stable form for negatives
    out[~pos] = e / (1.0 + e)                                      # equivalent expression
    return out                                                     # return the values

def softmax(z):                                                    # row-wise softmax
    """Return the row-wise softmax of a score matrix."""            # short docstring
    z = z - z.max(axis=1, keepdims=True)                           # subtract the row maximum
    e = np.exp(z)                                                  # safe exponentials
    return e / e.sum(axis=1, keepdims=True)                        # normalise

# ---------------------------------------------------------------- 3) the two-layer network
def init_params(hidden=16):                                        # weight initialisation
    """Return W1, b1, W2, b2 for a two-layer classifier."""         # short docstring
    W1 = np.random.default_rng(0).normal(0, 0.6, (2, hidden))      # input to hidden weights
    b1 = np.zeros(hidden)                                          # hidden bias
    W2 = np.random.default_rng(1).normal(0, 0.6, (hidden, 2))      # hidden to output weights
    b2 = np.zeros(2)                                               # output bias
    return W1, b1, W2, b2                                          # return the parameters

def forward(params, Xm):                                           # forward pass
    """Return the hidden activations, the scores and the cache for backprop."""   # short docstring
    W1, b1, W2, b2 = params                                        # unpack
    h_pre = Xm @ W1 + b1                                           # hidden pre-activation
    h = relu(h_pre)                                                # hidden activation
    scores = h @ W2 + b2                                           # output scores
    return h, scores, (Xm, h_pre, h, W2)                           # return what backprop needs

def loss_grad(params, Xm, yv):                                     # loss and gradients
    """Return the mean cross-entropy and the gradient of every parameter."""   # short docstring
    W1, b1, W2, b2 = params                                        # unpack
    h, scores, cache = forward(params, Xm)                         # forward pass
    probs = softmax(scores)                                        # class probabilities
    loss = -np.mean(np.log(np.clip(probs[np.arange(len(yv)), yv], 1e-12, None)))   # mean NLL
    dscores = probs.copy()                                         # start from the probabilities
    dscores[np.arange(len(yv)), yv] -= 1.0                         # subtract one at the true class
    dscores /= len(yv)                                             # average over the batch
    Xm_c, h_pre, h, W2 = cache                                     # unpack the cache
    dW2 = h.T @ dscores                                            # gradient for the second layer
    db2 = dscores.sum(axis=0)                                      # bias gradient
    dh = dscores @ W2.T                                            # back-propagate to the hidden layer
    dh_pre = dh * relu_grad(h_pre)                                 # through the activation
    dW1 = Xm_c.T @ dh_pre                                          # gradient for the first layer
    db1 = dh_pre.sum(axis=0)                                       # hidden bias gradient
    return float(loss), (dW1, db1, dW2, db2)                        # return loss and gradients

params = init_params(16)                                           # initialise
loss0, grads = loss_grad(params, X, y)                             # loss at the start
print("initial loss: %.4f (log 2 = %.4f)" % (loss0, np.log(2)))     # should be near log(2)

# ---------------------------------------------------------------- 4) numerical gradient check
W1, b1, W2, b2 = params                                            # unpack for the check
idx_i, idx_j = 0, 5                                                # one weight to test
eps = 1e-6                                                         # small step
plus = [W1.copy(), b1.copy(), W2.copy(), b2.copy()]                # copy for the plus perturbation
plus[0][idx_i, idx_j] += eps                                       # perturb upwards
minus = [W1.copy(), b1.copy(), W2.copy(), b2.copy()]               # copy for the minus perturbation
minus[0][idx_i, idx_j] -= eps                                      # perturb downwards
num = (loss_grad(tuple(plus), X, y)[0] - loss_grad(tuple(minus), X, y)[0]) / (2 * eps)   # central difference
ana = grads[0][idx_i, idx_j]                                       # analytic gradient
print("gradient check: analytic %.6e | numerical %.6e | gap %.2e" % (ana, num, abs(ana - num)))   # report
assert abs(ana - num) < 1e-5, "the manual gradient is wrong"        # hard check

# ---------------------------------------------------------------- 5) train and record the boundary
lr, epochs = 0.6, 600                                              # learning rate and epochs
history = []                                                       # loss per epoch
for _ in range(epochs):                                            # full-batch gradient descent
    loss, g = loss_grad(params, X, y)                              # evaluate
    history.append(loss)                                           # record
    params = tuple(p - lr * gp for p, gp in zip(params, g))         # step all parameters
acc = float(np.mean(softmax(forward(params, X)[1]).argmax(axis=1) == y))   # training accuracy
print("final loss %.4f | train accuracy %.3f" % (history[-1], acc))   # report
dump(pd.DataFrame({"epoch": np.arange(1, epochs + 1), "loss": np.round(history, 5)}), "l9_mlp_history.csv")   # persist

# ---------------------------------------------------------------- 6) the figures
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                # three panels
xs = np.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 260)     # horizontal grid
ys = np.linspace(X[:, 1].min() - 0.4, X[:, 1].max() + 0.4, 260)     # vertical grid
XX, YY = np.meshgrid(xs, ys)                                       # full grid
grid = np.c_[XX.ravel(), YY.ravel()]                               # flatten
scores_grid = forward(params, grid)[1]                             # scores on the grid
pred = softmax(scores_grid).argmax(axis=1).reshape(XX.shape)       # predicted class
axes[0].contourf(XX, YY, pred, levels=[-0.5, 0.5, 1.5], alpha=0.22, cmap="RdPu")   # decision regions
axes[0].scatter(X[y == 0, 0], X[y == 0, 1], s=12, color=OKABE[1], label="class 0")   # class 0
axes[0].scatter(X[y == 1, 0], X[y == 1, 1], s=12, color=OKABE[5], label="class 1")   # class 1
style_axes(axes[0], "Two-layer network: curved boundary", "x1", "x2")   # shared style
axes[0].legend(fontsize=7.2, loc="lower right")                     # compact legend
axes[1].plot(history, lw=1.8, color=OKABE[5])                       # loss curve
style_axes(axes[1], "Loss falls during training", "epoch", "cross-entropy")   # shared style
zz = np.linspace(-5, 5, 200)                                       # activation grid
axes[2].plot(zz, sigmoid(zz), lw=1.8, color=OKABE[2], label="sigmoid")   # sigmoid
axes[2].plot(zz, np.tanh(zz), lw=1.8, color=OKABE[3], label="tanh")   # tanh
axes[2].plot(zz, relu(zz), lw=1.8, color=OKABE[5], label="ReLU")     # ReLU
axes[2].plot(zz, zz * sigmoid(zz), lw=1.4, ls="--", color=OKABE[4], label="GELU (approx)")   # GELU-like curve
style_axes(axes[2], "Activation functions", "pre-activation", "activation")   # shared style
axes[2].legend(fontsize=7.0, loc="upper left")                      # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_mlp_boundary.png")                                  # save the boundary figure

# ---------------------------------------------------------------- 7) activation derivatives
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))                # two panels
axes[0].plot(zz, sigmoid(zz) * (1 - sigmoid(zz)), lw=1.8, color=OKABE[2], label="sigmoid'")   # sigmoid derivative
axes[0].plot(zz, 1 - np.tanh(zz) ** 2, lw=1.8, color=OKABE[3], label="tanh'")   # tanh derivative
axes[0].plot(zz, relu_grad(zz), lw=1.8, color=OKABE[5], label="ReLU'")   # ReLU derivative
style_axes(axes[0], "Derivatives: where gradients die", "pre-activation", "derivative")   # shared style
axes[0].legend(fontsize=7.2, loc="lower center")                    # compact legend
depth = np.arange(1, 31)                                           # network depths
axes[1].semilogy(depth, (0.25 ** depth), "-o", ms=4, color=OKABE[5], label="sigmoid factor")   # 0.25^depth
axes[1].semilogy(depth, np.ones_like(depth, dtype=float), "-s", ms=3, color=OKABE[2], label="ReLU factor")   # ReLU keeps 1
style_axes(axes[1], "Gradient magnitude with depth", "layers", "relative gradient (log)")   # shared style
axes[1].legend(fontsize=7.2, loc="upper right")                     # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_activations.png")                                   # save the activation figure
print("done: MLP from scratch, gradient check and activations")      # closing log
