"""l5_10_mlp_numpy.py -- a two-layer neural network written from scratch in NumPy.

Every line carries an English comment. The script implements forward propagation,
the cross-entropy loss, back propagation and mini-batch gradient descent for a
two-moons classification task, then compares the result with scikit-learn's MLP.
"""
import numpy as np                                                 # arrays and matrix algebra for the network
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # training curves and decision boundaries
from sklearn.datasets import make_moons                            # the same non-linear sample as lesson 5.6
from sklearn.metrics import accuracy_score                         # accuracy for the comparison
from sklearn.model_selection import train_test_split               # hold-out split
from sklearn.neural_network import MLPClassifier                   # scikit-learn reference implementation
from _common_l5 import RNG, save, dump, style_axes, OKABE          # shared helpers

# ---------------------------------------------------------------- 1) data
X, y = make_moons(n_samples=700, noise=0.25, random_state=42)      # two crescent shapes with label noise
X = X + RNG.normal(0, 0.12, X.shape)                               # add extra jitter
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)   # split
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)              # scaling statistics from the training split only
Xtr = (X_train - mu) / sigma                                       # standardised training features
Xte = (X_test - mu) / sigma                                        # apply the same transform to the test features
print("train/test:", Xtr.shape, Xte.shape)                        # report the shapes

# ---------------------------------------------------------------- 2) network definition and initialisation
n_in, n_hidden, n_out = Xtr.shape[1], 24, 2                        # input width, hidden width and number of classes
def init_params(seed=42):                                          # initialise the weights with a fixed seed
    """Return He-initialised weights and zero biases for a two-layer network."""   # short docstring
    rng = np.random.default_rng(seed)                              # local generator so the network is reproducible
    W1 = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_hidden))      # He scaling suits ReLU activations
    b1 = np.zeros(n_hidden)                                        # biases start at zero
    W2 = rng.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, n_out))   # second layer weights
    b2 = np.zeros(n_out)                                           # output biases start at zero
    return W1, b1, W2, b2                                          # return the four parameter arrays

def forward(Xb, params):                                           # forward propagation through the network
    """Return logits, hidden activations and the softmax probabilities."""   # short docstring
    W1, b1, W2, b2 = params                                        # unpack the parameters
    z1 = Xb @ W1 + b1                                              # linear combination of the inputs
    a1 = np.maximum(z1, 0.0)                                       # ReLU activation of the hidden layer
    logits = a1 @ W2 + b2                                          # linear output layer
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))       # stabilised exponentials for softmax
    proba = exp / exp.sum(axis=1, keepdims=True)                   # class probabilities
    return logits, a1, proba                                       # return intermediate values for back propagation

def loss_and_grads(Xb, yb, params, l2=1e-4):                       # cross-entropy loss plus its gradients
    """Return the mean loss and the gradients for a mini batch."""   # short docstring
    W1, b1, W2, b2 = params                                        # unpack the parameters
    logits, a1, proba = forward(Xb, params)                        # run the forward pass
    n = len(yb)                                                    # batch size
    loss = -np.log(np.clip(proba[np.arange(n), yb], 1e-12, 1.0)).mean()   # mean cross-entropy
    loss += l2 * (np.sum(W1 ** 2) + np.sum(W2 ** 2))               # add a small weight decay term
    dlogits = proba.copy()                                         # gradient of the loss with respect to the logits
    dlogits[np.arange(n), yb] -= 1.0                               # subtract one at the true class
    dlogits /= n                                                   # average over the batch
    dW2 = a1.T @ dlogits + 2 * l2 * W2                             # gradient for the second weight matrix
    db2 = dlogits.sum(axis=0)                                      # gradient for the second bias vector
    da1 = dlogits @ W2.T                                           # back propagate to the hidden activations
    dz1 = da1 * (a1 > 0)                                           # ReLU derivative: one where the unit fired
    dW1 = Xb.T @ dz1 + 2 * l2 * W1                                 # gradient for the first weight matrix
    db1 = dz1.sum(axis=0)                                          # gradient for the first bias vector
    return loss, (dW1, db1, dW2, db2)                              # return the loss and all gradients

# ---------------------------------------------------------------- 3) mini-batch gradient descent
params = init_params()                                             # start from random weights
lr, epochs, batch = 0.08, 220, 64                                  # learning rate, passes and batch size
history = []                                                       # record the loss and accuracies per epoch
rng = np.random.default_rng(7)                                     # shuffling generator
for epoch in range(1, epochs + 1):                                 # loop over the training epochs
    order = rng.permutation(len(Xtr))                              # reshuffle the training rows
    epoch_loss = 0.0                                               # accumulate the batch losses
    for start in range(0, len(Xtr), batch):                        # iterate over mini batches
        idx = order[start:start + batch]                           # indices of this batch
        loss, grads = loss_and_grads(Xtr[idx], y_train[idx], params)   # loss and gradients for the batch
        dW1, db1, dW2, db2 = grads                                 # unpack the gradients
        W1, b1, W2, b2 = params                                    # unpack the current parameters
        params = (W1 - lr * dW1, b1 - lr * db1,                    # gradient descent update for layer one
                  W2 - lr * dW2, b2 - lr * db2)                    # gradient descent update for layer two
        epoch_loss += loss * len(idx)                              # weight the batch loss by its size
    train_acc = accuracy_score(y_train, forward(Xtr, params)[2].argmax(axis=1))   # training accuracy
    test_acc = accuracy_score(y_test, forward(Xte, params)[2].argmax(axis=1))   # test accuracy
    history.append({"epoch": epoch, "loss": round(epoch_loss / len(Xtr), 4),   # record the epoch loss
                    "train_acc": round(float(train_acc), 4),       # record the training accuracy
                    "test_acc": round(float(test_acc), 4)})        # record the test accuracy
hist = pd.DataFrame(history)                                       # assemble the history table
dump(hist, "l5_mlp_history.csv")                                   # persist the training history
print("numpy MLP final: train %.3f | test %.3f" % (hist["train_acc"].iloc[-1], hist["test_acc"].iloc[-1]))   # report

# ---------------------------------------------------------------- 4) scikit-learn reference model
sk = MLPClassifier(hidden_layer_sizes=(24,), activation="relu",    # same architecture as the hand-written net
                   solver="adam", learning_rate_init=0.02,         # adam usually converges faster than plain SGD
                   max_iter=600, random_state=42).fit(Xtr, y_train)   # fit on the same training split
sk_test = accuracy_score(y_test, sk.predict(Xte))                  # test accuracy of the reference model
print("sklearn MLP test accuracy: %.3f" % sk_test)                  # report the comparison

# ---------------------------------------------------------------- 5) figure: curves, boundaries and weights
fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.5))                # four panels in one row
axes[0].plot(hist["epoch"], hist["loss"], lw=1.8, color=OKABE[5], label="training loss")   # loss curve
ax0b = axes[0].twinx()                                             # second axis for the accuracies
ax0b.plot(hist["epoch"], hist["train_acc"], lw=1.6, color=OKABE[2], label="train accuracy")   # training accuracy
ax0b.plot(hist["epoch"], hist["test_acc"], lw=1.6, color=OKABE[0], label="test accuracy")   # test accuracy
ax0b.set_ylim(0.5, 1.02)                                           # focus on the useful range
ax0b.grid(False)                                                   # avoid a doubled grid
ax0b.set_ylabel("Accuracy", fontsize=9)                            # label the right axis
style_axes(axes[0], "Training curves of the NumPy network", "Epoch", "Cross-entropy loss")   # shared style
axes[0].legend(fontsize=6.8, loc="upper right")                    # left axis legend
ax0b.legend(fontsize=6.8, loc="lower right")                       # right axis legend
x_min, x_max = Xtr[:, 0].min() - 0.5, Xtr[:, 0].max() + 0.5        # horizontal extent of the boundary plot
y_min, y_max = Xtr[:, 1].min() - 0.5, Xtr[:, 1].max() + 0.5        # vertical extent of the boundary plot
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 240),               # grid along the first axis
                     np.linspace(y_min, y_max, 240))               # grid along the second axis
grid = np.c_[xx.ravel(), yy.ravel()]                               # flatten the grid into query points
for ax, proba, title in [                                          # two boundary panels
        (axes[1], forward(grid, params)[2][:, 1], "NumPy MLP (from scratch)"),   # hand-written network
        (axes[2], sk.predict_proba(grid)[:, 1], "scikit-learn MLP (same width)")]:   # reference network
    zz = proba.reshape(xx.shape)                                   # reshape the probabilities onto the grid
    ax.contourf(xx, yy, zz, levels=12, cmap="RdPu", alpha=0.55)     # shaded probability surface
    ax.contour(xx, yy, zz, levels=[0.5], colors="#1E2630", linewidths=1.2)   # the decision boundary itself
    ax.scatter(Xtr[y_train == 0, 0], Xtr[y_train == 0, 1], s=10, color=OKABE[1], edgecolors="white", lw=0.3)   # class 0
    ax.scatter(Xtr[y_train == 1, 0], Xtr[y_train == 1, 1], s=10, color=OKABE[5], edgecolors="white", lw=0.3)   # class 1
    style_axes(ax, title, "feature 1 (standardised)", "feature 2 (standardised)")   # shared style
W1_final = params[0]                                               # first layer weights of the trained network
im = axes[3].imshow(W1_final.T, cmap="RdBu_r", aspect="auto",       # visualise the learned first layer filters
                    vmin=-np.abs(W1_final).max(), vmax=np.abs(W1_final).max())   # symmetric colour scale
axes[3].set_xlabel("input feature", fontsize=9)                    # label the input axis
axes[3].set_ylabel("hidden unit", fontsize=9)                      # label the hidden unit axis
axes[3].grid(False)                                                # a heatmap needs no grid
axes[3].set_title("Learned first-layer weights")                   # panel title
plt.colorbar(im, ax=axes[3], fraction=0.046)                       # colour bar for the weights
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_mlp_training.png")                                  # save the neural network figure
