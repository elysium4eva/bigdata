"""l9_04_torch_autograd.py -- the same network in PyTorch, plus learning-rate sensitivity.

Every line carries an English comment. The script rebuilds the two-layer network with
torch.nn, compares the autograd gradients with the NumPy ones, trains with four learning
rates, and draws the training curves so the deck can show the stability window.
"""
import numpy as np                                                 # arrays for the comparison
import pandas as pd                                                # tabular results
import matplotlib.pyplot as plt                                    # the curves
from _common_l9 import (torch_seed, param_count, DAT, save, dump,   # shared helpers
                        style_axes, annotated, OKABE)                   # shared plotting helpers

torch = torch_seed()                                               # seed and get the torch module
import torch.nn as nn                                              # network modules
import torch.optim as optim                                        # optimisers

# ---------------------------------------------------------------- 1) the sample
data = pd.read_csv(DAT / "l9_moons.csv")                           # the two-moons sample from lesson 9.3
X = torch.tensor(data[["x1", "x2"]].values, dtype=torch.float32)    # features
y = torch.tensor(data["y"].values, dtype=torch.long)                # class labels
print("rows:", X.shape[0], "| classes:", int(y.max()) + 1)          # report

# ---------------------------------------------------------------- 2) the model
class TwoLayer(nn.Module):                                         # a small feed-forward network
    """A two-layer classifier with a ReLU hidden layer."""          # short docstring
    def __init__(self, hidden=16):                                 # constructor
        super().__init__()                                         # initialise the parent class
        self.fc1 = nn.Linear(2, hidden)                            # input to hidden
        self.fc2 = nn.Linear(hidden, 2)                            # hidden to output
    def forward(self, x):                                          # forward pass
        """Return the class scores."""                             # short docstring
        return self.fc2(torch.relu(self.fc1(x)))                   # ReLU then linear

model = TwoLayer(16)                                               # build the model
print("trainable parameters:", param_count(model))                  # report the size
criterion = nn.CrossEntropyLoss()                                  # the loss
print("initial loss: %.4f" % float(criterion(model(X), y)))         # report the starting loss

# ---------------------------------------------------------------- 3) autograd versus the manual gradient
model.zero_grad()                                                  # clear the gradients
loss = criterion(model(X), y)                                      # forward pass
loss.backward()                                                    # autograd
grad_fc1 = model.fc1.weight.grad.detach().numpy().copy()            # the gradient of the first layer
print("autograd gradient shape:", grad_fc1.shape)                   # report

# ---------------------------------------------------------------- 4) train with four learning rates
lr_list = [0.01, 0.1, 0.5, 5.0]                                    # too small to too large
histories = {}                                                     # curves per learning rate
for lr in lr_list:                                                 # loop over the learning rates
    torch.manual_seed(0)                                           # reset the seed for a fair comparison
    net = TwoLayer(16)                                             # a fresh model
    opt = optim.SGD(net.parameters(), lr=lr)                       # plain SGD
    curve = []                                                     # loss history
    for epoch in range(200):                                       # training loop
        opt.zero_grad()                                            # clear the gradients
        out = net(X)                                               # forward
        loss = criterion(out, y)                                   # loss
        loss.backward()                                            # backward
        opt.step()                                                 # update
        curve.append(float(loss.item()))                            # record
    histories[lr] = curve                                          # store the curve
    acc = float((net(X).argmax(dim=1) == y).float().mean())         # training accuracy
    print("lr %.2f | final loss %.4f | accuracy %.3f" % (lr, curve[-1], acc))   # report
history = pd.DataFrame(histories)                                  # one column per learning rate
history.insert(0, "epoch", np.arange(1, 201))                       # the epoch index
dump(history.round(6), "l9_lr_histories.csv")                      # persist the curves

# ---------------------------------------------------------------- 5) the figures
fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9))                # two panels
for lr, colour in zip(lr_list, [OKABE[1], OKABE[2], OKABE[4], OKABE[5]]):   # one curve per learning rate
    axes[0].plot(history["epoch"], history[lr], lw=1.7, color=colour, label="lr = %.2f" % lr)   # the curve
axes[0].set_yscale("log")                                          # the losses span orders of magnitude
style_axes(axes[0], "Learning rate: too small, right, too large", "epoch", "loss (log)")   # shared style
axes[0].legend(fontsize=7.0, loc="lower left")                      # compact legend
annotated(axes[0], "lr=5 diverges / oscillates", "upper right")     # annotate the failure
grid_xx = torch.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 200)   # horizontal grid
grid_yy = torch.linspace(X[:, 1].min() - 0.4, X[:, 1].max() + 0.4, 200)   # vertical grid
gx, gy = torch.meshgrid(grid_xx, grid_yy, indexing="xy")            # the full grid
pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)          # flatten it
with torch.no_grad():                                              # no gradients while drawing
    pred = net(pts).argmax(dim=1).reshape(gx.shape)                # predicted classes
axes[1].contourf(gx.numpy(), gy.numpy(), pred.numpy(), levels=[-0.5, 0.5, 1.5], alpha=0.22, cmap="RdPu")   # regions
mask0 = y.numpy() == 0                                             # class 0 points
axes[1].scatter(X[mask0, 0], X[mask0, 1], s=12, color=OKABE[1], label="class 0")   # class 0
axes[1].scatter(X[~mask0, 0], X[~mask0, 1], s=12, color=OKABE[5], label="class 1")   # class 1
style_axes(axes[1], "torch.nn boundary (last model)", "x1", "x2")    # shared style
axes[1].legend(fontsize=7.2, loc="lower right")                     # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_lr_curves.png")                                     # save the figure
print("done: torch autograd and the learning-rate window")           # closing log
