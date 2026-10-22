"""l9_05_optimizers.py -- SGD, Momentum, RMSProp and Adam on the same task.

Every line carries an English comment. The script trains the same small network with four
optimisers, records the loss curves and the steps needed to reach a threshold, and draws
the comparison so the deck can argue that the optimiser changes the path, not the target.
"""
import numpy as np                                                 # arrays for the threshold search
import pandas as pd                                                # tabular results
import matplotlib.pyplot as plt                                    # the comparison figure
from _common_l9 import (torch_seed, DAT, save, dump, style_axes,   # shared helpers
                        annotated, OKABE)                               # shared plotting helpers

torch = torch_seed()                                               # seed and get torch
import torch.nn as nn                                              # network modules
import torch.optim as optim                                        # optimisers

# ---------------------------------------------------------------- 1) the data
data = pd.read_csv(DAT / "l9_moons.csv")                           # the two-moons sample
X = torch.tensor(data[["x1", "x2"]].values, dtype=torch.float32)    # features
y = torch.tensor(data["y"].values, dtype=torch.long)                # labels
print("rows:", X.shape[0])                                          # report

def make_model():                                                  # a fresh identical model
    """Return a two-layer network with a fixed seed."""             # short docstring
    torch.manual_seed(0)                                           # same start for every optimiser
    return nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))   # deep enough

# ---------------------------------------------------------------- 2) the four optimisers
configs = {"SGD": lambda p: optim.SGD(p, lr=0.5),                  # plain gradient descent
           "Momentum": lambda p: optim.SGD(p, lr=0.1, momentum=0.9),   # heavy-ball momentum
           "RMSProp": lambda p: optim.RMSprop(p, lr=0.01),         # adaptive per-parameter step
           "Adam": lambda p: optim.Adam(p, lr=0.01)}               # bias-corrected adaptive step
criterion = nn.CrossEntropyLoss()                                  # the loss
curves = {}                                                        # loss per optimiser
summary = []                                                       # one row per optimiser
for name, factory in configs.items():                              # loop over the optimisers
    model = make_model()                                           # a fresh model
    opt = factory(model.parameters())                              # its optimiser
    curve = []                                                     # loss history
    first_hit = None                                               # epoch when the loss first drops below 0.1
    for epoch in range(300):                                       # training loop
        opt.zero_grad()                                            # clear the gradients
        loss = criterion(model(X), y)                              # forward and loss
        loss.backward()                                            # backward
        opt.step()                                                 # update
        value = float(loss.item())                                 # the scalar loss
        curve.append(value)                                        # record
        if first_hit is None and value < 0.1:                      # the threshold
            first_hit = epoch + 1                                  # remember the epoch
    curves[name] = curve                                           # store the curve
    acc = float((model(X).argmax(dim=1) == y).float().mean())      # final accuracy
    summary.append({"optimizer": name, "final_loss": round(curve[-1], 5),   # final loss
                    "epochs_to_0.1": first_hit if first_hit is not None else ">300",   # steps to the threshold
                    "final_accuracy": round(acc, 4)})              # accuracy
    print("%-9s | final %.4f | hit 0.1 at %s | acc %.3f" % (name, curve[-1], first_hit, acc))   # report
table = pd.DataFrame(summary)                                      # assemble the table
dump(table, "l9_optimizers.csv")                                   # persist it
history = pd.DataFrame(curves)                                     # one column per optimiser
history.insert(0, "epoch", np.arange(1, 301))                       # the epoch index
dump(history.round(6), "l9_optimizer_histories.csv")               # persist the curves

# ---------------------------------------------------------------- 3) the figure
fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9))                # two panels
for name, colour in zip(configs, OKABE[:4]):                       # one curve per optimiser
    axes[0].plot(history["epoch"], history[name], lw=1.7, color=colour, label=name)   # the surviving curve
axes[0].set_yscale("log")                                          # losses span orders of magnitude
style_axes(axes[0], "Four optimisers, one task", "epoch", "loss (log)")   # shared style
axes[0].legend(fontsize=7.2, loc="lower left")                     # compact legend
axes[1].bar(table["optimizer"], table["final_loss"], color=OKABE[4], alpha=0.9)   # final losses
style_axes(axes[1], "Final loss after 300 epochs", "", "loss")     # shared style
for i, row in table.iterrows():                                    # annotate each bar
    axes[1].text(i, row["final_loss"], "%.3f" % row["final_loss"], ha="center", va="bottom", fontsize=8)   # value
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_optimizers.png")                                    # save the figure
print("done: four optimisers compared")                             # closing log
