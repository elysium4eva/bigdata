"""l9_10_training_engineering.py -- batch size, gradient accumulation and memory arithmetic.

Every line carries an English comment. The script measures training time for several batch
sizes, shows how gradient accumulation emulates a large batch, and prints the parameter,
gradient and optimiser memory of a model so the deck can discuss the engineering side
without pretending to have a GPU.
"""
import time                                                        # measure the training time
import numpy as np                                                 # arrays
import pandas as pd                                                # tabular results
import matplotlib.pyplot as plt                                    # the figures
from _common_l9 import (torch_seed, param_count, save, dump,       # shared helpers
                        style_axes, annotated, OKABE)                   # shared plotting helpers
                                                                        # shared plotting helpers
torch = torch_seed()                                               # seed and get torch
import torch.nn as nn                                              # network modules
import torch.optim as optim                                        # optimisers

# ---------------------------------------------------------------- 1) a small model and a synthetic batch
D_IN, D_HID, N = 256, 256, 8192                                    # input size, hidden size, samples
model = nn.Sequential(nn.Linear(D_IN, D_HID), nn.ReLU(), nn.Linear(D_HID, 10))   # a two-layer model
print("parameters:", param_count(model))                            # report
X = torch.randn(N, D_IN)                                           # the feature matrix
y = torch.randint(0, 10, (N,))                                     # random labels
criterion = nn.CrossEntropyLoss()                                  # the loss

# ---------------------------------------------------------------- 2) time per epoch by batch size
def time_epoch(batch_size, epochs=3):                              # measure the throughput
    """Return the mean seconds per epoch for one batch size."""     # short docstring
    opt = optim.SGD(model.parameters(), lr=0.01)                   # a fresh optimiser
    times = []                                                     # epoch timings
    for _ in range(epochs):                                        # a few epochs
        started = time.perf_counter()                              # start the clock
        for start in range(0, N, batch_size):                      # iterate over the batches
            xb = X[start:start + batch_size]                        # this batch
            yb = y[start:start + batch_size]                        # its labels
            opt.zero_grad()                                        # clear the gradients
            loss = criterion(model(xb), yb)                        # forward and loss
            loss.backward()                                        # backward
            opt.step()                                             # update
        times.append(time.perf_counter() - started)                # record the epoch
    return float(np.mean(times))                                   # mean epoch time

rows = []                                                          # one row per batch size
for bs in [16, 32, 64, 128, 256]:                                  # a range of batch sizes
    seconds = time_epoch(bs)                                       # time it
    rows.append({"batch_size": bs, "seconds_per_epoch": round(seconds, 3),   # the timing
                 "steps_per_epoch": int(np.ceil(N / bs))})          # the number of updates
    print("batch %3d | %.3f s/epoch | %d steps" % (bs, seconds, np.ceil(N / bs)))   # report
timing = pd.DataFrame(rows)                                        # assemble the table
dump(timing, "l9_batch_timing.csv")                                # persist it

# ---------------------------------------------------------------- 3) gradient accumulation emulates a large batch
def time_accum(batch_size, accum_steps, epochs=3):                 # accumulate the gradients
    """Return the mean seconds per epoch when accumulating accum_steps micro-batches."""   # short docstring
    opt = optim.SGD(model.parameters(), lr=0.01)                   # the optimiser
    times = []                                                     # epoch timings
    for _ in range(epochs):                                        # a few epochs
        started = time.perf_counter()                              # start the clock
        for start in range(0, N, batch_size * accum_steps):         # outer loop over the effective batch
            opt.zero_grad()                                        # clear once per effective batch
            for m in range(accum_steps):                           # inner loop over the micro-batches
                s = start + m * batch_size                         # the micro-batch start
                xb = X[s:s + batch_size]                           # micro-batch features
                yb = y[s:s + batch_size]                           # micro-batch labels
                loss = criterion(model(xb), yb) / accum_steps       # scale the loss
                loss.backward()                                    # accumulate the gradients
            opt.step()                                             # one update per effective batch
        times.append(time.perf_counter() - started)                # record
    return float(np.mean(times))                                   # the mean timing

accum_rows = []                                                    # results per configuration
for bs, accum in [(16, 8), (32, 4), (64, 2), (128, 1)]:            # all emulate a batch of 128
    seconds = time_accum(bs, accum)                                # time it
    accum_rows.append({"micro_batch": bs, "accum_steps": accum,     # the configuration
                       "effective_batch": bs * accum,              # the emulated batch size
                       "seconds_per_epoch": round(seconds, 3)})     # the timing
    print("micro %3d x %d | %.3f s/epoch" % (bs, accum, seconds))   # report
accum = pd.DataFrame(accum_rows)                                   # assemble the table
dump(accum, "l9_grad_accum.csv")                                   # persist it

# ---------------------------------------------------------------- 4) memory arithmetic
def memory_report(total_params, bytes_per_value=4):                # a simple memory estimate
    """Return the memory of the weights, their gradients and the Adam states."""   # short docstring
    weights = total_params * bytes_per_value                       # the parameters themselves
    grads = total_params * bytes_per_value                         # one gradient per parameter
    adam = total_params * bytes_per_value * 2                      # first and second moments
    return {"weights_mb": round(weights / 2 ** 20, 2),             # weights in MB
            "grads_mb": round(grads / 2 ** 20, 2),                 # gradients in MB
            "adam_mb": round(adam / 2 ** 20, 2),                   # optimiser state in MB
            "total_mb": round((weights + grads + adam) / 2 ** 20, 2)}   # the sum
estimates = pd.DataFrame([{"model": "this toy model", **memory_report(param_count(model))},   # the toy model
                          {"model": "7B model (fp16)", **memory_report(7_000_000_000, 2)},   # a large model
                          {"model": "7B model (4-bit weights)", **memory_report(7_000_000_000, 0.5)}])   # quantised
dump(estimates, "l9_memory_estimate.csv")                          # persist it
print(estimates.to_string(index=False))                            # report

# ---------------------------------------------------------------- 5) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                # three panels
axes[0].plot(timing["batch_size"], timing["seconds_per_epoch"], "-o", ms=5, color=OKABE[5])   # timing
style_axes(axes[0], "Bigger batches are not always faster", "batch size", "seconds per epoch")   # shared style
axes[1].bar(accum["effective_batch"].astype(str) + " (" + accum["micro_batch"].astype(str) + "x" + accum["accum_steps"].astype(str) + ")",   # labels
            accum["seconds_per_epoch"], color=OKABE[2], alpha=0.9)   # timings
axes[1].tick_params(axis="x", rotation=20, labelsize=7.2)           # rotate the labels
style_axes(axes[1], "Accumulation emulates a large batch", "", "seconds per epoch")   # shared style
components = ["weights", "grads", "adam"]                           # memory components
bottom = np.zeros(len(estimates))                                  # stacked bars start at zero
for comp in components:                                            # loop over the components
    values = estimates[comp + "_mb"].values                        # the component sizes
    axes[2].bar(estimates["model"], values, bottom=bottom, label=comp, alpha=0.9)   # the stack
    bottom += values                                               # advance the base
axes[2].set_yscale("log")                                          # the sizes span orders of magnitude
axes[2].tick_params(axis="x", rotation=20, labelsize=7.2)           # rotate the labels
style_axes(axes[2], "Memory: weights, gradients, optimiser", "", "MB (log)")   # shared style
axes[2].legend(fontsize=7.0, loc="lower right")                     # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_engineering.png")                                   # save the figure
print("done: batch timing, gradient accumulation and memory arithmetic")   # closing log
