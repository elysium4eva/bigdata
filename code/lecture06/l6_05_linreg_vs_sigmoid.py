"""l6_05_linreg_vs_sigmoid.py -- step, linear and sigmoid predictions, outliers and gradient descent.

Every line carries an English comment. The script compares the three candidate
predictions for a binary problem, shows how a single outlier distorts a regression
classifier, and trains logistic regression with plain gradient descent so that the
convergence curve and the learning-rate sensitivity can be plotted.
"""
import numpy as np                                                 # arrays for the synthetic problem
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # four figures for the story
from sklearn.linear_model import LogisticRegression               # reference optimiser
from _common_l6 import (RNG, save, dump, sigmoid, add_intercept,   # shared helpers, including the sigmoid
                        style_axes, annotated, OKABE)                   # # helper names imported from the lecture module

# ---------------------------------------------------------------- 1) a two-class problem with separability
N = 160                                                            # observations
x = RNG.normal(0, 1, N)                                            # single informative feature
y = (x + RNG.normal(0, 0.9, N) > 0).astype(int)                    # noisy threshold label
two = pd.DataFrame({"x": x.round(3), "y": y})                      # tabulate the clean sample
# Add three outliers that fight the trend: this is what breaks the regression classifier.
x_out = np.array([-2.6, -2.2, 2.4])                                # outlier locations
y_out = np.array([1, 1, 0])                                        # wrong-side labels
two_out = pd.concat([two, pd.DataFrame({"x": x_out, "y": y_out})], ignore_index=True)   # contaminated sample
dump(two_out, "l6_two_class.csv")                                  # persist the contaminated sample
X = add_intercept(two["x"].values)                                 # design matrix for the clean sample
Xo = add_intercept(two_out["x"].values)                            # augmented matrix of the contaminated sample (kept for reference)

# ---------------------------------------------------------------- 2) three candidate predictions
grid = np.linspace(-3.2, 3.2, 300)                                 # grid for drawing the curves
x_col = two[["x"]].values                                          # sklearn expects a 2D feature matrix
log_clean = LogisticRegression(max_iter=1000).fit(x_col, two["y"].values)   # logistic on the clean sample
bias_c = float(log_clean.intercept_[0])                            # fitted intercept of the clean model
slope_c = float(log_clean.coef_[0][0])                             # fitted slope of the clean model
linear_part = bias_c + slope_c * grid                              # the linear score w^T x
step_part = (linear_part > 0).astype(float)                        # the step function
sigmoid_part = sigmoid(linear_part)                                # the sigmoid squashing

# ---------------------------------------------------------------- 3) outlier damage: regression versus logistic
reg = np.polyfit(two_out["x"].values, two_out["y"].values, 1)      # least squares treated as a classifier
thr_reg = -reg[1] / reg[0]                                         # where the fitted line crosses 0.5
log_dirty = LogisticRegression(max_iter=1000).fit(two_out[["x"]].values, two_out["y"].values)   # contaminated
slope_d = float(log_dirty.coef_[0][0])                             # fitted slope with outliers present
print("regression threshold %.2f | logistic slope clean %.2f dirty %.2f" % (   # # report the two slopes
      thr_reg, slope_c, slope_d))                                  # report how much the outlier moved things

# ---------------------------------------------------------------- 4) gradient descent from scratch
def logistic_loss(w_vec, Xm, yv, l2=0.0):                          # mean cross-entropy plus optional L2
    """Return the mean cross-entropy of the logistic model on (Xm, yv)."""   # short docstring
    p = sigmoid(Xm @ w_vec)                                        # predicted probabilities
    eps = 1e-12                                                    # avoid log(0)
    loss = -np.mean(yv * np.log(p + eps) + (1 - yv) * np.log(1 - p + eps))   # cross-entropy
    loss += 0.5 * l2 * float(w_vec @ w_vec)                        # weight decay term
    return float(loss)                                             # return the scalar loss

def logistic_grad(w_vec, Xm, yv, l2=0.0):                          # analytic gradient
    """Return the gradient of the mean cross-entropy with respect to w."""   # short docstring
    p = sigmoid(Xm @ w_vec)                                        # predicted probabilities
    grad = (Xm.T @ (p - yv)) / len(yv)                             # X^T (p - y) / n
    grad += l2 * w_vec                                             # gradient of the weight decay
    return grad                                                    # return the gradient vector

def train_gd(learning_rate, epochs=1200, l2=0.0):                  # full-batch gradient descent
    """Run gradient descent and return the loss history and the final weights."""   # short docstring
    w_vec = np.zeros(X.shape[1])                                   # start from zero weights
    hist = []                                                      # loss history per epoch
    for _ in range(epochs):                                        # fixed number of full-batch steps
        hist.append(logistic_loss(w_vec, X, two["y"].values, l2))   # record the loss before the update
        w_vec = w_vec - learning_rate * logistic_grad(w_vec, X, two["y"].values, l2)   # gradient step
    return np.array(hist), w_vec                                   # return history and final weights

histories = {}                                                     # loss curves for several learning rates
for lr in [0.5, 2.0, 20.0]:                                        # too small, reasonable, past the divergence point
    histories[lr], final_w = train_gd(lr)                          # train with this learning rate
    print("lr %.1f | final loss %.4f | w %s" % (lr, histories[lr][-1], np.round(final_w, 3)))   # report
dump(pd.DataFrame({"epoch": np.arange(1, len(histories[0.5]) + 1),   # epoch index
                   "loss_lr_0.5": histories[0.5],                 # loss curve for lr = 0.5
                   "loss_lr_2.0": histories[2.0],                 # loss curve for lr = 2.0
                   "loss_lr_20.0": histories[20.0]}).round(6),     # loss curve for lr = 20.0
     "l6_gd_history.csv")                                          # persist the history

# ---------------------------------------------------------------- 5) figure one: the three predictions
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6))                # three panels
axes[0].scatter(two["x"], two["y"], s=14, color=OKABE[0], alpha=0.5, label="class 1")   # observations
axes[0].plot(grid, linear_part, lw=2.0, color=OKABE[3], label="linear $w^\\top x$")   # unbounded score
axes[0].plot(grid, step_part, lw=1.6, color=OKABE[5], ls="--", label="step")   # discontinuous decision
axes[0].plot(grid, sigmoid_part, lw=2.2, color=OKABE[2], label="sigmoid")   # smooth probability
style_axes(axes[0], "Three candidate predictions", "feature $x$", "prediction")   # shared style
axes[0].legend(fontsize=7.2, loc="center right")                   # compact legend
z = np.linspace(-6, 6, 400)                                        # grid for the sigmoid panel
axes[1].plot(z, sigmoid(z), lw=2.2, color=OKABE[2], label="$\\sigma(z)$")   # sigmoid curve
axes[1].plot(z, sigmoid(z) * (1 - sigmoid(z)), lw=2.0, color=OKABE[5], label="$\\sigma'(z)$")   # derivative
axes[1].axhline(0.5, ls="--", lw=1.0, color="#66717D")             # decision threshold
style_axes(axes[1], "Sigmoid and its derivative", "score $z=w^\\top x$", "value")   # shared style
axes[1].legend(fontsize=7.6, loc="upper left")                     # legend inside the panel
annotated(axes[1], "max slope $\\sigma'(0)=0.25$", "lower right")  # annotate the maximum slope
xg = np.linspace(-3.2, 3.2, 300)                                   # grid for the boundaries
axes[2].scatter(two_out["x"], two_out["y"], s=16, color="#B9B3B6", alpha=0.8, label="data + outliers")   # contaminated data
axes[2].plot(xg, reg[0] * xg + reg[1], lw=1.8, color=OKABE[5], label="least squares")   # damaged boundary
axes[2].plot(xg, sigmoid(bias_c + slope_c * xg), lw=2.2,   # clean logistic
             color=OKABE[2], label="logistic (clean)")             # legend label
style_axes(axes[2], "Outliers bend least squares, not logistic", "feature $x$", "predicted probability")   # style
axes[2].legend(fontsize=7.0, loc="center right")                   # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_three_curves.png")                                  # save the prediction comparison

# ---------------------------------------------------------------- 6) figure two: outlier effect in detail
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6))                # three panels
axes[0].scatter(two["x"], two["y"], s=14, color=OKABE[0], alpha=0.45, label="clean data")   # clean points
axes[0].scatter(x_out, y_out, s=90, marker="X", color=OKABE[5], label="outliers")   # outliers
axes[0].plot(xg, reg[0] * xg + reg[1], lw=1.8, color=OKABE[5], label="least squares fit")   # damaged line
style_axes(axes[0], "Three points move the regression line", "feature $x$", "label $y$")   # shared style
axes[0].legend(fontsize=7.2, loc="center left")                    # compact legend
axes[1].plot(grid, sigmoid(bias_c + slope_c * grid), lw=2.2,   # clean logistic
             color=OKABE[2], label="clean")                        # legend label
axes[1].plot(grid, sigmoid(float(log_dirty.intercept_[0]) + slope_d * grid), lw=2.2,   # contaminated logistic
             color=OKABE[5], ls="--", label="with outliers")       # legend label
axes[1].scatter(x_out, y_out, s=70, marker="X", color=OKABE[5])    # mark the outliers again
style_axes(axes[1], "Logistic regression is robust here", "feature $x$", "predicted probability")   # shared style
axes[1].legend(fontsize=7.6, loc="center right")                   # compact legend
axes[2].bar(["clean", "with outliers"], [slope_c, slope_d],   # slopes
            color=[OKABE[2], OKABE[5]], alpha=0.9)                 # one colour per condition
style_axes(axes[2], "Slope barely changes", "", "fitted slope $w_1$")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_outlier_effect.png")                                # save the outlier figure

# ---------------------------------------------------------------- 7) figure three: sigmoid properties
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6))                # three panels for the properties
zz = np.linspace(-8, 8, 500)                                       # wide grid for the sigmoid
axes[0].plot(zz, sigmoid(zz), lw=2.4, color=OKABE[2])              # the sigmoid itself
axes[0].axhline(1.0, ls="--", lw=1.0, color="#66717D")             # upper asymptote
axes[0].axhline(0.0, ls="--", lw=1.0, color="#66717D")             # lower asymptote
style_axes(axes[0], "Bounded: $\\sigma(a)\\in(0,1)$", "score $a$", "$\\sigma(a)$")   # shared style
annotated(axes[0], "never reaches 0 or 1", "lower right")          # annotate the bound
axes[1].plot(zz, sigmoid(zz) * (1 - sigmoid(zz)), lw=2.4, color=OKABE[5])   # derivative
axes[1].axvline(0, ls="--", lw=1.0, color="#66717D")               # centre line
axes[1].axhline(0.25, ls=":", lw=1.0, color="#66717D")             # maximum value
style_axes(axes[1], "Gradient: $\\sigma'(a)=\\sigma(a)(1-\\sigma(a))$", "score $a$", "$\\sigma'(a)$")   # style
annotated(axes[1], "max $0.25$ at $a=0$", "upper right")           # annotate the maximum
axes[2].plot(zz, 1 - sigmoid(zz), lw=4.0, color=OKABE[3], alpha=0.55, label="$1-\\sigma(a)$")   # thick line
axes[2].plot(zz, sigmoid(-zz), lw=1.6, ls="--", color=OKABE[4], label="$\\sigma(-a)$")   # dashed overlay
style_axes(axes[2], "Symmetry: $1-\\sigma(a)=\\sigma(-a)$", "score $a$", "value")   # shared style
axes[2].legend(fontsize=7.6, loc="center right")                   # the two curves coincide
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_sigmoid_props.png")                                 # save the property figure

# ---------------------------------------------------------------- 8) figure four: convergence and learning rates
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6))                # three panels
for lr, colour in zip([0.5, 2.0, 20.0], [OKABE[2], OKABE[3], OKABE[5]]):   # one curve per learning rate
    axes[0].plot(histories[lr], lw=1.8, color=colour, label="lr = %.1f" % lr)   # loss curve
axes[0].set_yscale("log")                                          # losses span orders of magnitude
style_axes(axes[0], "Gradient descent loss by learning rate", "epoch", "cross-entropy (log scale)")   # style
axes[0].legend(fontsize=7.6, loc="upper right")                    # compact legend
axes[1].plot(grid, sigmoid_part, lw=2.0, color=OKABE[2], label="logistic (GD)")   # fitted probability
axes[1].scatter(two["x"], two["y"], s=12, color="#B9B3B6", alpha=0.7)   # observations
style_axes(axes[1], "Fitted probability after GD", "feature $x$", "$P(y=1\\mid x)$")   # shared style
axes[1].legend(fontsize=7.6, loc="center right")                   # compact legend
axes[2].bar(["sklearn", "gd (lr=2)"],                             # compare the optimisers
            [slope_c, float(final_w[1])],     # fitted slopes
            color=[OKABE[2], OKABE[3]], alpha=0.9)                 # two colours
style_axes(axes[2], "Same solution, different route", "", "slope $w_1$")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_gd_convergence.png")                                # save the convergence figure
print("done: three curves, outlier effect, sigmoid properties and gradient descent")   # closing log
