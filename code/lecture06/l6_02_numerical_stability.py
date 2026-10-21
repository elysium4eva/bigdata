"""l6_02_numerical_stability.py -- why (X^T X)^{-1} X^T y is a bad algorithm.

Every line carries an English comment. The script builds designs whose columns are
almost collinear, then compares the normal equation against QR and SVD-based least
squares. It shows the coefficient error growing with the condition number and
records the results for the slides.
"""
import numpy as np                                                 # linear algebra and comparisons
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # figures for the error curves
from _common_l6 import (RNG, save, dump, add_intercept,            # shared helpers from the lecture module
                        normal_equation, rmse, style_axes, annotated, OKABE)   # # helper names imported from the lecture module

# ---------------------------------------------------------------- 1) a base sample with two useful columns
N = 120                                                            # number of observations
x1 = RNG.normal(0, 1, N)                                           # first predictor
x2 = RNG.normal(0, 1, N)                                           # second, nearly independent predictor
noise = RNG.normal(0, 0.5, N)                                      # irreducible noise
y = 1.5 * x1 - 0.8 * x2 + noise                                    # linear target

# ---------------------------------------------------------------- 2) sweep the collinearity level
levels = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]                 # noise added to the copied column
rows = []                                                          # one row per collinearity level
for level in levels:                                               # build a design for each level
    x3 = x1 + RNG.normal(0, level, N)                              # x3 tracks x1 with decreasing noise
    design = add_intercept(np.column_stack([x1, x2, x3]))          # augmented design matrix
    gram = design.T @ design                                       # gram matrix
    cond = float(np.linalg.cond(gram))                             # condition number of the normal equations
    w_ne = normal_equation(design, y)                              # normal-equation solution
    w_qr = np.linalg.lstsq(design, y, rcond=None)[0]               # QR/SVD based solution
    w_sv = np.linalg.pinv(design) @ y                              # explicit pseudo-inverse solution
    resid_ne = rmse(y, design @ w_ne)                              # residual of the normal-equation fit
    resid_qr = rmse(y, design @ w_qr)                              # residual of the stable fit
    rows.append({"collinearity_noise": level, "cond_gram": cond,   # record the condition number
                 "coef_gap_ne_vs_qr": float(np.max(np.abs(w_ne - w_qr))),   # largest coefficient gap
                 "rmse_ne": resid_ne, "rmse_qr": resid_qr,         # residuals of both solutions
                 "w_x3_ne": float(w_ne[3]), "w_x3_qr": float(w_qr[3])})   # the unstable coefficient itself
stability = pd.DataFrame(rows).round(6)                            # assemble the table
dump(stability, "l6_stability.csv")                                # persist it
print(stability[["collinearity_noise", "cond_gram", "coef_gap_ne_vs_qr"]].to_string(index=False))   # report

# ---------------------------------------------------------------- 3) a single ill-conditioned sample to share
x3_bad = x1 + RNG.normal(0, 0.01, N)                               # strongly collinear third column
ill = pd.DataFrame({"x1": x1.round(3), "x2": x2.round(3), "x3": x3_bad.round(3), "y": y.round(3)})   # table
dump(ill, "l6_illconditioned.csv")                                 # persist the ill-conditioned sample

# ---------------------------------------------------------------- 4) figure: error versus condition number
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6))                # three panels
axes[0].semilogx(stability["cond_gram"], stability["coef_gap_ne_vs_qr"], "-o", ms=5, color=OKABE[5])   # gap curve
axes[0].set_yscale("log")                                          # the gap spans orders of magnitude
style_axes(axes[0], "Coefficient gap grows with $\\kappa$", "cond$(X'^\\top X')$", "max |w_NE - w_QR|")   # style
axes[0].axhline(1e-6, ls="--", lw=1.0, color="#66717D")            # reference level for acceptable error
annotated(axes[0], "normal equation loses digits", "upper left")   # annotate the message
axes[1].plot(stability["collinearity_noise"], stability["rmse_ne"], "-o", ms=4,   # residual of the normal equation
             color=OKABE[5], label="normal equation")              # legend label
axes[1].plot(stability["collinearity_noise"], stability["rmse_qr"], "-s", ms=4,   # residual of the stable solver
             color=OKABE[2], label="QR / lstsq")                   # legend label
axes[1].set_xscale("log")                                          # collinearity noise on a log axis
style_axes(axes[1], "Residuals stay similar, coefficients do not", "collinearity noise", "RMSE")   # style
axes[1].legend(fontsize=7.6)                                       # compact legend
axes[2].plot(range(len(levels)), stability["w_x3_ne"], "-o", ms=4,   # unstable coefficient from the normal equation
             color=OKABE[5], label="normal equation")              # legend label
axes[2].plot(range(len(levels)), stability["w_x3_qr"], "-s", ms=4,   # the same coefficient from the stable solver
             color=OKABE[2], label="QR / lstsq")                   # legend label
axes[2].set_xticks(range(len(levels)), [str(l) for l in levels], rotation=45, fontsize=7.6)   # level labels
style_axes(axes[2], "The collinear coefficient explodes", "collinearity noise", "$w_3$")   # style
axes[2].legend(fontsize=7.6)                                       # compact legend
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_condition_number.png")                              # save the stability figure
print("done: stability sweep, %d levels" % len(levels))             # closing log
