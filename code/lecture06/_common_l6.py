"""_common_l6.py -- shared helpers for the Lecture 6 linear-model examples.

Every line carries an English comment. The module fixes fonts, colours, plotting
defaults and the random seed so that every figure in the deck is reproducible.
"""
import json                                                        # read and write JSON artefacts
import matplotlib                                                  # choose a backend before other plotting imports
matplotlib.use("Agg")                                              # headless backend: figures render without a display
import matplotlib.pyplot as plt                                    # figure and axes API
import numpy as np                                                 # numeric arrays and random numbers
import pandas as pd                                                # tabular data structures
from pathlib import Path                                           # portable path handling

SEED = 20260603                                                    # single seed shared by the whole lecture
RNG = np.random.default_rng(SEED)                                  # shared generator keeps samples reproducible
ROOT = Path(__file__).resolve().parents[2]                         # .../website (two levels above code/lecture06)
FIG = ROOT / "figures" / "lecture06"                               # output folder for figures
DAT = ROOT / "data" / "lecture06"                                  # output folder for sample data
OUT = DAT / "artifacts"                                            # tables, run logs and the glossary
for _d in (FIG, DAT, OUT):                                         # make sure every output folder exists
    _d.mkdir(parents=True, exist_ok=True)                          # create it when missing

# ---------------------------------------------------------------- shared visual style
FONT = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]                # CJK-capable font stack for Chinese labels
PALETTE = ["#8A0C3C", "#C9A227", "#66717D", "#2E6F8E", "#711426",  # SZU red, gold, grey, blue, deep red
           "#A6426B", "#4F7942", "#B5651D"]                        # four more distinguishable hues
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",               # Okabe-Ito colour-blind safe palette
         "#0072B2", "#D55E00", "#CC79A7", "#000000"]               # eight colours, still distinct in greyscale

plt.rcParams["font.sans-serif"] = FONT                             # use the CJK font stack globally
plt.rcParams["axes.unicode_minus"] = False                         # draw minus signs instead of tofu boxes
plt.rcParams["figure.dpi"] = 140                                   # on-screen resolution while drafting
plt.rcParams["savefig.dpi"] = 200                                  # higher resolution when saving for slides
plt.rcParams["savefig.bbox"] = "tight"                             # trim surrounding white space
plt.rcParams["axes.grid"] = True                                   # light grid helps reading values
plt.rcParams["grid.alpha"] = 0.25                                  # keep the grid subtle
plt.rcParams["axes.titlesize"] = 12.0                              # subplot title size
plt.rcParams["axes.labelsize"] = 10.5                              # axis label size
plt.rcParams["xtick.labelsize"] = 9.0                              # x tick label size
plt.rcParams["ytick.labelsize"] = 9.0                              # y tick label size
plt.rcParams["legend.fontsize"] = 8.6                              # legend text size
plt.rcParams["figure.titlesize"] = 13.5                            # figure level title size
plt.rcParams["axes.spines.top"] = False                            # drop the top spine to reduce chart junk
plt.rcParams["axes.spines.right"] = False                          # drop the right spine as well


def style_axes(ax, title=None, xlabel=None, ylabel=None):           # apply the shared style to one axes
    """Set title and axis labels with the shared style."""           # short docstring
    if title:                                                      # only set a title when provided
        ax.set_title(title)                                        # write the subplot title
    if xlabel:                                                     # only set an x label when provided
        ax.set_xlabel(xlabel)                                      # write the x axis label
    if ylabel:                                                     # only set a y label when provided
        ax.set_ylabel(ylabel)                                      # write the y axis label
    return ax                                                      # return the axes for chaining


def annotated(ax, text, loc="upper left"):                          # add an in-figure note (self-contained captions)
    """Place a short note inside the axes, with a light background box."""   # short docstring
    xy = {"upper left": (0.02, 0.96), "upper right": (0.98, 0.96),  # two positions on the top edge
          "lower left": (0.02, 0.06), "lower right": (0.98, 0.06)}.get(loc, (0.02, 0.96))   # fall back to top left
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, va="top",   # place text in axes coordinates
            ha="left" if "left" in loc else "right", fontsize=8.6,  # align left or right as requested
            color="#1E2630", bbox=dict(boxstyle="round,pad=0.25",   # add a pale rounded background
                                       fc="#F7F3F2", ec="#D6CDD0", lw=0.6))   # background and border colours
    return ax                                                      # return the axes for chaining


def save(fig, name):                                               # persist a figure into the figures folder
    """Save a figure under website/figures/lecture06 and close it."""   # short docstring
    path = FIG / name                                              # build the full output path
    fig.savefig(path, facecolor="white")                           # white background keeps slides readable
    plt.close(fig)                                                 # release the figure from memory
    print("[figure]", path)                                        # log the artefact path


def dump(df, name):                                                # persist a dataframe into the data folder
    """Write a dataframe to website/data/lecture06 as UTF-8 CSV."""   # short docstring
    path = DAT / name                                              # build the full output path
    df.to_csv(path, index=False, encoding="utf-8-sig")             # utf-8-sig opens cleanly in Excel
    print("[data]", path)                                          # log the artefact path


def dump_artifact(text, name):                                     # persist a text artefact (markdown, JSON)
    """Write a text artefact into data/lecture06/artifacts."""      # short docstring
    path = OUT / name                                              # build the path inside the artifacts folder
    path.write_text(text, encoding="utf-8")                        # write with UTF-8 so Chinese labels survive
    print("[artifact]", path)                                      # log the artefact path


def write_json(obj, name):                                         # persist a dictionary as JSON
    """Serialise a dictionary to artifacts/<name> with UTF-8 encoding."""   # short docstring
    dump_artifact(json.dumps(obj, ensure_ascii=False, indent=2), name)   # pretty print and keep Chinese readable


def add_intercept(X):                                              # add a constant column for the bias term
    """Return X with a leading column of ones (the augmented trick)."""   # short docstring
    X = np.asarray(X, dtype=float)                                 # work with a float array
    return np.column_stack([np.ones(X.shape[0]), X])               # ones first, then the original columns


def normal_equation(X, y):                                         # closed-form least squares
    """Solve w = (X^T X)^{-1} X^T y for an already augmented design matrix."""   # short docstring
    XtX = X.T @ X                                                  # gram matrix
    Xty = X.T @ y                                                  # cross-product with the target
    return np.linalg.solve(XtX, Xty)                               # solve instead of inverting for stability


def rmse(y_true, y_pred):                                          # root mean squared error helper
    """Return the root mean squared error of two numeric arrays."""   # short docstring
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))   # compute and cast to float


def mae(y_true, y_pred):                                           # mean absolute error helper
    """Return the mean absolute error of two numeric arrays."""     # short docstring
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))   # average absolute deviation


def r2_score(y_true, y_pred):                                      # coefficient of determination
    """Return R^2 computed from the residual and total sums of squares."""   # short docstring
    y_true = np.asarray(y_true, dtype=float)                       # ensure float input
    resid = float(np.sum((y_true - np.asarray(y_pred)) ** 2))      # residual sum of squares
    total = float(np.sum((y_true - y_true.mean()) ** 2))           # total sum of squares
    return 1.0 - resid / total                                     # share of variance explained


def sigmoid(a):                                                    # logistic squashing function
    """Return the numerically stable sigmoid of a scalar or array."""   # short docstring
    a = np.asarray(a, dtype=float)                                 # work with float arrays
    out = np.empty_like(a)                                         # allocate the output array
    pos = a >= 0                                                   # mask for non-negative inputs
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))                       # stable form for large positive values
    exp_a = np.exp(a[~pos])                                        # stable form for negative values
    out[~pos] = exp_a / (1.0 + exp_a)                              # equivalent to 1/(1+exp(-a)) without overflow
    return out                                                     # return the sigmoid values


def logsumexp(a, axis=None, keepdims=False):                       # stable log-sum-exp helper
    """Return log(sum(exp(a))) computed without overflow."""        # short docstring
    a = np.asarray(a, dtype=float)                                 # work with float arrays
    amax = np.max(a, axis=axis, keepdims=True)                     # subtract the maximum first
    shifted = np.exp(a - amax)                                     # safe exponentials
    out = amax + np.log(np.sum(shifted, axis=axis, keepdims=True))   # log-sum-exp of the shifted values
    if axis is None and not keepdims:                              # reduce to a plain scalar
        return float(np.squeeze(out))                              # return a python float
    if not keepdims:                                               # squeeze the reduced axis when asked
        out = np.squeeze(out, axis=axis)                           # drop the singleton dimension
    return out                                                     # return the stable result
