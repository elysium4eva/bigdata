"""_common_l7.py -- shared helpers for the Lecture 7 supervised-learning examples.

Every line carries an English comment. Fonts, colours, seeds and output folders are
fixed here so that every figure in the deck is reproducible from one command.
"""
import json                                                        # read and write JSON artefacts
import matplotlib                                                  # choose a backend before other plotting imports
matplotlib.use("Agg")                                              # headless backend: figures render without a display
import matplotlib.pyplot as plt                                    # figure and axes API
import numpy as np                                                 # numeric arrays and random numbers
import pandas as pd                                                # tabular data structures
from pathlib import Path                                           # portable path handling

SEED = 20260708                                                    # single seed shared by the whole lecture
RNG = np.random.default_rng(SEED)                                  # shared generator keeps samples reproducible
ROOT = Path(__file__).resolve().parents[2]                         # .../website (two levels above code/lecture07)
FIG = ROOT / "figures" / "lecture07"                               # output folder for figures
DAT = ROOT / "data" / "lecture07"                                  # output folder for sample data
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
    """Save a figure under website/figures/lecture07 and close it."""   # short docstring
    path = FIG / name                                              # build the full output path
    fig.savefig(path, facecolor="white")                           # white background keeps slides readable
    plt.close(fig)                                                 # release the figure from memory
    print("[figure]", path)                                        # log the artefact path


def dump(df, name):                                                # persist a dataframe into the data folder
    """Write a dataframe to website/data/lecture07 as UTF-8 CSV."""   # short docstring
    path = DAT / name                                              # build the full output path
    df.to_csv(path, index=False, encoding="utf-8-sig")             # utf-8-sig opens cleanly in Excel
    print("[data]", path)                                          # log the artefact path


def dump_artifact(text, name):                                     # persist a text artefact (markdown, JSON)
    """Write a text artefact into data/lecture07/artifacts."""      # short docstring
    path = OUT / name                                              # build the path inside the artifacts folder
    path.write_text(text, encoding="utf-8")                        # write with UTF-8 so Chinese labels survive
    print("[artifact]", path)                                      # log the artefact path


def write_json(obj, name):                                         # persist a dictionary as JSON
    """Serialise a dictionary to artifacts/<name> with UTF-8 encoding."""   # short docstring
    dump_artifact(json.dumps(obj, ensure_ascii=False, indent=2), name)   # pretty print and keep Chinese readable


def make_two_class(n=220):                                         # a two-dimensional binary problem
    """Return a two-dimensional sample with a curved boundary (for SVM demos)."""   # short docstring
    x1 = RNG.uniform(-2.2, 2.2, n)                                 # horizontal coordinate
    x2 = RNG.uniform(-2.2, 2.2, n)                                 # vertical coordinate
    score = x1 ** 2 + 0.8 * x2 ** 2 - 1.4 + RNG.normal(0, 0.35, n)   # a curved, non-linear boundary
    y = (score > 0).astype(int)                                    # binary label from the curved score
    return pd.DataFrame({"x1": x1.round(3), "x2": x2.round(3), "y": y})   # tabulate the sample


def make_two_class_linear(n=200):                                  # a roughly linearly separable problem
    """Return a two-dimensional sample with a nearly linear boundary (for margin demos)."""   # short docstring
    y = (RNG.random(n) < 0.5).astype(int)                          # balanced binary labels
    x1 = RNG.normal(0, 0.9, n) + 1.1 * y                           # separated along a diagonal
    x2 = RNG.normal(0, 0.9, n) + 0.9 * y                           # second coordinate follows the label
    return pd.DataFrame({"x1": x1.round(3), "x2": x2.round(3), "y": y})   # tabulate the sample


def make_knn_sample(n=240):                                        # a three-cluster sample for KNN maps
    """Return a two-dimensional three-class sample built from Gaussian clusters."""   # short docstring
    centres = np.array([[-1.6, 1.1], [1.5, 1.2], [0.0, -1.6]])     # three cluster centres
    labels = RNG.integers(0, 3, n)                                 # balanced random assignment
    X = centres[labels] + RNG.normal(0, 0.95, (n, 2))              # features around the centres
    return pd.DataFrame({"x1": X[:, 0].round(3), "x2": X[:, 1].round(3), "label": labels})   # tabulate


def make_poverty(n=600):                                           # the poverty-targeting sample
    """Return a synthetic household sample modelled on the poverty-targeting example."""   # short docstring
    log_income = RNG.normal(7.0, 0.7, n)                           # log household income
    assets = np.clip(RNG.normal(2.4, 1.1, n), 0, None)             # asset index
    size = np.clip(RNG.poisson(4.2, n), 1, 14)                     # household size
    educ = np.clip(RNG.normal(6.0, 2.4, n), 0, 16)                 # years of education of the head
    rural = (RNG.random(n) < 0.62).astype(int)                     # rural location indicator
    migration = np.clip(RNG.normal(0.35, 0.25, n), 0, 1)           # share of migrant income
    consumption = (1.8 * log_income + 0.55 * assets + 0.12 * size   # the true relationship
                   + 0.22 * educ - 0.9 * rural + 1.6 * migration   # remaining true effects
                   - 0.8 + RNG.normal(0, 0.9, n))                  # intercept and noise
    return pd.DataFrame({"log_income": log_income.round(3), "assets": assets.round(3),   # predictors
                         "size": size, "educ": educ.round(1), "rural": rural,            # more predictors
                         "migration": migration.round(3), "consumption": consumption.round(3)})   # target


def make_tabular(n=1400):                                          # a general tabular sample for GBDT work
    """Return a mixed-feature tabular sample with a non-linear target."""   # short docstring
    f1 = RNG.normal(0, 1, n)                                       # continuous feature one
    f2 = np.clip(RNG.normal(0, 1, n), -2.5, 2.5)                   # continuous feature two, capped tails
    f3 = RNG.integers(0, 4, n)                                     # ordinal feature with four levels
    f4 = RNG.choice(["east", "west", "north"], n)                  # nominal feature
    f5 = np.clip(RNG.normal(30, 8, n), 0, None)                    # age-like feature
    f6 = (RNG.random(n) < 0.3).astype(int)                         # binary flag
    region_effect = np.where(f4 == "east", 0.8, np.where(f4 == "west", -0.4, 0.2))   # nominal effect
    target = (1.2 * f1 + 0.6 * f2 ** 2 + 0.5 * f3 + region_effect  # the non-linear relationship
              + 0.05 * f5 + 0.4 * f6 + 0.7 * f1 * f6               # interaction term
              + RNG.normal(0, 0.7, n))                             # noise
    return pd.DataFrame({"f1": f1.round(3), "f2": f2.round(3), "f3": f3,   # continuous and ordinal
                         "f4": f4, "f5": f5.round(1), "f6": f6,        # nominal, age-like and binary
                         "target": target.round(3)})                   # target column


def encode_nominal(train_frame, test_frame, col="f4"):             # encode only the nominal column
    """Return numeric matrices with the nominal column mapped to integer codes."""   # short docstring
    levels = sorted(train_frame[col].unique().tolist())            # levels seen in the training split
    mapping = {k: i for i, k in enumerate(levels)}                 # stable code per level
    tr = train_frame.copy()                                        # copy the training frame
    te = test_frame.copy()                                         # copy the test frame
    tr[col] = tr[col].map(mapping).fillna(-1).astype(int)          # map the training rows
    te[col] = te[col].map(mapping).fillna(-1).astype(int)          # unknown levels become -1
    return tr.values.astype(float), te.values.astype(float), mapping   # return the matrices and the map


def accuracy(y_true, y_pred):                                      # simple accuracy helper
    """Return the share of correctly classified rows."""            # short docstring
    y_true = np.asarray(y_true)                                    # ensure an array
    y_pred = np.asarray(y_pred)                                    # ensure an array
    return float(np.mean(y_true == y_pred))                        # fraction of equal entries


def rmse(y_true, y_pred):                                          # root mean squared error helper
    """Return the root mean squared error of two numeric arrays."""   # short docstring
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))   # compute and cast


def r2_score(y_true, y_pred):                                      # coefficient of determination
    """Return R^2 computed from the residual and total sums of squares."""   # short docstring
    y_true = np.asarray(y_true, dtype=float)                       # ensure float input
    resid = float(np.sum((y_true - np.asarray(y_pred)) ** 2))      # residual sum of squares
    total = float(np.sum((y_true - y_true.mean()) ** 2))           # total sum of squares
    return 1.0 - resid / total                                     # share of variance explained
