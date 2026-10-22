"""_common_l9.py -- shared helpers for the Lecture 9 deep-learning examples.

Every line carries an English comment. Seeds, fonts, plotting defaults and the synthetic
data generators live here so that every figure in the deck is reproducible.
"""
import json                                                        # read and write JSON artefacts
import matplotlib                                                  # choose a backend before other plotting imports
matplotlib.use("Agg")                                              # headless backend: figures render without a display
import matplotlib.pyplot as plt                                    # figure and axes API
import numpy as np                                                 # numeric arrays and random numbers
import pandas as pd                                                # tabular data structures
from pathlib import Path                                           # portable path handling

SEED = 20261010                                                    # single seed shared by the whole lecture
RNG = np.random.default_rng(SEED)                                  # shared generator keeps samples reproducible
ROOT = Path(__file__).resolve().parents[2]                         # .../website (two levels above code/lecture09)
FIG = ROOT / "figures" / "lecture09"                               # output folder for figures
DAT = ROOT / "data" / "lecture09"                                  # output folder for sample data
OUT = DAT / "artifacts"                                            # tables, run logs and the glossary
for _d in (FIG, DAT, OUT):                                         # make sure every output folder exists
    _d.mkdir(parents=True, exist_ok=True)                          # create it when missing

# ---------------------------------------------------------------- shared visual style
FONT = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]                # CJK-capable font stack for Chinese labels
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
    """Save a figure under website/figures/lecture09 and close it."""   # short docstring
    path = FIG / name                                              # build the full output path
    fig.savefig(path, facecolor="white")                           # white background keeps slides readable
    plt.close(fig)                                                 # release the figure from memory
    print("[figure]", path)                                        # log the artefact path


def dump(df, name):                                                # persist a dataframe into the data folder
    """Write a dataframe to website/data/lecture09 as UTF-8 CSV."""   # short docstring
    path = DAT / name                                              # build the full output path
    df.to_csv(path, index=False, encoding="utf-8-sig")             # utf-8-sig opens cleanly in Excel
    print("[data]", path)                                          # log the artefact path


def dump_artifact(text, name):                                     # persist a text artefact (markdown, JSON)
    """Write a text artefact into data/lecture09/artifacts."""      # short docstring
    path = OUT / name                                              # build the path inside the artifacts folder
    path.write_text(text, encoding="utf-8")                        # write with UTF-8 so Chinese labels survive
    print("[artifact]", path)                                      # log the artefact path


def write_json(obj, name):                                         # persist a dictionary as JSON
    """Serialise a dictionary to artifacts/<name> with UTF-8 encoding."""   # short docstring
    dump_artifact(json.dumps(obj, ensure_ascii=False, indent=2), name)   # pretty print and keep Chinese readable


def make_shapes(n_per_class=60, size=16):                          # the synthetic image classification set
    """Return small grey-scale images of four shapes plus their labels."""   # short docstring
    images = []                                                    # image stack
    labels = []                                                    # class labels
    names = ["circle", "square", "triangle", "cross"]               # the four classes
    yy, xx = np.mgrid[0:size, 0:size]                              # pixel grid
    cy, cx = (size - 1) / 2, (size - 1) / 2                          # image centre
    for label, name in enumerate(names):                           # one class at a time
        for _ in range(n_per_class):                               # several variants per class
            img = np.zeros((size, size), dtype=float)               # blank canvas
            dy, dx = RNG.normal(0, 1.1, 2)                         # random offset from the centre
            oy, ox = cy + dy, cx + dx                              # shifted centre
            r = 4.0 + RNG.normal(0, 0.4)                           # radius or half side
            if name == "circle":                                   # a filled disc
                mask = ((yy - oy) ** 2 + (xx - ox) ** 2) <= r ** 2   # distance to the centre
                img[mask] = 1.0                                    # paint the disc
            elif name == "square":                                 # a filled square
                mask = (np.abs(yy - oy) <= r) & (np.abs(xx - ox) <= r)   # box test
                img[mask] = 1.0                                    # paint the square
            elif name == "triangle":                               # an upward triangle
                top = oy - r                                       # apex row
                base = oy + r                                      # base row
                width = r                                          # half width at the base
                for row in range(size):                            # scan the rows
                    frac = (row - top) / max(base - top, 1e-6)      # how far down the triangle
                    if 0 <= frac <= 1:                             # inside the vertical span
                        half = width * frac                        # half width at this row
                        left, right = int(ox - half), int(ox + half)   # horizontal extent
                        img[row, max(left, 0):min(right + 1, size)] = 1.0   # paint the row
            else:                                                  # a plus sign
                img[(np.abs(yy - oy) <= 1.2) & (np.abs(xx - ox) <= r)] = 1.0   # horizontal bar
                img[(np.abs(xx - ox) <= 1.2) & (np.abs(yy - oy) <= r)] = 1.0   # vertical bar
            img += RNG.normal(0, 0.06, (size, size))                # sensor noise
            images.append(np.clip(img, 0, 1))                      # keep it in [0, 1]
            labels.append(label)                                   # remember the class
    return np.asarray(images), np.asarray(labels), names            # return the stack


def make_moons(n=400, noise=0.18):                                 # the two-moons sample
    """Return a two-class non-linear sample shaped like two moons."""   # short docstring
    n_half = n // 2                                                # points per moon
    t = np.linspace(0, np.pi, n_half)                              # angle along the first arc
    x1 = np.c_[np.cos(t), np.sin(t)] + RNG.normal(0, noise, (n_half, 2))   # upper arc
    t2 = np.linspace(0, np.pi, n_half)                             # angle along the second arc
    x2 = np.c_[1 - np.cos(t2), 1 - np.sin(t2) - 0.5] + RNG.normal(0, noise, (n_half, 2))   # lower arc
    X = np.vstack([x1, x2])                                        # stack the two arcs
    y = np.r_[np.zeros(n_half), np.ones(n_half)].astype(int)        # binary labels
    return X, y                                                    # return features and labels


def make_delay_task(n=600, delay=6):                               # the delayed-recall sequence task
    """Return sequences where the model must remember a marker from earlier steps."""   # short docstring
    seqs = []                                                      # input sequences
    targets = []                                                   # the value to recall
    for _ in range(n):                                             # loop over the examples
        length = delay + RNG.integers(4, 12)                        # sequence length
        seq = RNG.normal(0, 0.15, (length, 2))                      # mostly noise channels
        pos = RNG.integers(0, max(length - delay, 1))               # where the marker sits
        marker = RNG.choice([0.0, 1.0])                            # the value to remember
        seq[pos, 0] = 1.0                                          # a marker channel
        seq[pos, 1] = marker                                       # the value rides on channel two
        seqs.append(seq)                                          # store the sequence
        targets.append(int(marker))                                # the answer
    return seqs, np.asarray(targets)                               # return the batch

# ---------------------------------------------------------------- tiny torch helpers
def torch_seed(seed=SEED):                                         # reproducible torch runs
    """Seed torch and numpy and return the torch module for convenience."""   # short docstring
    import torch                                                   # import lazily so numpy-only scripts stay light
    torch.manual_seed(seed)                                        # CPU and GPU seed
    np.random.seed(seed)                                           # numpy seed as well
    return torch                                                   # return the module


def param_count(model):                                            # count trainable parameters
    """Return the number of trainable parameters in a torch module."""   # short docstring
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))   # sum over tensors
