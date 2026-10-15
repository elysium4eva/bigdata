"""_common_l4.py -- shared helpers for the Lecture 4 (descriptive stats and visualization) examples.

Unify fonts, palettes and figure settings so the 13 figures keep one consistent, readable style;
fix the random seed so samples and figures are identical on every run.
"""
import json                                                        # read and write JSON (geo data and metadata)
import matplotlib                                                  # pick the backend before other plotting imports
matplotlib.use("Agg")                                              # headless backend: figures render without a display
import matplotlib.pyplot as plt                                    # figures and axes interface
import numpy as np                                                 # numeric arrays and random numbers
import pandas as pd                                                # tabular data structures
from pathlib import Path                                           # cross-platform path handling

SEED = 20260417                                                    # fixed random seed shared by this lecture
RNG = np.random.default_rng(SEED)                                  # shared generator: results stay reproducible
ROOT = Path(__file__).resolve().parents[2]                         # .../website (two levels up from code/lecture04)
FIG = ROOT / "figures" / "lecture04"                               # figure output directory
DAT = ROOT / "data" / "lecture04"                                  # sample data output directory
INT = FIG / "interactive"                                          # plotly interactive HTML output directory
for _d in (FIG, DAT, INT):                                         # make sure each directory exists
    _d.mkdir(parents=True, exist_ok=True)                          # create it when missing

# ---------------------------------------------------------------- unified visual style
FONT = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]                # Chinese font priority (Windows ships the first two)
PALETTE = ["#8A0C3C", "#C9A227", "#66717D", "#2E6F8E", "#711426",  # categorical palette: SZU lychee red + gold + grey + blue
           "#A6426B", "#4F7942", "#B5651D"]                        # four more distinguishable colors
SEQ = "RdPu"                                                       # sequential palette (single hue, good for choropleth)
DIV = "RdBu_r"                                                     # diverging palette (differences around a center)
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",               # Okabe-Ito colorblind-friendly palette (8 colors)
         "#0072B2", "#D55E00", "#CC79A7", "#000000"]               # colors stay distinguishable in grayscale

plt.rcParams["font.sans-serif"] = FONT                             # global Chinese font
plt.rcParams["axes.unicode_minus"] = False                         # show minus signs instead of boxes
plt.rcParams["figure.dpi"] = 140                                   # interactive preview resolution
plt.rcParams["savefig.dpi"] = 200                                  # saved resolution: sharper on a projector
plt.rcParams["savefig.bbox"] = "tight"                             # trim extra white margins when saving
plt.rcParams["axes.grid"] = True                                   # light grid helps reading values
plt.rcParams["grid.alpha"] = 0.25                                  # grid transparency
plt.rcParams["axes.titlesize"] = 12.5                              # subplot title size
plt.rcParams["axes.labelsize"] = 11                                # axis label size
plt.rcParams["xtick.labelsize"] = 9.5                              # x tick label size
plt.rcParams["ytick.labelsize"] = 9.5                              # y tick label size
plt.rcParams["legend.fontsize"] = 9                                # legend font size
plt.rcParams["figure.titlesize"] = 14                              # figure title size
plt.rcParams["axes.spines.top"] = False                            # drop the top spine: less chart junk
plt.rcParams["axes.spines.right"] = False                          # drop the right spine


def style_axes(ax, title=None, xlabel=None, ylabel=None):          # set title and axis labels of one subplot
    """Give one subplot the shared style: title and axis labels."""   # function contract
    if title:                                                      # set only when a title is given
        ax.set_title(title)                                        # set the subplot title
    if xlabel:                                                     # set only when an x label is given
        ax.set_xlabel(xlabel)                                      # set the x axis label
    if ylabel:                                                     # set only when a y label is given
        ax.set_ylabel(ylabel)                                      # set the y axis label
    return ax                                                      # return the same axes for chaining


def annotated(ax, text, loc="upper left"):                          # add an in-figure note (key to self-contained captions)
    """Add one explanatory line in a corner of the figure."""          # function contract
    xy = {"upper left": (0.02, 0.96), "upper right": (0.98, 0.96),  # the two upper positions
          "lower left": (0.02, 0.06), "lower right": (0.98, 0.06)}.get(loc, (0.02, 0.96))   # the two lower positions; unknown falls back to upper left
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, va="top",   # place the text in axes coordinates
            ha="left" if "left" in loc else "right", fontsize=8.6,  # left or right alignment
            color="#1E2630", bbox=dict(boxstyle="round,pad=0.25",   # a light box improves readability
                                       fc="#F7F3F2", ec="#D6CDD0", lw=0.6))   # fill and border colors
    return ax                                                      # return the same axes


def save(fig, name):                                               # save the figure and free memory
    """Save the figure under website/figures/lecture04 and close it."""   # function contract
    path = FIG / name                                              # build the full path
    fig.savefig(path, facecolor="white")                           # white background for slides
    plt.close(fig)                                                 # close the figure to free memory
    print("[figure]", path)                                        # print the output path


def dump(df, name):                                                # write a sample table into the data directory
    """Write the DataFrame as UTF-8 CSV under website/data/lecture04."""   # function contract
    path = DAT / name                                              # build the full path
    df.to_csv(path, index=False, encoding="utf-8-sig")             # utf-8-sig opens cleanly in Excel
    print("[data]", path)                                          # print the output path


def load_geojson(name):                                            # read the (optional) geographic boundary file
    """Read GeoJSON from the data directory; return None when missing."""   # function contract
    path = DAT / name                                              # build the full path
    if not path.exists():                                          # file is missing
        print("[geojson] not found, fall back to a symbol map:", path)   # note the fallback to a bubble map
        return None                                                # return nothing
    with open(path, "r", encoding="utf-8") as fh:                  # read as UTF-8
        return json.load(fh)                                       # return the parsed dictionary


def lorenz(values):                                                # compute the Lorenz curve (measures inequality)
    """Return cumulative population and value shares for the Lorenz curve."""   # function contract
    x = np.sort(np.asarray(values, dtype=float))                   # sort ascending
    cum = np.cumsum(x) / x.sum()                                   # cumulative share
    return np.insert(np.arange(1, len(x) + 1) / len(x), 0, 0), np.insert(cum, 0, 0)   # pad both ends with 0
