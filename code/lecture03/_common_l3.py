"""_common_l3.py -- shared helpers for the Lecture 3 (Data Preparation) examples.

Every line of code in this lecture carries an English comment, so the scripts can
be read as documentation. Fixed random seeds keep samples and figures identical
on every run, which is what makes the classroom examples reproducible.
"""
import matplotlib                                                  # select the plotting backend first
matplotlib.use("Agg")                                              # headless backend: works without a display
import matplotlib.pyplot as plt                                    # figure and axes API
import numpy as np                                                 # numerical arrays and random numbers
import pandas as pd                                                # tabular data structures
from pathlib import Path                                           # filesystem path handling

SEED = 20260317                                                    # one fixed seed for every script
RNG = np.random.default_rng(SEED)                                  # shared generator: reproducible draws
ROOT = Path(__file__).resolve().parents[2]                         # .../website (code/lecture03 -> website)
FIG = ROOT / "figures" / "lecture03"                               # figure output directory for Lecture 3
DAT = ROOT / "data" / "lecture03"                                  # sample-data output directory for Lecture 3
FIG.mkdir(parents=True, exist_ok=True)                             # create the figure directory when missing
DAT.mkdir(parents=True, exist_ok=True)                             # create the data directory when missing

plt.rcParams["figure.dpi"] = 140                                   # interactive/rendering resolution
plt.rcParams["savefig.dpi"] = 200                                  # saved resolution: crisp on a projector
plt.rcParams["savefig.bbox"] = "tight"                             # trim unused white margins on save
plt.rcParams["axes.grid"] = True                                   # light grid makes values easier to read
plt.rcParams["grid.alpha"] = 0.25                                  # keep the grid visually quiet
plt.rcParams["axes.titlesize"] = 13                                # title size tuned for slide readability
plt.rcParams["axes.labelsize"] = 11                                # axis label size
plt.rcParams["xtick.labelsize"] = 10                               # x tick label size
plt.rcParams["ytick.labelsize"] = 10                               # y tick label size
plt.rcParams["legend.fontsize"] = 9.2                              # compact legend text
plt.rcParams["font.size"] = 10                                     # default text size everywhere else


def save(fig, name):                                               # save one figure and release its memory
    """Write a figure into website/figures/lecture03 and close it."""   # helper contract
    path = FIG / name                                              # build the full output path
    fig.savefig(path, facecolor="white")                           # white background suits slide insertion
    plt.close(fig)                                                 # close the figure to free memory
    print("[figure]", path)                                        # log the artefact for the build output


def dump(df, name):                                                # write a sample table into the data folder
    """Write a DataFrame into website/data/lecture03 as UTF-8 CSV."""   # helper contract
    path = DAT / name                                              # build the full output path
    df.to_csv(path, index=False, encoding="utf-8-sig")             # utf-8-sig keeps Excel happy with CJK
    print("[data]", path)                                          # log the artefact for the build output


def money_to_float(series):                                        # normalise messy currency strings
    """Strip currency symbols, thousands separators and spaces, then cast to float."""   # helper contract
    cleaned = (series.astype(str)                                  # work on strings so regex can be applied
               .str.replace(r"[^\d.\-]", "", regex=True)           # keep digits, dot and minus only
               .replace({"": np.nan}))                             # empty strings become real missing values
    return pd.to_numeric(cleaned, errors="coerce")                 # unparsable leftovers also become NaN
