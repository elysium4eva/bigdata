"""_common_l3.py —— shared helpers for the Lecture 3 (data preparation) examples (every line carries a comment).

A fixed random seed keeps the samples and figures identical on every run, which is what makes the class examples reproducible.
"""
import matplotlib                                                  # pick the plotting backend, then import the rest
matplotlib.use("Agg")                                               # headless backend: figures render without a display
import matplotlib.pyplot as plt                                    # figure and axes interface
import numpy as np                                                 # numeric arrays and random numbers
import pandas as pd                                                # tabular data structures
from pathlib import Path                                           # cross-platform path handling

SEED = 20260317                                                    # fixed random seed shared by all scripts
RNG = np.random.default_rng(SEED)                                  # shared RNG: results are reproducible
ROOT = Path(__file__).resolve().parents[2]                         # .../website (two levels up from code/lecture03)
FIG = ROOT / "figures" / "lecture03"                               # figure output directory for Lecture 3
DAT = ROOT / "data" / "lecture03"                                  # sample data output directory for Lecture 3
FIG.mkdir(parents=True, exist_ok=True)                             # create the directory if missing
DAT.mkdir(parents=True, exist_ok=True)                             # create the directory if missing

plt.rcParams["figure.dpi"] = 140                                   # interactive preview resolution
plt.rcParams["savefig.dpi"] = 200                                  # save resolution: sharper when projected
plt.rcParams["savefig.bbox"] = "tight"                             # trim extra white margins when saving
plt.rcParams["axes.grid"] = True                                   # light grid makes values readable
plt.rcParams["grid.alpha"] = 0.25                                  # grid transparency, kept subtle
plt.rcParams["axes.titlesize"] = 13                                # subplot title size (projector readable)
plt.rcParams["axes.labelsize"] = 11                                # axis label font size
plt.rcParams["xtick.labelsize"] = 10                               # x tick label size
plt.rcParams["ytick.labelsize"] = 10                               # y tick label size
plt.rcParams["legend.fontsize"] = 9.2                              # legend font size
plt.rcParams["font.size"] = 10                                     # default size for other text


def save(fig, name):                                               # save one figure and free memory
    """Save the figure into website/figures/lecture03 and close it."""   # function contract
    path = FIG / name                                              # build the full output path
    fig.savefig(path, facecolor="white")                           # white background pastes into slides
    plt.close(fig)                                                 # close the figure to free memory
    print("[figure]", path)                                        # print the output path for checking


def dump(df, name):                                                # write the sample table to the data directory
    """Write the DataFrame as UTF-8 CSV into website/data/lecture03."""   # function contract
    path = DAT / name                                              # build the full output path
    df.to_csv(path, index=False, encoding="utf-8-sig")             # utf-8-sig so Excel shows Chinese
    print("[data]", path)                                          # print the output path for checking


def money_to_float(series):                                        # convert dirty amount strings to numbers
    """Drop currency symbols and separators, then cast to float."""   # function contract
    cleaned = (series.astype(str)                                  # cast to string first so regex works
               .str.replace(r"[^\d.\-]", "", regex=True)           # keep digits, dots and minus signs
               .replace({"": np.nan}))                             # empty strings become missing values
    return pd.to_numeric(cleaned, errors="coerce")                 # unparsable values become NaN
