"""_common.py -- shared helpers: Chinese fonts, output folders, save functions (every line commented)."""  # module docstring, states the purpose
import matplotlib                                                    # import the matplotlib package
matplotlib.use("Agg")                                                # headless backend, works without a display
import matplotlib.pyplot as plt                                      # import the plotting interface as plt
import numpy as np                                                   # import the numeric library as np
from pathlib import Path                                             # import the cross-platform path helper

RNG = np.random.default_rng(20260917)                                # fixed seed, identical results on every run
FIG = Path(__file__).resolve().parents[1] / "figures"                # figure output folder: website/figures
DAT = Path(__file__).resolve().parents[1] / "data"                   # data output folder: website/data
FIG.mkdir(parents=True, exist_ok=True)                               # create the figure folder if missing
DAT.mkdir(parents=True, exist_ok=True)                               # create the data folder if missing

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]  # Chinese font priority list (first two ship with Windows)
plt.rcParams["font.monospace"] = ["Consolas", "Microsoft YaHei", "DejaVu Sans Mono"]  # add Chinese fallbacks to monospace fonts too
plt.rcParams["axes.unicode_minus"] = False                           # render axis minus signs instead of boxes
plt.rcParams["figure.dpi"] = 140                                     # default screen and save resolution
plt.rcParams["savefig.dpi"] = 200                                    # resolution used when saving figures
plt.rcParams["savefig.bbox"] = "tight"                               # trim surrounding whitespace on save
plt.rcParams["axes.grid"] = True                                     # grid on by default for easier reading
plt.rcParams["grid.alpha"] = 0.25                                    # grid transparency, keeps data prominent
plt.rcParams["axes.titlesize"] = 13                                  # subplot title size for projection
plt.rcParams["axes.labelsize"] = 11                                  # axis label font size
plt.rcParams["xtick.labelsize"] = 10                                 # x tick label size
plt.rcParams["ytick.labelsize"] = 10                                 # y tick label size
plt.rcParams["legend.fontsize"] = 9.2                                # legend font size


def save(fig, name):                                                 # define the save helper: figure and file name
    """Save a figure into the FIG folder and close it to free memory."""                    # function docstring
    path = FIG / name                                                # build the full output path
    fig.savefig(path, facecolor="white")                             # save as PNG with a white background
    plt.close(fig)                                                   # close the figure and free memory
    print("[figure]", path)                                          # print the output path for checking


def dump_csv(df, name):                                              # define the CSV dump helper
    """Write a DataFrame into the DAT folder (UTF-8-SIG so Excel shows Chinese)."""  # function docstring
    path = DAT / name                                                # build the full output path
    df.to_csv(path, index=False, encoding="utf-8-sig")               # write CSV without an index, Excel friendly
    print("[data]", path)                                            # print the output path
