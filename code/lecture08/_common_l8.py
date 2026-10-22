"""_common_l8.py -- shared helpers for the Lecture 8 unsupervised-learning examples.

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

SEED = 20260909                                                    # single seed shared by the whole lecture
RNG = np.random.default_rng(SEED)                                  # shared generator keeps samples reproducible
ROOT = Path(__file__).resolve().parents[2]                         # .../website (two levels above code/lecture08)
FIG = ROOT / "figures" / "lecture08"                               # output folder for figures
DAT = ROOT / "data" / "lecture08"                                  # output folder for sample data
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
    """Save a figure under website/figures/lecture08 and close it."""   # short docstring
    path = FIG / name                                              # build the full output path
    fig.savefig(path, facecolor="white")                           # white background keeps slides readable
    plt.close(fig)                                                 # release the figure from memory
    print("[figure]", path)                                        # log the artefact path


def dump(df, name):                                                # persist a dataframe into the data folder
    """Write a dataframe to website/data/lecture08 as UTF-8 CSV."""   # short docstring
    path = DAT / name                                              # build the full output path
    df.to_csv(path, index=False, encoding="utf-8-sig")             # utf-8-sig opens cleanly in Excel
    print("[data]", path)                                          # log the artefact path


def dump_artifact(text, name):                                     # persist a text artefact (markdown, JSON)
    """Write a text artefact into data/lecture08/artifacts."""      # short docstring
    path = OUT / name                                              # build the path inside the artifacts folder
    path.write_text(text, encoding="utf-8")                        # write with UTF-8 so Chinese labels survive
    print("[artifact]", path)                                      # log the artefact path


def write_json(obj, name):                                         # persist a dictionary as JSON
    """Serialise a dictionary to artifacts/<name> with UTF-8 encoding."""   # short docstring
    dump_artifact(json.dumps(obj, ensure_ascii=False, indent=2), name)   # pretty print and keep Chinese readable


def make_smartphone(n=60):                                         # the smartphone feature table
    """Return a synthetic phone-spec sample modelled on the courseware table."""   # short docstring
    brands = ["Xiaomi", "iPhone", "Huawei", "Samsung", "OPPO", "vivo", "Honor", "OnePlus"]   # brand pool
    tiers = RNG.integers(0, 3, n)                                  # 0 entry, 1 mid, 2 flagship tiers
    cpu = np.choose(tiers, [55.0, 78.0, 96.0]) + RNG.normal(0, 4.5, n)   # CPU performance score
    gpu = np.choose(tiers, [50.0, 72.0, 94.0]) + RNG.normal(0, 5.0, n)   # GPU performance score
    cores = np.choose(tiers, [4, 8, 8]) + RNG.integers(0, 2, n)     # number of cores
    ram = np.choose(tiers, [4, 8, 12]) + RNG.integers(0, 5, n)      # RAM in GB
    pixels = np.choose(tiers, [24.0, 48.0, 64.0]) + RNG.normal(0, 3.0, n)   # main camera in megapixels
    ncams = np.choose(tiers, [2, 3, 4]) + RNG.integers(0, 2, n)     # number of cameras
    price = (1200 + 18.0 * cpu + 9.0 * ram + 22.0 * pixels        # a plausible price formula
             + RNG.normal(0, 220, n))                              # plus noise
    share = np.clip(RNG.normal(4.0, 3.0, n), 0.2, None)            # market share in per cent
    return pd.DataFrame({"brand": RNG.choice(brands, n),           # brand label
                         "cpu": cpu.round(1), "gpu": gpu.round(1),   # performance scores
                         "cores": cores, "ram_gb": ram,             # hardware counts
                         "camera_mp": pixels.round(1),              # main camera resolution
                         "n_cameras": ncams, "price": price.round(0),   # camera count and price
                         "market_share": share.round(2)})           # market share


def make_brands(n_brands=12, n_respondents=260):                   # the brand-perception sample
    """Return a synthetic perceptual-map sample: price versus performance ratings."""   # short docstring
    names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"][:n_brands]   # short brand labels
    base_price = np.linspace(-2.2, 2.2, n_brands) + RNG.normal(0, 0.15, n_brands)   # perceived price level
    base_perf = 0.75 * base_price + RNG.normal(0, 0.45, n_brands)   # perceived performance level
    rows = []                                                       # long-format rows
    for i, name in enumerate(names):                               # one brand at a time
        price = base_price[i] + RNG.normal(0, 0.35, n_respondents)   # respondent level ratings
        perf = base_perf[i] + RNG.normal(0, 0.45, n_respondents)     # respondent level ratings
        share = float(np.clip(RNG.normal(6.5, 2.5), 0.5, None))      # brand market share
        for j in range(n_respondents):                             # one row per rating
            rows.append({"brand": name, "price": round(float(price[j]), 3),   # perceived price
                         "performance": round(float(perf[j]), 3), "share": round(share, 2)})   # performance and share
    return pd.DataFrame(rows)                                      # return the long format


def make_blobs(n=420, k=4):                                        # a generic blob sample
    """Return a two-dimensional blob sample with k well-separated clusters."""   # short docstring
    centres = RNG.uniform(-6, 6, (k, 2))                           # random centres
    labels = RNG.integers(0, k, n)                                 # balanced assignment
    X = centres[labels] + RNG.normal(0, 1.0, (n, 2))               # points around the centres
    return pd.DataFrame({"x1": X[:, 0].round(3), "x2": X[:, 1].round(3), "label": labels})   # tabulate


def make_density(kind="rings", n=600):                             # density-based sample with noise
    """Return a sample with two rings, one blob and uniform noise (for DBSCAN demos)."""   # short docstring
    n_ring = n // 3                                                # points per ring
    theta = RNG.uniform(0, 2 * np.pi, n_ring)                      # angle around the first ring
    r1 = 3.0 + RNG.normal(0, 0.18, n_ring)                         # first ring radius
    theta2 = RNG.uniform(0, 2 * np.pi, n_ring)                     # angle around the second ring
    r2 = 6.4 + RNG.normal(0, 0.2, n_ring)                          # second ring radius
    n_blob = n - 2 * n_ring - n // 10                              # points in the central blob
    blob = RNG.normal(0, 0.75, (n_blob, 2))                        # a dense Gaussian blob
    noise = RNG.uniform(-8, 8, (n // 10, 2))                       # uniform background noise
    X = np.vstack([np.c_[r1 * np.cos(theta), r1 * np.sin(theta)],   # first ring
                   np.c_[r2 * np.cos(theta2), r2 * np.sin(theta2)],   # second ring
                   blob, noise])                                   # blob and noise
    return pd.DataFrame({"x1": X[:, 0].round(3), "x2": X[:, 1].round(3)})   # tabulate without labels


def make_chain(n=240):                                             # a chain sample for the single-link effect
    """Return a sample with two bridging points (the classic single-link failure)."""   # short docstring
    left = RNG.normal([-4.0, 0.0], 0.7, (n // 2, 2))               # first dense cluster
    right = RNG.normal([4.0, 0.0], 0.7, (n // 2, 2))               # second dense cluster
    bridge = np.array([[-1.6, 0.1], [-1.2, 0.15], [1.2, 0.12], [1.6, 0.1]])   # a thin bridge
    X = np.vstack([left, right, bridge])                           # stack everything
    return pd.DataFrame({"x1": X[:, 0].round(3), "x2": X[:, 1].round(3)})   # tabulate


def make_gmm(n=900):                                               # the Gaussian-mixture sample
    """Return three elongated Gaussian components (for GMM versus k-means)."""   # short docstring
    labels = RNG.integers(0, 3, n)                                 # component assignment
    means = np.array([[-4.0, 0.0], [4.0, 1.2], [0.0, -5.0]])       # component means
    covs = [np.array([[1.2, 0.75], [0.75, 0.7]]),                  # first covariance (tilted)
            np.array([[1.0, -0.5], [-0.5, 0.6]]),                  # second covariance (tilted)
            np.array([[1.4, 0.0], [0.0, 0.5]])]                    # third covariance
    X = np.vstack([RNG.multivariate_normal(means[k], covs[k], int((labels == k).sum())) for k in range(3)])   # draw
    y = np.concatenate([np.full(int((labels == k).sum()), k) for k in range(3)])   # matching labels
    return pd.DataFrame({"x1": X[:, 0].round(3), "x2": X[:, 1].round(3), "component": y})   # tabulate


def make_highdim(n=900, d=40, k=6):                                # high-dimensional sample for embeddings
    """Return a high-dimensional sample with k latent groups (for PCA/t-SNE/UMAP)."""   # short docstring
    labels = RNG.integers(0, k, n)                                 # latent group per point
    directions = RNG.normal(0, 1, (k, d))                          # one signal direction per group
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)   # normalise the directions
    X = 3.2 * directions[labels] + RNG.normal(0, 1.0, (n, d))      # group signal plus noise
    return pd.DataFrame(np.round(X, 3)), labels                    # return the matrix and the labels


def make_transactions(n=500, n_items=12):                          # transaction sample for association rules
    """Return a synthetic market-basket sample as a list of item sets."""   # short docstring
    items = ["milk", "bread", "eggs", "beer", "chips", "wine", "coffee",   # twelve products
             "tea", "butter", "cheese", "juice", "soda"]           # the full item pool
    baskets = []                                                   # one set per transaction
    for _ in range(n):                                             # loop over the transactions
        basket = set()                                             # start empty
        if RNG.random() < 0.45:                                    # a "breakfast" pattern
            basket.update(["milk", "bread", "eggs"])               # the core three items
            if RNG.random() < 0.5:                                 # sometimes add coffee or juice
                basket.add("coffee" if RNG.random() < 0.6 else "juice")   # one of the two
        if RNG.random() < 0.35:                                    # a "party" pattern
            basket.update(["beer", "chips", "soda"])               # the party core
            if RNG.random() < 0.4:                                 # sometimes add wine
                basket.add("wine")                                 # wine conditional on the party core
        for _ in range(RNG.integers(0, 3)):                        # random extra items
            basket.add(str(RNG.choice(items)))                     # one random item
        baskets.append(sorted(basket))                             # store the basket
    return baskets                                                 # return the list
