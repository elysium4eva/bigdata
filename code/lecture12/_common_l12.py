"""_common_l12.py -- shared helpers and synthetic causal data for Lecture 12.

Every line carries an English comment. All samples are generated here so that every figure and   # continuation of the call
every number in the deck can be regenerated from one seed.              # continuation of the call
"""
import json                                                        # read and write JSON artefacts
import matplotlib                                                  # choose a backend before pyplot
matplotlib.use("Agg")                                              # headless backend: no display needed
import matplotlib.pyplot as plt                                    # figure and axes API
import numpy as np                                                 # numeric arrays
import pandas as pd                                                # tabular structures
from pathlib import Path                                           # portable paths

SEED = 20270301                                                    # single seed for the whole lecture
RNG = np.random.default_rng(SEED)                                  # shared generator, reproducible draws
ROOT = Path(__file__).resolve().parents[2]                         # .../website (two levels above code/lecture12)
FIG = ROOT / "figures" / "lecture12"                               # output folder for figures
DAT = ROOT / "data" / "lecture12"                                  # output folder for data
OUT = DAT / "artifacts"                                            # tables, logs and the glossary
for _d in (FIG, DAT, OUT):                                         # make sure every folder exists
    _d.mkdir(parents=True, exist_ok=True)                          # create it when missing

# ---------------------------------------------------------------- shared visual style
FONT = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]                # CJK-capable font stack
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",               # colour-blind safe palette
         "#0072B2", "#D55E00", "#CC79A7", "#000000"]               # eight colours

plt.rcParams["font.sans-serif"] = FONT                             # use the CJK stack globally
plt.rcParams["axes.unicode_minus"] = False                         # draw real minus signs
plt.rcParams["figure.dpi"] = 140                                   # drafting resolution
plt.rcParams["savefig.dpi"] = 200                                  # slide resolution
plt.rcParams["savefig.bbox"] = "tight"                             # trim white space
plt.rcParams["axes.grid"] = True                                   # light grid for readability
plt.rcParams["grid.alpha"] = 0.25                                  # keep the grid subtle
plt.rcParams["axes.titlesize"] = 12.0                              # subplot title size
plt.rcParams["axes.labelsize"] = 10.5                              # axis label size
plt.rcParams["xtick.labelsize"] = 9.0                              # x tick size
plt.rcParams["ytick.labelsize"] = 9.0                              # y tick size
plt.rcParams["legend.fontsize"] = 8.6                              # legend size
plt.rcParams["figure.titlesize"] = 13.5                            # figure title size
plt.rcParams["axes.spines.top"] = False                            # drop the top spine
plt.rcParams["axes.spines.right"] = False                          # drop the right spine


def style_axes(ax, title=None, xlabel=None, ylabel=None):           # apply the shared style
    """Set title and axis labels with the shared style."""           # short docstring
    if title:                                                      # only when provided
        ax.set_title(title)                                        # write the title
    if xlabel:                                                     # only when provided
        ax.set_xlabel(xlabel)                                      # write the x label
    if ylabel:                                                     # only when provided
        ax.set_ylabel(ylabel)                                      # write the y label
    return ax                                                      # allow chaining


def annotated(ax, text, loc="upper left"):                          # in-figure note
    """Place a short note inside the axes with a pale box."""        # short docstring
    xy = {"upper left": (0.02, 0.96), "upper right": (0.98, 0.96),  # top positions
          "lower left": (0.02, 0.06), "lower right": (0.98, 0.06)}.get(loc, (0.02, 0.96))   # default top left
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, va="top",   # place in axes coordinates
            ha="left" if "left" in loc else "right", fontsize=8.6,  # alignment
            bbox=dict(boxstyle="round,pad=0.25", fc="#F7F3F2", ec="#D6CDD0", lw=0.6))   # pale box
    return ax                                                      # allow chaining


def save(fig, name):                                               # persist a figure
    """Save a figure into website/figures/lecture12 and close it."""   # short docstring
    path = FIG / name                                              # full path
    fig.savefig(path, facecolor="white")                           # white background
    plt.close(fig)                                                 # release memory
    print("[figure]", path)                                        # log it


def dump(df, name):                                                # persist a dataframe
    """Write a dataframe to website/data/lecture12 as UTF-8 CSV."""   # short docstring
    path = DAT / name                                              # full path
    df.to_csv(path, index=False, encoding="utf-8-sig")             # utf-8-sig opens cleanly in Excel
    print("[data]", path)                                          # log it


def dump_artifact(text, name):                                     # persist a text artefact
    """Write a text artefact into data/lecture12/artifacts."""       # short docstring
    path = OUT / name                                              # full path
    path.write_text(text, encoding="utf-8")                        # write UTF-8
    print("[artifact]", path)                                      # log it


def write_json(obj, name):                                         # persist a dictionary
    """Serialise a dictionary to artifacts/<name>."""               # short docstring
    dump_artifact(json.dumps(obj, ensure_ascii=False, indent=2), name)   # pretty, keep Chinese


# ---------------------------------------------------------------- 1) potential outcomes with confounding
def make_potential(n=2000, ate=2.0, confound=1.5):                  # a confounded sample
    """Return data with two potential outcomes, a confounded assignment and the observed Y."""   # short docstring
    x = RNG.normal(0, 1, n)                                        # an observed confounder
    u = RNG.normal(0, 1, n)                                        # an unobserved confounder
    # assignment depends on x and u: higher x and u raise the chance of treatment
    p_z = 1 / (1 + np.exp(-(0.0 + 0.9 * x + 0.7 * u)))               # the propensity score
    z = (RNG.random(n) < p_z).astype(int)                           # the realised assignment
    y0 = 1.0 + 1.2 * x + 0.8 * u + RNG.normal(0, 0.8, n)             # the control potential outcome
    y1 = y0 + ate + confound * x                                     # the treated potential outcome (heterogeneous)
    y = np.where(z == 1, y1, y0)                                     # the observed outcome
    return pd.DataFrame({"x": x.round(4), "u": u.round(4), "p_z": p_z.round(4),   # the frame
                         "z": z, "y0": y0.round(4), "y1": y1.round(4), "y": y.round(4)})   # outcomes


# ---------------------------------------------------------------- 2) Simpson's paradox data
def make_simpson(n=600):                                           # a dataset that reverses on aggregation
    """Return a two-group sample where the pooled slope has the opposite sign."""   # short docstring
    rows = []                                                      # one row per unit
    for g, (mu_x, shift) in enumerate([(2.0, 0.0), (8.0, 9.0)]):     # two groups with different centres
        x = RNG.normal(mu_x, 1.4, n // 2)                           # the predictor
        y = shift - 0.6 * x + RNG.normal(0, 1.0, n // 2)             # negative slope INSIDE each group
        for xi, yi in zip(x, y):                                    # loop over the units
            rows.append({"group": ["A", "B"][g], "x": round(float(xi), 3), "y": round(float(yi), 3)})   # the row
    return pd.DataFrame(rows)                                       # return the frame


# ---------------------------------------------------------------- 3) an A/B experiment
def make_experiment(n_per_arm=5000, p0=0.10, lift=0.012):           # two-arm experiment
    """Return per-user conversion outcomes for a control and a treatment arm."""   # short docstring
    rows = []                                                      # one row per user
    for arm, p in (("control", p0), ("treatment", p0 + lift)):       # loop over the arms
        conv = (RNG.random(n_per_arm) < p).astype(int)               # Bernoulli outcomes
        for k, c in enumerate(conv):                               # one row per user
            rows.append({"arm": arm, "user": "%s_%05d" % (arm[0], k + 1),   # the record
                         "converted": int(c)})                         # the outcome
    return pd.DataFrame(rows)                                       # return the frame


# ---------------------------------------------------------------- 4) a DID panel
def make_panel_did(n_units=400, n_periods=8, shock_at=4, effect=-3.0):   # a two-group panel
    """Return a panel where a subset of units is treated from period shock_at on."""   # short docstring
    rows = []                                                      # one row per unit-period
    for i in range(n_units):                                       # loop over units
        treated = int(i % 2 == 0)                                   # half the units are treated
        unit_fe = RNG.normal(0, 1.0)                                # a unit fixed effect
        for t in range(1, n_periods + 1):                            # loop over periods
            trend = 0.3 * t                                         # a common time trend
            post = int(t >= shock_at)                               # the post indicator
            y = 5.0 + unit_fe + trend                                  # the base outcome
            y += effect * treated * post                             # add the treatment effect
            y += RNG.normal(0, 0.6)                                 # idiosyncratic noise
            rows.append({"unit": i + 1, "period": t, "treated": treated,   # the record
                         "post": post, "y": round(float(y), 4)})     # the outcome
    return pd.DataFrame(rows)                                       # return the frame


# ---------------------------------------------------------------- 5) an IV sample
def make_iv(n=3000):                                               # instrument and endogenous regressor
    """Return an endogenous treatment, a valid instrument and an outcome."""   # short docstring
    z = RNG.binomial(1, 0.5, n)                                     # the binary instrument
    u = RNG.normal(0, 1, n)                                        # an unobserved confounder
    x = 0.8 * z + 0.9 * u + RNG.normal(0, 1, n)                     # the endogenous regressor
    y = 1.5 * x + 2.0 * u + RNG.normal(0, 1, n)                     # the outcome
    return pd.DataFrame({"z": z, "x": x.round(4), "u": u.round(4), "y": y.round(4)})   # the frame


# ---------------------------------------------------------------- 6) a regression-discontinuity sample
def make_rd(n=4000, cutoff=0.0, jump=4.0):                          # a sharp RD design
    """Return a running variable, a sharp treatment and an outcome with a jump at the cutoff."""   # short docstring
    running = RNG.uniform(-1, 1, n)                                 # the running variable
    treated = (running >= cutoff).astype(int)                       # sharp assignment at zero
    y = 10.0 + 2.0 * running + 0.5 * running ** 2 + jump * treated   # a smooth trend plus a jump
    y += RNG.normal(0, 0.9, n)                                     # noise
    return pd.DataFrame({"running": running.round(4), "treated": treated,   # the frame
                         "y": y.round(4)})                          # the outcome


# ---------------------------------------------------------------- 7) non-compliance
def make_compliance(n=4000):                                        # one-sided non-compliance
    """Return assignment, take-up, the compliance type and an outcome with a LATE."""   # short docstring
    z = RNG.binomial(1, 0.5, n)                                     # random assignment
    # compliers take the treatment only when assigned; never-takers never do; always-takers always do
    p_never = 0.25                                                  # share of never-takers
    p_always = 0.10                                                 # share of always-takers
    u = RNG.random(n)                                              # a uniform draw for the type
    never = u < p_never                                            # never-takers
    always = (u >= p_never) & (u < p_never + p_always)              # always-takers
    complier = u >= p_never + p_always                              # compliers
    d = np.where(complier, z, np.where(always, 1, 0))               # observed take-up
    base = RNG.normal(5, 1, n)                                     # baseline outcome
    y = base + 2.0 * d + 3.0 * complier * z * 0.0 + RNG.normal(0, 0.8, n)   # outcome with a 2.0 effect
    kind = np.where(complier, "顺从者", np.where(always, "始终接受", "从不接受"))   # the type label
    return pd.DataFrame({"z": z, "d": d, "type": kind, "y": y.round(4)})   # the frame


# ---------------------------------------------------------------- 8) heterogeneous effects
def make_cate(n=4000):                                              # a sample with treatment-effect heterogeneity
    """Return covariates, treatment and an outcome whose effect depends on x1."""   # short docstring
    x1 = RNG.normal(0, 1, n)                                        # the effect modifier
    x2 = RNG.normal(0, 1, n)                                        # a pure confounder
    p_z = 1 / (1 + np.exp(-(0.6 * x2)))                              # assignment depends on x2
    z = (RNG.random(n) < p_z).astype(int)                            # the treatment
    tau = 1.0 + 2.0 * x1                                            # the individual treatment effect
    y = 3.0 + 1.5 * x2 + tau * z + RNG.normal(0, 1.0, n)             # the outcome
    return pd.DataFrame({"x1": x1.round(4), "x2": x2.round(4), "z": z,   # the frame
                         "y": y.round(4), "tau_true": tau.round(4)})   # the truth
