"""_common_l13.py -- shared helpers and synthetic financial data for Lecture 13.

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

SEED = 20270220                                                    # single seed for the whole lecture
RNG = np.random.default_rng(SEED)                                  # shared generator, reproducible draws
ROOT = Path(__file__).resolve().parents[2]                         # .../website (two levels above code/lecture13)
FIG = ROOT / "figures" / "lecture13"                               # output folder for figures
DAT = ROOT / "data" / "lecture13"                                  # output folder for data
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
    """Save a figure into website/figures/lecture13 and close it."""   # short docstring
    path = FIG / name                                              # full path
    fig.savefig(path, facecolor="white")                           # white background
    plt.close(fig)                                                 # release memory
    print("[figure]", path)                                        # log it


def dump(df, name):                                                # persist a dataframe
    """Write a dataframe to website/data/lecture13 as UTF-8 CSV."""   # short docstring
    path = DAT / name                                              # full path
    df.to_csv(path, index=False, encoding="utf-8-sig")             # utf-8-sig opens cleanly in Excel
    print("[data]", path)                                          # log it


def dump_artifact(text, name):                                     # persist a text artefact
    """Write a text artefact into data/lecture13/artifacts."""       # short docstring
    path = OUT / name                                              # full path
    path.write_text(text, encoding="utf-8")                        # write UTF-8
    print("[artifact]", path)                                      # log it


def write_json(obj, name):                                         # persist a dictionary
    """Serialise a dictionary to artifacts/<name>."""               # short docstring
    dump_artifact(json.dumps(obj, ensure_ascii=False, indent=2), name)   # pretty, keep Chinese


# ---------------------------------------------------------------- 1) a GARCH-like price panel
TICKERS = ["600519", "000858", "601318", "600036", "000333",           # twenty large caps
           "600276", "002415", "300750", "601012", "600030",           # more tickers
           "000651", "600887", "601888", "002594", "600009",           # more tickers
           "601166", "000001", "600016", "601288", "600028"]           # more tickers


def make_prices(n_days=2000, n_stocks=20):                         # the daily price panel
    """Return a price panel with a market factor, GARCH volatility and fat-tailed shocks."""   # short docstring
    market = np.zeros(n_days)                                      # market log returns
    vol = np.zeros(n_days)                                         # conditional volatility
    vol[0] = 0.011                                                 # start at about 1.1% a day
    omega, alpha, beta = 1.6e-6, 0.09, 0.88                         # GARCH(1,1) parameters
    for t in range(1, n_days):                                     # simulate the market
        vol[t] = np.sqrt(omega + alpha * market[t - 1] ** 2 + beta * vol[t - 1] ** 2)   # the variance recursion
        shock = RNG.standard_t(5) / np.sqrt(5 / 3)                  # a standardised t(5) shock
        market[t] = vol[t] * shock                                 # the return
    common = market * 0.9 + RNG.normal(0, 0.004, n_days)            # a common component plus noise
    rows = []                                                      # one row per stock-day
    for i, ticker in enumerate(TICKERS[:n_stocks]):                # loop over the stocks
        beta_i = float(RNG.uniform(0.6, 1.5))                       # each stock's beta
        idio = RNG.standard_t(6, n_days) / np.sqrt(6 / 4) * 0.012   # idiosyncratic fat-tailed noise
        ret = beta_i * common + idio                                # the stock's log return
        price = 20.0 * np.exp(np.cumsum(ret))                       # a price path starting at 20
        for t in range(n_days):                                    # loop over the days
            rows.append({"date": "D%04d" % t, "ticker": ticker,      # identifiers
                         "ret": round(float(ret[t]), 6),            # the log return
                         "price": round(float(price[t]), 3),        # the close price
                         "mkt_ret": round(float(market[t]), 6)})    # the market return
    return pd.DataFrame(rows), market                              # the frame and the market series


def make_yield_curve(n_days=500):                                  # the government bond curve
    """Return daily yields for five tenors with a normal, flat and inverted regime."""   # short docstring
    rows = []                                                      # one row per day
    level = 3.2 + np.cumsum(RNG.normal(0, 0.035, n_days))           # the level factor
    slope = 1.1 + np.cumsum(RNG.normal(0, 0.018, n_days))           # the slope factor
    for t in range(n_days):                                        # loop over the days
        inverted = -1.0 if (300 <= t < 380) else 0.0                 # a temporary inversion episode
        base = level[t] - 0.2                                          # the short rate
        curve = [base,                                        # 1 year
                 base + (slope[t] + inverted) * 0.35,          # 2 years
                 base + (slope[t] + inverted) * 0.75,          # 5 years
                 base + (slope[t] + inverted) * 1.00,          # 10 years
                 base + (slope[t] + inverted) * 1.25]          # 30 years
        row = {"day": t + 1, "regime": "倒挂" if inverted else "正常"}   # the row
        for tenor, y in zip(["1Y", "2Y", "5Y", "10Y", "30Y"], curve):   # the five tenors
            row[tenor] = round(float(y), 3)                          # the yield
        rows.append(row)                                             # append it
    return pd.DataFrame(rows)                                        # return the frame


# ---------------------------------------------------------------- 2) options and implied volatility
def make_option_chain(spot=100.0, r=0.02, n_strikes=17):            # a synthetic option chain
    """Return strikes, mid prices and the implied volatility smile."""   # short docstring
    strikes = np.linspace(80, 120, n_strikes)                        # strikes around the spot
    base_iv = 0.22                                                   # the at-the-money volatility
    rows = []                                                        # one row per strike
    for k in strikes:                                                # loop over strikes
        moneyness = np.log(k / spot)                                  # log moneyness
        iv = base_iv + 0.55 * moneyness ** 2 - 0.12 * moneyness        # a skewed smile
        iv = float(max(iv, 0.06))                                     # keep it positive
        from math import log, sqrt, exp, erf                          # local imports keep the top clean
        def norm_cdf(x):                                              # the standard normal CDF
            return 0.5 * (1 + erf(x / sqrt(2)))                        # via the error function
        d1 = (log(spot / k) + (r + 0.5 * iv ** 2)) / (iv * sqrt(1.0))  # one year to expiry
        d2 = d1 - iv * sqrt(1.0)                                     # the second argument
        call = spot * norm_cdf(d1) - k * exp(-r) * norm_cdf(d2)       # the Black-Scholes call
        rows.append({"strike": round(float(k), 2), "iv": round(iv, 4),   # the record
                     "call_price": round(float(call), 4)})            # the price
    return pd.DataFrame(rows)                                        # return the chain


def black_scholes_call(s, k, r, sigma, t=1.0):                      # the pricing formula
    """Return the Black-Scholes price of a European call."""           # short docstring
    from math import log, sqrt, exp, erf                              # local imports
    def cdf(x):                                                      # the standard normal CDF
        return 0.5 * (1 + erf(x / sqrt(2)))                            # via erf
    if sigma <= 0:                                                   # guard against zero volatility
        return max(s - k * exp(-r * t), 0.0)                          # the intrinsic value
    d1 = (log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * sqrt(t))   # d1
    d2 = d1 - sigma * sqrt(t)                                        # d2
    return s * cdf(d1) - k * exp(-r * t) * cdf(d2)                    # the call price


# ---------------------------------------------------------------- 3) credit risk sample
def make_credit(n=3000):                                           # a corporate credit sample
    """Return firm characteristics and a default flag from a logistic process."""   # short docstring
    leverage = RNG.normal(0.45, 0.18, n).clip(0.02, 1.4)             # debt over assets
    coverage = RNG.gamma(3.0, 1.6, n)                                # interest coverage
    size = np.exp(RNG.normal(9.0, 1.2, n))                            # total assets
    growth = RNG.normal(0.06, 0.15, n)                               # revenue growth
    logit = (-2.6 + 2.8 * leverage - 0.42 * coverage + 0.18 * growth   # the default process
             - 0.22 * np.log(size) + RNG.normal(0, 0.6, n))          # plus noise
    prob = 1 / (1 + np.exp(-logit))                                   # the default probability
    default = (RNG.random(n) < prob).astype(int)                       # the realised default
    return pd.DataFrame({"leverage": leverage.round(3), "coverage": coverage.round(2),   # assemble the frame
                         "size": size.round(0), "growth": growth.round(3),   # continuation of the call
                         "pd_true": prob.round(4), "default": default})   # the frame


# ---------------------------------------------------------------- 4) a fraud graph
def make_fraud_graph(n_nodes=900, n_rings=6):                       # a transaction/friendship graph
    """Return nodes with labels and an edge list containing a few dense fraud rings."""   # short docstring
    labels = np.zeros(n_nodes, dtype=int)                             # 0 = normal, 1 = fraud
    edges = []                                                        # the edge list
    for _ in range(n_nodes * 3):                                     # random background edges
        a, b = RNG.integers(0, n_nodes, 2)                            # two nodes
        if a != b:                                                   # no self loops
            edges.append((int(a), int(b)))                            # add the edge
    # normal communities: dense but not fraudulent, so raw degree cannot separate the classes
    for _ in range(n_rings * 2):                                     # twice as many normal communities
        members = list(RNG.choice(n_nodes, 16, replace=False))        # pick the members
        for i in range(len(members)):                                # make them moderately dense
            for j in range(i + 1, len(members)):                     # every unordered pair
                if RNG.random() < 0.45:                              # only some edges
                    edges.append((int(members[i]), int(members[j])))   # the edge
    ring_size = 18                                                   # each fraud ring has 18 members
    for r in range(n_rings):                                         # loop over the rings
        members = list(RNG.choice(n_nodes, ring_size, replace=False))   # pick the members
        for m in members:                                            # label them
            labels[m] = 1                                             # fraud
        for i in range(len(members)):                                # make the ring dense
            for j in range(len(members)):                            # every pair
                if i != j:                                           # no self loops
                    edges.append((int(members[i]), int(members[j])))   # a dense edge
    frame = pd.DataFrame({"node": np.arange(n_nodes), "label": labels})   # the node table
    edge_frame = pd.DataFrame(edges, columns=["src", "dst"]).drop_duplicates()   # the edge table
    return frame, edge_frame                                          # return both


# ---------------------------------------------------------------- 5) filings for the RAG demo
def make_filings(n=120):                                            # synthetic annual-report snippets
    """Return Chinese filing snippets with revenue, profit and a risk paragraph."""   # short docstring
    rows = []                                                        # one row per filing
    for i in range(n):                                               # loop over filings
        company = "公司%03d" % (i + 1)                                # a company name
        revenue = float(np.exp(RNG.normal(6.2, 0.8)))                 # revenue in hundred million
        profit = revenue * float(RNG.uniform(0.05, 0.22))             # net profit
        risk = str(RNG.choice(["原材料价格波动", "汇率波动", "下游需求放缓", "应收账款回收"]))   # continuation of the call
        text = ("%s 2026 年度报告：营业收入 %.2f 亿元，同比增长 %.1f%%；"                # continuation of the call
                "净利润 %.2f 亿元。风险提示：%s。") % (                             # continuation of the call
                company, revenue, float(RNG.uniform(-10, 35)), profit, risk)   # the sentence
        rows.append({"doc_id": "R%03d" % (i + 1), "company": company,   # identifiers
                     "text": text, "revenue": round(revenue, 2),      # the text and the truth
                     "profit": round(profit, 2), "risk": risk})       # more truth
    return pd.DataFrame(rows)                                        # return the frame
