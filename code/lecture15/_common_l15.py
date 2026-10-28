"""_common_l15.py -- shared helpers and synthetic marketing data for Lecture 15.

Every line carries an English comment. All samples are generated here so that every figure
and every number in the deck can be regenerated from one seed.
"""
import json                                                        # read and write JSON artefacts
import matplotlib                                                  # choose a backend before pyplot
matplotlib.use("Agg")                                              # headless backend: no display needed
import matplotlib.pyplot as plt                                    # figure and axes API
import numpy as np                                                 # numeric arrays
import pandas as pd                                                # tabular structures
from pathlib import Path                                           # portable paths

SEED = 20261216                                                    # single seed for the whole lecture
RNG = np.random.default_rng(SEED)                                  # shared generator, reproducible draws
ROOT = Path(__file__).resolve().parents[2]                         # .../website (two levels above code/lecture15)
FIG = ROOT / "figures" / "lecture15"                               # output folder for figures
DAT = ROOT / "data" / "lecture15"                                  # output folder for data
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
    """Save a figure into website/figures/lecture15 and close it."""   # short docstring
    path = FIG / name                                              # full path
    fig.savefig(path, facecolor="white")                           # white background
    plt.close(fig)                                                 # release memory
    print("[figure]", path)                                        # log it


def dump(df, name):                                                # persist a dataframe
    """Write a dataframe to website/data/lecture15 as UTF-8 CSV."""   # short docstring
    path = DAT / name                                              # full path
    df.to_csv(path, index=False, encoding="utf-8-sig")             # utf-8-sig opens cleanly in Excel
    print("[data]", path)                                          # log it


def dump_artifact(text, name):                                     # persist a text artefact
    """Write a text artefact into data/lecture15/artifacts."""       # short docstring
    path = OUT / name                                              # full path
    path.write_text(text, encoding="utf-8")                        # write UTF-8
    print("[artifact]", path)                                      # log it


def write_json(obj, name):                                         # persist a dictionary
    """Serialise a dictionary to artifacts/<name>."""               # short docstring
    dump_artifact(json.dumps(obj, ensure_ascii=False, indent=2), name)   # pretty, keep Chinese


# ---------------------------------------------------------------- 1) survey / scale data
SURVEY_ITEMS = {                                                   # three constructs, four items each
    "品牌信任": ["这个品牌值得信赖", "这个品牌不会让我失望", "我相信这个品牌的质量", "这个品牌口碑好"],        # brand-trust items
    "价格价值": ["这个价格很划算", "物有所值", "性价比高", "促销时更愿意买"],                     # price-value items
    "服务体验": ["客服响应很快", "售后让我放心", "问题能一次解决", "门店体验舒服"],                  # service items
}                                                                       # end of the construct dictionary


def make_survey(n=600):                                            # a Likert-scale survey sample
    """Return a survey dataframe with three latent constructs and demographics."""   # short docstring
    latent = RNG.normal(0, 1, (n, 3))                              # three latent scores per respondent
    loadings = np.array([[0.86, 0.06, 0.02], [0.82, 0.05, 0.03],    # item loadings on the factors
                         [0.78, 0.04, 0.05], [0.74, 0.02, 0.06],   # brand trust items
                         [0.03, 0.83, 0.04], [0.04, 0.79, 0.02],   # price-value items
                         [0.05, 0.75, 0.06], [0.02, 0.70, 0.03],   # price-value items
                         [0.02, 0.04, 0.85], [0.03, 0.06, 0.80],   # service items
                         [0.04, 0.03, 0.76], [0.02, 0.05, 0.72]])  # service items
    noise = RNG.normal(0, 0.55, (n, 12))                           # measurement noise
    raw = latent @ loadings.T + noise                              # continuous item responses
    items = np.clip(np.round(3.0 + raw * 0.9), 1, 5).astype(int)    # map to a 1-5 Likert scale
    frame = pd.DataFrame(items, columns=list(range(1, 13)))        # item columns i1..i12
    frame.columns = ["q%02d" % c for c in frame.columns]           # rename to q01..q12
    frame["age"] = RNG.integers(18, 66, n)                         # respondent age
    frame["gender"] = RNG.choice(["女", "男"], n)                   # respondent gender
    frame["city_tier"] = RNG.choice(["一线", "二线", "三线"], n, p=[0.32, 0.42, 0.26])   # city tier
    frame["overall"] = np.clip(np.round(3.0 + latent.mean(axis=1) * 0.95 + RNG.normal(0, 0.4, n)), 1, 5)   # overall rating
    frame["intent"] = np.clip(np.round(2.8 + latent.mean(axis=1) * 1.05 + RNG.normal(0, 0.5, n)), 1, 5)   # purchase intent
    return frame                                                    # return the frame


# ---------------------------------------------------------------- 2) conjoint design and ratings
ATTRS = {"续航": ["500km", "650km", "800km"],                       # attribute 1 and its levels
         "价格": ["18万", "23万", "28万"],                          # attribute 2
         "快充": ["30分钟", "20分钟", "15分钟"],                     # attribute 3
         "智驾": ["基础", "增强", "旗舰"]}                           # attribute 4
TRUE_UTIL = {                                                      # the generating part-worths
    "续航": {"500km": -0.55, "650km": 0.15, "800km": 0.60},         # longer range helps
    "价格": {"18万": 0.70, "23万": 0.05, "28万": -0.75},            # higher price hurts
    "快充": {"30分钟": -0.30, "20分钟": 0.10, "15分钟": 0.35},       # faster charging helps
    "智驾": {"基础": -0.25, "增强": 0.20, "旗舰": 0.45},             # better assist helps
}                                                                       # end of the attribute dictionary


def make_conjoint_profiles(n_profiles=18):                         # a balanced, orthogonal design
    """Return a balanced profile design where every level appears equally often."""   # short docstring
    names = list(ATTRS)                                            # the four attributes
    if n_profiles % 3 != 0:                                         # balance needs a multiple of three
        n_profiles = 18                                            # fall back to eighteen profiles
    columns = {}                                                   # attribute -> level list
    for k, attr in enumerate(names):                                # build each attribute independently
        levels = ATTRS[attr] * (n_profiles // 3)                    # every level appears n/3 times
        columns[attr] = list(RNG.permutation(levels))               # shuffle independently: no collinearity
    design = pd.DataFrame(columns)                                  # the profile table
    design.insert(0, "profile", ["P%02d" % (i + 1) for i in range(len(design))])   # profile ids
    return design                                                  # return the design

def make_conjoint_responses(design, n_respondents=300, noise=0.55):   # simulate ratings
    """Return a long dataframe of respondent ratings for every profile."""   # short docstring
    base = np.zeros((n_respondents, len(design)))                   # rating matrix
    for a in ATTRS:                                                # sum the part-worths
        for lv, u in TRUE_UTIL[a].items():                          # level by level
            mask = (design[a] == lv).values                         # profiles at this level
            base[:, mask] += u                                      # add the utility
    base += 3.6                                                     # shift to the 1-7 rating scale
    base += RNG.normal(0, noise, base.shape)                        # respondent noise
    ratings = np.clip(np.round(base), 1, 7).astype(int)             # round to integers
    rows = []                                                      # long-format rows
    for r in range(n_respondents):                                 # loop over respondents
        for p in range(len(design)):                               # loop over profiles
            rows.append({"resp": "R%03d" % (r + 1), "profile": design["profile"].iloc[p],   # ids
                         "rating": int(ratings[r, p])})             # the rating
    return pd.DataFrame(rows)                                      # return the long frame


# ---------------------------------------------------------------- 3) discrete choice data
def make_choice_data(n_choices=800):                               # individual-level choice data
    """Return a wide choice frame: three alternatives with attributes and the chosen one."""   # short docstring
    rows = []                                                      # one row per choice occasion
    betas = {"price": -0.22, "range": 0.004, "fast": 0.09, "brand": 0.55}   # true coefficients
    brands = ["A", "B", "C"]                                       # three brands
    for i in range(n_choices):                                     # loop over occasions
        alts = []                                                  # the three alternatives
        for b in brands:                                           # build each alternative
            price = float(RNG.choice([18.0, 23.0, 28.0]))           # price in 万元
            rng_km = float(RNG.choice([500, 650, 800]))            # range in km
            fast = float(RNG.choice([15, 20, 30]))                 # charging minutes
            brand_eff = {"A": 0.0, "B": 0.35, "C": -0.25}[b]       # brand intercept
            u = (betas["price"] * price + betas["range"] * rng_km   # deterministic utility
                 + betas["fast"] * (30 - fast) + brand_eff + RNG.gumbel(0, 1))   # plus a Gumbel shock
            alts.append({"brand": b, "price": price, "range": rng_km, "fast": fast, "u": u})   # store it
        best = max(range(3), key=lambda k: alts[k]["u"])            # the utility-maximising choice
        row = {"choice_id": "C%04d" % (i + 1)}                      # the record
        for k, a in enumerate(alts):                               # write each alternative
            row["brand_%d" % (k + 1)] = a["brand"]                  # brand label
            row["price_%d" % (k + 1)] = a["price"]                  # price
            row["range_%d" % (k + 1)] = a["range"]                  # range
            row["fast_%d" % (k + 1)] = a["fast"]                    # charging time
        row["chosen"] = alts[best]["brand"]                         # the chosen brand
        rows.append(row)                                           # append the record
    return pd.DataFrame(rows)                                      # return the frame


# ---------------------------------------------------------------- 4) hedonic housing data
def make_housing(n=1200):                                          # a hedonic price sample
    """Return housing transactions with the attributes used in the hedonic regression."""   # short docstring
    area = RNG.normal(95, 28, n).clip(38, 260)                      # floor area in square metres
    school = RNG.choice([0, 1], n, p=[0.62, 0.38])                  # key school district flag
    metro = RNG.exponential(1.6, n).clip(0.1, 8.0)                  # distance to the metro in km
    age = RNG.integers(0, 31, n)                                    # building age in years
    floor = RNG.integers(1, 33, n)                                  # floor number
    log_price = (3.05 + 0.0092 * area + 0.185 * school - 0.055 * metro   # the data-generating process
                 - 0.0075 * age + 0.0022 * floor + RNG.normal(0, 0.16, n))   # plus log-normal noise
    price = np.exp(log_price) * 1e4                                 # price in yuan
    return pd.DataFrame({"area": area.round(1), "school": school,    # the frame
                         "metro_km": metro.round(2), "age": age,     # more attributes
                         "floor": floor, "price": price.round(0)})   # the outcome


# ---------------------------------------------------------------- 5) clickstream and attribution
CHANNELS = ["自然搜索", "信息流广告", "搜索广告", "社交媒体", "直接访问"]   # five marketing channels
STAGES = ["曝光", "点击", "加购", "下单", "支付"]                    # the five funnel stages


def make_clickstream(n_sessions=6000):                             # event-level clickstream
    """Return a session table with a channel path, a funnel depth and a conversion flag."""   # short docstring
    rows = []                                                      # one row per session
    base_conv = {"自然搜索": 0.052, "信息流广告": 0.028, "搜索广告": 0.061,   # base conversion by channel
                 "社交媒体": 0.021, "直接访问": 0.071}               # more channels
    for i in range(n_sessions):                                    # loop over sessions
        path_len = int(RNG.integers(1, 5))                          # touches before the visit
        path = list(RNG.choice(CHANNELS, path_len, p=[0.24, 0.26, 0.18, 0.22, 0.10]))   # the channel path
        last = path[-1]                                            # the last touch
        p_conv = base_conv[last] * (1.0 + 0.12 * (path_len - 1))     # more touches lift conversion
        converted = RNG.random() < p_conv                          # did the session convert
        depth = 5 if converted else int(RNG.integers(1, 5))         # funnel depth reached
        rows.append({"session_id": "S%05d" % (i + 1), "path": " > ".join(path),   # the record
                     "touches": path_len, "last_channel": last,     # path features
                     "depth": depth, "converted": int(converted),   # funnel outcome
                     "revenue": float(np.round(RNG.gamma(2.0, 120.0), 0)) if converted else 0.0})   # order value
    return pd.DataFrame(rows)                                      # return the frame


# ---------------------------------------------------------------- 6) recommender rating matrix
def make_rating_matrix(n_users=500, n_items=60, n_factors=4, density=0.16):   # implicit-ish ratings
    """Return a sparse rating frame generated from latent factors."""   # short docstring
    U = RNG.normal(0, 0.9, (n_users, n_factors))                    # user factors
    V = RNG.normal(0, 0.9, (n_items, n_factors))                    # item factors
    scores = U @ V.T + 3.2                                          # baseline plus interaction
    scores += RNG.normal(0, 0.5, scores.shape)                      # noise
    mask = RNG.random(scores.shape) < density                        # observed entries
    for u in range(n_users):                                        # guarantee a few items per user
        mask[u, RNG.choice(n_items, 6, replace=False)] = True        # six observed items minimum
    u_idx, i_idx = np.where(mask)                                          # the observed indices
    frame = pd.DataFrame({"user": ["U%03d" % (u + 1) for u in u_idx],   # the long frame
                          "item": ["I%02d" % (i + 1) for i in i_idx],   # item ids
                          "rating": np.clip(scores[u_idx, i_idx], 1, 5).round(2)})   # the ratings
    return frame, scores                                              # ratings and the latent truth


# ---------------------------------------------------------------- 7) price and sales
def make_price_sales(n_weeks=104):                                 # weekly price and sales
    """Return weekly price, promotion flag and units sold for an elasticity estimate."""   # short docstring
    price = 39.0 + RNG.normal(0, 2.4, n_weeks)                      # regular price
    promo = (RNG.random(n_weeks) < 0.18).astype(int)                 # promotion weeks
    price = np.where(promo == 1, price * 0.82, price)               # discount in promotion weeks
    elasticity = -1.85                                              # true own-price elasticity
    log_q = 10.2 + elasticity * np.log(price) + 0.35 * promo         # log-linear demand
    log_q += 0.12 * np.sin(np.arange(n_weeks) / 8.0)                 # seasonality
    log_q += RNG.normal(0, 0.09, n_weeks)                            # noise
    units = np.clip(np.exp(log_q).round(0), 1, None)                # units sold (never zero)
    return pd.DataFrame({"week": np.arange(1, n_weeks + 1),          # the frame
                         "price": price.round(2), "promo": promo,    # price and promotion
                         "units": units.astype(int)})                # outcome


# ---------------------------------------------------------------- 8) churn data
def make_churn(n=3000):                                            # customer churn sample
    """Return customer features and a churn flag from a logistic process."""   # short docstring
    tenure = RNG.integers(1, 49, n)                                 # months as a customer
    usage = RNG.gamma(2.2, 22.0, n).round(1)                        # monthly usage minutes
    complaints = RNG.poisson(0.45, n)                               # complaints in the last quarter
    plan = RNG.choice(["基础", "标准", "尊享"], n, p=[0.42, 0.38, 0.20])   # plan tier
    plan_eff = np.where(plan == "基础", 0.55, np.where(plan == "标准", 0.0, -0.62))   # plan effect
    logit = (-1.05 - 0.052 * tenure - 0.011 * usage + 0.62 * complaints + plan_eff   # churn logit
             + RNG.normal(0, 0.5, n))                               # noise
    prob = 1 / (1 + np.exp(-logit))                                 # churn probability
    churn = (RNG.random(n) < prob).astype(int)                       # realised churn
    return pd.DataFrame({"tenure": tenure, "usage": usage,           # the frame
                         "complaints": complaints, "plan": plan,     # features
                         "churn_prob_true": prob.round(4), "churn": churn})   # labels


# ---------------------------------------------------------------- 9) experiment data
def make_ab(n_per_arm=5000, p_control=0.050, lift=0.006):          # a two-arm experiment
    """Return per-user conversion outcomes for a control and a treatment arm."""   # short docstring
    p_treat = p_control + lift                                      # the treatment rate
    rows = []                                                      # one row per user
    for arm, p in (("control", p_control), ("treatment", p_treat)):   # loop over the arms
        conv = (RNG.random(n_per_arm) < p).astype(int)               # Bernoulli outcomes
        for k, c in enumerate(conv):                               # one row per user
            rows.append({"arm": arm, "user": "%s_%05d" % (arm[0], k + 1),   # the record
                         "converted": int(c), "revenue": float(c * RNG.gamma(2.0, 60.0))})   # order value
    return pd.DataFrame(rows)                                      # return the frame
