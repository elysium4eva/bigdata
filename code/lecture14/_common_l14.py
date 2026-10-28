"""_common_l14.py -- shared helpers and synthetic trade data for Lecture 14.

Every line carries an English comment. All samples are generated here so that every figure   # continuation of the call
and every number in the deck can be regenerated from one seed.          # continuation of the call
"""
import json                                                        # read and write JSON artefacts
import matplotlib                                                  # choose a backend before pyplot
matplotlib.use("Agg")                                              # headless backend: no display needed
import matplotlib.pyplot as plt                                    # figure and axes API
import numpy as np                                                 # numeric arrays
import pandas as pd                                                # tabular structures
from pathlib import Path                                           # portable paths

SEED = 20270113                                                    # single seed for the whole lecture
RNG = np.random.default_rng(SEED)                                  # shared generator, reproducible draws
ROOT = Path(__file__).resolve().parents[2]                         # .../website (two levels above code/lecture14)
FIG = ROOT / "figures" / "lecture14"                               # output folder for figures
DAT = ROOT / "data" / "lecture14"                                  # output folder for data
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
    """Save a figure into website/figures/lecture14 and close it."""   # short docstring
    path = FIG / name                                              # full path
    fig.savefig(path, facecolor="white")                           # white background
    plt.close(fig)                                                 # release memory
    print("[figure]", path)                                        # log it


def dump(df, name):                                                # persist a dataframe
    """Write a dataframe to website/data/lecture14 as UTF-8 CSV."""   # short docstring
    path = DAT / name                                              # full path
    df.to_csv(path, index=False, encoding="utf-8-sig")             # utf-8-sig opens cleanly in Excel
    print("[data]", path)                                          # log it


def dump_artifact(text, name):                                     # persist a text artefact
    """Write a text artefact into data/lecture14/artifacts."""       # short docstring
    path = OUT / name                                              # full path
    path.write_text(text, encoding="utf-8")                        # write UTF-8
    print("[artifact]", path)                                      # log it


def write_json(obj, name):                                         # persist a dictionary
    """Serialise a dictionary to artifacts/<name>."""               # short docstring
    dump_artifact(json.dumps(obj, ensure_ascii=False, indent=2), name)   # pretty, keep Chinese


# ---------------------------------------------------------------- 1) the country panel
ISO = ["CN", "US", "JP", "DE", "GB", "FR", "IN", "BR", "RU", "IT",   # twenty large economies
       "CA", "KR", "AU", "MX", "ID", "TR", "SA", "ZA", "AR", "NG",   # more large economies
       "VN", "TH", "MY", "SG", "PH", "EG", "IR", "PK", "BD", "PL",   # emerging economies
       "NL", "BE", "ES", "SE", "CH", "AT", "NO", "DK", "FI", "IE",   # European economies
       "PT", "GR", "CZ", "HU", "RO", "UA", "IL", "AE", "CL", "CO",   # more economies
       "PE", "NZ", "KZ", "MA", "KE", "ET", "GH", "TZ", "UY", "EC"]   # sixty countries in total
NAME_ZH = {"CN": "中国", "US": "美国", "JP": "日本", "DE": "德国",   # Chinese names for the figures
           "GB": "英国", "FR": "法国", "IN": "印度", "BR": "巴西",   # more names
           "RU": "俄罗斯", "IT": "意大利", "KR": "韩国", "AU": "澳大利亚",   # more names
           "MX": "墨西哥", "CA": "加拿大", "ID": "印度尼西亚", "SG": "新加坡",   # more names
           "NL": "荷兰", "VN": "越南", "TH": "泰国", "MY": "马来西亚"}   # names used in plots


def make_country_panel(n=60, years=(2000, 2019)):                  # the exporter/importer panel
    """Return a country-year panel with GDP, population and geography."""   # short docstring
    years = list(range(years[0], years[1] + 1))                    # the years to generate
    rows = []                                                      # one row per country-year
    base_gdp = np.exp(RNG.normal(26.5, 0.55, n))                    # baseline GDP in logs (tighter spread)
    # the first ten ISO codes are the large economies, the next twelve mid-sized, the rest small
    scale = np.array([3.4 if i < 10 else (1.0 if i < 22 else 0.22) for i in range(n)])   # size tiers
    base_gdp = base_gdp * scale                                     # apply the tiering
    growth = RNG.normal(0.035, 0.02, n)                            # country growth rates
    for i, iso in enumerate(ISO[:n]):                              # loop over countries
        for t, year in enumerate(years):                           # loop over years
            gdp = float(base_gdp[i] * np.exp(growth[i] * t + RNG.normal(0, 0.05)))   # GDP with noise
            rows.append({"iso": iso, "year": year,                 # identifiers
                         "gdp": round(gdp, 1),                      # GDP in billions of USD
                         "pop": round(float(np.exp(RNG.normal(17.0, 1.4))), 1),   # population in millions
                         "island": int(RNG.random() < 0.18)})       # island dummy
    return pd.DataFrame(rows)                                      # return the panel


def make_geography(n=60):                                          # bilateral geography
    """Return a directed dyad table with distance, contiguity and shared attributes."""   # short docstring
    iso = ISO[:n]                                                  # the country list
    lat = RNG.uniform(-40, 60, n)                                  # latitudes
    lon = RNG.uniform(-120, 150, n)                                # longitudes
    rows = []                                                      # one row per directed pair
    for a in range(n):                                             # exporter
        for b in range(n):                                         # importer
            if a == b:                                             # skip the diagonal
                continue                                           # next pair
            d_lat = np.radians(lat[a] - lat[b])                     # latitude difference
            d_lon = np.radians(lon[a] - lon[b])                     # longitude difference
            hav = (np.sin(d_lat / 2) ** 2 + np.cos(np.radians(lat[a]))   # haversine formula
                   * np.cos(np.radians(lat[b])) * np.sin(d_lon / 2) ** 2)   # the other term
            dist = 6371 * 2 * np.arcsin(np.sqrt(hav))               # great-circle distance in km
            rows.append({"iso_o": iso[a], "iso_d": iso[b],          # the dyad
                         "dist": round(float(dist), 1),             # distance
                         "contig": int(dist < 1500),                # a simple contiguity proxy
                         "comlang": int(RNG.random() < 0.12),       # shared official language
                         "colony": int(RNG.random() < 0.05)})       # colonial link
    return pd.DataFrame(rows)                                      # return the dyad table


# ---------------------------------------------------------------- 2) gravity data-generating process
TRUE_BETA = {"lgdp_o": 0.97, "lgdp_d": 0.85, "ldist": -0.89,       # the "true" elasticities
             "contig": 0.49, "comlang": 0.49, "colony": 0.91,      # more coefficients
             "rta": 0.47, "cur": 0.87, "border_home": 1.93}        # home bias


def make_trade_flows(countries, geo, years=(2000, 2019), zero_share=0.28):   # the bilateral flow panel
    """Return the dyad-year trade panel with zeros generated by a selection rule."""   # short docstring
    years = list(range(years[0], years[1] + 1))                    # the years
    gdp = countries.set_index(["iso", "year"])["gdp"]               # GDP lookup
    geo = geo.copy()                                               # do not touch the caller's frame
    geo["ld"] = np.log(geo["dist"])                                 # log distance per dyad
    remote_o = geo.groupby("iso_o")["ld"].mean()                    # exporter remoteness (multilateral resistance)
    remote_d = geo.groupby("iso_d")["ld"].mean()                    # importer remoteness
    rem_centre = float(np.mean(list(remote_o.values) + list(remote_d.values)))   # a centring constant
    rows = []                                                      # one row per dyad-year
    for _, g in geo.iterrows():                                    # loop over dyads
        a, b, dist = g["iso_o"], g["iso_d"], g["dist"]              # the pair
        rta = float(RNG.random() < 0.35)                            # RTA membership (time invariant here)
        cur = float(RNG.random() < 0.10)                            # shared currency
        for year in years:                                         # loop over years
            ga = float(gdp.loc[(a, year)])                          # exporter GDP
            gb = float(gdp.loc[(b, year)])                          # importer GDP
            lgdp_o, lgdp_d = np.log(ga), np.log(gb)                 # logs
            ldist = np.log(dist)                                   # log distance
            v = (TRUE_BETA["lgdp_o"] * lgdp_o + TRUE_BETA["lgdp_d"] * lgdp_d   # deterministic part
                 + TRUE_BETA["ldist"] * ldist                       # distance
                 + TRUE_BETA["contig"] * g["contig"]                # contiguity
                 + TRUE_BETA["comlang"] * g["comlang"]              # language
                 + TRUE_BETA["colony"] * g["colony"]                # colonial link
                 + TRUE_BETA["rta"] * rta + TRUE_BETA["cur"] * cur)   # policy variables
            # multilateral resistance: an exporter-level term that a naive regression omits
            v += -1.10 * (float(remote_o.loc[a]) - rem_centre)          # the omitted multilateral-resistance term
            v += -6.2 + RNG.normal(0, 0.35)                         # scale plus noise
            # selection: distant, small pairs are more likely to record no trade at all
            p_zero = min(0.92, max(0.02, zero_share + 0.10 * (ldist - 8.0) / 1.0   # the zero-trade probability
                                   - 0.05 * (lgdp_o + lgdp_d - 52.0) / 2.0))   # zero probability
            is_zero = RNG.random() < p_zero                         # draw the selection
            flow = 0.0 if is_zero else float(np.exp(v))             # observed flow (millions USD)
            rows.append({"iso_o": a, "iso_d": b, "year": year,      # identifiers
                         "trade": round(flow, 3),                   # the flow
                         "dist": dist, "contig": g["contig"],        # geography
                         "comlang": g["comlang"], "colony": g["colony"],   # more geography
                         "rta": rta, "cur": cur,                    # policy
                         "lgdp_o": round(lgdp_o, 4), "lgdp_d": round(lgdp_d, 4),   # GDP in logs
                         "ldist": round(ldist, 4)})                 # distance in logs
    return pd.DataFrame(rows)                                      # return the panel


# ---------------------------------------------------------------- 3) firms and customs matching
def make_firms(n_firms=4000):                                      # the industrial survey sample
    """Return firms with output, employment and a name-like identifier."""   # short docstring
    size = np.exp(RNG.normal(3.2, 1.5, n_firms))                    # firm size (employment)
    rows = []                                                      # one row per firm
    for i in range(n_firms):                                       # loop over firms
        rows.append({"firm_id": "F%05d" % (i + 1),                  # firm identifier
                     "name": "企业%05d" % (i + 1),                   # a Chinese-style name
                     "phone": "0755-%07d" % int(RNG.integers(1000000, 9999999)),   # phone number
                     "postcode": "%06d" % int(RNG.integers(100000, 999999)),      # postcode
                     "output": round(float(np.exp(RNG.normal(9.4, 1.4))), 1),    # output
                     "employ": int(max(1, round(size[i]))),       # employment
                     "province": int(RNG.integers(1, 32))})        # province code
    return pd.DataFrame(rows)                                      # return the frame


def make_customs(firms, n_records=6000):                           # the customs transaction sample
    """Return customs records where a subset links back to the firm survey."""   # short docstring
    rows = []                                                      # one row per record
    for i in range(n_records):                                     # loop over records
        f = firms.iloc[int(RNG.integers(0, len(firms)))]            # pick a firm
        linked = int(RNG.random() < 0.62)                           # 62% of records can be matched
        rows.append({"record_id": "C%06d" % (i + 1),                # record identifier
                     "firm_id": f["firm_id"] if linked else "",      # the linked firm (or blank)
                     "name": f["name"] if linked else "个体户%05d" % i,   # the reported name
                     "hs8": str(int(RNG.integers(10000000, 99999999))),   # an eight-digit code
                     "partner": str(RNG.choice(["US", "JP", "DE", "VN", "KR", "IN"])),   # partner
                     "value": round(float(np.exp(RNG.normal(6.6, 1.5))), 2),   # value in thousand USD
                     "year": int(RNG.integers(2000, 2020))})        # year
    return pd.DataFrame(rows)                                      # return the frame


# ---------------------------------------------------------------- 4) a tiny world input-output table
SECTORS = ["农业", "制造", "服务"]                                  # three sectors
WIO_COUNTRIES = ["CN", "US", "DE", "JP"]                            # four countries


def make_wiot():                                                   # a simplified WIOT
    """Return a small WIOT: intermediate flows, final demand and total output."""   # short docstring
    n_ind = len(WIO_COUNTRIES) * len(SECTORS)                       # number of industries
    Z = np.zeros((n_ind, n_ind))                                    # intermediate use matrix
    for i in range(n_ind):                                         # row (producer)
        for j in range(n_ind):                                     # column (user)
            ci, cj = i // len(SECTORS), j // len(SECTORS)           # country of each side
            base = 120.0 if ci == cj else 45.0                      # domestic flows are larger
            Z[i, j] = base * np.exp(RNG.normal(0, 0.45))            # random but structured
    fd = (Z.sum(axis=1) * RNG.uniform(0.15, 0.45, n_ind))           # final demand per industry
    x = Z.sum(axis=1) + fd                                          # total output
    va = x - Z.sum(axis=0)                                          # value added (residual)
    labels = ["%s-%s" % (c, s) for c in WIO_COUNTRIES for s in SECTORS]   # industry labels
    return {"Z": Z, "fd": fd, "x": x, "va": va, "labels": labels}     # return the pieces


# ---------------------------------------------------------------- 5) ports and shipping
PORTS = {"上海": (31.2, 121.5), "新加坡": (1.35, 103.8), "鹿特丹": (51.9, 4.5),   # major ports
         "洛杉矶": (33.7, -118.2), "釜山": (35.1, 129.0), "迪拜": (25.2, 55.3),   # more ports
         "汉堡": (53.5, 10.0), "宁波": (29.9, 121.6), "深圳": (22.5, 114.1),      # more ports
         "巴生港": (3.0, 101.4), "安特卫普": (51.2, 4.4), "长滩": (33.8, -118.1)}  # more ports


def make_ais(n_routes=120):                                        # synthetic AIS-derived voyages
    """Return voyage records: origin port, destination port, days and delay."""   # short docstring
    names = list(PORTS)                                            # port names
    rows = []                                                      # one row per voyage
    for i in range(n_routes):                                      # loop over voyages
        o, d = RNG.choice(names, 2, replace=False)                  # two distinct ports
        (lat1, lon1), (lat2, lon2) = PORTS[o], PORTS[d]             # their coordinates
        dist = 6371 * 2 * np.arcsin(np.sqrt(                          # great-circle distance
            np.sin(np.radians(lat2 - lat1) / 2) ** 2                    # the haversine term
            + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))       # the second haversine term
            * np.sin(np.radians(lon2 - lon1) / 2) ** 2))            # the haversine term
        days = dist / 620.0                                           # 620 km per day average
        delay = float(max(0.0, RNG.normal(1.4, 1.1)))                # port congestion delay
        rows.append({"voyage_id": "V%04d" % (i + 1), "origin": o, "dest": d,   # the record
                     "distance_km": round(float(dist), 0),           # distance
                     "days": round(float(days + delay), 1),          # sailing days including delay
                     "delay_days": round(delay, 1)})                 # the delay component
    return pd.DataFrame(rows)                                      # return the frame


# ---------------------------------------------------------------- 6) customs declaration texts
def make_declaration_texts(n=400):                                 # synthetic declaration sentences
    """Return Chinese declaration texts with the fields the LLM extractor should find."""   # short docstring
    rows = []                                                      # one row per declaration
    goods = ["锂电池", "光伏组件", "数控机床", "服装", "咖啡豆", "汽车零部件"]   # goods
    ports = ["上海港", "深圳港", "宁波舟山港", "青岛港"]                  # ports
    incoterms = ["FOB", "CIF", "EXW", "DAP"]                        # incoterms
    for i in range(n):                                             # loop over declarations
        hs = str(int(RNG.integers(10000000, 99999999)))             # an eight-digit HS code
        value = float(np.exp(RNG.normal(4.8, 0.9)))                 # value in thousand USD
        origin = str(RNG.choice(ISO[:20]))                          # origin country
        text = ("报关单：品名%s，HS 编码 %s，申报总值 %.1f 千美元，原产国 %s，"               # the declaration sentence
                "出口口岸%s，成交方式 %s，运输方式海运。") % (                           # the rest of the sentence
                str(RNG.choice(goods)), hs, value, origin, str(RNG.choice(ports)),   # goods and numbers
                str(RNG.choice(incoterms)))                         # incoterm
        rows.append({"doc_id": "D%04d" % (i + 1), "text": text,      # the record
                     "true_hs": hs, "true_value": round(value, 1),   # the ground truth
                     "true_origin": origin})                        # the ground truth
    return pd.DataFrame(rows)                                      # return the frame


# ---------------------------------------------------------------- 7) geopolitical risk index
def make_gpr(months=120):                                          # a monthly risk index
    """Return a synthetic geopolitical-risk index built from keyword counts."""   # short docstring
    t = np.arange(months)                                          # the month index
    base = 100 + 12 * np.sin(t / 9.0) + RNG.normal(0, 6, months)   # a seasonal baseline
    shocks = np.zeros(months)                                      # discrete shock spikes
    for at, size in [(28, 55), (63, 80), (95, 45)]:                # three geopolitical events
        if at < months:                                            # only if inside the window
            shocks[at:at + 4] += size * np.exp(-np.arange(4) / 1.6)   # decaying spike
    index = base + shocks                                          # the composite index
    keywords = (index - index.min()) / (index.max() - index.min()) * 40 + 8   # keyword counts
    return pd.DataFrame({"month": t + 1, "index": index.round(1),    # the frame
                         "keyword_count": keywords.round(1),         # raw counts
                         "event": (shocks > 20).astype(int)})        # an event flag
