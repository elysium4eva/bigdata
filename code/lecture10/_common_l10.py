"""_common_l10.py -- shared helpers for the Lecture 10 text/NLP examples.

Every line carries an English comment. The synthetic bilingual corpora, the fonts, the
seeds and the output folders live here so that every figure is reproducible.
"""
import json                                                        # read and write JSON artefacts
import matplotlib                                                  # choose a backend before other plotting imports
matplotlib.use("Agg")                                              # headless backend: figures render without a display
import matplotlib.pyplot as plt                                    # figure and axes API
import numpy as np                                                 # numeric arrays and random numbers
import pandas as pd                                                # tabular data structures
from pathlib import Path                                           # portable path handling

SEED = 20261111                                                    # single seed shared by the whole lecture
RNG = np.random.default_rng(SEED)                                  # shared generator keeps corpora reproducible
ROOT = Path(__file__).resolve().parents[2]                         # .../website (two levels above code/lecture10)
FIG = ROOT / "figures" / "lecture10"                               # output folder for figures
DAT = ROOT / "data" / "lecture10"                                  # output folder for corpora
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
    """Save a figure under website/figures/lecture10 and close it."""   # short docstring
    path = FIG / name                                              # build the full output path
    fig.savefig(path, facecolor="white")                           # white background keeps slides readable
    plt.close(fig)                                                 # release the figure from memory
    print("[figure]", path)                                        # log the artefact path


def dump(df, name):                                                # persist a dataframe into the data folder
    """Write a dataframe to website/data/lecture10 as UTF-8 CSV."""   # short docstring
    path = DAT / name                                              # build the full output path
    df.to_csv(path, index=False, encoding="utf-8-sig")             # utf-8-sig opens cleanly in Excel
    print("[data]", path)                                          # log the artefact path


def dump_artifact(text, name):                                     # persist a text artefact (markdown, JSON)
    """Write a text artefact into data/lecture10/artifacts."""      # short docstring
    path = OUT / name                                              # build the path inside the artifacts folder
    path.write_text(text, encoding="utf-8")                        # write with UTF-8 so Chinese labels survive
    print("[artifact]", path)                                      # log the artefact path


def write_json(obj, name):                                         # persist a dictionary as JSON
    """Serialise a dictionary to artifacts/<name> with UTF-8 encoding."""   # short docstring
    dump_artifact(json.dumps(obj, ensure_ascii=False, indent=2), name)   # pretty print and keep Chinese readable


# ---------------------------------------------------------------- positive and negative review templates
ZH_POS = ["这家店的服务非常好，物流也很快，下次还会买。",           # service and speed
          "质量超出预期，包装很仔细，五星好评。",                   # quality and packaging
          "性价比很高，用了一个月没有任何问题。",                   # value and durability
          "客服态度好，退货也方便，值得推荐。",                     # support and returns
          "款式好看，颜色和图片一致，很满意。",                     # appearance
          "发货速度快，第二天就到了，非常喜欢。",                   # delivery
          "做工精细，手感很好，价格公道。",                         # workmanship
          "第二次回购了，家里人都说不错。"]                         # repeat purchase
ZH_NEG = ["快递太慢了，等了十天才到，服务也差。",                   # slow delivery and service
          "质量很差，用两天就坏了，要求退款。",                     # quality and refund
          "和描述完全不符，颜色差别很大，很失望。",                 # mismatch
          "客服态度恶劣，问题一直不解决。",                         # support
          "包装破损，里面的商品也磕坏了。",                         # packaging damage
          "价格偏贵，性价比不高，不推荐。",                         # value
          "尺码不标准，退换货很麻烦。",                             # sizing
          "刚买就降价，感觉很坑。"]                                 # pricing complaint
ZH_NEU = ["收到了，还没来得及用，先给个评价。",                      # neutral receipt
          "物流正常，包装完整，具体效果待观察。",                   # neutral logistics
          "东西一般，没有想象中好，也没有多差。"]                   # neutral quality
EN_POS = ["Great quality and fast shipping, will buy again.",       # positive template
          "Excellent value for the price, works perfectly.",       # positive template
          "The support team replied quickly and solved my issue.",  # positive template
          "Looks exactly like the pictures, very satisfied.",      # positive template
          "Solid build, arrived early, no complaints at all."]     # positive template
EN_NEG = ["Cheap material, broke after two days, want a refund.",   # negative template
          "Delivery took forever and the box was damaged.",        # negative template
          "Does not match the description at all, disappointed.",  # negative template
          "Support never answered my emails, terrible service.",   # negative template
          "Overpriced for what you get, would not recommend."]     # negative template


def make_reviews_zh(n=600, pos_share=0.45, neg_share=0.35):         # a bilingual review corpus
    """Return Chinese reviews with three sentiment labels and aspect tags."""   # short docstring
    rows = []                                                      # one row per review
    for i in range(n):                                             # loop over the reviews
        u = RNG.random()                                           # uniform draw for the label
        if u < pos_share:                                          # positive
            label, pool = "正面", ZH_POS                            # label and template pool
        elif u < pos_share + neg_share:                            # negative
            label, pool = "负面", ZH_NEG                            # label and template pool
        else:                                                      # neutral
            label, pool = "中性", ZH_NEU                            # label and template pool
        text = str(RNG.choice(pool))                               # pick a template
        if RNG.random() < 0.35:                                    # sometimes append an aspect clause
            aspect = str(RNG.choice(["物流", "质量", "客服", "价格", "包装"]))   # a random aspect
            text = text + "（%s方面%s）" % (aspect, str(RNG.choice(["不错", "一般", "失望"])))   # aspect note
        stars = {"正面": 5, "中性": 3, "负面": 1}[label] + int(RNG.integers(-1, 2))   # noisy star rating
        if RNG.random() < 0.12:                                    # simulate annotation noise
            label = str(RNG.choice(["正面", "中性", "负面"]))         # a random label for a few rows
        rows.append({"doc_id": "zh%03d" % i, "text": text, "label": label,   # the record
                     "stars": int(np.clip(stars, 1, 5)), "lang": "zh"})   # clipped stars and language
    return pd.DataFrame(rows)                                      # return the frame


def make_reviews_en(n=300):                                        # the English half of the corpus
    """Return English reviews labelled positive or negative."""      # short docstring
    rows = []                                                      # one row per review
    for i in range(n):                                             # loop over the reviews
        positive = RNG.random() < 0.5                               # balanced labels
        text = str(RNG.choice(EN_POS if positive else EN_NEG))      # pick a template
        rows.append({"doc_id": "en%03d" % i, "text": text,          # the record
                     "label": "正面" if positive else "负面",        # keep the Chinese label vocabulary
                     "stars": 5 if positive else 1, "lang": "en"})   # stars and language
    return pd.DataFrame(rows)                                      # return the frame


def make_topic_corpus_zh(n=360):                                   # a three-topic Chinese corpus
    """Return Chinese documents generated from three topic word pools."""   # short docstring
    topics = {"数码": ["手机", "屏幕", "电池", "芯片", "拍照", "系统", "充电", "性能"],   # digital gadgets
              "食品": ["口味", "包装", "保质期", "配料", "甜度", "零食", "冷链", "新鲜"],   # food
              "服务": ["客服", "物流", "退款", "售后", "配送", "态度", "时效", "投诉"]}     # service
    rows = []                                                      # one row per document
    for i in range(n):                                             # loop over the documents
        topic = list(topics)[i % 3]                                 # round-robin topics
        words = RNG.choice(topics[topic], size=int(RNG.integers(8, 16)))   # sample the topic words
        filler = ["感觉", "整体", "大概", "比较", "一点", "还行"]      # neutral filler
        words = list(words) + list(RNG.choice(filler, size=3))        # add filler words
        RNG.shuffle(words)                                          # shuffle the order
        rows.append({"doc_id": "t%03d" % i, "text": "，".join(str(w) for w in words) + "。",   # join the words
                     "true_topic": topic})                          # remember the generating topic
    return pd.DataFrame(rows)                                      # return the frame


def make_contracts_zh(n=60):                                       # contracts for the NER demo
    """Return Chinese contract and news snippets with companies, amounts and dates."""   # short docstring
    companies = ["华为技术有限公司", "比亚迪股份有限公司", "深圳市腾讯计算机系统有限公司",   # company names
                 "小米科技有限责任公司", "阿里巴巴网络技术有限公司"]                        # more names
    amounts = ["人民币 1,200 万元", "3,450 万元", "人民币 86 亿元", "5,000 万元", "12.6 亿元"]   # money
    dates = ["2026 年 3 月 15 日", "2026-04-01", "2025 年 12 月 31 日", "2026 年 6 月"]      # dates
    rows = []                                                      # one row per snippet
    for i in range(n):                                             # loop over the snippets
        a, b = RNG.choice(companies, 2, replace=False)              # two distinct parties
        amount = str(RNG.choice(amounts))                           # a random amount
        date = str(RNG.choice(dates))                               # a random date
        template = str(RNG.choice([                                     # the template pool for the snippet
            "{a}与{b}于{date}签署合作协议，合同金额为{amount}。",     # cooperation agreement
            "根据{date}公告，{a}拟收购{b}持有的子公司股权，交易对价{amount}。",   # acquisition
            "{a}于{date}发布公告，向{b}提供总额{amount}的授信额度。",  # credit line
            "本协议由{a}与{b}于{date}签订，标的金额{amount}，履行期限两年。"]))   # contract
        text = template.format(a=a, b=b, date=date, amount=amount)   # fill the template
        rows.append({"doc_id": "c%03d" % i, "text": text})           # the record
    return pd.DataFrame(rows)                                      # return the frame


def make_lexicon_sentences():                                      # sentences with domain terms
    """Return sentences containing product names that a default segmenter splits."""   # short docstring
    return ["华为Mate60Pro在京东的销量很高。",                         # a phone model name
            "比亚迪海獭在欧洲市场表现不错。",                          # a car model name
            "我在小红书上看到vivoX100的测评。",                        # a brand plus model
            "李宁䨻科技的跑鞋很轻。",                                  # a rare character term
            "该公司的ESG报告提到碳中和目标。"]                          # an acronym


def make_analogy_pairs():                                          # analogy pairs for word vectors
    """Return a few analogy triples used to evaluate word vectors."""   # short docstring
    return [("国王", "男人", "女人", "女王"),                           # king - man + woman = queen
            ("北京", "中国", "日本", "东京"),                           # capital city relation
            ("手机", "苹果", "电脑", "微软"),                           # brand to product category
            ("好评", "满意", "差评", "失望")]                           # sentiment polarity
