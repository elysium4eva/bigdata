"""_common_l11.py -- a tiny local chat model plus shared helpers for Lecture 11.

Every line carries an English comment. There is no network and no API key: `LocalChatModel`   # continuation of the call
mimics the shape of the OpenAI chat interface (messages, temperature, max_tokens) on top of a   # continuation of the call
word-level n-gram model, so every example in this lecture is reproducible offline. Swap the   # continuation of the call
`model=` argument for a real endpoint and the calling code does not change.   # continuation of the call
"""
import json                                                        # read and write JSON artefacts
import math                                                        # log, exp and entropy
import re                                                          # light text cleaning
import matplotlib                                                  # choose a backend before pyplot
matplotlib.use("Agg")                                              # headless backend: no display needed
import matplotlib.pyplot as plt                                    # figure and axes API
import numpy as np                                                 # numeric arrays
import pandas as pd                                                # tabular structures
from pathlib import Path                                           # portable paths

SEED = 20270210                                                    # single seed for the whole lecture
RNG = np.random.default_rng(SEED)                                  # shared generator, reproducible draws
ROOT = Path(__file__).resolve().parents[2]                         # .../website (two levels above code/lecture11)
FIG = ROOT / "figures" / "lecture11"                               # output folder for figures
DAT = ROOT / "data" / "lecture11"                                  # output folder for data
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
    """Save a figure into website/figures/lecture11 and close it."""   # short docstring
    path = FIG / name                                              # full path
    fig.savefig(path, facecolor="white")                           # white background
    plt.close(fig)                                                 # release memory
    print("[figure]", path)                                        # log it


def dump(df, name):                                                # persist a dataframe
    """Write a dataframe to website/data/lecture11 as UTF-8 CSV."""   # short docstring
    path = DAT / name                                              # full path
    df.to_csv(path, index=False, encoding="utf-8-sig")             # utf-8-sig opens cleanly in Excel
    print("[data]", path)                                          # log it


def dump_artifact(text, name):                                     # persist a text artefact
    """Write a text artefact into data/lecture11/artifacts."""       # short docstring
    path = OUT / name                                              # full path
    path.write_text(text, encoding="utf-8")                        # write UTF-8
    print("[artifact]", path)                                      # log it


def write_json(obj, name):                                         # persist a dictionary
    """Serialise a dictionary to artifacts/<name>."""               # short docstring
    dump_artifact(json.dumps(obj, ensure_ascii=False, indent=2), name)   # pretty, keep Chinese


# ---------------------------------------------------------------- 1) the synthetic corpus
CORPUS = [                                                              # continuation of the call
    "跨境电商的物流时效是影响复购率的关键因素之一",                                           # continuation of the call
    "大数据分析可以帮助企业识别高价值客户并优化投放预算",                                        # continuation of the call
    "大语言模型在客户服务场景中适合做意图识别与话术生成",                                        # continuation of the call
    "退货率上升往往与商品描述不符和物流延迟同时出现",                                          # continuation of the call
    "人工智能可以自动生成商品标题和详情页文案",                                             # continuation of the call
    "数据质量决定了分析结论的可信度与可复现性",                                             # continuation of the call
    "跨境贸易需要关注汇率波动合规要求与本地化运营",                                           # continuation of the call
    "模型评测需要区分准确率拒答率和格式合规率",                                             # continuation of the call
    "检索增强生成可以降低幻觉并给出可核查的引用",                                            # continuation of the call
    "提示工程的核心是把任务写清楚并给出输出格式",                                            # continuation of the call
    "外贸客户开发需要结合海关数据与行业展会信息",                                            # continuation of the call
    "智能客服的护栏规则应当在生成之后再做一次校验",                                           # continuation of the call
    "预测模型的价值取决于它在决策中的可执行性",                                             # continuation of the call
    "小样本场景下few-shot提示通常优于零样本",                                          # continuation of the call
    "生成式模型不会真正理解因果只会拟合文本分布",                                            # continuation of the call
    "企业部署大模型时需要评估数据边界与合规风险",                                            # continuation of the call
]                                                                       # continuation of the call

# A small bilingual vocabulary keeps the n-gram model easy to read.
def build_vocab(sentences):                                        # build the vocabulary
    """Return a sorted list of whitespace/single-character tokens seen in the corpus."""   # short docstring
    tokens = set()                                                 # the vocabulary
    for s in sentences:                                            # loop over sentences
        tokens.update(tokenize(s))                                  # split and add
    return sorted(tokens)                                          # a stable order for reproducibility


def tokenize(text):                                                # a deliberately simple tokenizer
    """Split Chinese into single characters and English into lower-case words."""   # short docstring
    out = []                                                       # the token list
    for chunk in re.findall(r"[A-Za-z]+|\d+|[\u4e00-\u9fff]|[^\sA-Za-z\d\u4e00-\u9fff]", text):   # three classes
        out.append(chunk.lower() if chunk.isascii() and chunk.isalpha() else chunk)   # normalise ASCII words
    return out                                                     # the tokens


VOCAB = build_vocab(CORPUS)                                        # the shared vocabulary
VOCAB_INDEX = {tok: i for i, tok in enumerate(VOCAB)}               # token to index


# ---------------------------------------------------------------- 2) the local chat model
class LocalChatModel:                                                   # continuation of the call
    """A word-bigram language model that mimics the OpenAI chat interface."""   # short docstring

    def __init__(self, corpus=None, order=2, seed=SEED):            # constructor
        """Fit the bigram counts on a small corpus."""               # short docstring
        self.corpus = corpus or CORPUS                              # the training text
        self.order = order                                          # 2 = bigram, 3 = trigram
        self.vocab = build_vocab(self.corpus)                        # the vocabulary
        self.counts = {}                                            # context -> {next: count}
        self.totals = {}                                            # context -> total count
        for sentence in self.corpus:                               # loop over sentences
            toks = tokenize(sentence)                                # the tokens
            padded = ["<s>"] * (order - 1) + toks + ["</s>"]          # pad both ends
            for i in range(order - 1, len(padded)):                  # loop over positions
                ctx = tuple(padded[i - order + 1:i])                  # the context tuple
                nxt = padded[i]                                      # the next token
                self.counts.setdefault(ctx, {})                       # ensure the dict
                self.counts[ctx][nxt] = self.counts[ctx].get(nxt, 0) + 1   # increment
                self.totals[ctx] = self.totals.get(ctx, 0) + 1        # increment the total
        self.rng = np.random.default_rng(seed)                       # the sampler

    def logits(self, prefix):                                      # the raw next-token scores
        """Return a dict from token to count-based score given a prefix string."""   # short docstring
        toks = tokenize(prefix)                                     # tokenise the prefix
        ctx = tuple((["<s>"] * (self.order - 1) + toks)[-(self.order - 1):])   # the last context
        table = self.counts.get(ctx, {})                            # the observed continuations
        if not table:                                              # fall back to a unigram estimate
            flat = {}                                              # the unigram table
            for counts in self.counts.values():                     # loop over all contexts
                for tok, c in counts.items():                       # loop over tokens
                    flat[tok] = flat.get(tok, 0) + c                 # accumulate
            total = sum(flat.values()) or 1                          # the denominator
            return {tok: math.log(c / total) for tok, c in flat.items()}   # log probabilities
        total = sum(table.values())                                 # the denominator
        return {tok: math.log(c / total) for tok, c in table.items()}   # log probabilities

    def probs(self, prefix, temperature=1.0, top_p=1.0):            # temperature and nucleus
        """Return a probability dict after temperature scaling and top-p truncation."""   # short docstring
        logits = self.logits(prefix)                                 # the raw scores
        if not logits:                                              # nothing to sample
            return {"</s>": 1.0}                                    # end the sequence
        temp = max(float(temperature), 1e-3)                         # guard against zero
        scaled = {t: v / temp for t, v in logits.items()}             # temperature scaling
        m = max(scaled.values())                                    # numerical stability
        exps = {t: math.exp(v - m) for t, v in scaled.items()}        # exponentiate
        z = sum(exps.values())                                      # the partition function
        probs = {t: v / z for t, v in exps.items()}                  # normalise
        if top_p < 1.0:                                             # nucleus sampling
            ordered = sorted(probs.items(), key=lambda kv: -kv[1])    # sort by probability
            kept, cumulative = {}, 0.0                               # accumulators
            for tok, p in ordered:                                  # walk down the sorted list
                kept[tok] = p                                       # keep the token
                cumulative += p                                     # grow the mass
                if cumulative >= top_p:                             # stop at the nucleus
                    break                                           # done
            z2 = sum(kept.values()) or 1.0                            # renormalise
            probs = {t: p / z2 for t, p in kept.items()}             # the truncated distribution
        return probs                                               # the final distribution

    def entropy(self, prefix, temperature=1.0):                     # the uncertainty of the next token
        """Return the Shannon entropy of the next-token distribution."""   # short docstring
        probs = self.probs(prefix, temperature)                      # the distribution
        return float(-sum(p * math.log(p) for p in probs.values() if p > 0))   # -sum p log p

    def sample(self, prefix, temperature=1.0, top_p=1.0, max_tokens=24):   # generate text
        """Return a generated continuation using temperature and top-p sampling."""   # short docstring
        out = tokenize(prefix)                                      # start from the prefix
        text = []                                                   # the produced tokens
        for _ in range(max_tokens):                                 # bounded generation
            probs = self.probs("".join(out + text), temperature, top_p)   # next-token distribution
            if not probs:                                           # nothing to add
                break                                               # stop
            toks = list(probs.keys())                                # the candidates
            weights = np.array([probs[t] for t in toks])             # their probabilities
            nxt = str(self.rng.choice(toks, p=weights / weights.sum()))   # sample one
            if nxt == "</s>":                                        # end of sentence
                break                                               # stop
            text.append(nxt)                                        # append it
        return "".join(text)                                        # return the text

    def chat(self, messages, temperature=0.7, max_tokens=24, top_p=1.0):   # the OpenAI-shaped call
        """Return a chat-completion style dict for a list of role/content messages."""   # short docstring
        prompt = "".join(m.get("content", "") for m in messages)      # flatten the conversation
        content = self.sample(prompt, temperature, top_p, max_tokens)   # generate
        return {"model": "local-ngram-1",                            # the model name
                "choices": [{"index": 0,                                # one choice
                             "message": {"role": "assistant", "content": content},   # the reply
                             "finish_reason": "stop"}],                  # why it stopped
                "usage": {"prompt_tokens": len(tokenize(prompt)),       # the reported usage
                          "completion_tokens": len(tokenize(content)),  # the completion
                          "total_tokens": len(tokenize(prompt)) + len(tokenize(content))}}   # the sum


MODEL = None                                                        # lazily built singleton


def get_model():                                                    # build once and reuse
    """Return the shared LocalChatModel instance."""                 # short docstring
    global MODEL                                                    # the module-level cache
    if MODEL is None:                                               # first call
        MODEL = LocalChatModel()                                    # fit it
    return MODEL                                                    # return it


# ---------------------------------------------------------------- 3) a synthetic document library for RAG
TOPICS = ["跨境物流", "退货率", "客户流失", "广告投放", "汇率风险", "数据质量", "模型评测", "提示工程"]   # continuation of the call


def make_documents(n=40):                                          # build the retrieval corpus
    """Return a small document library with ids, titles and bodies."""   # short docstring
    rows = []                                                       # one row per document
    for i in range(n):                                             # loop over documents
        topic = TOPICS[i % len(TOPICS)]                              # rotate through the topics
        body = "%s 相关说明：%s。%s。" % (                                     # continuation of the call
            topic, CORPUS[i % len(CORPUS)], CORPUS[(i * 3 + 1) % len(CORPUS)])   # a two-sentence body
        rows.append({"doc_id": "D%03d" % (i + 1),                    # the identifier
                     "title": "%s 手册 第 %d 节" % (topic, i // len(TOPICS) + 1),   # the title
                     "body": body,                                    # the text
                     "topic": topic})                                 # the topic tag
    return pd.DataFrame(rows)                                       # return the frame


def make_qa_set(docs, n=60):                                       # build a question set with answers
    """Return questions whose answer is a specific document, for retrieval evaluation."""   # short docstring
    rows = []                                                       # one row per question
    for i in range(n):                                             # loop over the questions
        doc = docs.iloc[i % len(docs)]                               # the gold document
        q = "%s 的主要结论是什么？（编号 %d）" % (doc["topic"], i + 1)   # a templated question
        rows.append({"q_id": "Q%03d" % (i + 1), "question": q,        # the question
                     "gold_doc": doc["doc_id"], "topic": doc["topic"]})   # the gold answer
    return pd.DataFrame(rows)                                       # return the frame
