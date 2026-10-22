"""l10_01_preprocess.py -- Chinese segmentation, normalisation and the n-gram coverage curve.

Every line carries an English comment. The script builds the bilingual review and topic
corpora, runs jieba with and without a domain lexicon, shows the segmentation of a few
sentences, and measures how n-gram coverage grows with corpus size.
"""
import numpy as np                                                 # arrays for the coverage curve
import pandas as pd                                                # tabular data
import matplotlib.pyplot as plt                                    # the figures
import jieba                                                       # Chinese word segmentation
from collections import Counter                                    # count tokens and n-grams
from _common_l10 import (make_reviews_zh, make_reviews_en,         # shared helpers
                         make_topic_corpus_zh, make_lexicon_sentences,   # shared generators
                         save, dump, style_axes, annotated, OKABE)      # shared plotting helpers

# ---------------------------------------------------------------- 1) the corpora
zh = make_reviews_zh(600)                                          # Chinese reviews with labels
en = make_reviews_en(300)                                          # English reviews
topics = make_topic_corpus_zh(360)                                 # three-topic Chinese corpus
reviews = pd.concat([zh, en], ignore_index=True)                    # the full bilingual corpus
dump(reviews, "l10_reviews.csv")                                   # persist it
dump(topics, "l10_topics_corpus.csv")                              # persist the topic corpus
print("reviews:", len(reviews), "| Chinese:", (reviews["lang"] == "zh").sum())   # report
print("topic docs:", len(topics))                                   # report

# ---------------------------------------------------------------- 2) segmentation with and without a lexicon
stopwords = set("的 了 也 很 都 就 和 与 在 是 我 有 这 那 一个 还 要 会 说 去 来".split())   # a small stop list
def tokenize(text, use_lexicon=False):                             # tokenise one Chinese sentence
    """Return the tokens of a Chinese sentence, optionally with domain terms added."""   # short docstring
    if use_lexicon:                                                # add the domain terms first
        for term in ["华为Mate60Pro", "比亚迪海獭", "vivoX100", "李宁䨻科技"]:   # product names
            jieba.add_word(term)                                   # teach the segmenter the term
    return [t for t in jieba.lcut(text) if t.strip() and t not in stopwords]   # cut and drop stopwords
samples = make_lexicon_sentences()                                 # sentences with domain terms
rows = []                                                          # one row per sentence
for s in samples:                                                  # loop over the sentences
    plain = " / ".join(jieba.lcut(s))                              # default segmentation
    rows.append({"sentence": s, "default_seg": plain})              # record it
dump(pd.DataFrame(rows), "l10_segmentation_demo.csv")               # persist the demo
for r in rows:                                                     # report the segmentation
    print(r["sentence"], "->", r["default_seg"])                   # the cut result

# ---------------------------------------------------------------- 3) token statistics
all_tokens = []                                                    # every token in the Chinese corpus
for text in zh["text"]:                                            # loop over the reviews
    all_tokens.extend(tokenize(text))                              # tokenise and collect
counts = Counter(all_tokens)                                       # token frequency
print("tokens:", len(all_tokens), "| vocabulary:", len(counts))     # report
top = pd.DataFrame(counts.most_common(20), columns=["token", "count"])   # the twenty most frequent
dump(top, "l10_top_tokens.csv")                                    # persist

# ---------------------------------------------------------------- 4) n-gram coverage versus corpus size
sizes = [10, 25, 50, 100, 200, 400, 600]                           # increasing training sizes
test_docs = zh["text"].iloc[-100:]                                 # a fixed held-out set
test_tokens = [t for text in test_docs for t in tokenize(text)]     # its tokens
test_bigrams = set((test_tokens[i], test_tokens[i + 1]) for i in range(len(test_tokens) - 1))   # its bigram types
coverage = []                                                      # share of test bigrams covered
for size in sizes:                                                 # loop over the training sizes
    train_docs = zh["text"].iloc[:size]                             # the first `size` documents
    train_tokens = [t for text in train_docs for t in tokenize(text)]   # their tokens
    train_bigrams = set((train_tokens[i], train_tokens[i + 1]) for i in range(len(train_tokens) - 1))   # their bigram types
    covered = len(test_bigrams & train_bigrams) / max(len(test_bigrams), 1)   # share of test types seen
    coverage.append({"docs": size, "test_bigram_coverage": round(float(covered), 4),   # record the coverage
                     "train_bigram_types": len(train_bigrams)})     # and the number of training types
    print("docs %3d | test bigram coverage %.3f | train types %d" % (size, covered, len(train_bigrams)))   # report
cov = pd.DataFrame(coverage)                                       # assemble the table
dump(cov, "l10_ngram_coverage.csv")                                # persist

# ---------------------------------------------------------------- 5) the figures
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                # three panels
labels = reviews["label"].value_counts()                            # label balance
axes[0].bar(labels.index, labels.values, color=[OKABE[2], OKABE[5], OKABE[3]], alpha=0.9)   # the bars
style_axes(axes[0], "Review labels in the corpus", "", "documents")   # shared style
axes[1].plot(cov["docs"], cov["test_bigram_coverage"], "-o", ms=5, color=OKABE[5])   # coverage curve
style_axes(axes[1], "Test bigram coverage versus training size", "training documents", "share of test bigrams seen")   # shared style
annotated(axes[1], "more data helps, but slowly", "lower right")            # annotate the shape
axes[2].barh(top["token"][::-1], top["count"][::-1], color=OKABE[3], alpha=0.9)   # top tokens
style_axes(axes[2], "Most frequent tokens", "count", "")            # shared style
axes[2].tick_params(axis="y", labelsize=7.0)                        # small labels
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_text_uses.png")                                     # save the corpus figure

# ---------------------------------------------------------------- 6) the linguistic levels figure
fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))                # two panels
levels = ["音系\nphonetics", "形态\nmorphology", "词性\nPOS", "句法\nsyntax", "语义\nsemantics", "语篇\ndiscourse"]   # six levels
examples = ["品牌名的读音", "“䨻”这种生僻字", "“很”是副词", "合同条款的从句", "“便宜”是好是坏", "多句指代"]   # business examples
axes[0].barh(levels[::-1], np.arange(len(levels))[::-1] + 1, color=OKABE[4], alpha=0.85)   # a schematic ladder
axes[0].set_xticks([])                                             # no numeric axis needed
style_axes(axes[0], "Six levels of linguistic knowledge", "", "")   # shared style
axes[1].axis("off")                                               # the example list needs no axes
for i, (level, example) in enumerate(zip(levels, examples)):        # write the examples
    axes[1].text(0.02, 0.92 - i * 0.16, "%s → %s" % (level.replace("\n", " "), example), fontsize=9)   # one line each
axes[1].set_title("Each level has a business example")             # panel title
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_levels.png")                                        # save the levels figure

# ---------------------------------------------------------------- 7) the ambiguity figure
fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.6))                # two panels
axes[0].axis("off")                                                # text only
sentence = "我看好这个手机的拍照"                                    # an ambiguous sentence
axes[0].text(0.5, 0.8, sentence, ha="center", fontsize=14)          # the sentence
axes[0].text(0.5, 0.55, "读法一：我 看好 这个 手机 的 拍照", ha="center", fontsize=10, color=OKABE[2])   # reading one
axes[0].text(0.5, 0.38, "读法二：我 看 好 这个 手机 的 拍照", ha="center", fontsize=10, color=OKABE[5])   # reading two
axes[0].text(0.5, 0.12, "分词歧义会直接改变下游特征", ha="center", fontsize=9, color="#66717D")   # the consequence
axes[0].set_title("Word segmentation ambiguity")                   # panel title
sents = ["把它放到桌上", "他说的话我听不懂", "这家店的东西便宜"]                              # three ambiguous sentences
counts_bar = [2, 2, 2]                                             # two readings each (schematic)
axes[1].bar(range(3), counts_bar, color=OKABE[3], alpha=0.9)       # schematic bar chart
axes[1].set_xticks(range(3), ["句 1", "句 2", "句 3"], fontsize=8)   # sentence labels
style_axes(axes[1], "Ambiguous sentences you can collect", "", "readings")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_ambiguity.png")                                     # save the ambiguity figure
print("done: segmentation, token statistics and coverage")           # closing log
