"""l10_03_word2vec.py -- training word vectors with gensim and looking at what they learn.

Every line carries an English comment. The script trains a skip-gram model on the bilingual
review corpus, prints nearest neighbours and analogies, projects the vectors into two
dimensions, and draws the semantic clusters so the deck can show that meaning is geometry.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tabular data
import matplotlib.pyplot as plt                                    # the projection figure
import jieba                                                       # Chinese segmentation
from gensim.models import Word2Vec                                 # the word-vector model
from sklearn.decomposition import PCA                              # the two-dimensional projection
from _common_l10 import (DAT, make_analogy_pairs, save, dump,      # shared helpers
                         style_axes, annotated, OKABE)                  # shared plotting helpers

# ---------------------------------------------------------------- 1) build the tokenised corpus
reviews = pd.read_csv(DAT / "l10_reviews.csv")                     # the bilingual reviews
def cut(text):                                                     # tokenise one document
    """Return a list of tokens for a review."""                     # short docstring
    if any("\u4e00" <= ch <= "\u9fff" for ch in str(text)):         # Chinese text
        return [t for t in jieba.lcut(str(text)) if t.strip()]      # segment with jieba
    return str(text).lower().split()                               # otherwise split on whitespace

sentences = [cut(t) for t in reviews["text"]]                      # one token list per review
print("sentences:", len(sentences), "| avg length: %.1f" % np.mean([len(s) for s in sentences]))   # report

# ---------------------------------------------------------------- 2) train skip-gram
model = Word2Vec(sentences=sentences, vector_size=60, window=4, min_count=1,   # small vectors for speed
                 sg=1, negative=5, epochs=80, seed=20261111, workers=1)          # skip-gram with negative sampling
print("vocabulary:", len(model.wv), "| vector size:", model.wv.vector_size)   # report
dump(pd.DataFrame({"token": list(model.wv.index_to_key)}), "l10_w2v_vocab.csv")   # persist the vocabulary

# ---------------------------------------------------------------- 3) nearest neighbours and analogies
probes = ["物流", "质量", "客服", "价格", "发货", "退货", "满意", "失望"]   # words to explore
rows = []                                                          # neighbour records
for word in probes:                                                # loop over the probes
    if word not in model.wv:                                       # skip words outside the vocabulary
        continue                                                   # next probe
    neighbours = model.wv.most_similar(word, topn=5)               # the five nearest words
    rows.append({"word": word, "neighbours": " / ".join("%s(%.2f)" % (w, s) for w, s in neighbours)})   # record
    print("%s -> %s" % (word, rows[-1]["neighbours"]))             # report
dump(pd.DataFrame(rows), "l10_word_neighbours.csv")                # persist
analogy_rows = []                                                  # analogy results
for a, b, c, d in make_analogy_pairs():                            # loop over the analogy triples
    try:                                                           # the analogy needs all words in the vocabulary
        guess, score = model.wv.most_similar(positive=[b, c], negative=[a], topn=1)[0]   # a - b + c
        analogy_rows.append({"a": a, "b": b, "c": c, "expected": d, "got": guess,   # the record
                             "score": round(float(score), 3), "hit": guess == d})   # whether it matched
    except KeyError:                                               # a word is missing from the vocabulary
        analogy_rows.append({"a": a, "b": b, "c": c, "expected": d, "got": "-", "score": 0.0, "hit": False})   # record a miss
analogies = pd.DataFrame(analogy_rows)                             # assemble the table
dump(analogies, "l10_analogies.csv")                               # persist
print(analogies.to_string(index=False))                            # report

# ---------------------------------------------------------------- 4) the vector-space figure
words = [w for w in ["物流", "发货", "快递", "配送", "质量", "做工", "手感", "包装",    # four semantic groups
                     "客服", "售后", "态度", "退款", "价格", "性价比", "便宜", "贵"] if w in model.wv]   # groups
vectors = np.array([model.wv[w] for w in words])                   # the corresponding vectors
coords = PCA(n_components=2, random_state=42).fit_transform(vectors)   # project to two dimensions
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))                # two panels
colours = [OKABE[0], OKABE[0], OKABE[0], OKABE[0], OKABE[1], OKABE[1], OKABE[1], OKABE[1],   # one colour per group
           OKABE[2], OKABE[2], OKABE[2], OKABE[2], OKABE[4], OKABE[4], OKABE[4], OKABE[4]][:len(words)]   # group colours
axes[0].scatter(coords[:, 0], coords[:, 1], s=90, color=colours, alpha=0.9)   # the projected words
for i, w in enumerate(words):                                      # label every point
    axes[0].annotate(w, (coords[i, 0], coords[i, 1]), xytext=(5, 4), textcoords="offset points", fontsize=8)   # label
style_axes(axes[0], "Word vectors projected to two dimensions", "PC1", "PC2")   # shared style
annotated(axes[0], "similar words cluster", "upper left")          # annotate the structure
sims = pd.DataFrame([{"pair": "%s-%s" % (a, b), "cosine": float(model.wv.similarity(a, b))}   # cosine similarity per pair
                     for a, b in [("物流", "发货"), ("质量", "做工"), ("客服", "售后"), ("价格", "性价比"), ("物流", "质量")] if a in model.wv and b in model.wv])   # similarity pairs
axes[1].barh(sims["pair"][::-1], sims["cosine"][::-1], color=OKABE[3], alpha=0.9)   # cosine similarity bars
style_axes(axes[1], "Cosine similarity inside and across groups", "cosine", "")   # shared style
axes[1].tick_params(axis="y", labelsize=7.4)                        # small labels
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_word_vectors.png")                                  # save the vector figure

# ---------------------------------------------------------------- 5) the analogy figure
fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))                # two panels
hit_counts = analogies["hit"].value_counts()                        # how many analogies worked
axes[0].bar(["hit", "miss"], [int(hit_counts.get(True, 0)), int(hit_counts.get(False, 0))],   # hit and miss counts
            color=[OKABE[2], OKABE[5]], alpha=0.9)                  # the two bars
style_axes(axes[0], "Analogy accuracy on a tiny corpus", "", "count")   # shared style
axes[1].barh(analogies["a"] + " - " + analogies["b"] + " + " + analogies["c"], analogies["score"],   # scores
             color=OKABE[4], alpha=0.9)                             # horizontal bars
style_axes(axes[1], "Similarity score of the retrieved word", "cosine", "")   # shared style
axes[1].tick_params(axis="y", labelsize=7.0)                        # small labels
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_analogy.png")                                       # save the analogy figure
print("done: word2vec training, neighbours and analogies")          # closing log
