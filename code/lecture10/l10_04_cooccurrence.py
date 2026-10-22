"""l10_04_cooccurrence.py -- the co-occurrence matrix, its SVD, and the GloVe idea.

Every line carries an English comment. The script builds a window co-occurrence matrix by
hand, applies a truncated SVD to get dense vectors, and contrasts count-based vectors with
the prediction-based ones from lesson 10.3 so the deck can explain GloVe's motivation.
"""
import numpy as np                                                 # arrays for the matrix
import pandas as pd                                                # tabular data
import matplotlib.pyplot as plt                                    # the heatmap
import seaborn as sns                                              # heatmap plotting
import jieba                                                       # Chinese segmentation
from sklearn.decomposition import TruncatedSVD                     # the sparse SVD
from scipy.sparse import csr_matrix                                # sparse matrix container
from _common_l10 import (DAT, save, dump, style_axes, annotated, OKABE)   # shared helpers

# ---------------------------------------------------------------- 1) the corpus and vocabulary
reviews = pd.read_csv(DAT / "l10_reviews.csv")                     # the bilingual reviews
def cut(text):                                                     # tokenise one document
    """Return a list of tokens for a review."""                     # short docstring
    if any("\u4e00" <= ch <= "\u9fff" for ch in str(text)):         # Chinese text
        return [t for t in jieba.lcut(str(text)) if t.strip()]      # segment with jieba
    return str(text).lower().split()                               # otherwise split on whitespace

docs = [cut(t) for t in reviews["text"]]                           # tokenised documents
from collections import Counter                                    # count tokens
freq = Counter(tok for d in docs for tok in d)                      # global frequency
vocab = [w for w, _ in freq.most_common(60)]                       # the sixty most frequent tokens
index = {w: i for i, w in enumerate(vocab)}                        # word to row index
print("vocabulary for the matrix:", len(vocab))                     # report

# ---------------------------------------------------------------- 2) window co-occurrence by hand
window = 2                                                         # context window on each side
X = np.zeros((len(vocab), len(vocab)), dtype=float)                 # the co-occurrence matrix
for doc in docs:                                                   # loop over the documents
    for i, word in enumerate(doc):                                 # loop over the positions
        if word not in index:                                      # skip out-of-vocabulary tokens
            continue                                               # next token
        for j in range(max(0, i - window), min(len(doc), i + window + 1)):   # the context window
            if j == i:                                             # skip the centre token
                continue                                           # next position
            other = doc[j]                                         # the context token
            if other in index:                                     # keep only vocabulary tokens
                X[index[word], index[other]] += 1.0                 # increment the count
print("co-occurrence matrix:", X.shape, "| non-zero share: %.3f" % (X > 0).mean())   # report
dump(pd.DataFrame(X.astype(int), index=vocab, columns=vocab), "l10_cooccurrence.csv")   # persist

# ---------------------------------------------------------------- 3) truncated SVD to dense vectors
svd = TruncatedSVD(n_components=10, random_state=42).fit(csr_matrix(X))   # ten dense dimensions
dense = svd.transform(csr_matrix(X))                               # the dense vectors
explained = svd.explained_variance_ratio_                          # share of variance per component
print("SVD explained variance (first five):", np.round(explained[:5], 3))   # report
dump(pd.DataFrame({"component": np.arange(1, len(explained) + 1), "share": np.round(explained, 4)}),   # the table
     "l10_svd_variance.csv")                                       # persist
vec = pd.DataFrame(dense, index=vocab)                             # the vectors as a frame
dump(vec.round(4).reset_index().rename(columns={"index": "token"}), "l10_svd_vectors.csv")   # persist

# ---------------------------------------------------------------- 4) cosine similarities from the count space
def cosine(a, b):                                                  # cosine similarity of two rows
    """Return the cosine similarity between two count rows."""      # short docstring
    na, nb = np.linalg.norm(a), np.linalg.norm(b)                  # the two norms
    return float(a @ b / max(na * nb, 1e-12))                      # the normalised dot product
pairs = [("物流", "发货"), ("质量", "做工"), ("客服", "售后"), ("价格", "性价比"), ("物流", "质量")]   # pairs to score
rows = []                                                          # similarity records
for a, b in pairs:                                                 # loop over the pairs
    if a in index and b in index:                                  # only when both are in the matrix
        rows.append({"pair": "%s-%s" % (a, b), "cosine_counts": round(cosine(X[index[a]], X[index[b]]), 4),   # count space
                     "cosine_svd": round(cosine(dense[index[a]], dense[index[b]]), 4)})   # SVD space
        print("%s: %.3f (counts) | %.3f (svd)" % (rows[-1]["pair"], rows[-1]["cosine_counts"], rows[-1]["cosine_svd"]))   # report
sims = pd.DataFrame(rows)                                          # assemble the table
dump(sims, "l10_cooccurrence_similarity.csv")                      # persist

# ---------------------------------------------------------------- 5) the figures
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4))                # two panels
sub = X[:24, :24]                                                  # the head of the matrix for the heatmap
sns.heatmap(np.log1p(sub), cmap="RdPu", cbar=True, ax=axes[0], xticklabels=vocab[:24], yticklabels=vocab[:24])   # log counts
axes[0].tick_params(axis="x", rotation=70, labelsize=6.4)          # small x labels
axes[0].tick_params(axis="y", labelsize=6.4)                       # small y labels
axes[0].set_title("Window co-occurrence (log counts)")             # panel title
axes[1].plot(np.arange(1, len(explained) + 1), explained, "-o", ms=5, color=OKABE[5])   # scree
style_axes(axes[1], "SVD explains most of the count variance", "component", "variance share")   # shared style
annotated(axes[1], "a few components carry the structure", "upper right")   # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_cooccurrence.png")                                  # save the figure

# ---------------------------------------------------------------- 6) the count versus prediction figure
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))                # two panels
xpos = np.arange(len(sims))                                        # bar positions
axes[0].bar(xpos - 0.19, sims["cosine_counts"], width=0.38, color=OKABE[1], alpha=0.9, label="count space")   # counts
axes[0].bar(xpos + 0.19, sims["cosine_svd"], width=0.38, color=OKABE[2], alpha=0.9, label="SVD space")   # svd
axes[0].set_xticks(xpos, sims["pair"], rotation=25, fontsize=7.2)   # pair labels
style_axes(axes[0], "Same pairs, two vector spaces", "", "cosine")   # shared style
axes[0].legend(fontsize=7.4, loc="upper right")                     # compact legend
methods = ["counts\n(GloVe/LSA)", "prediction\n(word2vec)"]          # the two families
axes[1].bar(methods, [1.0, 1.0], color=[OKABE[4], OKABE[3]], alpha=0.9)   # a schematic comparison
axes[1].set_yticks([])                                             # no numeric axis
style_axes(axes[1], "Two ways to get word vectors", "", "")         # shared style
annotated(axes[1], "GloVe combines both: counts + weighted least squares", "upper left")   # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_count_vs_prediction.png")                           # save the comparison figure
print("done: co-occurrence, SVD and the GloVe motivation")          # closing log
