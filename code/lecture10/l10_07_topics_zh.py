"""l10_07_topics_zh.py -- topic modelling on a Chinese corpus.

Every line carries an English comment. The script runs LDA on the three-topic Chinese corpus,
prints the top words of every topic, draws the topic-word heatmap, and scores how well the
topics recover the generating pools.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tabular data
import matplotlib.pyplot as plt                                    # the figures
import jieba                                                       # Chinese segmentation
from sklearn.decomposition import LatentDirichletAllocation        # LDA
from sklearn.feature_extraction.text import CountVectorizer        # the word counts
from _common_l10 import (DAT, save, dump, style_axes, annotated, OKABE)   # shared helpers

# ---------------------------------------------------------------- 1) the corpus and its tokens
corpus = pd.read_csv(DAT / "l10_topics_corpus.csv")                # the three-topic corpus
def cut(text):                                                     # tokenise one document
    """Return the tokens of a Chinese document."""                  # short docstring
    return [t for t in jieba.lcut(str(text)) if t.strip() and t not in "，。"]   # segment and drop punctuation
docs = [" ".join(cut(t)) for t in corpus["text"]]                   # pre-tokenised documents
print("documents:", len(docs), "| true topics:", corpus["true_topic"].nunique())   # report

# ---------------------------------------------------------------- 2) fit LDA
counts = CountVectorizer(max_features=200).fit_transform(docs)      # the document-term matrix
lda = LatentDirichletAllocation(n_components=3, random_state=42, learning_method="batch").fit(counts)   # three topics
vocab = CountVectorizer(max_features=200).fit(docs).get_feature_names_out()   # the vocabulary
print("vocabulary:", len(vocab), "| perplexity: %.1f" % lda.perplexity(counts))   # report
rows = []                                                          # topic-word rows
for ti, comp in enumerate(lda.components_):                        # loop over the topics
    order = np.argsort(comp)[::-1][:8]                              # the eight strongest words
    rows.append({"topic": "T%d" % (ti + 1), "top_words": " / ".join(vocab[i] for i in order)})   # record them
    print("T%d: %s" % (ti + 1, rows[-1]["top_words"]))               # report
topics = pd.DataFrame(rows)                                        # assemble the table
dump(topics, "l10_topics_zh.csv")                                  # persist

# ---------------------------------------------------------------- 3) document-topic weights and purity
doc_topic = lda.transform(counts)                                  # the weights per document
assigned = doc_topic.argmax(axis=1)                                # the dominant topic
put_rows = []                                                      # one row per document
for i in range(len(docs)):                                         # loop over the documents
    put_rows.append({"doc_id": corpus["doc_id"].iloc[i], "true_topic": corpus["true_topic"].iloc[i],   # the truth
                     "topic": "T%d" % (assigned[i] + 1), "weight": round(float(doc_topic[i].max()), 3)})   # the assignment
dump(pd.DataFrame(put_rows), "l10_doc_topics_zh.csv")              # persist
purity = 0.0                                                       # compute the cluster purity
for t in range(3):                                                 # loop over the topics
    mask = assigned == t                                           # documents assigned to this topic
    if mask.sum():                                                 # avoid empty topics
        purity += float(corpus["true_topic"][mask].value_counts().iloc[0])   # the majority count
purity /= len(docs)                                                # normalise
print("purity against the generating topics: %.3f" % purity)        # report

# ---------------------------------------------------------------- 4) the figures
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                # three panels
comp = lda.components_[:, :20]                                     # the head of the topic-word matrix
im = axes[0].imshow(comp, cmap="RdPu", aspect="auto")              # the heatmap
axes[0].set_yticks([0, 1, 2], ["T1", "T2", "T3"], fontsize=8)      # topic labels
axes[0].set_xticks(range(0, 20, 2), vocab[:20:2], rotation=70, fontsize=6.6)   # word labels
axes[0].grid(False)                                                # a heatmap needs no grid
axes[0].set_title("Topic-word weights (head of the vocabulary)")   # panel title
plt.colorbar(im, ax=axes[0], fraction=0.046)                       # colour bar
colours = {"数码": OKABE[0], "食品": OKABE[1], "服务": OKABE[2]}    # one colour per true topic
for topic, colour in colours.items():                              # scatter documents by truth
    mask = corpus["true_topic"] == topic                            # the documents of this pool
    axes[1].scatter(doc_topic[mask, 0], doc_topic[mask, 1], s=12, color=colour, alpha=0.75, label=topic)   # the points
style_axes(axes[1], "Documents in the first two topics", "T1 weight", "T2 weight")   # shared style
axes[1].legend(fontsize=7.2, loc="upper right")                     # compact legend
axes[2].bar(["purity"], [purity], color=OKABE[5], alpha=0.9)        # purity bar
axes[2].set_ylim(0, 1.05)                                          # keep room for the label
axes[2].text(0, purity, "%.3f" % purity, ha="center", va="bottom", fontsize=10)   # annotate
style_axes(axes[2], "Do the topics match the truth?", "", "purity")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_topics_zh.png")                                     # save the topic figure
print("done: LDA on the Chinese corpus")                            # closing log
