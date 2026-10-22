"""l8_13_topics.py -- topic modelling without a transformer: NMF and LDA on a synthetic corpus.

Every line carries an English comment. The script generates a small corpus with three
latent topics, fits NMF and Latent Dirichlet Allocation, prints the top words per topic,
and draws the topic-word matrix so the deck can show what "topic modelling" actually
returns before introducing BERTopic.
"""
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # two panels
from sklearn.decomposition import NMF, LatentDirichletAllocation   # the two factorisers
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer   # the two encoders
from _common_l8 import (RNG, save, dump, style_axes, annotated, OKABE)   # shared helpers

# ---------------------------------------------------------------- 1) a synthetic corpus with three topics
TOPICS = {                                                         # three topic word pools
    "travel": ["flight", "hotel", "beach", "ticket", "tour", "island",   # travel words
               "passport", "luggage", "resort", "trip"],                # more travel words
    "tech": ["laptop", "battery", "screen", "chip", "software", "update",   # technology words
             "wifi", "gadget", "firmware", "camera"],                   # more technology words
    "food": ["noodle", "spicy", "restaurant", "dessert", "coffee", "menu",   # food words
             "delivery", "recipe", "snack", "breakfast"],                # more food words
}                                                                       # end of the topic word pools
docs = []                                                          # the corpus
true_topic = []                                                    # the generating topic
for i in range(360):                                               # 360 documents
    topic = list(TOPICS)[i % 3]                                     # round-robin topics
    words = RNG.choice(TOPICS[topic], size=RNG.integers(8, 18))      # sample the topic words
    filler = ["today", "people", "really", "think", "good", "make"]   # neutral filler words
    words = list(words) + list(RNG.choice(filler, size=RNG.integers(1, 4)))   # add filler
    RNG.shuffle(words)                                              # shuffle the order
    docs.append(" ".join(str(w) for w in words))                    # one document
    true_topic.append(topic)                                        # remember the truth
print("documents:", len(docs), "| vocabulary size:", len({w for d in docs for w in d.split()}))   # report

# ---------------------------------------------------------------- 2) two factorisations
tfidf = TfidfVectorizer().fit_transform(docs)                       # TF-IDF for NMF
counts = CountVectorizer().fit_transform(docs)                      # counts for LDA
vocab = CountVectorizer().fit(docs).get_feature_names_out()         # the vocabulary
nmf = NMF(n_components=3, random_state=42, init="nndsvda").fit(tfidf)   # non-negative factorisation
lda = LatentDirichletAllocation(n_components=3, random_state=42, learning_method="batch").fit(counts)   # LDA
def top_words(model, words, n=6):                                  # print the top words per component
    """Return the n highest-weight words for every component of a topic model."""   # short docstring
    out = []                                                       # one list per component
    for comp in model.components_:                                 # loop over the components
        idx = np.argsort(comp)[::-1][:n]                           # the highest weights
        out.append([str(words[i]) for i in idx])                    # the words themselves
    return out                                                     # return the lists
nmf_words = top_words(nmf, vocab)                                  # NMF topics
lda_words = top_words(lda, vocab)                                  # LDA topics
for i, (a, b) in enumerate(zip(nmf_words, lda_words)):              # report both
    print("topic %d | NMF: %s" % (i, ", ".join(a)))                 # NMF words
    print("         | LDA: %s" % ", ".join(b))                      # LDA words
topic_table = pd.DataFrame({"component": [1, 2, 3],                # component ids
                            "nmf_top_words": [", ".join(w) for w in nmf_words],   # NMF words
                            "lda_top_words": [", ".join(w) for w in lda_words]})   # LDA words
dump(topic_table, "l8_topics.csv")                                 # persist the table

# ---------------------------------------------------------------- 3) document-topic matrix and agreement
doc_topic = nmf.transform(tfidf)                                   # document weights (NMF)
dominant = doc_topic.argmax(axis=1)                                # the most likely component
purity = 0.0                                                       # compute purity properly
for comp in range(3):                                              # for every component
    mask = dominant == comp                                        # documents assigned to it
    if mask.sum():                                                 # avoid division by zero
        most_common = pd.Series([true_topic[i] for i in range(len(docs)) if mask[i]]).value_counts().iloc[0]   # majority truth
        purity += most_common                                             # accumulate
purity /= len(docs)                                                # normalise
print("cluster purity versus the generating topics: %.3f" % purity)   # report
dump(pd.DataFrame({"doc": np.arange(len(docs)), "true_topic": true_topic,   # one row per document
                   "nmf_component": dominant,                      # assigned component
                   "weight_1": np.round(doc_topic[:, 0], 3),       # weights
                   "weight_2": np.round(doc_topic[:, 1], 3),            # second component weight
                   "weight_3": np.round(doc_topic[:, 2], 3)}),     # the three weights
     "l8_doc_topics.csv")                                          # persist

# ---------------------------------------------------------------- 4) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                # three panels
im = axes[0].imshow(nmf.components_, cmap="RdPu", aspect="auto")   # topic-word matrix
axes[0].set_yticks([0, 1, 2], ["topic 1", "topic 2", "topic 3"], fontsize=8)   # component labels
axes[0].set_xlim(-0.5, 24.5)                                       # show only the head of the vocabulary
axes[0].set_xticks(range(0, 25, 2), [str(w) for w in vocab[:25:2]], rotation=70, fontsize=6.4)   # word labels
axes[0].grid(False)                                                # a heatmap needs no grid
axes[0].set_title("NMF: topic-word weights")                       # panel title
plt.colorbar(im, ax=axes[0], fraction=0.046)                       # colour bar
colours = {"travel": OKABE[0], "tech": OKABE[1], "food": OKABE[2]}   # one colour per true topic
for topic, colour in colours.items():                              # scatter the documents by truth
    mask = np.array([t == topic for t in true_topic])               # mask for this topic
    axes[1].scatter(doc_topic[mask, 0], doc_topic[mask, 1], s=8, color=colour, alpha=0.7, label=topic)   # points
style_axes(axes[1], "Documents in the first two topics", "topic 1 weight", "topic 2 weight")   # shared style
axes[1].legend(fontsize=7.2, loc="upper right")                     # compact legend
axes[2].bar(["purity"], [purity], color=OKABE[5], alpha=0.9)        # purity bar
axes[2].set_ylim(0, 1.05)                                          # keep room for the label
axes[2].text(0, purity, "%.3f" % purity, ha="center", va="bottom", fontsize=10)   # annotate
style_axes(axes[2], "Do the topics match the truth?", "", "purity")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_topics.png")                                        # save the figure
print("done: NMF and LDA topics on the synthetic corpus")           # closing log
