"""l8_12_faiss_ann.py -- approximate nearest neighbours with FAISS: recall versus speed.

Every line carries an English comment. The script indexes a few thousand vectors with an
exact index and with an inverted-file index, measures recall against the exact answer and
the query throughput, so the deck can show the recall-latency trade-off that governs every
vector database.
"""
import time                                                        # measure query throughput
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # two panels
import faiss                                                       # the vector search library
from _common_l8 import (make_highdim, save, dump, style_axes,      # shared helpers
                        annotated, OKABE)                               # shared plotting helpers

# ---------------------------------------------------------------- 1) the vectors
frame, labels = make_highdim(4000, 32, 6)                          # four thousand vectors in 32 dimensions
X = frame.values.astype("float32")                                 # FAISS wants float32
print("corpus:", X.shape)                                          # report the corpus
queries = X[:120].copy()                                           # use the first rows as queries
k = 10                                                             # neighbours per query

# ---------------------------------------------------------------- 2) the exact index
exact = faiss.IndexFlatL2(X.shape[1])                              # brute-force L2 index
exact.add(X)                                                       # add the corpus
started = time.perf_counter()                                      # start the clock
D_exact, I_exact = exact.search(queries, k)                        # exact search
exact_ms = (time.perf_counter() - started) * 1000 / len(queries)     # milliseconds per query
print("exact: %.2f ms per query" % exact_ms)                        # report

# ---------------------------------------------------------------- 3) the approximate index
n_lists = 40                                                       # number of inverted lists
quantizer = faiss.IndexFlatL2(X.shape[1])                          # the coarse quantiser
ivf = faiss.IndexIVFFlat(quantizer, X.shape[1], n_lists, faiss.METRIC_L2)   # inverted-file index
ivf.train(X)                                                       # train the coarse quantiser
ivf.add(X)                                                         # add the corpus
rows = []                                                          # one row per nprobe setting
for nprobe in [1, 2, 4, 8, 16, 40]:                                # probes per query
    ivf.nprobe = nprobe                                            # set the search breadth
    started = time.perf_counter()                                  # start the clock
    D_ann, I_ann = ivf.search(queries, k)                          # approximate search
    ms = (time.perf_counter() - started) * 1000 / len(queries)      # milliseconds per query
    recall = float(np.mean([len(set(I_exact[i]).intersection(set(I_ann[i]))) / k for i in range(len(queries))]))   # recall at k
    rows.append({"nprobe": nprobe, "recall_at_10": round(recall, 4),   # recall
                 "ms_per_query": round(ms, 3), "speedup": round(exact_ms / ms, 2)})   # latency and speedup
    print("nprobe=%2d | recall %.3f | %.3f ms | speedup x%.1f" % (nprobe, recall, ms, exact_ms / ms))   # report
sweep = pd.DataFrame(rows)                                         # assemble the table
dump(sweep, "l8_faiss_sweep.csv")                                  # persist it

# ---------------------------------------------------------------- 4) the figure
fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))                # two panels
axes[0].plot(sweep["ms_per_query"], sweep["recall_at_10"], "-o", ms=6, color=OKABE[5])   # recall against latency
for _, row in sweep.iterrows():                                    # annotate each configuration
    axes[0].annotate("nprobe=%d" % row["nprobe"], (row["ms_per_query"], row["recall_at_10"]),   # label
                     xytext=(6, -8), textcoords="offset points", fontsize=7.2)   # small offset
style_axes(axes[0], "Recall versus latency (ANN trade-off)", "milliseconds per query", "recall at 10")   # style
axes[0].axhline(1.0, ls="--", lw=1.0, color="#66717D")             # perfect recall reference
axes[1].bar(sweep["nprobe"].astype(str), sweep["speedup"], color=OKABE[2], alpha=0.9)   # speedup bars
style_axes(axes[1], "Speed-up over the exact index", "nprobe", "speed-up (x)")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_ann.png")                                           # save the figure
print("done: exact versus approximate nearest neighbours")          # closing log
