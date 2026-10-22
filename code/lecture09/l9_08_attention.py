"""l9_08_attention.py -- scaled dot-product attention and multi-head attention from scratch.

Every line carries an English comment. The script implements attention with NumPy, checks
the softmax rows sum to one, draws the attention weight matrix for a short sentence, and
compares single-head with multi-head so the deck can show what the weights mean.
"""
import numpy as np                                                 # arrays for the attention maths
import pandas as pd                                                # tabular weights
import matplotlib.pyplot as plt                                    # the heatmap
from _common_l9 import (RNG, save, dump, style_axes, annotated, OKABE)   # shared helpers

# ---------------------------------------------------------------- 1) a short "sentence" of token vectors
tokens = ["the", "cat", "sat", "on", "the", "mat"]                  # six tokens
d_model, d_k = 8, 8                                                # embedding and key dimensions
emb = RNG.normal(0, 0.6, (len(tokens), d_model))                   # synthetic embeddings
Wq = RNG.normal(0, 0.5, (d_model, d_k))                            # query projection
Wk = RNG.normal(0, 0.5, (d_model, d_k))                            # key projection
Wv = RNG.normal(0, 0.5, (d_model, d_k))                            # value projection
print("tokens:", len(tokens), "| d_model:", d_model, "| d_k:", d_k)   # report

# ---------------------------------------------------------------- 2) the attention function
def softmax_rows(z):                                               # row-wise softmax
    """Return the row-wise softmax of a score matrix."""            # short docstring
    z = z - z.max(axis=1, keepdims=True)                           # subtract the row maximum
    e = np.exp(z)                                                  # safe exponentials
    return e / e.sum(axis=1, keepdims=True)                        # normalise

def attention(Q, K, V):                                            # scaled dot-product attention
    """Return the attention weights and the context vectors."""     # short docstring
    scores = Q @ K.T / np.sqrt(K.shape[-1])                         # scaled scores
    weights = softmax_rows(scores)                                 # softmax over the keys
    return weights, weights @ V                                    # weights and the weighted values

Q, K, V = emb @ Wq, emb @ Wk, emb @ Wv                             # project the embeddings
weights, context = attention(Q, K, V)                              # single-head attention
row_sums = weights.sum(axis=1)                                     # every row should sum to one
print("row sums:", np.round(row_sums, 6))                           # report
assert np.allclose(row_sums, 1.0), "each attention row must sum to one"   # hard check
dump(pd.DataFrame(weights, index=tokens, columns=tokens).round(4), "l9_attention_weights.csv")   # persist

# ---------------------------------------------------------------- 3) multi-head attention
def multi_head(emb, h=2):                                          # split into h heads
    """Return per-head attention weights for the same input."""     # short docstring
    outputs = []                                                   # one entry per head
    for head in range(h):                                          # loop over the heads
        Wq_h = RNG.normal(0, 0.5, (d_model, d_k // 2))              # head-specific projections
        Wk_h = RNG.normal(0, 0.5, (d_model, d_k // 2))              # head-specific projections
        Wv_h = RNG.normal(0, 0.5, (d_model, d_k // 2))              # head-specific projections
        w, c = attention(emb @ Wq_h, emb @ Wk_h, emb @ Wv_h)        # attention within the head
        outputs.append((w, c))                                     # store it
    return outputs                                                 # return the list

heads = multi_head(emb, 2)                                         # two heads
print("head 1 argmax per token:", heads[0][0].argmax(axis=1).tolist())   # report what each token attends to
print("head 2 argmax per token:", heads[1][0].argmax(axis=1).tolist())   # report the second head

# ---------------------------------------------------------------- 4) complexity illustration
lengths = np.arange(4, 65, 4)                                      # sequence lengths
self_attn = lengths ** 2                                           # quadratic cost
recurrent = lengths                                                # linear cost
print("attention cost at n=64: %d vs recurrent %d" % (self_attn[-1], recurrent[-1]))   # report

# ---------------------------------------------------------------- 5) the figures
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9))                # three panels
im = axes[0].imshow(weights, cmap="RdPu", vmin=0, vmax=weights.max())   # attention weights
axes[0].set_xticks(range(len(tokens)), tokens, fontsize=8)          # key labels
axes[0].set_yticks(range(len(tokens)), tokens, fontsize=8)          # query labels
axes[0].grid(False)                                                # a heatmap needs no grid
axes[0].set_xlabel("key")                                          # axis label
axes[0].set_ylabel("query")                                        # axis label
axes[0].set_title("Self-attention weights")                        # panel title
plt.colorbar(im, ax=axes[0], fraction=0.046)                       # colour bar
axes[1].semilogy(lengths, self_attn, "-o", ms=4, color=OKABE[5], label="self-attention $O(n^2)$")   # quadratic
axes[1].semilogy(lengths, recurrent, "-s", ms=4, color=OKABE[2], label="RNN $O(n)$")   # linear
style_axes(axes[1], "Cost per token scales differently", "sequence length", "operations (log)")   # shared style
axes[1].legend(fontsize=7.4, loc="upper left")                     # compact legend
axes[2].bar(["head 1", "head 2"], [float(heads[0][0].max()), float(heads[1][0].max())],   # peak weight
            color=[OKABE[1], OKABE[3]], alpha=0.9)                  # one colour per head
style_axes(axes[2], "Different heads focus differently", "", "peak weight")   # shared style
annotated(axes[2], "heads learn different patterns", "upper right")   # annotate
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_attention.png")                                     # save the attention figure
print("done: scaled dot-product and multi-head attention")          # closing log
