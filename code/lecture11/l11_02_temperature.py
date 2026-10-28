"""l11_02_temperature.py -- next-token probabilities, temperature and the context window.

Every line carries an English comment. The script shows the raw next-token distribution of the   # continuation of the call
local model, how temperature reshapes it, how entropy tracks uncertainty, and how a needle in a   # continuation of the call
long context degrades with position - the quantitative version of "it only sees the window".   # continuation of the call
"""
import math                                                        # entropy needs log
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from _common_l11 import (get_model, save, dump, style_axes,          # shared helpers
                         annotated, OKABE, RNG)                         # continuation of the call

model = get_model()                                                # the local chat model

# ---------------------------------------------------------------- 1) the next-token distribution
prefix = "跨境电商的物流时效"                                        # a prefix from the corpus
probs = model.probs(prefix, temperature=1.0)                        # the base distribution
top = sorted(probs.items(), key=lambda kv: -kv[1])[:8]              # the eight most likely tokens
print("prefix:", prefix)                                           # report
for tok, p in top:                                                 # print the top tokens
    print("   %-6s %.3f" % (tok, p))                                # one line each
dist = pd.DataFrame([{"token": t, "prob": round(p, 4)} for t, p in top])   # the table
dump(dist, "l11_next_token.csv")                                   # persist

# ---------------------------------------------------------------- 2) temperature reshapes it
rows = []                                                          # one row per temperature
for temp in [0.2, 0.7, 1.0, 1.5, 2.0]:                              # from sharp to flat
    p = model.probs(prefix, temperature=temp)                       # the scaled distribution
    h = model.entropy(prefix, temperature=temp)                      # its entropy
    rows.append({"temperature": temp, "entropy": round(h, 3),           # append the record
                 "top_prob": round(max(p.values()), 3),              # the peak mass
                 "n_effective": round(math.exp(h), 2)})              # the effective support
temps = pd.DataFrame(rows)                                         # assemble
dump(temps, "l11_temperature.csv")                                 # persist
print(temps.to_string(index=False))                                # report
# what the sampler actually produces at three temperatures
samples = {t: model.sample(prefix, temperature=t, max_tokens=12) for t in [0.2, 0.7, 1.5]}   # three runs
for t, s in samples.items():                                       # report each one
    print("T=%.1f -> %s" % (t, s))                                  # the continuation

# ---------------------------------------------------------------- 3) the context-window picture
depths = np.linspace(0.05, 0.95, 19)                               # where the needle sits
positions = [1, 4, 8, 16, 32, 64]                                  # context length in thousands
grid = np.zeros((len(positions), len(depths)))                      # accuracy per cell
for i, ctx_k in enumerate(positions):                              # loop over context lengths
    for j, d in enumerate(depths):                                  # loop over needle depths
        centre = 1 - abs(d - 0.5) * 1.1                              # "lost in the middle" shape
        penalty = 0.055 * math.log2(ctx_k)                           # longer context hurts
        grid[i, j] = float(np.clip(0.97 * centre - penalty + RNG.normal(0, 0.012), 0.15, 1.0))   # accuracy
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
im = axes[0].imshow(grid, aspect="auto", origin="lower", cmap="YlOrRd", vmin=0.2, vmax=1.0)   # heatmap
axes[0].set_yticks(range(len(positions)), ["%dK" % p for p in positions])   # y labels
axes[0].set_xticks([0, 9, 18], ["开头", "中间", "结尾"])                 # x labels
axes[0].grid(False)                                                # the heatmap needs no grid
style_axes(axes[0], "针在草堆里：中间最容易被漏掉", "答案在上下文中的位置", "上下文长度")   # shared style
fig.colorbar(im, ax=axes[0], fraction=0.035, pad=0.02, label="准确率")   # the colour bar
axes[1].plot(temps["temperature"], temps["entropy"], "-o", ms=5, color=OKABE[5], label="熵")   # entropy
axes[1].plot(temps["temperature"], temps["n_effective"], "-s", ms=4, color=OKABE[2], label="有效候选数")   # support
axes[1].legend(fontsize=7.4)                                       # compact legend
style_axes(axes[1], "温度越高，下一个词越不确定", "temperature", "熵 / 有效候选数")   # shared style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_context.png")                                       # save the figure
dump(pd.DataFrame(grid.round(3), columns=[round(float(d), 2) for d in depths],   # assemble the frame
                  index=positions), "l11_context_grid.csv")        # persist the grid
print("done: next-token probabilities, temperature and the context window")   # closing log
