"""02_datatype_categorical.py -- categorical data (nominal vs ordinal): samples and plots."""  # module comment
import numpy as np                                                    # numeric computation
import pandas as pd                                                   # table handling
import matplotlib.pyplot as plt                                       # plotting
from _common import RNG, save, dump_csv                               # shared helpers

# ---------------------------------------------------------------- 1) generate samples
industries = ["电子", "医药", "机械", "消费", "金融", "能源"]           # nominal categories: industries, no natural order
ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "C"]                   # ordinal categories: credit ratings, clearly ordered
n = 300                                                               # sample size: 300 listed firms
ind = RNG.choice(industries, size=n, p=[0.28, 0.13, 0.18, 0.16, 0.15, 0.10])  # draw industries with the given probabilities (nominal)
rate = RNG.choice(ratings, size=n, p=[0.03, 0.09, 0.18, 0.30, 0.22, 0.13, 0.05])  # draw ratings with the given probabilities (ordinal)
year = RNG.choice([2023, 2024, 2025], size=n)                          # keep the year for cross tabulation
df = pd.DataFrame({"industry": ind, "rating": rate, "year": year})     # assemble the sample table
dump_csv(df, "categorical_samples.csv")                                # dump as CSV into the repo data/ folder

# ---------------------------------------------------------------- 2) visualization
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))                    # one row, three columns: nominal bars / ordinal bars / cross heatmap
counts_ind = df["industry"].value_counts().reindex(industries)         # count each industry, keeping the given order
axes[0].bar(industries, counts_ind.values, color="#8A0C3C", alpha=0.85)  # left: frequency bars of the nominal categories
axes[0].set_title("名义型：行业分布")                                    # left panel title
axes[0].tick_params(axis="x", rotation=30)                             # rotate x labels to avoid overlap

counts_rate = df["rating"].value_counts().reindex(ratings)              # count each rating in AAA to C order
axes[1].bar(ratings, counts_rate.values, color="#C9A227", alpha=0.9)    # middle: frequency bars of the ordinal categories
axes[1].set_title("有序型：信用评级（顺序有意义）")                        # middle panel title

cross = pd.crosstab(df["industry"], df["rating"]).reindex(index=industries, columns=ratings)  # industry by rating cross table
im = axes[2].imshow(cross.values, cmap="RdPu", aspect="auto")           # right: heatmap of the cross table
axes[2].set_xticks(range(len(ratings)), ratings, fontsize=8)            # right panel x axis: rating ticks
axes[2].set_yticks(range(len(industries)), industries, fontsize=8)      # right panel y axis: industry ticks
axes[2].set_title("类别交叉：行业 × 评级")                               # right panel title
axes[2].grid(False)                                                     # a heatmap needs no grid lines
plt.colorbar(im, ax=axes[2], fraction=0.046)                            # add a colour bar for the counts
plt.tight_layout()                                                      # adjust the layout
save(fig, "fig_categorical.png")                                        # save the figure
