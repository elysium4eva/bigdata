"""l10_08_llm_agreement.py -- checking annotation agreement when an LLM does the labelling.

Every line carries an English comment. The script simulates repeated LLM labelling with
different temperatures, compares it with a human reference, and reports Cohen's kappa and
the agreement curve so the deck can show how to audit machine annotation.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tabular data
import matplotlib.pyplot as plt                                    # the figures
from sklearn.metrics import cohen_kappa_score                       # agreement metric
from _common_l10 import (DAT, save, dump, write_json, style_axes,   # shared helpers
                         annotated, OKABE)                              # shared plotting helpers

# ---------------------------------------------------------------- 1) the corpus and the human labels
reviews = pd.read_csv(DAT / "l10_reviews.csv")                     # the bilingual reviews
human = reviews["label"].values                                    # the human reference labels
texts = reviews["text"].values                                     # the texts
print("documents:", len(texts), "| classes:", pd.Series(human).nunique())   # report

# ---------------------------------------------------------------- 2) simulate an LLM annotator
def simulate_labels(texts, temperature, rng):                      # a noisy rule-based "LLM"
    """Return labels where the noise level grows with the temperature."""   # short docstring
    labels = []                                                    # the generated labels
    positive = ["好", "满意", "推荐", "不错", "喜欢", "great", "excellent", "perfect"]   # positive cues
    negative = ["差", "失望", "慢", "坏", "退款", "terrible", "broke", "overpriced"]      # negative cues
    for text in texts:                                             # loop over the documents
        low = str(text).lower()                                    # lowercase for the English cues
        pos_hit = sum(1 for w in positive if w in low)              # positive cue count
        neg_hit = sum(1 for w in negative if w in low)              # negative cue count
        if pos_hit > neg_hit:                                      # the rule says positive
            base = "正面"                                          # the base label
        elif neg_hit > pos_hit:                                    # the rule says negative
            base = "负面"                                          # the base label
        else:                                                      # no cue
            base = "中性"                                          # neutral
        if rng.random() < min(0.05 + temperature * 0.18, 0.55):     # flip with probability rising in T
            base = str(rng.choice(["正面", "中性", "负面"]))          # a random label
        labels.append(base)                                        # record it
    return np.array(labels)                                        # return the labels

rows = []                                                          # one row per temperature
rng = np.random.default_rng(20261111)                              # local generator
for temp in [0.0, 0.3, 0.7, 1.0, 1.5]:                             # five temperature settings
    labels = simulate_labels(texts, temp, rng)                     # generate the labels
    kappa = float(cohen_kappa_score(human, labels))                # agreement with the human labels
    agreement = float(np.mean(labels == human))                    # raw agreement
    rows.append({"temperature": temp, "agreement": round(agreement, 4),   # the record
                 "cohen_kappa": round(kappa, 4)})                  # the kappa
    print("T=%.1f | agreement %.3f | kappa %.3f" % (temp, agreement, kappa))   # report
table = pd.DataFrame(rows)                                         # assemble the table
dump(table, "l10_llm_agreement.csv")                               # persist
best = table.loc[table["cohen_kappa"].idxmax()]                    # the most consistent setting
write_json({"best_temperature": float(best["temperature"]),        # run log
            "best_kappa": float(best["cohen_kappa"]),              # the kappa
            "n_documents": len(texts)}, "l10_run_log_08.json")     # write the log

# ---------------------------------------------------------------- 3) repeated sampling at one temperature
rng = np.random.default_rng(7)                                     # a second generator
samples = [simulate_labels(texts, 0.7, rng) for _ in range(5)]      # five repeated runs
pair_rows = []                                                     # pairwise agreement
for i in range(len(samples)):                                      # loop over the runs
    for j in range(i + 1, len(samples)):                           # every later run
        pair_rows.append({"run_i": i + 1, "run_j": j + 1,           # the pair
                          "kappa": round(float(cohen_kappa_score(samples[i], samples[j])), 4)})   # its kappa
pairs = pd.DataFrame(pair_rows)                                    # assemble the table
dump(pairs, "l10_llm_pairwise_kappa.csv")                          # persist
print("mean pairwise kappa: %.3f" % pairs["kappa"].mean())          # report

# ---------------------------------------------------------------- 4) the figures
fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9))                # two panels
axes[0].plot(table["temperature"], table["agreement"], "-o", ms=5, color=OKABE[2], label="raw agreement")   # agreement curve
axes[0].plot(table["temperature"], table["cohen_kappa"], "-s", ms=5, color=OKABE[5], label="Cohen's kappa")   # kappa curve
style_axes(axes[0], "Agreement falls as the temperature rises", "temperature", "score")   # shared style
axes[0].legend(fontsize=7.4, loc="lower left")                      # compact legend
axes[1].hist(pairs["kappa"], bins=8, color=OKABE[4], alpha=0.9)     # pairwise kappa distribution
style_axes(axes[1], "Repeated runs: pairwise kappa", "kappa", "pairs")   # shared style
annotated(axes[1], "mean %.2f" % pairs["kappa"].mean(), "upper right")   # annotate the mean
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_llm_agreement.png")                                 # save the agreement figure
print("done: LLM annotation agreement audit")                       # closing log
