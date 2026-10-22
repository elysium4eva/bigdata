"""l10_10_domain_lexicon.py -- why a domain lexicon matters for Chinese segmentation.

Every line carries an English comment. The script segments the same sentences with a default
and an augmented dictionary, measures how often the domain terms survive segmentation, and
draws the before/after comparison so the deck can argue that segmentation is a business choice.
"""
import numpy as np                                                 # arrays
import pandas as pd                                                # tabular data
import matplotlib.pyplot as plt                                    # the figure
import jieba                                                       # Chinese segmentation
from _common_l10 import (DAT, make_lexicon_sentences, save, dump,   # shared helpers
                         style_axes, annotated, OKABE)                  # shared plotting helpers

# ---------------------------------------------------------------- 1) the sentences and the lexicon
sentences = make_lexicon_sentences()                               # sentences with domain terms
terms = ["华为Mate60Pro", "比亚迪海獭", "vivoX100", "李宁䨻科技", "ESG", "碳中和"]   # the domain lexicon
print("sentences:", len(sentences), "| terms:", len(terms))         # report

# ---------------------------------------------------------------- 2) segment before and after
def segment(text):                                                 # the default segmentation
    """Return the tokens produced by the default dictionary."""     # short docstring
    return list(jieba.lcut(text))                                  # cut with the current dictionary
before = [segment(s) for s in sentences]                            # default cuts
for term in terms:                                                 # add every domain term
    jieba.add_word(term)                                           # to the dictionary
after = [segment(s) for s in sentences]                            # cuts with the domain lexicon
rows = []                                                          # one row per sentence
for s, b, a in zip(sentences, before, after):                      # loop over the sentences
    kept_before = [t for t in b if t in terms]                     # domain terms that survived (default)
    kept_after = [t for t in a if t in terms]                       # domain terms that survived (augmented)
    rows.append({"sentence": s, "default_cut": " / ".join(b),       # the record
                 "augmented_cut": " / ".join(a),                    # the augmented cut
                 "terms_before": len(kept_before), "terms_after": len(kept_after)})   # survival counts
    print(s)                                                        # report the sentence
    print("  default: ", " / ".join(b))                             # the default cut
    print("  augmented:", " / ".join(a))                            # the augmented cut
compare = pd.DataFrame(rows)                                       # assemble the table
dump(compare, "l10_lexicon_compare.csv")                           # persist
survival = compare[["terms_before", "terms_after"]].sum()           # the totals
print("domain terms surviving -- default %d / augmented %d" % (survival["terms_before"], survival["terms_after"]))   # report

# ---------------------------------------------------------------- 3) a tokenisation error rate
test_terms = ["物流时效", "退换货", "性价比", "续航", "售后"]          # five more business terms
probes = ["这个产品的物流时效很好", "退换货流程很麻烦", "性价比不错", "续航一般", "售后服务可以"]   # their sentences
error_rows = []                                                    # error records
for term, probe in zip(test_terms, probes):                        # loop over the probes
    default = jieba.lcut(probe)                                    # the default cut
    hit = term in default                                          # whether the term survived
    error_rows.append({"term": term, "sentence": probe,             # the record
                       "default_hit": bool(hit), "tokens": " / ".join(default)})   # survival and tokens
errors = pd.DataFrame(error_rows)                                  # assemble the table
dump(errors, "l10_segmentation_errors.csv")                        # persist

# ---------------------------------------------------------------- 4) the figures
fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9))                # two panels
axes[0].bar(["默认词典", "领域词典"], [int(survival["terms_before"]), int(survival["terms_after"])],   # survival counts
            color=[OKABE[5], OKABE[2]], alpha=0.9)                  # survival counts
style_axes(axes[0], "Domain terms surviving segmentation", "", "count")   # shared style
for i, v in enumerate([int(survival["terms_before"]), int(survival["terms_after"])]):   # annotate
    axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=10)   # the value
axes[1].axis("off")                                               # the example needs no axes
axes[1].text(0.02, 0.92, "同一句话，两种分词：", fontsize=11, fontweight="bold")   # heading
y = 0.78                                                          # vertical cursor
for i in range(min(3, len(compare))):                              # show three examples
    axes[1].text(0.02, y, compare["default_cut"].iloc[i][:34], fontsize=8, color=OKABE[5])   # default cut
    axes[1].text(0.02, y - 0.07, compare["augmented_cut"].iloc[i][:34], fontsize=8, color=OKABE[2])   # augmented cut
    y -= 0.22                                                     # move down
axes[1].set_title("red = default, green = augmented")               # panel title
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_lexicon.png")                                       # save the lexicon figure
print("done: domain lexicon effect on segmentation")                # closing log
