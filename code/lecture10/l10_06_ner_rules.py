"""l10_06_ner_rules.py -- named-entity recognition with a lexicon, regex and BIO rules.

Every line carries an English comment. The script extracts companies, amounts and dates from
synthetic contract snippets using only a lexicon and regular expressions (no downloaded
model), writes a BIO-tagged table, and draws the extraction results so the deck can compare
rules with a learned tagger.
"""
import re                                                          # regular expressions
import numpy as np                                                 # arrays
import pandas as pd                                                # tabular data
import matplotlib.pyplot as plt                                    # the figure
from _common_l10 import (make_contracts_zh, save, dump,            # shared helpers
                         style_axes, annotated, OKABE)                  # shared plotting helpers

# ---------------------------------------------------------------- 1) the snippets
contracts = make_contracts_zh(60)                                  # synthetic contracts and news
dump(contracts, "l10_contracts.csv")                               # persist them
print("snippets:", len(contracts))                                  # report

# ---------------------------------------------------------------- 2) the three extractors
COMPANY = re.compile(r"[\u4e00-\u9fa5]{2,12}(?:有限公司|股份有限公司|有限责任公司)")   # company names
MONEY = re.compile(r"(?:人民币\s*)?[\d,\.]+\s*(?:万元|亿元|元)")   # money expressions
DATE = re.compile(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|\d{4}-\d{2}-\d{2}|\d{4}\s*年\s*\d{1,2}\s*月")   # dates
def extract(text):                                                 # run all three extractors
    """Return the company, money and date spans found in one snippet."""   # short docstring
    return {"company": COMPANY.findall(text), "money": MONEY.findall(text), "date": DATE.findall(text)}   # the spans
rows = []                                                          # one row per snippet
for _, row in contracts.iterrows():                                # loop over the snippets
    found = extract(row["text"])                                   # extract the entities
    rows.append({"doc_id": row["doc_id"], "text": row["text"],      # the snippet
                 "companies": " | ".join(found["company"]),        # extracted companies
                 "amounts": " | ".join(found["money"]),            # extracted amounts
                 "dates": " | ".join(found["date"])})              # extracted dates
    print(row["doc_id"], "->", found["company"], found["money"], found["date"])   # report
extracted = pd.DataFrame(rows)                                     # assemble the table
dump(extracted, "l10_ner_results.csv")                             # persist

# ---------------------------------------------------------------- 3) BIO tagging for one snippet
def bio_tag(text):                                                 # build a BIO sequence
    """Return token, tag pairs for the companies found in one snippet."""   # short docstring
    tags = []                                                      # the (token, tag) pairs
    spans = [(m.start(), m.end()) for m in COMPANY.finditer(text)]   # company spans
    covered = set()                                                # positions inside an entity
    for start, end in spans:                                       # mark the entity positions
        covered.update(range(start, end))                          # add every character index
    i = 0                                                          # cursor
    while i < len(text):                                           # walk the snippet
        if i in covered:                                           # inside an entity
            start = i                                              # remember the start
            while i < len(text) and i in covered:                  # advance to the end of the entity
                i += 1                                             # step forward
            entity = text[start:i]                                 # the entity text
            tags.append((entity, "B-ORG" if start == min(s for s, _ in spans) else "I-ORG"))   # B for the first, I otherwise
        else:                                                      # outside an entity
            start = i                                              # remember the start
            while i < len(text) and i not in covered:              # advance to the next entity
                i += 1                                             # step forward
            tags.append((text[start:i], "O"))                      # outside tag
    return tags                                                    # return the pairs
first = contracts.iloc[0]["text"]                                  # one snippet for the figure
tagged = pd.DataFrame(bio_tag(first), columns=["token", "tag"])     # the BIO frame
dump(tagged, "l10_bio_example.csv")                                # persist
print(tagged.to_string(index=False))                               # report

# ---------------------------------------------------------------- 4) the figures
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))                # two panels
counts = pd.DataFrame({"companies": extracted["companies"].str.count(r"\|") + 1,   # entity counts
                       "amounts": extracted["amounts"].str.count(r"\|") + 1,      # amount counts
                       "dates": extracted["dates"].str.count(r"\|") + 1})          # date counts
mean_counts = counts.mean()                                        # average entities per snippet
axes[0].bar(mean_counts.index, mean_counts.values, color=[OKABE[4], OKABE[5], OKABE[2]], alpha=0.9)   # the bars
style_axes(axes[0], "Entities per snippet (mean)", "", "count")    # shared style
axes[1].axis("off")                                               # the tagged example needs no axes
axes[1].text(0.02, 0.9, "BIO 标注示例：", fontsize=11, fontweight="bold")   # heading
y = 0.78                                                          # vertical cursor
for token, tag in tagged.itertuples(index=False):                  # write the tags
    if token.strip():                                             # skip whitespace tokens
        colour = OKABE[5] if tag.startswith("B") else (OKABE[2] if tag.startswith("I") else "#66717D")   # tag colour
        axes[1].text(0.02, y, "%s  →  %s" % (token[:24], tag), fontsize=8.6, color=colour)   # one line
        y -= 0.06                                                 # move down
        if y < 0.02:                                              # stop when the panel is full
            break                                                  # leave the loop
axes[1].set_title("One snippet, char-level BIO tags")              # panel title
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_ner.png")                                           # save the NER figure
print("done: rule-based NER and BIO tagging")                       # closing log
