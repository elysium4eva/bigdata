"""l14_09_text_extract.py -- extracting trade fields from declaration texts.

Every line carries an English comment. The script builds synthetic Chinese customs declarations,   # continuation of the call
extracts HS code, value, origin, port and incoterm with regular expressions (the same task an LLM   # continuation of the call
would be asked to do), measures field-level accuracy and the agreement with the ground truth,   # continuation of the call
and draws the results the deck uses for the "trade documents" section.   # continuation of the call
"""
import re                                                          # pattern matching
import numpy as np                                                 # arrays
import pandas as pd                                                # tables
import matplotlib.pyplot as plt                                    # figures
from sklearn.metrics import cohen_kappa_score                       # agreement metric
from _common_l14 import (make_declaration_texts, save, dump,        # shared generators
                         style_axes, annotated, OKABE)                  # continuation of the call

# ---------------------------------------------------------------- 1) the documents
docs = make_declaration_texts(400)                                 # 400 synthetic declarations
dump(docs, "l14_declarations.csv")                                 # persist
print("documents:", len(docs))                                      # report
print(docs["text"].iloc[0])                                        # show one example

# ---------------------------------------------------------------- 2) the extractors
HS = re.compile(r"HS\s*编码\s*(\d{8})")                              # eight-digit HS code
VALUE = re.compile(r"申报总值\s*([\d.]+)\s*千美元")                    # declared value in thousand USD
ORIGIN = re.compile(r"原产国\s*([A-Z]{2})")                          # two-letter origin
PORT = re.compile(r"出口口岸\s*([\u4e00-\u9fa5]+港)")                  # a Chinese port name
INCOTERM = re.compile(r"成交方式\s*(FOB|CIF|EXW|DAP)")                # the incoterm

def extract_one(text):                                             # run every extractor on one text
    """Return a dict of extracted fields (blank when a pattern misses)."""   # short docstring
    def first(pattern):                                            # return the first capture group or blank
        m = pattern.search(text)                                   # run the pattern
        return m.group(1) if m else ""                             # the value or a blank
    return {"hs8": first(HS), "value": first(VALUE), "origin": first(ORIGIN),   # the fields
            "port": first(PORT), "incoterm": first(INCOTERM)}      # the remaining fields

rows = []                                                          # one row per document
for _, r in docs.iterrows():                                       # loop over the documents
    found = extract_one(r["text"])                                 # extract the fields
    rows.append({"doc_id": r["doc_id"], **found,                    # the extraction
                 "true_hs": r["true_hs"], "true_value": r["true_value"],   # the ground truth
                 "true_origin": r["true_origin"]})                 # the ground truth
extracted = pd.DataFrame(rows)                                     # assemble
dump(extracted, "l14_extracted.csv")                               # persist

# ---------------------------------------------------------------- 3) field-level accuracy
checks = []                                                        # one row per field
hs_ok = (extracted["hs8"] == extracted["true_hs"]).mean()           # HS code accuracy
val_ok = np.mean([abs(float(a) - float(b)) < 0.15 for a, b in           # continuation of the call
                  zip(extracted["value"], extracted["true_value"]) if a != ""])   # value within 0.15
origin_ok = (extracted["origin"] == extracted["true_origin"]).mean()   # origin accuracy
nonempty = np.mean([(extracted[c] != "").mean() for c in ["port", "incoterm"]])   # port and incoterm filled
checks = pd.DataFrame([                                                 # assemble the frame
    {"field": "HS 编码（8 位）", "accuracy": round(float(hs_ok), 4)},   # the HS field
    {"field": "申报总值（容差 0.15）", "accuracy": round(float(val_ok), 4)},   # the value field
    {"field": "原产国", "accuracy": round(float(origin_ok), 4)},        # the origin field
    {"field": "口岸/成交方式（非空率）", "accuracy": round(float(nonempty), 4)},   # the other fields
])                                                                      # continuation of the call
dump(checks, "l14_field_accuracy.csv")                             # persist
print(checks.to_string(index=False))                               # report

# ---------------------------------------------------------------- 4) agreement with a "human" pass
# Simulate a second annotator that makes a few mistakes, then compute Cohen's kappa on the HS code.
human = extracted["true_hs"].astype(str).tolist()                   # the reference labels
machine = extracted["hs8"].astype(str).tolist()                     # the extractor labels
kappa = float(cohen_kappa_score(human, machine))                    # agreement on the string labels
print("kappa (HS code): %.3f" % kappa)                              # report
dump(pd.DataFrame([{"field": "HS 编码", "kappa": round(kappa, 4),         # assemble the frame
                    "raw_agreement": round(float(hs_ok), 4)}]), "l14_kappa.csv")   # persist

# ---------------------------------------------------------------- 5) the figure
fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9))                 # two panels
axes[0].barh(checks["field"][::-1], checks["accuracy"][::-1], color=OKABE[2], alpha=0.92)   # accuracies
axes[0].set_xlim(0, 1.05)                                          # full scale
style_axes(axes[0], "字段级准确率", "accuracy", "")                   # shared style
for i, v in enumerate(checks["accuracy"][::-1]):                     # annotate
    axes[0].text(v, i, "%.3f" % v, va="center", fontsize=8.6)         # the value
mis = (extracted["hs8"] != extracted["true_hs"]).sum()               # how many HS codes were wrong
axes[1].bar(["正确", "错误"], [len(extracted) - int(mis), int(mis)], color=[OKABE[2], OKABE[5]], alpha=0.92)   # the split
style_axes(axes[1], "HS 编码：正确 vs 错误（%d 份）" % len(extracted), "", "份数")   # shared style
annotated(axes[1], "kappa %.3f" % kappa, "upper right")               # annotate the agreement
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_text_extract.png")                                  # save the figure
print("done: field extraction, accuracy and agreement")               # closing log
