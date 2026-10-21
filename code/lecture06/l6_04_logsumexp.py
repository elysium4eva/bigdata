"""l6_04_logsumexp.py -- numerical stability: underflow and the log-sum-exp trick.

Every line carries an English comment. The script demonstrates how naive probability
products underflow, then implements a stable softmax and a stable log-sum-exp, and
records the numbers so the slides can show the actual magnitudes.
"""
import numpy as np                                                 # arrays for the overflow demo
import pandas as pd                                                # tabular data handling
from _common_l6 import dump, write_json, logsumexp                 # shared helpers

# ---------------------------------------------------------------- 1) a product of many small probabilities
n = 300                                                            # number of factors
probs = np.full(n, 0.01)                                           # each factor is one per cent
naive_product = float(np.prod(probs))                              # naive product underflows to zero
log_product = float(np.sum(np.log(probs)))                         # working in logs keeps the value
print("naive product: %.3e | log product: %.3f" % (naive_product, log_product))   # report the contrast
assert naive_product == 0.0, "expected underflow for 300 factors of 0.01"   # document the failure mode

# ---------------------------------------------------------------- 2) stable versus naive softmax
def naive_softmax(z):                                              # textbook softmax, prone to overflow
    """Return the softmax of z computed without shifting."""        # short docstring
    ez = np.exp(z)                                                 # exponentials may overflow
    return ez / np.sum(ez)                                         # normalise

def stable_softmax(z):                                             # shifted softmax used in practice
    """Return the softmax of z after subtracting the maximum."""    # short docstring
    z = np.asarray(z, dtype=float)                                 # work with a float array
    shifted = z - np.max(z)                                        # the shift removes the overflow
    ez = np.exp(shifted)                                           # safe exponentials
    return ez / np.sum(ez)                                         # normalise

small = np.array([1.0, 2.0, 3.0])                                  # well-scaled logits
large = np.array([1000.0, 1001.0, 1002.0])                         # huge logits
print("naive small:", np.round(naive_softmax(small), 4))           # fine for small inputs
print("stable small:", np.round(stable_softmax(small), 4))         # identical result
with np.errstate(over="ignore"):                                   # silence the overflow warning
    print("naive large:", naive_softmax(large))                    # nan because of inf/inf
print("stable large:", np.round(stable_softmax(large), 4))         # the shifted version is fine

# ---------------------------------------------------------------- 3) stable log-sum-exp versus the naive one
z = np.array([800.0, 801.0, 802.0])                                # logits that break the naive version
with np.errstate(over="ignore"):                                   # silence the warning
    naive_lse = float(np.log(np.sum(np.exp(z))))                   # overflows to inf
stable_lse = float(logsumexp(z))                                   # our stable helper
print("naive logsumexp: %s | stable: %.4f" % (naive_lse, stable_lse))   # report the contrast

# ---------------------------------------------------------------- 4) record the demonstration
table = pd.DataFrame({"case": ["product of 300 p=0.01", "logit 1000/1001/1002",   # the three experiments
                              "logsumexp 800/801/802"],            # third experiment
                      "naive_value": [naive_product, float("nan"), naive_lse],   # what the naive code returns
                      "stable_value": [float(np.exp(log_product)), float("nan"), stable_lse]})   # the stable result
dump(table, "l6_logsumexp_demo.csv")                               # persist the table
write_json({"log_product": log_product,                            # run log for the slides
            "stable_logsumexp": stable_lse,                        # the stable value
            "note": "naive products underflow, naive softmax overflows"}, "l6_run_log_04.json")   # write the log
print("done: log-sum-exp demonstration")                           # closing log
