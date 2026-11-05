"""l6_15_entropy.py -- entropy, cross-entropy and KL divergence, drawn and computed.

Every line carries an English comment. The script draws the binary entropy curve, shows that the
cross-entropy is minimised exactly where the prediction matches the truth, and prints the numbers
(bits and nats) that the deck quotes for the three quantities.
"""
import numpy as np                                                 # arrays
import matplotlib                                                  # backend before pyplot
matplotlib.use("Agg")                                              # headless backend
import matplotlib.pyplot as plt                                    # figures
from pathlib import Path                                           # paths

ROOT = Path(__file__).resolve().parents[2]                         # website/
FIG = ROOT / "figures" / "lecture06"                               # figure folder
DAT = ROOT / "data" / "lecture06"                                  # data folder
for d in (FIG, DAT):                                               # ensure both exist
    d.mkdir(parents=True, exist_ok=True)                           # create when missing

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]   # CJK stack
plt.rcParams["axes.unicode_minus"] = False                         # real minus signs
plt.rcParams["savefig.dpi"] = 200                                  # slide resolution
plt.rcParams["savefig.bbox"] = "tight"                             # trim white space
plt.rcParams["axes.grid"] = True                                   # light grid
plt.rcParams["grid.alpha"] = 0.25                                  # subtle grid
plt.rcParams["axes.spines.top"] = False                            # drop the top spine
plt.rcParams["axes.spines.right"] = False                          # drop the right spine
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]


def entropy(p):                                                    # Shannon entropy in nats
    """Return -sum p log p, ignoring zero entries."""               # short docstring
    p = np.asarray(p, dtype=float)                                 # to array
    p = p[p > 0]                                                   # zeros contribute nothing
    return float(-(p * np.log(p)).sum())                           # the sum


def cross_entropy(p, q):                                           # cross-entropy in nats
    """Return -sum p log q."""                                      # short docstring
    p, q = np.asarray(p, float), np.asarray(q, float)               # to arrays
    mask = p > 0                                                   # only the support of p matters
    return float(-(p[mask] * np.log(q[mask])).sum())                # the sum


def kl(p, q):                                                      # KL divergence in nats
    """Return sum p log(p/q)."""                                    # short docstring
    p, q = np.asarray(p, float), np.asarray(q, float)               # to arrays
    mask = p > 0                                                   # only where p has mass
    return float((p[mask] * np.log(p[mask] / q[mask])).sum())       # the sum


# ---------------------------------------------------------------- 1) the numbers the deck quotes
cases = {"one-hot (0,1)": [1.0, 0.0], "skewed (0.9,0.1)": [0.9, 0.1], "fair coin (0.5,0.5)": [0.5, 0.5]}
rows = []                                                          # one row per distribution
for name, p in cases.items():                                      # loop over the cases
    h = entropy(p)                                                 # nats
    rows.append({"distribution": name, "H_nats": round(h, 4), "H_bits": round(h / np.log(2), 4)})   # record
uniform4 = entropy([0.25] * 4)                                     # four equally likely outcomes
rows.append({"distribution": "uniform over 4", "H_nats": round(uniform4, 4),
             "H_bits": round(uniform4 / np.log(2), 4)})            # add it
print("entropy table:")                                            # report
for r in rows:                                                     # print the table
    print("  %-24s %.4f nats  %.4f bits" % (r["distribution"], r["H_nats"], r["H_bits"]))

p_true = np.array([0.2, 0.8])                                      # the truth
print("H(p)=%.4f nats (%.4f bits)" % (entropy(p_true), entropy(p_true) / np.log(2)))   # report
for q in ([0.2, 0.8], [0.5, 0.5], [0.05, 0.95]):                    # three predictions
    ce = cross_entropy(p_true, q)                                  # cross-entropy
    d = kl(p_true, q)                                              # KL divergence
    print("  q=%s -> H(p,q)=%.4f  D_KL=%.4f  check %.4f" % (q, ce, d, ce - entropy(p_true) - d))   # report

# ---------------------------------------------------------------- 2) the figure
grid = np.linspace(0.01, 0.99, 400)                                # probabilities for the x axis
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8))                # three panels
h_bin = [entropy([g, 1 - g]) for g in grid]                        # binary entropy
axes[0].plot(grid, h_bin, lw=2.2, color=OKABE[4])                   # the curve
axes[0].axhline(np.log(2), ls="--", lw=1.0, color="#66717D")         # the maximum
axes[0].axhline(0, ls="-", lw=0.8, color="#2B2B2B")                 # the floor
axes[0].set_xlabel("$p$（事件发生的概率）"); axes[0].set_ylabel("$H(p)$（nats）")   # labels
axes[0].set_title("熵：均匀时最大 $\\ln 2=0.693$，确定时为 0")            # panel title
axes[0].annotate("最大 .693", xy=(0.5, np.log(2)), xytext=(0.58, 0.62), fontsize=8.4,
                 arrowprops=dict(arrowstyle="->", lw=0.9, color="#2B2B2B"))   # annotate the peak
ce_curve = [cross_entropy([0.2, 0.8], [g, 1 - g]) for g in grid]     # cross-entropy against q
axes[1].plot(grid, ce_curve, lw=2.2, color=OKABE[5])                 # the curve
axes[1].axhline(entropy([0.2, 0.8]), ls="--", lw=1.0, color="#66717D", label="H(p) 下界")   # the floor
axes[1].axvline(0.2, ls=":", lw=1.0, color="#2B2B2B")                 # the minimiser
axes[1].legend(fontsize=7.6)                                        # compact legend
axes[1].set_xlabel("$q$（模型预测）"); axes[1].set_ylabel("$H(p,q)$（nats）")   # labels
axes[1].set_title("交叉熵在 $q=p=0.2$ 处最小，等于 $H(p)$")              # panel title
kl_curve = [kl([0.2, 0.8], [g, 1 - g]) for g in grid]                # KL against q
axes[2].plot(grid, kl_curve, lw=2.2, color=OKABE[2])                 # the curve
axes[2].axvline(0.2, ls=":", lw=1.0, color="#2B2B2B")                 # the zero
axes[2].axhline(0, ls="-", lw=0.8, color="#2B2B2B")                   # the floor
axes[2].set_xlabel("$q$（模型预测）"); axes[2].set_ylabel("$D_{KL}(p\\Vert q)$（nats）")   # labels
axes[2].set_title("KL ≥ 0，且只在 $q=p$ 时为 0")                       # panel title
axes[2].annotate("q=p 处为 0", xy=(0.2, 0.0), xytext=(0.33, 0.35), fontsize=8.4,
                 arrowprops=dict(arrowstyle="->", lw=0.9, color="#2B2B2B"))   # annotate
plt.tight_layout()                                                 # tidy the layout
fig.savefig(FIG / "fig_entropy.png", facecolor="white")              # save
plt.close(fig)                                                     # release memory
print("[figure]", FIG / "fig_entropy.png")                           # log it

# ---------------------------------------------------------------- 3) persist the tables
import pandas as pd                                                # only needed here
pd.DataFrame(rows).to_csv(DAT / "l6_entropy_table.csv", index=False, encoding="utf-8-sig")   # persist
checks = pd.DataFrame([{"q": str(q), "H_pq": round(cross_entropy(p_true, q), 4),
                        "D_kl": round(kl(p_true, q), 4)} for q in ([0.2, 0.8], [0.5, 0.5], [0.05, 0.95])])
checks.to_csv(DAT / "l6_cross_entropy_checks.csv", index=False, encoding="utf-8-sig")   # persist
print("done: entropy, cross-entropy and KL")                        # closing log
