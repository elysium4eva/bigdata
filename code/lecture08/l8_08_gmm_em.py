"""l8_08_gmm_em.py -- Gaussian mixtures and the EM algorithm.

Every line carries an English comment. The script fits a Gaussian mixture on a sample of
three elongated components, shows the soft assignment (posterior probabilities), tracks
the log-likelihood across EM iterations, and contrasts it with k-means on the same data.
"""
import numpy as np                                                 # numeric helpers
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # three panels
import matplotlib.cm as cm                                         # colour maps for the soft assignment
from sklearn.mixture import GaussianMixture                       # the EM implementation
from sklearn.cluster import KMeans                                 # the hard-assignment baseline
from sklearn.metrics import adjusted_rand_score                    # agreement with the truth
from _common_l8 import (make_gmm, save, dump, style_axes,          # shared helpers
                        annotated, OKABE)                               # shared plotting helpers

# ---------------------------------------------------------------- 1) the sample
data = make_gmm(900)                                               # three elongated components
dump(data, "l8_gmm.csv")                                           # persist it
X = data[["x1", "x2"]].values                                      # feature matrix
truth = data["component"].values                                   # the generating component
print("rows:", X.shape[0], "| components:", len(set(truth)))        # report

# ---------------------------------------------------------------- 2) fit the mixture for several k
rows = []                                                          # one row per k
for k in [2, 3, 4]:                                                # under, correct and over
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42).fit(X)   # full covariance
    labels = gmm.predict(X)                                        # hard assignment from the posteriors
    rows.append({"components": k, "bic": round(float(gmm.bic(X)), 1),   # BIC: lower is better
                 "aic": round(float(gmm.aic(X)), 1),               # AIC
                 "log_likelihood": round(float(gmm.score(X) * len(X)), 1),   # total log-likelihood
                 "ari_vs_truth": round(float(adjusted_rand_score(truth, labels)), 4)})   # agreement
    print("k=%d | BIC %.1f | AIC %.1f | ARI %.3f" % (                   # report the row for this k
        k, gmm.bic(X), gmm.aic(X), adjusted_rand_score(truth, labels)))   # report
comparison = pd.DataFrame(rows)                                    # assemble the table
dump(comparison, "l8_gmm_k.csv")                                   # persist it
best_k = int(comparison.loc[comparison["bic"].idxmin(), "components"])   # k by BIC
print("best k by BIC:", best_k)                                    # report

# ---------------------------------------------------------------- 3) the EM trajectory (manual, for the picture)
def em_trace(Xm, k=3, iters=25):                                   # a small EM implementation
    """Return the log-likelihood after every iteration of a Gaussian mixture."""   # short docstring
    means = Xm[np.random.default_rng(42).choice(len(Xm), k, replace=False)]   # initialise on random points
    covs = [np.cov(Xm.T) for _ in range(k)]                         # shared initial covariance
    weights = np.full(k, 1.0 / k)                                  # uniform mixture weights
    trace = []                                                     # likelihood history
    for _ in range(iters):                                         # EM iterations
        resp = np.zeros((len(Xm), k))                              # responsibilities
        for j in range(k):                                         # E step
            diff = Xm - means[j]                                   # centred data
            inv = np.linalg.inv(covs[j])                           # inverse covariance
            det = np.linalg.det(covs[j])                           # determinant
            norm = 1.0 / np.sqrt((2 * np.pi) ** 2 * det)           # Gaussian normaliser
            resp[:, j] = weights[j] * norm * np.exp(-0.5 * np.einsum("ni,ij,nj->n", diff, inv, diff))   # density
        resp /= np.maximum(resp.sum(axis=1, keepdims=True), 1e-300)   # normalise the responsibilities
        trace.append(float(np.log(np.maximum(resp.sum(axis=1), 1e-300)).mean()))   # record the average log-density
        nk = resp.sum(axis=0) + 1e-10                              # effective counts
        weights = nk / nk.sum()                                    # M step: weights
        means = (resp.T @ Xm) / nk[:, None]                        # M step: means
        for j in range(k):                                         # M step: covariances
            diff = Xm - means[j]                                   # centred data
            covs[j] = (resp[:, j, None] * diff).T @ diff / nk[j] + 1e-6 * np.eye(2)   # regularised covariance
    return trace                                                   # return the history

np.random.seed(42)                                                 # the manual EM uses numpy randomness
trace = em_trace(X, k=3, iters=25)                                 # run the manual version
dump(pd.DataFrame({"iteration": np.arange(1, len(trace) + 1), "avg_log_density": np.round(trace, 5)}),   # table
     "l8_em_trace.csv")                                            # persist

# ---------------------------------------------------------------- 4) the figure
fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8))                # three panels
gmm3 = GaussianMixture(n_components=3, covariance_type="full", random_state=42).fit(X)   # the correct k
post = gmm3.predict_proba(X)                                       # soft assignment
labels = post.argmax(axis=1)                                       # hard assignment
axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap="RdPu", s=14, alpha=0.85)   # hard labels
axes[0].scatter(gmm3.means_[:, 0], gmm3.means_[:, 1], s=150, marker="X", color="#1E2630", label="means")   # means
style_axes(axes[0], "GMM with k = 3: hard assignment", "x1", "x2")   # shared style
axes[0].legend(fontsize=7.2, loc="upper left")                     # compact legend
conf = post.max(axis=1)                                            # confidence per point
sc = axes[1].scatter(X[:, 0], X[:, 1], c=conf, cmap="viridis", s=16, alpha=0.9)   # colour by confidence
style_axes(axes[1], "Soft assignment: posterior confidence", "x1", "x2")   # shared style
plt.colorbar(sc, ax=axes[1], fraction=0.046, label="max posterior")   # colour bar
axes[2].plot(np.arange(1, len(trace) + 1), trace, "-o", ms=3.5, color=OKABE[5])   # EM trajectory
style_axes(axes[2], "EM: log-likelihood rises every step", "iteration", "average log density")   # shared style
annotated(axes[2], "monotone improvement", "lower right")           # annotate the property
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_gmm.png")                                           # save the figure
print("done: Gaussian mixture, soft assignment and the EM trace")   # closing log
