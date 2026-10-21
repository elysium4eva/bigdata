"""l6_03_mle_gaussian.py -- maximum likelihood: from a Gaussian sample to the NLL surface.

Every line carries an English comment. The script shows the negative log-likelihood
of a Gaussian sample as a function of the mean and of the standard deviation, draws
the two-dimensional surface, and contrasts it with the Bernoulli likelihood used by
logistic regression.
"""
import numpy as np                                                 # arrays and the likelihood grid
import pandas as pd                                                # tabular data handling
import matplotlib.pyplot as plt                                    # four panels of likelihood geometry
from scipy.optimize import minimize                                # numerical maximisation for the check
from _common_l6 import (RNG, save, dump, style_axes, annotated, OKABE)   # shared helpers

# ---------------------------------------------------------------- 1) a Gaussian sample to estimate
N = 60                                                             # sample size
mu_true, sd_true = 2.5, 1.4                                        # generating parameters
sample = RNG.normal(mu_true, sd_true, N)                           # draw the sample
mle_mu = float(sample.mean())                                      # MLE of the mean equals the sample mean
mle_sd = float(sample.std(ddof=0))                                 # MLE of the sd divides by n, not n-1
sd_unbiased = float(sample.std(ddof=1))                            # unbiased estimator for comparison
print("sample mean %.4f | MLE sd %.4f | unbiased sd %.4f" % (mle_mu, mle_sd, sd_unbiased))   # report

def nll_gaussian(params, data):                                    # negative log-likelihood of a Gaussian
    """Return the negative log-likelihood for (mu, log sigma) given data."""   # short docstring
    mu, log_sigma = params                                         # unpack the parameter vector
    sigma = np.exp(log_sigma)                                      # keep sigma positive by construction
    z = (data - mu) / sigma                                        # standardised residuals
    return float(np.sum(0.5 * z ** 2 + log_sigma + 0.5 * np.log(2 * np.pi)))   # sum of NLL terms

fit = minimize(nll_gaussian, x0=[0.0, 0.0], args=(sample,), method="Nelder-Mead")   # numerical MLE
print("numerical MLE: mu %.4f | sigma %.4f" % (fit.x[0], np.exp(fit.x[1])))   # should match the closed form

# ---------------------------------------------------------------- 2) one-dimensional scans
mu_grid = np.linspace(sample.min() - 0.5, sample.max() + 0.5, 300)   # grid over the mean
nll_mu = [nll_gaussian([m, np.log(mle_sd)], sample) for m in mu_grid]   # NLL as a function of the mean
sd_grid = np.linspace(0.3, 3.0, 300)                               # grid over the standard deviation
nll_sd = [nll_gaussian([mle_mu, np.log(s)], sample) for s in sd_grid]   # NLL as a function of sigma

# ---------------------------------------------------------------- 3) two-dimensional surface
mu_axis = np.linspace(mle_mu - 1.2, mle_mu + 1.2, 120)             # mean axis for the contour plot
sd_axis = np.linspace(0.7, 2.2, 120)                               # sigma axis for the contour plot
MM, SS = np.meshgrid(mu_axis, sd_axis)                             # build the grid
NLL = np.empty_like(MM)                                            # allocate the NLL surface
for i in range(MM.shape[0]):                                       # loop over sigma values
    for j in range(MM.shape[1]):                                   # loop over mean values
        NLL[i, j] = nll_gaussian([MM[i, j], np.log(SS[i, j])], sample)   # evaluate the NLL on the grid
dump(pd.DataFrame({"mu": MM.ravel().round(4), "sigma": SS.ravel().round(4),   # flatten the surface
                   "nll": NLL.ravel().round(4)}), "l6_nll_surface.csv")   # persist it for the slides

# ---------------------------------------------------------------- 4) Bernoulli likelihood for classification
k, n = 37, 60                                                      # successes and trials
p_grid = np.linspace(0.01, 0.99, 400)                              # grid over the success probability
log_lik = k * np.log(p_grid) + (n - k) * np.log(1 - p_grid)        # Bernoulli log-likelihood
nll_bern = -log_lik                                                # negative log-likelihood
p_mle = k / n                                                      # the MLE is the observed frequency
print("Bernoulli MLE p = %.4f" % p_mle)                             # report the MLE
dump(pd.DataFrame({"k": [k], "n": [n], "p_mle": [p_mle]}), "l6_spam.csv")   # persist the counts

# ---------------------------------------------------------------- 5) figure one: NLL surfaces
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6))                # three panels
axes[0].plot(mu_grid, nll_mu, lw=2.0, color=OKABE[5])              # NLL as a function of the mean
axes[0].axvline(mle_mu, ls="--", lw=1.2, color="#66717D")           # mark the MLE
style_axes(axes[0], "NLL over the mean (sigma fixed)", "mean $\\mu$", "negative log-likelihood")   # style
annotated(axes[0], "$\\hat\\mu$ = %.3f" % mle_mu, "upper right")     # annotate the estimate
axes[1].plot(sd_grid, nll_sd, lw=2.0, color=OKABE[3])              # NLL as a function of sigma
axes[1].axvline(mle_sd, ls="--", lw=1.2, color="#66717D", label="MLE (1/n)")   # mark the biased MLE
axes[1].axvline(sd_unbiased, ls=":", lw=1.2, color=OKABE[5], label="unbiased (1/(n-1))")   # mark the unbiased one
style_axes(axes[1], "NLL over sigma (mu fixed)", "standard deviation $\\sigma$", "negative log-likelihood")   # style
axes[1].legend(fontsize=7.4)                                       # compact legend
cs = axes[2].contourf(MM, SS, NLL, levels=18, cmap="RdPu")         # the two-dimensional NLL surface
axes[2].contour(MM, SS, NLL, levels=10, colors="white", linewidths=0.5, alpha=0.6)   # contour lines
axes[2].scatter([mle_mu], [mle_sd], s=60, marker="*", color="#1E2630", zorder=3, label="MLE")   # mark the MLE
style_axes(axes[2], "2D NLL surface: a single well", "mean $\\mu$", "standard deviation $\\sigma$")   # style
plt.colorbar(cs, ax=axes[2], fraction=0.046, label="NLL")           # colour bar for the surface
axes[2].legend(fontsize=7.4, loc="upper right")                    # legend inside the panel
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_nll_surface.png")                                   # save the likelihood figure

# ---------------------------------------------------------------- 6) figure two: Bernoulli likelihood
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6))                # three panels
axes[0].plot(p_grid, np.exp(log_lik), lw=2.0, color=OKABE[2])      # likelihood curve
axes[0].axvline(p_mle, ls="--", lw=1.2, color="#66717D")           # mark the MLE
style_axes(axes[0], "Bernoulli likelihood (k=%d, n=%d)" % (k, n), "p", "likelihood")   # style
axes[1].plot(p_grid, nll_bern, lw=2.0, color=OKABE[5])             # negative log-likelihood
axes[1].axvline(p_mle, ls="--", lw=1.2, color="#66717D")           # mark the MLE
style_axes(axes[1], "Negative log-likelihood = cross-entropy", "p", "NLL")   # style
annotated(axes[1], "$\\hat p$ = %.3f" % p_mle, "upper center")      # annotate the MLE
w_grid = np.linspace(-4, 4, 200)                                   # grid over a one-dimensional weight
x_feat = RNG.normal(0.0, 1.0, 200)                                 # a synthetic feature column
y_feat = (x_feat + RNG.normal(0, 0.6, 200) > 0).astype(float)      # a noisy threshold label, fixed once
loss_curve = []                                                    # collect the logistic loss
for w in w_grid:                                                   # sweep the weight
    prob = 1.0 / (1.0 + np.exp(-w * x_feat))                       # sigmoid prediction for each observation
    loss_curve.append(float(-np.mean(y_feat * np.log(prob + 1e-12)   # cross-entropy for y = 1
                                     + (1 - y_feat) * np.log(1 - prob + 1e-12))))   # and for y = 0
axes[2].plot(w_grid, loss_curve, lw=2.0, color=OKABE[4])           # logistic loss as a function of w
axes[2].axvline(0, ls="--", lw=1.0, color="#66717D")               # zero reference
style_axes(axes[2], "Logistic loss is convex in w", "weight $w$", "mean cross-entropy")   # style
plt.tight_layout()                                                 # tidy the layout
save(fig, "fig_bernoulli_nll.png")                                 # save the Bernoulli figure
print("done: NLL surfaces and the Bernoulli likelihood")            # closing log
