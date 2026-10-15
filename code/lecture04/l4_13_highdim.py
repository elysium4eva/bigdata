"""l4_13_highdim.py -- high-dimensional visualization: projections, parallel coordinates, clustered heatmap, explanations.

A 12-dimensional customer behaviour dataset (600 rows) answers three questions:
(1) How do we "see" high-dimensional data? -- PCA / t-SNE / UMAP compared, plus their pitfalls;
(2) How do we compare multivariate profiles? -- parallel coordinates, radar chart and a cluster-ordered heatmap;
(3) How do we show what a model relies on? -- permutation importance, partial dependence (PDP) and ICE curves.
"""
import numpy as np                                                 # numeric computation
import pandas as pd                                                # data tables
import matplotlib.pyplot as plt                                    # static figures
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list   # hierarchical clustering and reordering
from sklearn.decomposition import PCA                              # principal component analysis (linear reduction)
from sklearn.manifold import TSNE, trustworthiness                 # t-SNE and neighbourhood trustworthiness
from sklearn.ensemble import RandomForestClassifier                # model used for the explanation plots
from sklearn.inspection import permutation_importance              # permutation importance
from sklearn.preprocessing import StandardScaler                   # standardisation (required before reduction)
from _common_l4 import RNG, OKABE, INT, save, dump                 # reuse shared helpers and the interactive dir
try:                                                               # UMAP is an optional dependency
    from umap import UMAP                                          # non-linear reduction (local and global structure)
    HAS_UMAP = True                                                # mark it as available
except Exception as exc:                                           # fall back when it is not installed
    print("[umap] not available, fall back to Isomap:", exc)        # report the fallback
    from sklearn.manifold import Isomap                            # use Isomap as a same-family substitute
    HAS_UMAP = False                                               # mark it as unavailable

# ---------------------------------------------------------------- 1) synthetic 12-dimensional customer behaviour data
N = 600                                                            # number of customers
SEGMENTS = ["高价值忠诚", "价格敏感", "新客试探"]                    # three latent segments (used to color the projections)
seg = RNG.choice(SEGMENTS, N, p=[0.3, 0.4, 0.3])                   # random segment assignment
FEATURES = {                                                       # feature name -> (mean, sd) baseline
    "recency_days": (60, 25),                                      # days since the last purchase
    "frequency": (12, 5),                                          # purchase frequency
    "monetary": (2600, 900),                                       # cumulative spend
    "avg_order_value": (220, 60),                                  # average order value
    "return_rate": (0.08, 0.04),                                   # return rate
    "discount_share": (0.24, 0.10),                                # share of discounted orders
    "category_breadth": (4.5, 1.6),                                # number of categories bought
    "session_count": (28, 10),                                     # number of sessions
    "avg_session_min": (7.5, 2.5),                                 # average session length (minutes)
    "email_open_rate": (0.32, 0.12),                               # email open rate
    "support_tickets": (1.2, 1.0),                                 # support tickets
    "tenure_months": (22, 9),                                      # tenure in months
}                                                                  # end of the feature dictionary
rows = {}                                                          # generate feature by feature
for name, (mu, sd) in FEATURES.items():                            # loop over the features
    base = RNG.normal(mu, sd, N)                                   # baseline values
    if name in ("recency_days", "return_rate", "support_tickets"):   # these three are lower for high-value customers
        base -= np.where(seg == "高价值忠诚", sd * 0.9, 0.0)         # loyal customers are more active, return and complain less
        base += np.where(seg == "新客试探", sd * 0.7, 0.0)           # new customers are less active
    if name in ("frequency", "monetary", "avg_order_value", "tenure_months", "category_breadth"):   # engagement features
        base += np.where(seg == "高价值忠诚", sd * 1.1, 0.0)        # high-value customers are clearly higher
        base -= np.where(seg == "价格敏感", sd * 0.3, 0.0)          # price-sensitive customers are slightly lower
    if name in ("discount_share",):                                # discount reliance is a single-feature effect
        base += np.where(seg == "价格敏感", sd * 1.4, 0.0)          # price-sensitive customers depend on discounts
    rows[name] = np.round(np.clip(base, 0, None), 3)                # drop negatives and keep three decimals
customers = pd.DataFrame(rows)                                     # assemble the feature table
customers["segment"] = seg                                         # latent segment (teaching color only)
# build a churn label from several features, for importance / PDP / ICE later
logit = (-0.035 * customers["recency_days"] + 0.09 * customers["support_tickets"]      # longer since purchase and more tickets raise churn
         - 0.10 * customers["frequency"] - 0.0004 * customers["monetary"]              # buying more often and spending more lowers churn
         + 2.2 * customers["return_rate"] + 1.1 * customers["discount_share"]          # returns and discount dependence raise churn
         + 0.35 * np.sin(customers["avg_session_min"] / 3.0))                          # non-linear effect of session length
prob = 1 / (1 + np.exp(-(logit - logit.mean())))                   # normalise into a probability
customers["churn"] = (RNG.random(N) < prob).astype(int)             # draw the 0/1 label from that probability
dump(customers, "l4_highdim_customers.csv")                        # write the sample
X = customers.drop(columns=["segment", "churn"]).values            # feature matrix (12 dimensions)
X_std = StandardScaler().fit_transform(X)                          # standardisation: required before reduction
print("high-dim data:", X.shape, "| segments:", dict(pd.Series(seg).value_counts()),   # data overview: shape and groups
      "| churn rate:", round(float(customers['churn'].mean()), 3))   # print the data overview

# ---------------------------------------------------------------- 2) three dimensionality reduction projections
pca = PCA(n_components=6, random_state=42).fit(X_std)               # keep 6 principal components
Z_pca = pca.transform(X_std)[:, :3]                                # first three components (2D uses two, 3D uses three)
evr = pca.explained_variance_ratio_ * 100                          # variance explained by each component (%)
Z_tsne = TSNE(n_components=2, perplexity=30, init="pca",            # t-SNE: emphasises local neighbourhoods
              learning_rate="auto", random_state=42).fit_transform(X_std)   # fixed seed keeps it reproducible
if HAS_UMAP:                                                       # use UMAP when available
    Z_umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2,    # neighbours and minimum distance are the two key parameters
                  random_state=42).fit_transform(X_std)            # balances local and global structure
    umap_label = "UMAP"                                            # legend name
else:                                                              # fall back to Isomap
    Z_umap = Isomap(n_neighbors=15, n_components=2).fit_transform(X_std)   # isometric mapping (same family)
    umap_label = "Isomap（UMAP 替代）"                               # label it truthfully
# neighbourhood trustworthiness: do high-dimensional neighbours stay close (closer to 1 is better)
tw_pca = trustworthiness(X_std, Z_pca[:, :2], n_neighbors=10)       # PCA trustworthiness
tw_tsne = trustworthiness(X_std, Z_tsne, n_neighbors=10)            # t-SNE trustworthiness
tw_umap = trustworthiness(X_std, Z_umap, n_neighbors=10)            # UMAP trustworthiness
metrics = pd.DataFrame({                                           # summarise projection quality
    "方法": ["PCA", "t-SNE", umap_label],                           # the three methods
    "邻域保真度(10-NN)": [round(tw_pca, 3), round(tw_tsne, 3), round(tw_umap, 3)],   # trustworthiness values
    "解释性": ["方差可解释（前两成分为 %.1f%% + %.1f%%）" % (evr[0], evr[1]),       # PCA has a variance reading
               "只能读局部结构", "局部+全局，参数敏感"],             # qualitative notes for the other two
})                                                                 # end of the metrics frame
dump(metrics, "l4_highdim_projection_metrics.csv")                 # write the projection metrics
print(metrics.to_string(index=False))                              # print to the terminal

# ---------------------------------------------------------------- 3) figure 1: three projections compared
seg_colors = {"高价值忠诚": OKABE[0], "价格敏感": OKABE[1], "新客试探": OKABE[2]}   # three colorblind-friendly colors
fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.5))                # three projections plus one quality panel
for name, c in seg_colors.items():                                 # draw points segment by segment
    m = seg == name                                                # mask for that segment
    axes[0].scatter(Z_pca[m, 0], Z_pca[m, 1], s=9, color=c, alpha=0.6, label=name)   # PCA projection
axes[0].set_xlabel(f"PC1（{evr[0]:.1f}%）", fontsize=9)             # annotate the explained variance
axes[0].set_ylabel(f"PC2（{evr[1]:.1f}%）", fontsize=9)             # annotate the explained variance
axes[0].set_title("① PCA：线性、方差可解释")                        # subplot title
axes[0].legend(fontsize=7.4, markerscale=1.6)                      # legend
for name, c in seg_colors.items():                                 # t-SNE projection
    m = seg == name                                                # mask for that segment
    axes[1].scatter(Z_tsne[m, 0], Z_tsne[m, 1], s=9, color=c, alpha=0.6)   # draw the points
axes[1].set_title("② t-SNE：局部邻域清楚，距离不可读")               # subplot title
axes[1].set_xticks([]); axes[1].set_yticks([])                      # reduced coordinates have no units
for name, c in seg_colors.items():                                 # UMAP / Isomap projection
    m = seg == name                                                # mask for that segment
    axes[2].scatter(Z_umap[m, 0], Z_umap[m, 1], s=9, color=c, alpha=0.6)   # draw the points
axes[2].set_title(f"③ {umap_label}：局部+全局兼顾")                 # subplot title
axes[2].set_xticks([]); axes[2].set_yticks([])                      # reduced coordinates have no units
axes[3].bar(range(1, 7), evr, color=OKABE[3], alpha=0.9)            # scree plot: explained share per component
axes[3].plot(range(1, 7), np.cumsum(evr), "-o", ms=4, color=OKABE[5], label="累计")   # cumulative explained share
axes[3].set_title("④ 碎石图与邻域保真度")                            # subplot title
axes[3].set_xlabel("主成分序号", fontsize=9); axes[3].set_ylabel("解释方差 %", fontsize=9)   # axis labels
axes[3].legend(fontsize=7.6)                                       # legend
axes[3].text(0.03, 0.95, f"保真度：PCA {tw_pca:.2f}｜t-SNE {tw_tsne:.2f}｜{umap_label} {tw_umap:.2f}",   # fidelity note
             transform=axes[3].transAxes, va="top", fontsize=7.4, color="#1E2630")   # compare inside the figure
plt.tight_layout()                                                 # adjust the four panels
save(fig, "fig_highdim_projections.png")                           # save: projection comparison figure

# ---------------------------------------------------------------- 4) figure 2: parallel coordinates, clustered heatmap and radar chart
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6), gridspec_kw={"width_ratios": [1.5, 1.1, 1.0]})   # three panels
show_cols = list(FEATURES.keys())[:8]                              # parallel coordinates show the first eight features
Xs = pd.DataFrame(X_std, columns=list(FEATURES.keys()))[show_cols]   # standardised subset (for the parallel coordinates)
Xall = pd.DataFrame(X_std, columns=list(FEATURES.keys()))          # all 12 standardised dimensions (the radar chart needs later features)
for name, c in seg_colors.items():                                 # draw one profile line per segment
    m = (seg == name)                                              # segment mask
    axes[0].plot(range(len(show_cols)), Xs[m].mean().values, "-o", ms=4, lw=1.8, color=c, label=name)   # mean profile of the segment
axes[0].set_xticks(range(len(show_cols)))                          # one tick per feature
axes[0].set_xticklabels(show_cols, rotation=35, ha="right", fontsize=7.2)   # feature names (rotated to avoid overlap)
axes[0].set_ylabel("标准化取值（z）", fontsize=9)                    # y axis: standardised, so unitless
axes[0].set_title("① 平行坐标：多变量剖面一目了然")                   # subplot title
axes[0].legend(fontsize=7.6)                                       # legend
axes[0].axhline(0, color="#66717D", lw=0.8, ls="--")               # mean reference line
corr = Xs.corr().values                                            # feature correlation matrix
Z = linkage(Xs.T.values, method="average")                         # hierarchical clustering of the variables
order = leaves_list(Z)                                             # variable order after clustering (used to reorder the heatmap)
im = axes[1].imshow(corr[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)   # reordered correlation heatmap
axes[1].set_xticks(range(len(order)), [show_cols[i] for i in order], rotation=45, ha="right", fontsize=6.6)   # x axis labels
axes[1].set_yticks(range(len(order)), [show_cols[i] for i in order], fontsize=6.6)   # y axis labels
axes[1].grid(False)                                                # no grid needed on a heatmap
axes[1].set_title("② 聚类重排热力图：成块=成组")                     # subplot title
plt.colorbar(im, ax=axes[1], fraction=0.046)                       # color bar
radar_cols = ["frequency", "monetary", "avg_order_value", "return_rate", "discount_share", "tenure_months"]   # radar axes
angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False)   # radar chart angles
angles_closed = np.concatenate([angles, angles[:1]])               # close the angle loop
ax_r = plt.subplot(1, 3, 3, polar=True)                            # switch the third panel to polar coordinates
for name, c in seg_colors.items():                                 # draw one radar per segment
    vals = Xall.assign(seg=seg).groupby("seg")[radar_cols].mean().loc[name].values   # segment means
    vals = np.concatenate([vals, vals[:1]])                        # close the data loop
    ax_r.plot(angles_closed, vals, color=c, lw=1.8, label=name)     # draw the line
    ax_r.fill(angles_closed, vals, color=c, alpha=0.12)            # semi-transparent fill
ax_r.set_xticks(angles, [c.replace("_", "\n") for c in radar_cols], fontsize=6.4)   # axis labels
ax_r.set_title("③ 雷达图：慎用（面积会放大差异）", fontsize=10.5, pad=14)   # subplot title
ax_r.tick_params(labelsize=6.4)                                    # tick font size
plt.tight_layout()                                                 # adjust the three panels
save(fig, "fig_parallel_cluster.png")                              # save: multivariate profile figure

# ---------------------------------------------------------------- 5) figure 3: permutation importance, PDP and ICE
clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=4,   # a random forest as an explainable model
                             random_state=42).fit(X_std, customers["churn"])   # train it on the churn label
perm = permutation_importance(clf, X_std, customers["churn"],        # permutation importance: shuffle a feature and watch the score drop
                              n_repeats=10, random_state=42, scoring="roc_auc")   # ten repeats give the error
imp = pd.DataFrame({"feature": list(FEATURES.keys()),                # feature names
                    "importance": perm.importances_mean.round(4),     # mean importance
                    "std": perm.importances_std.round(4)})           # importance standard deviation
imp = imp.sort_values("importance", ascending=False)                 # sort from high to low
dump(imp, "l4_highdim_importance.csv")                              # write the importance table
print("top-3 importance:\n", imp.head(3).to_string(index=False))     # print the top three
top = imp["feature"].head(3).tolist()                               # the three most important features
axes[0].barh(imp["feature"][::-1][:8], imp["importance"][::-1][:8],   # horizontal bars for the top eight
             xerr=imp["std"][::-1][:8], capsize=3, color=OKABE[5], alpha=0.9)   # with error bars
axes[0].set_xlabel("ROC-AUC 下降幅度", fontsize=9)                   # axis label
axes[0].set_title("① 置换重要性（含 10 次重复的误差）")                # subplot title
axes[0].tick_params(axis="y", labelsize=7.4)                        # feature name font size
grid_vals = np.linspace(0, 1, 60)                                   # quantile grid (0-1)
pdp_curves = {}                                                     # keep the PDP curves
for feat in top[:2]:                                                # one PDP for each of the top two features
    j = list(FEATURES.keys()).index(feat)                           # column index of that feature
    qs = np.quantile(X_std[:, j], grid_vals)                        # actual quantiles of that feature
    preds = []                                                      # mean prediction at each quantile
    for v in qs:                                                    # loop over the quantiles
        X_tmp = X_std.copy()                                        # copy the feature matrix
        X_tmp[:, j] = v                                             # replace the whole column with v
        preds.append(clf.predict_proba(X_tmp)[:, 1].mean())          # mean churn probability
    pdp_curves[feat] = (qs, np.array(preds))                        # record the curve
for (feat, (xs, ys)), c in zip(pdp_curves.items(), [OKABE[0], OKABE[1]]):   # draw the PDP curves
    axes[1].plot(xs, ys, lw=2.0, color=c, label=feat)                # curve
axes[1].set_xlabel("标准化特征取值（z）", fontsize=9)                 # axis label
axes[1].set_ylabel("平均预测流失概率", fontsize=9)                    # axis label
axes[1].set_title("② PDP：平均而言特征怎么影响预测")                  # subplot title
axes[1].legend(fontsize=7.6)                                        # legend
ice_feat = top[0]                                                   # ICE uses the most important feature
j = list(FEATURES.keys()).index(ice_feat)                           # column index of that feature
qs = np.quantile(X_std[:, j], grid_vals)                            # quantile grid
idx_sample = RNG.choice(len(X_std), 60, replace=False)               # draw 60 individuals for ICE
ice = np.zeros((len(idx_sample), len(qs)))                          # store the ICE curves
for r, i in enumerate(idx_sample):                                  # individual by individual
    for k, v in enumerate(qs):                                      # quantile by quantile
        X_tmp = X_std[i:i + 1].copy()                               # take just that individual's row
        X_tmp[0, j] = v                                             # change the feature value
        ice[r, k] = clf.predict_proba(X_tmp)[0, 1]                  # that individual's predicted probability
    axes[2].plot(qs, ice[r], color="#66717D", alpha=0.25, lw=0.9)    # grey lines: individual curves
axes[2].plot(qs, ice.mean(axis=0), color=OKABE[5], lw=2.2, label="平均（=PDP）")   # the mean line is the PDP
axes[2].set_xlabel(f"{ice_feat}（标准化）", fontsize=9)              # axis label
axes[2].set_ylabel("预测流失概率", fontsize=9)                        # axis label
axes[2].set_title("③ ICE：个体曲线揭示异质性与交叉")                  # subplot title
axes[2].legend(fontsize=7.6)                                        # legend
plt.tight_layout()                                                  # adjust the three panels
save(fig, "fig_interpretability.png")                               # save: explanation figure

# ---------------------------------------------------------------- 6) interactive 3D embedding (plotly)
import plotly.graph_objects as go                                   # low-level API for the 3D scatter
fig3d = go.Figure()                                                 # new canvas
for name, c in seg_colors.items():                                  # add one layer per segment
    m = seg == name                                                 # segment mask
    fig3d.add_trace(go.Scatter3d(x=Z_pca[m, 0], y=Z_pca[m, 1], z=Z_pca[m, 2],   # the first three principal components
                                 mode="markers", name=name,          # legend name
                                 marker=dict(size=3, color=c, opacity=0.6)))    # point style
fig3d.update_layout(title="PCA 三维嵌入（可旋转缩放：高维结构的交互探索）",   # title
                    scene=dict(xaxis_title=f"PC1 {evr[0]:.0f}%",     # 3D axis labels
                               yaxis_title=f"PC2 {evr[1]:.0f}%",     # second axis
                               zaxis_title=f"PC3 {evr[2]:.0f}%"))    # third axis
fig3d.write_html(INT / "embedding_3d.html", include_plotlyjs="cdn")   # export the rotatable 3D embedding
print("[interactive]", INT / "embedding_3d.html")                    # print the output path
print("high-dim figures done:", ["fig_highdim_projections.png", "fig_parallel_cluster.png", "fig_interpretability.png"])   # closing log
