"""l4_13_highdim.py —— 高维数据可视化：降维投影、平行坐标、聚类热图与解释性可视化。

一份 12 维客户行为数据（600 行）回答三个问题：
① 高维数据怎么"看"？——PCA / t-SNE / UMAP 三种投影的对照与陷阱；
② 多变量剖面怎么比较？——平行坐标、雷达图与聚类重排的热力图；
③ 模型的判断依据怎么展示？——置换重要性、部分依赖（PDP）与 ICE 曲线。
"""
import numpy as np                                                 # 数值计算
import pandas as pd                                                # 数据表
import matplotlib.pyplot as plt                                    # 静态图
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list   # 层次聚类与重排
from sklearn.decomposition import PCA                              # 主成分分析（线性降维）
from sklearn.manifold import TSNE, trustworthiness                 # t-SNE 与邻域保真度
from sklearn.ensemble import RandomForestClassifier                # 用于解释性可视化的模型
from sklearn.inspection import permutation_importance              # 置换重要性
from sklearn.preprocessing import StandardScaler                   # 标准化（降维前必做）
from _common_l4 import RNG, OKABE, INT, save, dump                 # 复用公共工具与交互目录
try:                                                               # UMAP 为可选依赖
    from umap import UMAP                                          # 非线性降维（局部+全局结构）
    HAS_UMAP = True                                                # 标记可用
except Exception as exc:                                           # 未安装时退化
    print("[umap] not available, fall back to Isomap:", exc)        # 提示退化
    from sklearn.manifold import Isomap                            # 用 Isomap 作为同族替代
    HAS_UMAP = False                                               # 标记不可用

# ---------------------------------------------------------------- 1) 合成 12 维客户行为数据
N = 600                                                            # 客户数
SEGMENTS = ["高价值忠诚", "价格敏感", "新客试探"]                    # 三个潜在群体（用于给投影上色）
seg = RNG.choice(SEGMENTS, N, p=[0.3, 0.4, 0.3])                   # 随机分配群体
FEATURES = {                                                       # 特征名 → (均值, 标准差) 的基线
    "recency_days": (60, 25),                                      # 最近一次购买距今天数
    "frequency": (12, 5),                                          # 购买频次
    "monetary": (2600, 900),                                       # 累计金额
    "avg_order_value": (220, 60),                                  # 客单价
    "return_rate": (0.08, 0.04),                                   # 退货率
    "discount_share": (0.24, 0.10),                                # 折扣订单占比
    "category_breadth": (4.5, 1.6),                                # 购买品类数
    "session_count": (28, 10),                                     # 会话次数
    "avg_session_min": (7.5, 2.5),                                 # 平均会话时长（分钟）
    "email_open_rate": (0.32, 0.12),                               # 邮件打开率
    "support_tickets": (1.2, 1.0),                                 # 客服工单数
    "tenure_months": (22, 9),                                      # 在网月数
}
rows = {}                                                          # 逐特征生成
for name, (mu, sd) in FEATURES.items():                            # 遍历特征
    base = RNG.normal(mu, sd, N)                                   # 基线取值
    if name in ("recency_days", "return_rate", "support_tickets"):   # 这三项：高价值群体更低
        base -= np.where(seg == "高价值忠诚", sd * 0.9, 0.0)         # 忠诚客户更活跃、更少退货与工单
        base += np.where(seg == "新客试探", sd * 0.7, 0.0)           # 新客更不活跃
    if name in ("frequency", "monetary", "avg_order_value", "tenure_months", "category_breadth"):
        base += np.where(seg == "高价值忠诚", sd * 1.1, 0.0)        # 高价值群体显著更高
        base -= np.where(seg == "价格敏感", sd * 0.3, 0.0)          # 价格敏感群体略低
    if name in ("discount_share",):
        base += np.where(seg == "价格敏感", sd * 1.4, 0.0)          # 价格敏感群体折扣依赖高
    rows[name] = np.round(np.clip(base, 0, None), 3)                # 去掉负值并保留三位小数
customers = pd.DataFrame(rows)                                     # 组装特征表
customers["segment"] = seg                                         # 潜在群体（仅用于教学上色）
# 造一个"流失"标签：由若干特征非线性决定，便于后面做重要性 / PDP / ICE
logit = (-0.035 * customers["recency_days"] + 0.09 * customers["support_tickets"]      # 越久未买、工单越多越易流失
         - 0.10 * customers["frequency"] - 0.0004 * customers["monetary"]              # 越常买、花得越多越不易流失
         + 2.2 * customers["return_rate"] + 1.1 * customers["discount_share"]          # 退货与折扣依赖抬高流失
         + 0.35 * np.sin(customers["avg_session_min"] / 3.0))                          # 会话时长的非线性效应
prob = 1 / (1 + np.exp(-(logit - logit.mean())))                   # 归一化为概率
customers["churn"] = (RNG.random(N) < prob).astype(int)             # 按概率生成 0/1 标签
dump(customers, "l4_highdim_customers.csv")                        # 落盘样本
X = customers.drop(columns=["segment", "churn"]).values            # 特征矩阵（12 维）
X_std = StandardScaler().fit_transform(X)                          # 标准化：降维前的必要步骤
print("high-dim data:", X.shape, "| segments:", dict(pd.Series(seg).value_counts()),
      "| churn rate:", round(float(customers['churn'].mean()), 3))   # 打印数据概况

# ---------------------------------------------------------------- 2) 三种降维投影
pca = PCA(n_components=6, random_state=42).fit(X_std)               # 保留 6 个主成分
Z_pca = pca.transform(X_std)[:, :3]                                # 取前三个主成分（2D 用前两个，3D 交互用三个）
evr = pca.explained_variance_ratio_ * 100                          # 各主成分解释的方差比例（%）
Z_tsne = TSNE(n_components=2, perplexity=30, init="pca",            # t-SNE：强调局部邻域
              learning_rate="auto", random_state=42).fit_transform(X_std)   # 固定种子保证可复现
if HAS_UMAP:                                                       # UMAP 可用则用之
    Z_umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2,    # 邻居数与最小距离是两个关键参数
                  random_state=42).fit_transform(X_std)            # 兼顾局部与全局结构
    umap_label = "UMAP"                                            # 图例名称
else:                                                              # 退化为 Isomap
    Z_umap = Isomap(n_neighbors=15, n_components=2).fit_transform(X_std)   # 等距映射（同族方法）
    umap_label = "Isomap（UMAP 替代）"                               # 如实标注
# 邻域保真度：高维中的近邻在低维中还是近邻吗（越接近 1 越好）
tw_pca = trustworthiness(X_std, Z_pca[:, :2], n_neighbors=10)       # PCA 的保真度
tw_tsne = trustworthiness(X_std, Z_tsne, n_neighbors=10)            # t-SNE 的保真度
tw_umap = trustworthiness(X_std, Z_umap, n_neighbors=10)            # UMAP 的保真度
metrics = pd.DataFrame({                                           # 汇总投影质量
    "方法": ["PCA", "t-SNE", umap_label],                           # 三种方法
    "邻域保真度(10-NN)": [round(tw_pca, 3), round(tw_tsne, 3), round(tw_umap, 3)],   # 保真度数值
    "解释性": ["方差可解释（前两成分为 %.1f%% + %.1f%%）" % (evr[0], evr[1]),       # PCA 有方差口径
               "只能读局部结构", "局部+全局，参数敏感"],             # 另两种的定性说明
})
dump(metrics, "l4_highdim_projection_metrics.csv")                 # 落盘投影指标
print(metrics.to_string(index=False))                              # 终端打印

# ---------------------------------------------------------------- 3) 图一：三种投影对照
seg_colors = {"高价值忠诚": OKABE[0], "价格敏感": OKABE[1], "新客试探": OKABE[2]}   # 色盲友好的三色
fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.5))                # 三张投影 + 一张质量对照
for name, c in seg_colors.items():                                 # 逐群体画点
    m = seg == name                                                # 该群体的掩码
    axes[0].scatter(Z_pca[m, 0], Z_pca[m, 1], s=9, color=c, alpha=0.6, label=name)   # PCA 投影
axes[0].set_xlabel(f"PC1（{evr[0]:.1f}%）", fontsize=9)             # 标注解释方差
axes[0].set_ylabel(f"PC2（{evr[1]:.1f}%）", fontsize=9)             # 标注解释方差
axes[0].set_title("① PCA：线性、方差可解释")                        # 子图标题
axes[0].legend(fontsize=7.4, markerscale=1.6)                      # 图例
for name, c in seg_colors.items():                                 # t-SNE 投影
    m = seg == name                                                # 该群体掩码
    axes[1].scatter(Z_tsne[m, 0], Z_tsne[m, 1], s=9, color=c, alpha=0.6)   # 画点
axes[1].set_title("② t-SNE：局部邻域清楚，距离不可读")               # 子图标题
axes[1].set_xticks([]); axes[1].set_yticks([])                      # 降维坐标没有单位
for name, c in seg_colors.items():                                 # UMAP / Isomap 投影
    m = seg == name                                                # 该群体掩码
    axes[2].scatter(Z_umap[m, 0], Z_umap[m, 1], s=9, color=c, alpha=0.6)   # 画点
axes[2].set_title(f"③ {umap_label}：局部+全局兼顾")                 # 子图标题
axes[2].set_xticks([]); axes[2].set_yticks([])                      # 降维坐标没有单位
axes[3].bar(range(1, 7), evr, color=OKABE[3], alpha=0.9)            # 碎石图：各主成分解释率
axes[3].plot(range(1, 7), np.cumsum(evr), "-o", ms=4, color=OKABE[5], label="累计")   # 累计解释率
axes[3].set_title("④ 碎石图与邻域保真度")                            # 子图标题
axes[3].set_xlabel("主成分序号", fontsize=9); axes[3].set_ylabel("解释方差 %", fontsize=9)   # 轴标签
axes[3].legend(fontsize=7.6)                                       # 图例
axes[3].text(0.03, 0.95, f"保真度：PCA {tw_pca:.2f}｜t-SNE {tw_tsne:.2f}｜{umap_label} {tw_umap:.2f}",
             transform=axes[3].transAxes, va="top", fontsize=7.4, color="#1E2630")   # 图内比较
plt.tight_layout()                                                 # 调整四格
save(fig, "fig_highdim_projections.png")                           # 保存：投影对照图

# ---------------------------------------------------------------- 4) 图二：平行坐标、聚类热图与雷达图
fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6), gridspec_kw={"width_ratios": [1.5, 1.1, 1.0]})
show_cols = list(FEATURES.keys())[:8]                              # 平行坐标展示前八个特征
Xs = pd.DataFrame(X_std, columns=list(FEATURES.keys()))[show_cols]   # 标准化后的子集（用于平行坐标）
Xall = pd.DataFrame(X_std, columns=list(FEATURES.keys()))          # 全部 12 维标准化结果（雷达图要用到后面的特征）
for name, c in seg_colors.items():                                 # 逐群体画剖面线
    m = (seg == name)                                              # 群体掩码
    axes[0].plot(range(len(show_cols)), Xs[m].mean().values, "-o", ms=4, lw=1.8, color=c, label=name)   # 群体均值剖面
axes[0].set_xticks(range(len(show_cols)))                          # 每个特征一个刻度
axes[0].set_xticklabels(show_cols, rotation=35, ha="right", fontsize=7.2)   # 特征名（旋转避免重叠）
axes[0].set_ylabel("标准化取值（z）", fontsize=9)                    # 纵轴：标准化后无量纲
axes[0].set_title("① 平行坐标：多变量剖面一目了然")                   # 子图标题
axes[0].legend(fontsize=7.6)                                       # 图例
axes[0].axhline(0, color="#66717D", lw=0.8, ls="--")               # 均值参考线
corr = Xs.corr().values                                            # 特征相关矩阵
Z = linkage(Xs.T.values, method="average")                         # 对变量做层次聚类
order = leaves_list(Z)                                             # 聚类后的变量顺序（用于重排热力图）
im = axes[1].imshow(corr[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)   # 重排后的相关热力图
axes[1].set_xticks(range(len(order)), [show_cols[i] for i in order], rotation=45, ha="right", fontsize=6.6)   # 横轴标签
axes[1].set_yticks(range(len(order)), [show_cols[i] for i in order], fontsize=6.6)   # 纵轴标签
axes[1].grid(False)                                                # 热力图不需要网格
axes[1].set_title("② 聚类重排热力图：成块=成组")                     # 子图标题
plt.colorbar(im, ax=axes[1], fraction=0.046)                       # 颜色条
radar_cols = ["frequency", "monetary", "avg_order_value", "return_rate", "discount_share", "tenure_months"]
angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False)   # 雷达图角度
angles_closed = np.concatenate([angles, angles[:1]])               # 闭合角度
ax_r = plt.subplot(1, 3, 3, polar=True)                            # 第三个面板改为极坐标
for name, c in seg_colors.items():                                 # 逐群体画雷达
    vals = Xall.assign(seg=seg).groupby("seg")[radar_cols].mean().loc[name].values   # 群体均值
    vals = np.concatenate([vals, vals[:1]])                        # 闭合数据
    ax_r.plot(angles_closed, vals, color=c, lw=1.8, label=name)     # 画线
    ax_r.fill(angles_closed, vals, color=c, alpha=0.12)            # 半透明填充
ax_r.set_xticks(angles, [c.replace("_", "\n") for c in radar_cols], fontsize=6.4)   # 轴标签
ax_r.set_title("③ 雷达图：慎用（面积会放大差异）", fontsize=10.5, pad=14)   # 子图标题
ax_r.tick_params(labelsize=6.4)                                    # 刻度字号
plt.tight_layout()                                                 # 调整三格
save(fig, "fig_parallel_cluster.png")                              # 保存：多变量剖面图

# ---------------------------------------------------------------- 5) 图三：置换重要性、PDP 与 ICE
clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=4,   # 一棵随机森林作为可解释模型
                             random_state=42).fit(X_std, customers["churn"])   # 用流失标签训练
perm = permutation_importance(clf, X_std, customers["churn"],        # 置换重要性：打乱某特征看性能掉多少
                              n_repeats=10, random_state=42, scoring="roc_auc")   # 十次重复给出误差
imp = pd.DataFrame({"feature": list(FEATURES.keys()),                # 特征名
                    "importance": perm.importances_mean.round(4),     # 重要性均值
                    "std": perm.importances_std.round(4)})           # 重要性标准差
imp = imp.sort_values("importance", ascending=False)                 # 从高到低排序
dump(imp, "l4_highdim_importance.csv")                              # 落盘重要性表
print("top-3 importance:\n", imp.head(3).to_string(index=False))     # 打印前三名
top = imp["feature"].head(3).tolist()                               # 取前三重要特征
axes[0].barh(imp["feature"][::-1][:8], imp["importance"][::-1][:8],   # 画前八名的水平条形
             xerr=imp["std"][::-1][:8], capsize=3, color=OKABE[5], alpha=0.9)   # 带误差棒
axes[0].set_xlabel("ROC-AUC 下降幅度", fontsize=9)                   # 轴标签
axes[0].set_title("① 置换重要性（含 10 次重复的误差）")                # 子图标题
axes[0].tick_params(axis="y", labelsize=7.4)                        # 特征名字号
grid_vals = np.linspace(0, 1, 60)                                   # 分位点网格（0—1）
pdp_curves = {}                                                     # 记录 PDP 曲线
for feat in top[:2]:                                                # 前两个特征各画一条 PDP
    j = list(FEATURES.keys()).index(feat)                           # 该特征在矩阵中的列号
    qs = np.quantile(X_std[:, j], grid_vals)                        # 取该特征的实际分位点
    preds = []                                                      # 每个分位点的平均预测
    for v in qs:                                                    # 遍历分位点
        X_tmp = X_std.copy()                                        # 复制特征矩阵
        X_tmp[:, j] = v                                             # 把该列整体替换为 v
        preds.append(clf.predict_proba(X_tmp)[:, 1].mean())          # 平均流失概率
    pdp_curves[feat] = (qs, np.array(preds))                        # 记录曲线
for (feat, (xs, ys)), c in zip(pdp_curves.items(), [OKABE[0], OKABE[1]]):   # 画 PDP 曲线
    axes[1].plot(xs, ys, lw=2.0, color=c, label=feat)                # 曲线
axes[1].set_xlabel("标准化特征取值（z）", fontsize=9)                 # 轴标签
axes[1].set_ylabel("平均预测流失概率", fontsize=9)                    # 轴标签
axes[1].set_title("② PDP：平均而言特征怎么影响预测")                  # 子图标题
axes[1].legend(fontsize=7.6)                                        # 图例
ice_feat = top[0]                                                   # ICE 用最重要的特征
j = list(FEATURES.keys()).index(ice_feat)                           # 该特征列号
qs = np.quantile(X_std[:, j], grid_vals)                            # 分位点网格
idx_sample = RNG.choice(len(X_std), 60, replace=False)               # 抽 60 个个体画 ICE
ice = np.zeros((len(idx_sample), len(qs)))                          # 存 ICE 曲线
for r, i in enumerate(idx_sample):                                  # 逐个体
    for k, v in enumerate(qs):                                      # 逐分位点
        X_tmp = X_std[i:i + 1].copy()                               # 只取该个体一行
        X_tmp[0, j] = v                                             # 改变该特征取值
        ice[r, k] = clf.predict_proba(X_tmp)[0, 1]                  # 该个体的预测概率
    axes[2].plot(qs, ice[r], color="#66717D", alpha=0.25, lw=0.9)    # 灰线：个体曲线
axes[2].plot(qs, ice.mean(axis=0), color=OKABE[5], lw=2.2, label="平均（=PDP）")   # 均值线即 PDP
axes[2].set_xlabel(f"{ice_feat}（标准化）", fontsize=9)              # 轴标签
axes[2].set_ylabel("预测流失概率", fontsize=9)                        # 轴标签
axes[2].set_title("③ ICE：个体曲线揭示异质性与交叉")                  # 子图标题
axes[2].legend(fontsize=7.6)                                        # 图例
plt.tight_layout()                                                  # 调整三格
save(fig, "fig_interpretability.png")                               # 保存：解释性可视化

# ---------------------------------------------------------------- 6) 交互式三维嵌入（plotly）
import plotly.graph_objects as go                                   # 三维散点用低级接口
fig3d = go.Figure()                                                 # 新建画布
for name, c in seg_colors.items():                                  # 逐群体加一层
    m = seg == name                                                 # 群体掩码
    fig3d.add_trace(go.Scatter3d(x=Z_pca[m, 0], y=Z_pca[m, 1], z=Z_pca[m, 2],   # 前三个主成分
                                 mode="markers", name=name,          # 图例名称
                                 marker=dict(size=3, color=c, opacity=0.6)))    # 点样式
fig3d.update_layout(title="PCA 三维嵌入（可旋转缩放：高维结构的交互探索）",   # 标题
                    scene=dict(xaxis_title=f"PC1 {evr[0]:.0f}%",     # 三维轴标签
                               yaxis_title=f"PC2 {evr[1]:.0f}%",     # 第二轴
                               zaxis_title=f"PC3 {evr[2]:.0f}%"))    # 第三轴
fig3d.write_html(INT / "embedding_3d.html", include_plotlyjs="cdn")   # 导出可旋转的三维嵌入
print("[interactive]", INT / "embedding_3d.html")                    # 打印产物路径
print("high-dim figures done:", ["fig_highdim_projections.png", "fig_parallel_cluster.png", "fig_interpretability.png"])
