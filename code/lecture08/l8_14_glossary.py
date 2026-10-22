"""l8_14_glossary.py -- the bilingual glossary for Lecture 8.

Every line carries an English comment. The script writes the machine-readable term table
(Chinese, English, symbol, formula, note, part) that feeds the two glossary frames in the
deck, the site table and the printable Word/Excel handout.
"""
import pandas as pd                                                # build and persist the table
from _common_l8 import dump, OUT                                   # shared helpers

# Columns: term_zh, term_en, symbol, formula, note, part.
TERMS = [                                                               # term list: zh, en, symbol, formula, note, part
    ("无监督学习", "unsupervised learning", "", "", "没有标签，只能靠数据结构与解释", "1 版图"),        # concept
    ("降维", "dimensionality reduction", "", "", "把高维压到低维，保留主要结构", "1 版图"),           # concept
    ("聚类", "clustering", "", "", "把相似样本分到一组", "1 版图"),                                # concept
    ("异常检测", "anomaly detection", "", "", "找出与大多数样本不同的点", "1 版图"),                # concept
    ("关联规则", "association rules", "", "", "挖掘常一起出现的商品组合", "1 版图"),                    # concept
    ("矩阵分解", "matrix factorisation", "", r"$X\approx WH$", "把数据矩阵拆成低秩因子", "1 版图"),   # concept
    ("距离度量", "distance metric", "", "", "欧氏/L1/余弦/相关；决定什么算相近", "1 版图"),              # concept
    ("标准化", "standardisation", "", "", "去掉量纲，避免大方差变量主导", "1 版图"),                  # concept
    ("内部指标", "internal metric", "", "", "只用数据与标签评分（轮廓等）", "1 版图"),                # concept
    ("外部指标", "external metric", "", "", "与真标签比对（ARI 等）", "1 版图"),                     # concept
    ("主成分分析", "principal component analysis (PCA)", "", "", "找方差最大的正交方向", "2 PCA"),    # pca
    ("协方差矩阵", "covariance matrix", r"$\Sigma$", "", "变量两两之间的线性相关程度", "2 PCA"),       # pca
    ("特征向量", "eigenvector", r"$v$", r"$\Sigma v=\lambda v$", "主成分的方向", "2 PCA"),          # pca
    ("特征值", "eigenvalue", r"$\lambda$", "", "该方向解释的方差量", "2 PCA"),                       # pca
    ("方差解释比例", "explained variance ratio", "", r"$\lambda_k/\sum_j\lambda_j$", "每个成分的贡献", "2 PCA"),   # pca
    ("载荷", "loading", "", "", "原变量在成分上的权重", "2 PCA"),                                   # pca
    ("碎石图", "scree plot", "", "", "特征值随成分序号下降的曲线", "2 PCA"),                        # pca
    ("投影", "projection", "", r"$z=V_k^\top x$", "把样本投到前 k 个成分上", "2 PCA"),              # pca
    ("重构误差", "reconstruction error", "", r"$\lVert x-\hat x\rVert^2$", "丢掉成分后信息损失", "2 PCA"),   # pca
    ("增量 PCA", "incremental PCA", "", "", "用 partial_fit 分块处理超大数据", "2 PCA"),            # pca
    ("核 PCA", "kernel PCA", "", "", "用核函数做非线性降维", "2 PCA"),                              # pca
    ("因子分析", "factor analysis", "", r"$x=\Lambda f+\epsilon$", "用少数潜在因子解释相关", "3 因子"),   # factor
    ("潜在因子", "latent factor", r"$f$", "", "观测不到的公共原因", "3 因子"),                       # factor
    ("因子载荷", "factor loading", r"$\Lambda$", "", "变量对因子的依赖强度", "3 因子"),               # factor
    ("唯一性方差", "uniqueness", r"$\psi$", "", "变量中因子解释不了的部分", "3 因子"),                # factor
    ("因子提取", "factor extraction", "", "", "从相关矩阵里估计因子（PCA/PFA/ML）", "3 因子"),        # factor
    ("旋转", "rotation", "", "", "让载荷更易解释（Varimax/Promax）", "3 因子"),                     # factor
    ("因子得分", "factor score", "", "", "每个样本在各因子上的取值", "3 因子"),                       # factor
    ("三因子模型", "three-factor model", "", r"$E(R)-R_f=\beta_m(\cdot)+\beta_{SMB}\mathrm{SMB}+\beta_{HML}\mathrm{HML}$", "Fama-French 的资产定价模型", "3 因子"),   # factor
    ("规模因子", "size factor (SMB)", "", "", "小市值减低市值的历史超额收益", "3 因子"),               # factor
    ("价值因子", "value factor (HML)", "", "", "高账面市值比减低的历史超额收益", "3 因子"),            # factor
    ("感知图", "perceptual map", "", "", "把品牌放到两个可解释维度上的定位图", "3 因子"),             # factor
    ("定位", "positioning", "", "", "消费者心中品牌相对竞品的位置", "3 因子"),                        # factor
    ("簇内凝聚", "cohesion", "", "", "同簇样本彼此接近的程度", "4 聚类"),                            # cluster
    ("簇间分离", "separation", "", "", "不同簇彼此远离的程度", "4 聚类"),                            # cluster
    ("k 均值", "k-means", "", r"$\min\sum_k\sum_{x\in C_k}\lVert x-\mu_k\rVert^2$", "交替分配与更新质心", "4 聚类"),   # cluster
    ("质心", "centroid", r"$\mu_k$", "", "簇内样本的均值", "4 聚类"),                                # cluster
    ("Voronoi 单元", "Voronoi cell", "", "", "离某个质心最近的点集", "4 聚类"),                      # cluster
    ("层次聚类", "hierarchical clustering", "", "", "自下而上凝聚或自上而下分裂", "4 聚类"),           # cluster
    ("树状图", "dendrogram", "", "", "展示合并顺序与距离的树", "4 聚类"),                            # cluster
    ("连接方式", "linkage", "", "", "单链/全链/平均/Ward 等簇间距离定义", "4 聚类"),                  # cluster
    ("单链接效应", "single-link effect", "", "", "链式粘连，把桥上的点当成同簇", "4 聚类"),            # cluster
    ("密度聚类", "density-based clustering", "", "", "把高密度区域当作簇", "4 聚类"),                 # cluster
    ("核心点", "core point", "", "", "半径内样本数不少于 minPts 的点", "4 聚类"),                     # cluster
    ("边界点", "border point", "", "", "在核心点邻域内但自身不是核心点", "4 聚类"),                    # cluster
    ("噪声点", "noise point", "", "", "不属于任何簇的点", "4 聚类"),                                 # cluster
    ("高斯混合", "Gaussian mixture model (GMM)", "", "", "每个簇一个高斯分布", "4 聚类"),              # cluster
    ("期望最大化", "expectation-maximisation (EM)", "", "", "交替估计责任度与参数", "4 聚类"),          # cluster
    ("软分配", "soft assignment", "", "", "给出属于每个簇的概率", "4 聚类"),                          # cluster
    ("轮廓系数", "silhouette coefficient", "", r"$(b-a)/\max(a,b)$", "簇内紧、簇间远的综合分数", "5 评估"),   # metrics
    ("调整兰德指数", "adjusted Rand index (ARI)", "", "", "与真标签一致度，随机为 0", "5 评估"),        # metrics
    ("标准化互信息", "normalized mutual information (NMI)", "", "", "标签之间的信息共享程度", "5 评估"),   # metrics
    ("稳定性", "stability", "", "", "重抽样后分群是否一致", "5 评估"),                               # metrics
    ("孤立森林", "isolation forest", "", "", "随机切分，越早被孤立的越异常", "5 评估"),                # metrics
    ("局部离群因子", "local outlier factor (LOF)", "", "", "与邻域密度比较判断异常", "5 评估"),         # metrics
    ("支持度", "support", "", "", "包含某集合的交易占比", "5 评估"),                                 # metrics
    ("置信度", "confidence", "", r"$P(B\mid A)$", "买了 A 之后也买 B 的比例", "5 评估"),             # metrics
    ("提升度", "lift", "", "", "置信度相对独立时的倍数", "5 评估"),                                  # metrics
    ("t-SNE", "t-distributed stochastic neighbour embedding", "", "", "保局部的非线性降维", "6 嵌入"),   # embed
    ("困惑度", "perplexity", "", "", "t-SNE 的有效邻居数", "6 嵌入"),                                # embed
    ("UMAP", "uniform manifold approximation and projection", "", "", "更快的非线性降维，保全局更多", "6 嵌入"),   # embed
    ("邻域保持", "neighbourhood preservation", "", "", "降维后近邻关系是否保留", "6 嵌入"),            # embed
    ("近似最近邻", "approximate nearest neighbour (ANN)", "", "", "牺牲少量召回换取速度", "6 嵌入"),    # embed
    ("倒排文件索引", "inverted file index (IVF)", "", "", "先粗聚类再在少数簇内搜索", "6 嵌入"),         # embed
    ("HNSW", "hierarchical navigable small world", "", "", "图结构的 ANN 索引", "6 嵌入"),          # embed
    ("召回率", "recall at k", "", "", "近似结果中真近邻的比例", "6 嵌入"),                             # embed
    ("主题建模", "topic modelling", "", "", "从文档集合中提取主题", "6 嵌入"),                         # embed
    ("潜在狄利克雷分配", "latent Dirichlet allocation (LDA)", "", "", "词袋上的生成式主题模型", "6 嵌入"),   # embed
    ("非负矩阵分解", "non-negative matrix factorisation (NMF)", "", r"$X\approx WH,\;W,H\ge0$", "可加的部分-整体分解", "6 嵌入"),   # embed
    ("BERTopic", "BERTopic", "", "", "嵌入 + 聚类 + 主题词的三步主题建模", "6 嵌入"),                  # embed
    ("自监督学习", "self-supervised learning", "", "", "用数据自身构造监督信号", "6 嵌入"),             # embed
    ("对比学习", "contrastive learning", "", "", "拉近正样本对、推远负样本对", "6 嵌入"),              # embed
]                                                                       # end of the term list

# ---------------------------------------------------------------- persist the table
glossary = pd.DataFrame(TERMS, columns=["term_zh", "term_en", "symbol", "formula", "note", "part"])   # build the frame
dump(glossary, "l8_glossary.csv")                                  # machine-readable CSV in the data folder
counts = glossary["part"].value_counts().sort_index()              # terms per part
print("glossary terms:", len(glossary))                            # report the total
print(counts.to_string())                                          # report the split by part
assert len(glossary) >= 60, "the glossary should hold at least sixty terms"   # acceptance check
md_lines = ["# Lecture 8 glossary", "",                            # markdown title
            "| 术语 (zh) | Term (en) | Symbol | Formula | Note | Part |",   # header row
            "| --- | --- | --- | --- | --- | --- |"]                # separator row
for _, r in glossary.iterrows():                                   # one row per term
    md_lines.append("| %s | %s | %s | %s | %s | %s |" % (r["term_zh"], r["term_en"],   # Chinese and English
                                                         r["symbol"], r["formula"], r["note"], r["part"]))   # rest
(OUT / "l8_glossary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")   # markdown table
print("[artifact]", OUT / "l8_glossary.md")                        # log the markdown path
print("done: glossary with %d terms" % len(glossary))              # closing log
