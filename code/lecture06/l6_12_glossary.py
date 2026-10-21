"""l6_12_glossary.py -- the bilingual glossary for Lecture 6.

Every line carries an English comment. The script writes the machine-readable term
table (Chinese, English, symbol, formula, note, part) that feeds the three glossary
frames in the deck, the site table and the printable Word/Excel handout.
"""
import pandas as pd                                                # build and persist the table
from _common_l6 import dump, OUT                                   # shared helpers

# ---------------------------------------------------------------- 1) the term list
# Columns: term_zh, term_en, symbol, formula, note, part.
TERMS = [                                                               # # term list: zh, en, symbol, formula, note, part
    ("机器学习", "machine learning", "", "", "从数据中学习函数以完成预测或决策", "1 概念"),                           # term
    ("监督学习", "supervised learning", "", "", "训练数据带标签 (x_i, y_i)", "1 概念"),                      # term
    ("无监督学习", "unsupervised learning", "", "", "训练数据无标签，发现结构", "1 概念"),                         # term
    ("二分类", "binary classification", "", "", "输出两个类别，如欺诈/正常", "1 概念"),                          # term
    ("多分类", "multi-class classification", "", "", "输出三个及以上类别", "1 概念"),                         # term
    ("训练数据", "training data", "", "", "用于学习函数的数据", "1 概念"),                                     # term
    ("测试数据", "test data", "", "", "用于评估的未见数据", "1 概念"),                                         # term
    ("特征向量", "feature vector", r"$x_i$", "", "一个样本的输入表示", "1 概念"),                              # term
    ("标签", "label", r"$y_i$", "", "样本的目标值", "1 概念"),                                            # term
    ("假设类", "hypothesis class", r"$\mathcal{H}$", "", "候选函数的集合，决定模型族", "1 概念"),                 # term
    ("独立同分布", "independent and identically distributed (i.i.d.)", "", "", "训练与测试同分布且相互独立", "1 概念"),   # term
    ("期望损失", "expected loss / risk", r"$L(f)$", r"$E_{x,y\sim D}[l(f,x,y)]$", "对数据分布的期望误差", "1 概念"),   # term
    ("经验损失", "empirical loss / empirical risk", r"$\hat L(f)$", r"$\frac1n\sum_i l(f,x_i,y_i)$", "样本平均误差", "1 概念"),   # term
    ("0-1 损失", "0-1 loss", "", r"$\mathbb{I}[f(x)\ne y]$", "误分类即 1，否则 0", "1 概念"),              # term
    ("平方损失", "squared loss / $l_2$ loss", "", r"$(f(x)-y)^2$", "回归的默认损失", "1 概念"),              # term
    ("泛化差距", "generalization gap", "", "", "期望损失与经验损失之差", "1 概念"),                              # term
    ("过拟合", "overfitting", "", "", "拟合训练噪声，样本外变差", "1 概念"),                                     # term
    ("表征学习", "representation learning", "", "", "自动学习特征表示", "1 概念"),                            # term
    ("线性回归", "linear regression", r"$f_w(x)=w^\top x$", "", "线性假设类 + 平方损失", "2 回归"),            # term
    ("均方误差", "mean squared error (MSE)", "", r"$\frac1n\sum_i(w^\top x_i-y_i)^2$", "平方损失的平均", "2 回归"),   # term
    ("正规方程", "normal equation", "", r"$w=(X^\top X)^{-1}X^\top y$", "梯度为零得到的闭式解", "2 回归"),      # term
    ("设计矩阵", "design matrix", r"$X$", "", "第 i 行为 $x_i^\top$ 的矩阵", "2 回归"),                     # term
    ("截距", "intercept / bias", r"$b$", "", "常数项，可用增广列实现", "2 回归"),                              # term
    ("增广技巧", "augmented trick", r"$x'=(x,1)$", "", "把截距并入权重向量", "2 回归"),                        # term
    ("投影", "projection", "", "", "最小二乘解是 y 在列空间的投影", "2 回归"),                                   # term
    ("条件数", "condition number", r"$\kappa$", "", "输入微小扰动引起的解的相对变化", "2 回归"),                    # term
    ("多重共线性", "multicollinearity", "", "", "特征高度线性相关，$X^\top X$ 病态", "2 回归"),                   # term
    ("极大似然估计", "maximum likelihood estimation (MLE)", r"$\hat\theta_{ML}$", r"$\arg\max_\theta\prod_i P_\theta(x_i,y_i)$", "选让数据最可能的参数", "3 似然"),   # term
    ("似然", "likelihood", "", r"$P_\theta(x_i,y_i)$", "参数对数据的拟合程度", "3 似然"),                     # likelihood
    ("对数似然", "log-likelihood", "", r"$\sum_i\log P_\theta(x_i,y_i)$", "把乘积变求和，数值更稳", "3 似然"),   # term
    ("负对数似然", "negative log-likelihood (NLL)", "", r"$-\sum_i\log P_\theta(x_i,y_i)$", "常用作损失函数", "3 似然"),   # term
    ("条件似然", "conditional likelihood", "", r"$P_\theta(y\mid x)$", "判别式建模，只关心 y 给定 x", "3 似然"),   # term
    ("判别式模型", "discriminative model", "", r"$P(y\mid x)$", "直接建模条件分布", "3 似然"),                 # term
    ("生成式模型", "generative model", "", r"$P(x,y)$", "建模联合分布", "3 似然"),                           # term
    ("交叉熵", "cross-entropy", "", r"$-\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]$", "分类的 NLL 形式", "3 似然"),   # term
    ("KL 散度", "Kullback-Leibler divergence", "", r"$D_{KL}(p\Vert q)$", "两个分布的差异，交叉熵减熵", "3 似然"),   # term
    ("下溢", "underflow", "", "", "概率连乘趋近于 0，需取对数", "3 似然"),                                      # term
    ("对数求和指数", "log-sum-exp", "", r"$\log\sum_k e^{z_k}$", "稳定计算对数概率和的技巧", "3 似然"),             # term
    ("线性分类器", "linear classifier", "", r"$w^\top x=0$", "用超平面划分两类", "4 分类"),                    # term
    ("超平面", "hyperplane", "", r"$w^\top x+b=0$", "线性决策边界", "4 分类"),                             # term
    ("间隔", "margin", "", "", "样本到决策边界的距离", "4 分类"),                                             # term
    ("阶跃函数", "step function", "", r"$\mathrm{step}(z)=\mathbb{I}[z>0]$", "硬分类器，不可导", "4 分类"),   # term
    ("sigmoid 函数", "sigmoid function", r"$\sigma(z)$", r"$1/(1+e^{-z})$", "把实数压到 (0,1)", "4 分类"),   # term
    ("逻辑回归", "logistic regression", "", r"$P(y=1\mid x)=\sigma(w^\top x)$", "sigmoid + 交叉熵（MLE）", "4 分类"),   # term
    ("对数几率", "log-odds", "", r"$\log\frac{p}{1-p}=w^\top x$", "系数解释为对数几率的变化", "4 分类"),          # term
    ("几率比", "odds ratio", "", r"$e^{w_j}$", "特征增加一单位对几率的倍数", "4 分类"),                           # term
    ("梯度下降", "gradient descent", "", r"$w\leftarrow w-\eta\nabla L$", "沿负梯度迭代下降", "4 分类"),      # term
    ("学习率", "learning rate", r"$\eta$", "", "步长，过大发散、过小慢", "4 分类"),                             # term
    ("批量", "batch", "", "", "一次更新使用的样本数", "4 分类"),                                              # term
    ("自动微分", "automatic differentiation", "", "", "由计算图自动求梯度（如 PyTorch）", "4 分类"),              # term
    ("凸函数", "convex function", "", "", "局部最优即全局最优", "4 分类"),                                    # term
    ("softmax", "softmax", "", r"$e^{z_k}/\sum_j e^{z_j}$", "多分类的概率输出", "4 分类"),                # term
    ("类别不平衡", "class imbalance", "", "", "正类稀少，准确率会误导", "4 分类"),                                # term
    ("阈值", "threshold", "", "", "概率转硬标签的临界值", "4 分类"),                                          # term
    ("第一类错误", "Type I error (false positive)", "", "", "误报：把负例判为正", "4 分类"),                    # term
    ("第二类错误", "Type II error (false negative)", "", "", "漏判：把正例判为负", "4 分类"),                   # term
    ("混淆矩阵", "confusion matrix", "", "", "TP/FP/FN/TN 四格表", "4 分类"),                            # term
    ("ROC 曲线", "ROC curve", "", "", "不同阈值下的 TPR 与 FPR", "4 分类"),                                # term
    ("AUC", "area under the ROC curve", "", "", "排序能力的单一指标", "4 分类"),                           # term
    ("精确率", "precision", "", r"$TP/(TP+FP)$", "预测为正中有多少真的正", "4 分类"),                           # term
    ("召回率", "recall / sensitivity", "", r"$TP/(TP+FN)$", "真的正中有多少被找到", "4 分类"),                 # term
    ("概率校准", "probability calibration", "", "", "预测概率与实际频率一致", "4 分类"),                         # term
    ("等渗回归", "isotonic regression", "", "", "单调校准方法", "4 分类"),                                  # term
    ("Platt 缩放", "Platt scaling", "", "", "用逻辑回归做校准", "4 分类"),                                  # term
    ("布里尔分数", "Brier score", "", r"$\frac1n\sum(p_i-y_i)^2$", "概率预测的均方误差", "4 分类"),             # term
    ("正则化", "regularization", "", "", "用惩罚项控制模型复杂度", "5 正则"),                                   # term
    ("惩罚回归", "penalized regression", "", r"$\text{RSS}+\lambda\,\Omega(w)$", "损失加惩罚项", "5 正则"),   # term
    ("L1 惩罚", "L1 penalty / LASSO", "", r"$\lambda\sum_j|w_j|$", "产生稀疏解，自动选择特征", "5 正则"),       # term
    ("L2 惩罚", "L2 penalty / Ridge", "", r"$\lambda\sum_j w_j^2$", "向零收缩但不为零", "5 正则"),          # term
    ("弹性网", "elastic net", "", r"$\lambda(\alpha\sum|w_j|+(1-\alpha)\sum w_j^2)$", "L1 与 L2 的混合", "5 正则"),   # term
    ("超参数", "hyper-parameter", r"$\lambda$", "", "学习前需设定的参数", "5 正则"),                          # term
    ("稀疏解", "sparse solution", "", "", "多数系数恰为零", "5 正则"),                                      # term
    ("收缩", "shrinkage", "", "", "系数向零靠拢", "5 正则"),                                              # term
    ("偏差-方差分解", "bias-variance decomposition", "", r"$E[(y-\hat f)^2]=\text{Bias}^2+\text{Var}+\sigma^2$", "期望误差的三个来源", "5 正则"),   # term
    ("近似误差", "approximation error", "", "", "函数类表达能力的限制", "5 正则"),                              # term
    ("交叉验证", "cross-validation", "", "", "轮流留出验证折评估模型", "5 正则"),                                # term
    ("标准误", "standard error", "", r"$\hat\sigma/\sqrt n$", "估计量的抽样波动", "5 正则"),                 # term
    ("置信区间", "confidence interval", "", "", "参数的区间估计", "5 正则"),                                 # term
    ("p 值", "p-value", "", "", "在原假设下观测到该统计量的概率", "5 正则"),                                      # term
]                                                                       # # end of the term list

# ---------------------------------------------------------------- 2) persist the table
glossary = pd.DataFrame(TERMS, columns=["term_zh", "term_en", "symbol", "formula", "note", "part"])   # build the frame
dump(glossary, "l6_glossary.csv")                                  # machine-readable CSV in the data folder
counts = glossary["part"].value_counts().sort_index()             # terms per part
print("glossary terms:", len(glossary))                            # report the total
print(counts.to_string())                                          # report the split by part
assert len(glossary) >= 60, "the glossary should hold at least sixty terms"   # acceptance check
md_lines = ["# Lecture 6 glossary", "",                               # markdown title
            "| 术语 (zh) | Term (en) | Symbol | Formula | Note | Part |",   # header row
            "| --- | --- | --- | --- | --- | --- |"]                 # separator row
for _, r in glossary.iterrows():                                   # one row per term
    md_lines.append("| %s | %s | %s | %s | %s | %s |" % (r["term_zh"], r["term_en"],   # Chinese and English
                                                         r["symbol"], r["formula"], r["note"], r["part"]))   # rest
(OUT / "l6_glossary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")   # markdown table, no extra dependency
print("[artifact]", OUT / "l6_glossary.md")                        # log the markdown path
print("done: glossary with %d terms" % len(glossary))              # closing log
