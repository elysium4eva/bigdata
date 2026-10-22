"""l7_12_glossary.py -- the bilingual glossary for Lecture 7.

Every line carries an English comment. The script writes the machine-readable term
table (Chinese, English, symbol, formula, note, part) that feeds the two glossary
frames in the deck, the site table and the printable Word/Excel handout.
"""
import pandas as pd                                                # build and persist the table
from _common_l7 import dump, OUT                                   # shared helpers

# Columns: term_zh, term_en, symbol, formula, note, part.
TERMS = [                                                               # term list: zh, en, symbol, formula, note, part
    ("监督学习", "supervised learning", "", "", "从带标签数据学习映射", "1 框架"),               # concept
    ("函数类", "function class", r"$\mathcal{H}$", "", "候选函数的集合，决定表达能力", "1 框架"),   # concept
    ("正则项", "regularizer", r"$\Omega(w)$", "", "惩罚复杂度，限制表达能力", "1 框架"),           # concept
    ("优化算法", "optimizer", "", "", "在函数类里找最优解的手段", "1 框架"),                      # concept
    ("归纳偏置", "inductive bias", "", "", "模型对未见数据的先验假设", "1 框架"),                  # concept
    ("留出样本", "hold-out sample", "", "", "不参与训练、只用于评价的数据", "1 框架"),              # concept
    ("交叉验证", "cross-validation", "", "", "轮流留出一折做验证", "1 框架"),                     # concept
    ("防火墙原则", "firewall principle", "", "", "训练与评价样本必须隔离", "1 框架"),               # concept
    ("超参数", "hyper-parameter", "", "", "训练前设定、不通过梯度学习的参数", "1 框架"),            # concept
    ("调参", "hyper-parameter tuning", "", "", "用验证集选超参数", "1 框架"),                     # concept
    ("网格搜索", "grid search", "", "", "穷举离散组合", "1 框架"),                                # concept
    ("随机搜索", "random search", "", "", "随机采样组合，更省预算", "1 框架"),                     # concept
    ("贝叶斯优化", "Bayesian optimisation", "", "", "用代理模型指导下一次试验（Optuna 的 TPE）", "1 框架"),   # concept
    ("基线", "baseline", "", "", "先跑通的最简单参照模型", "1 框架"),                              # concept
    ("支持向量机", "support vector machine (SVM)", "", "", "以最大间隔划分两类的线性/核方法", "2 SVM"),   # svm
    ("超平面", "hyperplane", "", r"$w^\top x+b=0$", "线性决策边界", "2 SVM"),                     # svm
    ("间隔", "margin", "", r"$2/\lVert w\rVert$", "两类样本到边界的距离带", "2 SVM"),              # svm
    ("最大间隔", "maximum margin", "", r"$\min\frac12\lVert w\rVert^2$", "SVM 的优化目标", "2 SVM"),   # svm
    ("支持向量", "support vector", "", "", "决定边界的那几个样本", "2 SVM"),                        # svm
    ("软间隔", "soft margin", "", r"$\xi_i\ge0$", "允许少量样本越过间隔", "2 SVM"),                # svm
    ("惩罚参数", "penalty parameter", r"$C$", "", "C 越大对误分类惩罚越重", "2 SVM"),               # svm
    ("核函数", "kernel function", "", r"$K(x,z)$", "把内积换成相似度，隐式升维", "2 SVM"),          # svm
    ("核技巧", "kernel trick", "", "", "不用显式计算高维映射", "2 SVM"),                           # svm
    ("多项式核", "polynomial kernel", "", r"$(\gamma x^\top z+r)^d$", "常用核之一", "2 SVM"),      # svm
    ("径向基核", "radial basis function (RBF) kernel", "", r"$e^{-\gamma\lVert x-z\rVert^2}$", "最常用的核", "2 SVM"),   # svm
    ("拉格朗日对偶", "Lagrangian dual", "", "", "把约束优化转成只含内积的形式", "2 SVM"),            # svm
    ("稀疏性", "sparsity", "", "", "只有少数样本/系数起作用", "2 SVM"),                            # svm
    ("概率校准", "probability calibration", "", "", "让输出概率与实际频率一致（Platt）", "2 SVM"),   # svm
    ("k 近邻", "k-nearest neighbours (k-NN)", "", "", "按最近的 k 个样本投票或取平均", "3 KNN"),    # knn
    ("非参数方法", "non-parametric method", "", "", "不假设固定函数形式、参数随数据增长", "3 KNN"),   # knn
    ("距离度量", "distance metric", "", "", "欧氏/L1/余弦等，决定近邻是谁", "3 KNN"),            # knn
    ("余弦相似度", "cosine similarity", "", r"$\frac{x^\top z}{\lVert x\rVert\lVert z\rVert}$", "文本与稀疏数据常用", "3 KNN"),   # knn
    ("维度灾难", "curse of dimensionality", "", "", "高维下距离集中，近邻失去意义", "3 KNN"),        # knn
    ("原型", "prototype", "", "", "数据约简后保留的代表点", "3 KNN"),                              # knn
    ("被吸收点", "absorbed point", "", "", "可被原型正确分类、因而可删除的点", "3 KNN"),            # knn
    ("类别离群点", "class outlier", "", "", "被异类包围的样本，会制造噪声", "3 KNN"),                # knn
    ("决策树", "decision tree", "", "", "递归二分形成的规则模型", "4 树"),                          # tree
    ("分类回归树", "classification and regression tree (CART)", "", "", "可做分类也可做回归的树", "4 树"),   # tree
    ("基尼不纯度", "Gini impurity", "", r"$1-\sum_k p_k^2$", "分类分裂准则", "4 树"),              # tree
    ("信息增益", "information gain", "", r"$H(p)-\sum\frac{n_k}{n}H(p_k)$", "熵的下降量", "4 树"),   # tree
    ("熵", "entropy", "", r"$-\sum_k p_k\log p_k$", "不确定性的度量", "4 树"),                     # tree
    ("分裂准则", "splitting criterion", "", "", "每个节点选哪一刀的依据", "4 树"),                   # tree
    ("贪心分裂", "greedy splitting", "", "", "每步只选当下最优，非全局最优", "4 树"),               # tree
    ("剪枝", "pruning", "", "", "砍掉不带来增益的分支", "4 树"),                                    # tree
    ("预剪枝", "pre-pruning", "", "", "生长时提前停止（深度、叶大小）", "4 树"),                     # tree
    ("后剪枝", "post-pruning", "", "", "先长满再回剪", "4 树"),                                    # tree
    ("代价复杂度剪枝", "cost-complexity pruning", "", r"$R_\alpha(T)=R(T)+\alpha|T|$", "用 alpha 控制树的大小", "4 树"),   # tree
    ("叶节点", "leaf", "", "", "树的终端结点，输出预测值", "4 树"),                                 # tree
    ("分段常数", "piecewise constant", "", "", "树的函数形式：每个叶一个常数", "4 树"),             # tree
    ("集成学习", "ensemble learning", "", "", "组合多个模型的预测", "5 集成"),                      # ensemble
    ("袋装", "bagging", "", "", "自助抽样并行训练，再平均/投票", "5 集成"),                        # ensemble
    ("自助抽样", "bootstrap", "", "", "有放回抽样构造训练子集", "5 集成"),                          # ensemble
    ("袋外样本", "out-of-bag (OOB)", "", "", "没被抽到的样本，可当免费验证集", "5 集成"),            # ensemble
    ("随机森林", "random forest", "", "", "树 + 自助抽样 + 特征子采样", "5 集成"),                  # ensemble
    ("特征子采样", "feature subsampling", "", "", "每次分裂只看部分特征，降低相关性", "5 集成"),       # ensemble
    ("提升", "boosting", "", "", "串行训练，每轮纠正前面的错误", "5 集成"),                         # ensemble
    ("梯度提升", "gradient boosting", "", "", "在函数空间做梯度下降", "5 集成"),                    # ensemble
    ("学习率", "learning rate (shrinkage)", r"$\eta$", "", "每棵树的贡献权重，越小越稳", "5 集成"),   # ensemble
    ("早停", "early stopping", "", "", "验证误差不再下降就停", "5 集成"),                            # ensemble
    ("弱学习器", "weak learner", "", "", "比随机猜略好的模型（如树桩）", "5 集成"),                  # ensemble
    ("堆叠", "stacking", "", "", "用元模型组合多个基模型", "5 集成"),                                # ensemble
    ("置换重要性", "permutation importance", "", "", "打乱某特征后误差的上升量", "5 集成"),          # ensemble
    ("SHAP 值", "SHAP value", "", "", "基于 Shapley 值的特征贡献分解", "5 集成"),                   # ensemble
    ("摘要图", "summary plot", "", "", "SHAP 的全局重要性可视化", "5 集成"),                        # ensemble
    ("依赖图", "dependence plot", "", "", "某特征取值与其 SHAP 值的关系", "5 集成"),                # ensemble
    ("代价敏感学习", "cost-sensitive learning", "", r"$C_{FN}FN+C_{FP}FP$", "按错误代价加权训练/决策", "6 取舍"),   # tuning
    ("类别不平衡", "class imbalance", "", "", "正类稀少，准确率会误导", "6 取舍"),                   # tuning
    ("阈值", "threshold", "", "", "概率转硬标签的临界值", "6 取舍"),                                # tuning
    ("表格数据", "tabular data", "", "", "行列结构的数据，GBDT 的主场", "6 取舍"),                   # tuning
    ("嵌入", "embeddings", "", "", "把高维输入映射为稠密向量", "6 取舍"),                            # tuning
    ("微调", "fine-tuning", "", "", "在预训练模型上继续训练", "6 取舍"),                             # tuning
    ("线性探测", "linear probing", "", "", "冻结主干、只训练分类头", "6 取舍"),                      # tuning
]                                                                       # end of the term list

# ---------------------------------------------------------------- persist the table
glossary = pd.DataFrame(TERMS, columns=["term_zh", "term_en", "symbol", "formula", "note", "part"])   # build the frame
dump(glossary, "l7_glossary.csv")                                  # machine-readable CSV in the data folder
counts = glossary["part"].value_counts().sort_index()              # terms per part
print("glossary terms:", len(glossary))                            # report the total
print(counts.to_string())                                          # report the split by part
assert len(glossary) >= 60, "the glossary should hold at least sixty terms"   # acceptance check
md_lines = ["# Lecture 7 glossary", "",                            # markdown title
            "| 术语 (zh) | Term (en) | Symbol | Formula | Note | Part |",   # header row
            "| --- | --- | --- | --- | --- | --- |"]                # separator row
for _, r in glossary.iterrows():                                   # one row per term
    md_lines.append("| %s | %s | %s | %s | %s | %s |" % (r["term_zh"], r["term_en"],   # Chinese and English
                                                         r["symbol"], r["formula"], r["note"], r["part"]))   # rest
(OUT / "l7_glossary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")   # markdown table
print("[artifact]", OUT / "l7_glossary.md")                        # log the markdown path
print("done: glossary with %d terms" % len(glossary))              # closing log
