"""l15_11_glossary.py -- the bilingual glossary for Lecture 15.

Every line carries an English comment. The script writes the machine-readable term table
(Chinese, English, symbol, formula, note, part) that feeds the two glossary frames, the site
table and the printable Word/Excel handout.
"""
import pandas as pd                                                # build and persist the table
from _common_l15 import dump, OUT                                  # shared helpers

# Columns: term_zh, term_en, symbol, formula, note, part.
TERMS = [                                                               # term list
    ("构念", "construct", "", "", "要测但不可直接观测的概念", "1 测量"),                              # measurement
    ("操作化", "operationalisation", "", "", "把构念变成可测量的题项", "1 测量"),                      # measurement
    ("李克特量表", "Likert scale", "", "", "1-5 或 1-7 的同意程度", "1 测量"),                        # measurement
    ("信度", "reliability", "", "", "重复测量的一致性", "1 测量"),                                    # measurement
    ("克隆巴赫 α", "Cronbach's alpha", r"$\alpha$", r"$\frac{k}{k-1}(1-\frac{\sum\sigma_i^2}{\sigma_T^2})$", "内部一致性系数", "1 测量"),   # measurement
    ("效度", "validity", "", "", "测到的确实是构念", "1 测量"),                                      # measurement
    ("内容效度", "content validity", "", "", "题项覆盖构念的全部维度", "1 测量"),                      # measurement
    ("收敛效度", "convergent validity", "", "", "同构念的不同测量应高度相关", "1 测量"),               # measurement
    ("区分效度", "discriminant validity", "", "", "不同构念的测量应低相关", "1 测量"),                 # measurement
    ("探索性因子分析", "exploratory factor analysis (EFA)", "", "", "用降维找潜在维度", "1 测量"),      # measurement
    ("抽样框", "sampling frame", "", "", "抽样的名单", "1 测量"),                                     # measurement
    ("分层抽样", "stratified sampling", "", "", "按层分别抽样再合并", "1 测量"),                       # measurement
    ("配额抽样", "quota sampling", "", "", "按比例凑样本，非概率", "1 测量"),                          # measurement
    ("无回答偏差", "non-response bias", "", "", "回答者与不回答者系统不同", "1 测量"),                  # measurement
    ("模式效应", "mode effect", "", "", "访问方式改变答案分布", "1 测量"),                             # measurement
    ("共同方法偏差", "common method bias", "", "", "同一测量方式带来的系统性偏差", "1 测量"),            # measurement
    ("测量误差", "measurement error", "", "", "观测值 = 真值 + 误差", "1 测量"),                        # measurement
    ("联合分析", "conjoint analysis", "", "", "评价属性组合而非单个属性", "2 选择"),                    # choice
    ("属性", "attribute", "", "", "产品的可比较维度", "2 选择"),                                       # choice
    ("水平", "level", "", "", "属性的具体取值", "2 选择"),                                            # choice
    ("正交设计", "orthogonal design", "", "", "各属性水平不相关，减少轮廓数", "2 选择"),               # choice
    ("部分因子设计", "fractional factorial design", "", "", "只取全因子的一部分", "2 选择"),           # choice
    ("D-最优设计", "D-optimal design", "", "", "按信息矩阵优化设计", "2 选择"),                        # choice
    ("效用", "utility (part-worth)", "", r"$u_{jl}$", "某一属性水平的边际价值", "2 选择"),        # choice
    ("属性重要性", "attribute importance", "", r"$\frac{\max u-\min u}{\sum(\max u-\min u)}$", "效用极差占比", "2 选择"),   # choice
    ("离散选择模型", "discrete choice model", "", "", "在有限方案中选择", "2 选择"),                   # choice
    ("随机效用", "random utility", "", r"$U=V+\varepsilon$", "可观测效用加随机项", "2 选择"),      # choice
    ("logit 模型", "logit model", "", r"$P_j=\frac{e^{V_j}}{\sum_k e^{V_k}}$", "Gumbel 误差下的闭式解", "2 选择"),   # choice
    ("probit 模型", "probit model", "", "", "正态误差下的选择模型", "2 选择"),                          # choice
    ("无关选项独立性", "independence of irrelevant alternatives (IIA)", "", "", "加入新选项按比例分走份额", "2 选择"),   # choice
    ("嵌套 logit", "nested logit", "", "", "允许组内相关的方法", "2 选择"),                            # choice
    ("混合 logit", "mixed logit", "", "", "随机系数的 logit", "2 选择"),                               # choice
    ("选择集", "choice set", "", "", "消费者实际考虑的方案集合", "2 选择"),                            # choice
    ("边际效应", "marginal effect", "", r"$\partial P_j/\partial x$", "某属性变化对概率的影响", "2 选择"),   # choice
    ("特征价格", "hedonic pricing", "", r"$\ln p=\beta_0+\sum\beta_kx_k$", "把价格拆成属性价值", "3 反应"),   # response
    ("半弹性", "semi-elasticity", "", r"$e^{\beta}-1$", "对数-线性模型里属性的百分比影响", "3 反应"),   # response
    ("分位数回归", "quantile regression", "", "", "看不同价位上的属性价值", "3 反应"),                  # response
    ("响应模型", "response model", "", "", "营销投入与销量的关系", "3 反应"),                           # response
    ("饱和效应", "saturation", "", "", "投入越多边际效果越小", "3 反应"),                              # response
    ("adstock", "adstock", "", r"$A_t=S_t+\lambda A_{t-1}$", "广告的滞后与衰减", "3 反应"),            # response
    ("结转率", "carryover rate", r"$\lambda$", "", "上一期效果的保留比例", "3 反应"),                   # response
    ("基线销量", "baseline sales", "", "", "没有营销活动时的销量", "3 反应"),                           # response
    ("增量", "increment", "", "", "活动带来的超出基线的部分", "3 反应"),                                # response
    ("点击流", "clickstream", "", "", "用户在页面上的点击序列", "4 客户"),                              # customer
    ("会话", "session", "", "", "一次连续访问", "4 客户"),                                             # customer
    ("漏斗", "funnel", "", "", "分阶段的转化路径", "4 客户"),                                          # customer
    ("转化率", "conversion rate", "", "", "完成目标行为的比例", "4 客户"),                              # customer
    ("RFM", "recency, frequency, monetary (RFM)", "", "", "最近/频次/金额的客户打分", "4 客户"),        # customer
    ("客户旅程", "customer journey", "", "", "从认知到复购的全过程", "4 客户"),                         # customer
    ("归因", "attribution", "", "", "把转化功劳分给各触点", "4 客户"),                                  # customer
    ("末次点击", "last-click attribution", "", "", "全部功劳给最后一次触点", "4 客户"),                 # customer
    ("线性归因", "linear attribution", "", "", "各触点平分功劳", "4 客户"),                             # customer
    ("时间衰减归因", "time-decay attribution", "", "", "越近的触点权重越高", "4 客户"),                 # customer
    ("数据驱动归因", "data-driven attribution", "", "", "用模型分摊贡献", "4 客户"),                    # customer
    ("夏普利值", "Shapley value", "", "", "博弈论的分摊方法", "4 客户"),                                # customer
    ("马尔可夫归因", "Markov attribution", "", "", "用吸收态马尔可夫链分摊", "4 客户"),                 # customer
    ("围墙花园", "walled garden", "", "", "平台内闭环、数据不外流", "4 客户"),                          # customer
    ("身份解析", "identity resolution", "", "", "把多设备多来源的 ID 打通", "4 客户"),                  # customer
    ("隐私增强技术", "privacy-enhancing technology (PET)", "", "", "差分隐私、联邦学习、可信执行环境", "4 客户"),   # customer
    ("一方数据", "first-party data", "", "", "企业自己收集的数据", "4 客户"),                            # customer
    ("协同过滤", "collaborative filtering", "", "", "用相似用户/物品做推荐", "5 推荐"),                 # recsys
    ("矩阵分解", "matrix factorisation", "", r"$R\approx UV^\top$", "把评分矩阵拆成隐因子", "5 推荐"),   # recsys
    ("交替最小二乘", "alternating least squares (ALS)", "", "", "轮流固定一侧求解另一侧", "5 推荐"),       # recsys
    ("冷启动", "cold start", "", "", "新用户/新物品缺少历史", "5 推荐"),                                # recsys
    ("精确率@k", "precision@k", "", "", "前 k 个推荐里命中的比例", "5 推荐"),                           # recsys
    ("NDCG", "normalised discounted cumulative gain", "", "", "考虑排序位置的命中质量", "5 推荐"),       # recsys
    ("双塔模型", "two-tower model", "", "", "用户塔与物品塔分别编码", "5 推荐"),                        # recsys
    ("价格弹性", "price elasticity", r"$\varepsilon$", r"$\frac{\partial\ln q}{\partial\ln p}$", "价格变动 1% 时需求变动的百分比", "5 推荐"),   # recsys
    ("交叉弹性", "cross-price elasticity", "", "", "另一商品价格变化的影响", "5 推荐"),                 # recsys
    ("加成定价", "markup pricing", "", r"$p^*=\frac{c\varepsilon}{1+\varepsilon}$", "由弹性推出的最优价", "5 推荐"),   # recsys
    ("边际成本", "marginal cost", r"$c$", "", "多卖一件增加的成本", "5 推荐"),                          # recsys
    ("促销增量", "promotion uplift", "", "", "促销带来的额外销量", "5 推荐"),                           # recsys
    ("价格歧视", "price discrimination", "", "", "对不同人定不同价（有法律边界）", "5 推荐"),            # recsys
    ("流失", "churn", "", "", "客户停止使用或离开", "6 实验"),                                         # experiment
    ("期望损失", "expected loss", "", "", "把概率换算成钱的决策准则", "6 实验"),                         # experiment
    ("客户终身价值", "customer lifetime value (CLV)", "", r"$\sum r^t s_t m_t$", "折现后的未来贡献", "6 实验"),   # experiment
    ("留存曲线", "retention curve", r"$s_t$", "", "第 t 期仍留下的比例", "6 实验"),                     # experiment
    ("折现率", "discount rate", r"$r$", "", "未来一块钱折成现在的价值", "6 实验"),                       # experiment
    ("A/B 实验", "A/B test", "", "", "随机分组的对照实验", "6 实验"),                                   # experiment
    ("统计功效", "statistical power", "", "", "真有效应时能检出的概率", "6 实验"),                       # experiment
    ("最小可检测效应", "minimum detectable effect (MDE)", "", "", "给定样本量能稳定检出的最小差异", "6 实验"),   # experiment
    ("CUPED", "controlled-experiment using pre-experiment data", "", "", "用前期协变量降方差", "6 实验"),   # experiment
    ("增量实验", "incrementality test", "", "", "测广告带来的真实增量", "6 实验"),                       # experiment
    ("提升模型", "uplift model", "", "", "估计个体层面的处理效应", "6 实验"),                            # experiment
    ("营销组合模型", "marketing mix model (MMM)", "", "", "用回归分解各渠道贡献", "6 实验"),              # experiment
]                                                                       # end of the term list

# ---------------------------------------------------------------- persist the table
glossary = pd.DataFrame(TERMS, columns=["term_zh", "term_en", "symbol", "formula", "note", "part"])   # build the frame
dump(glossary, "l15_glossary.csv")                                 # machine-readable CSV
counts = glossary["part"].value_counts().sort_index()              # terms per part
print("glossary terms:", len(glossary))                            # report the total
print(counts.to_string())                                          # report the split
assert len(glossary) >= 60, "the glossary should hold at least sixty terms"   # acceptance check
md_lines = ["# Lecture 15 glossary", "",                            # markdown title
            "| 术语 (zh) | Term (en) | Symbol | Formula | Note | Part |",   # header
            "| --- | --- | --- | --- | --- | --- |"]                # separator
for _, r in glossary.iterrows():                                   # one row per term
    md_lines.append("| %s | %s | %s | %s | %s | %s |" % (r["term_zh"], r["term_en"],   # Chinese and English
                                                         r["symbol"], r["formula"], r["note"], r["part"]))   # the rest
(OUT / "l15_glossary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")   # markdown table
print("[artifact]", OUT / "l15_glossary.md")                        # log the path
print("done: glossary with %d terms" % len(glossary))              # closing log
