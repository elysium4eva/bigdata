"""l12_11_glossary.py -- the bilingual glossary for Lecture 12.

Every line carries an English comment. The script writes the machine-readable term table   # continuation of the call
(Chinese, English, symbol, formula, note, part) that feeds the two glossary frames, the site   # continuation of the call
table and the printable Word/Excel handout.                             # continuation of the call
"""
import pandas as pd                                                # build and persist the table
from _common_l12 import dump, OUT                                  # shared helpers

# Columns: term_zh, term_en, symbol, formula, note, part.
TERMS = [                                                               # continuation of the call
    ("相关", "association", "", "", "两个变量一起变，不代表一个导致另一个", "1 概念"),            # concept
    ("因果", "causation", "", "", "干预一个变量会改变另一个", "1 概念"),                          # concept
    ("反事实", "counterfactual", "", "", "如果没有做这件事，结果会怎样", "1 概念"),               # concept
    ("潜在结果", "potential outcome", r"$Y_i(z)$", "", "同一单位在不同处理下的结果", "1 概念"),   # concept
    ("处理", "treatment", r"$Z$", "", "被研究的干预", "1 概念"),                                 # concept
    ("混杂", "confounder", r"$U$", "", "同时影响处理与结果的变量", "1 概念"),                     # concept
    ("中介", "mediator", "", "", "位于处理与结果之间、传递效应的变量", "1 概念"),                  # concept
    ("对撞点", "collider", "", "", "被两个变量共同影响的变量", "1 概念"),                          # concept
    ("因果图", "directed acyclic graph (DAG)", "", "", "用箭头表示因果假设的图", "1 概念"),        # concept
    ("后门准则", "back-door criterion", "", "", "找一组变量切断所有后门路径", "1 概念"),            # concept
    ("前门准则", "front-door criterion", "", "", "用中介识别因果的另一条路", "1 概念"),            # concept
    ("辛普森悖论", "Simpson's paradox", "", "", "合并与分层的结论相反", "1 概念"),                 # concept
    ("生态谬误", "ecological fallacy", "", "", "用群体结论推断个体", "1 概念"),                    # concept
    ("Cornfield 不等式", "Cornfield inequality", r"$RR_{ZU}\ge RR_{ZY}$", "", "混杂必须多强才能解释掉关联", "1 概念"),   # concept
    ("关联强度", "risk ratio", r"$RR$", "", "两组风险之比", "1 概念"),                            # concept
    ("识别", "identification", "", "", "在什么假设下因果量可被数据确定", "2 识别"),                # ident
    ("识别假设", "identifying assumption", "", "", "把不可观测量变成可观测量所依赖的假设", "2 识别"),   # ident
    ("无混杂", "unconfoundedness", r"$Z\perp Y(z)\mid X$", "", "控制 X 后处理与潜在结果独立", "2 识别"),   # ident
    ("重叠", "overlap", "", "", "每种协变量取值下都有处理与对照", "2 识别"),                       # ident
    ("SUTVA", "stable unit treatment value assumption", "", "", "一个人的处理不影响别人", "2 识别"),   # ident
    ("一致性", "consistency", r"$Y=Y(Z)$", "", "观测结果等于对应处理下的潜在结果", "2 识别"),       # ident
    ("分配机制", "assignment mechanism", r"$p(Z\mid X,Y(0),Y(1))$", "", "决定谁被处理的概率规则", "2 识别"),   # ident
    ("倾向得分", "propensity score", r"$e(X)=p(Z=1\mid X)$", "", "给定协变量被处理的概率", "2 识别"),   # ident
    ("平均处理效应", "average treatment effect (ATE)", r"$E[Y(1)-Y(0)]$", "", "全体的平均因果效应", "2 识别"),   # ident
    ("处理组平均效应", "average treatment effect on the treated (ATT)", "", "", "对实际被处理者的效应", "2 识别"),   # ident
    ("未处理组平均效应", "ATU", "", "", "对未被处理者的效应", "2 识别"),                            # ident
    ("局部平均处理效应", "local average treatment effect (LATE)", "", "", "顺从者的平均效应", "2 识别"),   # ident
    ("条件平均处理效应", "conditional average treatment effect (CATE)", r"$\tau(x)$", "", "给定特征的效应", "2 识别"),   # ident
    ("根本问题", "fundamental problem of causal inference", "", "", "同一单位只能观测到一种结果", "2 识别"),   # ident
    ("完美医生", "perfect doctor", "", "", "分配依赖潜在结果，差值估计失效", "2 识别"),             # ident
    ("选择偏差", "selection bias", "", "", "进入样本的方式造成的偏差", "2 识别"),                  # ident
    ("平行趋势", "parallel trends", "", "", "没有处理时两组走势相同", "3 策略"),                    # strategy
    ("双重差分", "difference-in-differences (DID)", "", r"$E[Y_{1}-Y_{0}\mid T]-E[Y_{1}-Y_{0}\mid C]$", "用时间换识别", "3 策略"),   # strategy
    ("事件研究", "event study", "", "", "逐期估计处理前后的效应", "3 策略"),                       # strategy
    ("匹配", "matching", "", "", "给处理组找可比的对照组", "3 策略"),                              # strategy
    ("标准化差异", "standardised difference", "", "", "平衡性的常用度量，小于 0.1 为好", "3 策略"),   # strategy
    ("共同支撑域", "common support", "", "", "处理组与对照组倾向得分的重叠区间", "3 策略"),          # strategy
    ("工具变量", "instrumental variable (IV)", r"$Z$", "", "只通过处理影响结果的外生变量", "3 策略"),   # strategy
    ("相关性条件", "relevance", "", "", "工具必须影响处理", "3 策略"),                             # strategy
    ("排他性", "exclusion restriction", "", "", "工具只能通过处理影响结果", "3 策略"),              # strategy
    ("两阶段最小二乘", "two-stage least squares (2SLS)", "", "", "先预测处理，再回归结果", "3 策略"),   # strategy
    ("弱工具", "weak instrument", "", "", "第一阶段太弱，估计偏且不稳", "3 策略"),                  # strategy
    ("第一阶段 F", "first-stage F", "", "", "弱工具的常用检验，$F<10$ 要警惕", "3 策略"),           # strategy
    ("断点回归", "regression discontinuity (RD)", "", "", "用门槛附近的跳跃识别", "3 策略"),        # strategy
    ("运行变量", "running variable", "", "", "决定是否跨过门槛的变量", "3 策略"),                   # strategy
    ("带宽", "bandwidth", "", "", "断点两侧取多宽的窗口", "3 策略"),                               # strategy
    ("合成控制", "synthetic control", "", "", "用加权组合构造反事实", "3 策略"),   # strategy
    ("敏感性分析", "sensitivity analysis", "", "", "需要多强的混杂才能推翻结论", "3 策略"),   # strategy
    ("E 值", "E-value", "", "", "推翻结论所需的最小关联强度", "3 策略"),                            # strategy
    ("随机实验", "randomised experiment", "", "", "随机化切断所有指向处理的箭头", "4 实验"),        # experiment
    ("A/B 测试", "A/B test", "", "", "两组随机的对照实验", "4 实验"),                              # experiment
    ("统计功效", "statistical power", r"$1-\beta$", "", "真有效应时能检出的概率", "4 实验"),         # experiment
    ("最小可检测效应", "minimum detectable effect (MDE)", "", "", "给定样本量能稳定检出的效应", "4 实验"),   # experiment
    ("第一类错误", "type I error", r"$\alpha$", "", "原假设为真却拒绝", "4 实验"),                  # experiment
    ("序贯检验", "sequential testing", "", "", "边做边看要校正阈值", "4 实验"),                    # experiment
    ("多重比较", "multiple comparisons", "", "", "指标越多，假阳性越多", "4 实验"),                 # experiment
    ("Bonferroni 校正", "Bonferroni correction", "", r"$\alpha/m$", "把显著性阈值除以检验次数", "4 实验"),   # experiment
    ("错误发现率", "false discovery rate (FDR)", "", "", "被拒绝的假设中假的比例", "4 实验"),       # experiment
    ("聚类随机", "cluster randomisation", "", "", "按群（城市/门店）随机化", "4 实验"),             # experiment
    ("AA 测试", "AA test", "", "", "两组都给旧方案，检查分流是否均衡", "4 实验"),                   # experiment
    ("外部有效性", "external validity", "", "", "实验结论能否推广到别处", "4 实验"),                # experiment
    ("不依从", "non-compliance", "", "", "分配的处理没有被真正接受", "5 依从"),                    # compliance
    ("顺从者", "complier", "", "", "分配才接受的人", "5 依从"),                                    # compliance
    ("从不接受者", "never-taker", "", "", "无论分配都不接受的人", "5 依从"),                        # compliance
    ("始终接受者", "always-taker", "", "", "无论分配都接受的人", "5 依从"),                         # compliance
    ("反抗者", "defier", "", "", "分配才不接受的人；单调性假设排除", "5 依从"),                     # compliance
    ("单调性", "monotonicity", "", "", "没有反抗者", "5 依从"),                                    # compliance
    ("意向性治疗", "intention-to-treat (ITT)", "", "", "按分配而非接受来比较", "5 依从"),           # compliance
    ("Wald 估计", "Wald estimator", "", r"$\frac{ITT}{\text{take-up}}$", "LATE 的比值形式", "5 依从"),   # compliance
    ("信息性删失", "informative censoring", "", "", "失访与结果相关", "5 依从"),                    # compliance
    ("健康工人效应", "healthy worker bias", "", "", "在职者本身更健康", "5 依从"),                  # compliance
    ("自选择", "self-selection", "", "", "自己决定是否进入样本/处理", "5 依从"),                    # compliance
    ("双重机器学习", "double machine learning (DML)", "", "", "正交化 + 交叉拟合", "6 前沿"),   # frontier
    ("交叉拟合", "cross-fitting", "", "", "分折估计 nuisance 再预测留出折", "6 前沿"),               # frontier
    ("正交化", "orthogonalisation", "", "", "用残差对残差去掉 nuisance 偏差", "6 前沿"),            # frontier
    ("Neyman 正交", "Neyman orthogonality", "", "", "对 nuisance 的一阶导为零", "6 前沿"),          # frontier
    ("因果森林", "causal forest", "", "", "用森林估 CATE，需要 honest 切分", "6 前沿"),             # frontier
    ("诚实切分", "honest splitting", "", "", "估计效应与选分裂用不同子样本", "6 前沿"),              # frontier
    ("T-learner", "T-learner", "", "", "分别对处理组与对照组建模型再相减", "6 前沿"),                # frontier
    ("增量模型", "uplift model", "", "", "直接预测个体处理效应", "6 前沿"),                         # frontier
    ("异质性处理效应", "heterogeneous treatment effect", "", "", "效应因人而异", "6 前沿"),         # frontier
    ("亚组分析", "subgroup analysis", "", "", "按特征分组估计效应，注意多重比较", "6 前沿"),         # frontier
    ("模型治理", "model governance", "", "", "模型卡、监控、漂移与审计", "6 前沿"),                  # frontier
    ("知情同意", "informed consent", "", "", "实验参与者的知情与自愿", "6 前沿"),                   # frontier
]                                                                       # continuation of the call

# ---------------------------------------------------------------- persist the table
glossary = pd.DataFrame(TERMS, columns=["term_zh", "term_en", "symbol", "formula", "note", "part"])   # build the frame
dump(glossary, "l12_glossary.csv")                                 # machine-readable CSV
counts = glossary["part"].value_counts().sort_index()              # terms per part
print("glossary terms:", len(glossary))                            # report the total
print(counts.to_string())                                          # report the split
assert len(glossary) >= 60, "the glossary should hold at least sixty terms"   # acceptance check
md_lines = ["# Lecture 12 glossary", "",                            # markdown title
            "| 术语 (zh) | Term (en) | Symbol | Formula | Note | Part |",   # header
            "| --- | --- | --- | --- | --- | --- |"]                # separator
for _, r in glossary.iterrows():                                   # one row per term
    md_lines.append("| %s | %s | %s | %s | %s | %s |" % (r["term_zh"], r["term_en"],   # Chinese and English
                                                         r["symbol"], r["formula"], r["note"], r["part"]))   # the rest
(OUT / "l12_glossary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")   # markdown table
print("[artifact]", OUT / "l12_glossary.md")                        # log the path
print("done: glossary with %d terms" % len(glossary))              # closing log
