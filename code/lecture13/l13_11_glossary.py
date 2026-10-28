"""l13_11_glossary.py -- the bilingual glossary for Lecture 13.

Every line carries an English comment. The script writes the machine-readable term table   # continuation of the call
(Chinese, English, symbol, formula, note, part) that feeds the two glossary frames, the site   # continuation of the call
table and the printable Word/Excel handout.                             # continuation of the call
"""
import pandas as pd                                                # build and persist the table
from _common_l13 import dump, OUT                                  # shared helpers

# Columns: term_zh, term_en, symbol, formula, note, part.
TERMS = [                                                               # continuation of the call
    ("资产类别", "asset class", "", "", "股票/债券/衍生品/另类", "1 资产"),                        # assets
    ("固定收益", "fixed income", "", "", "现金流确定的资产", "1 资产"),                            # assets
    ("到期收益率", "yield to maturity (YTM)", r"$y$", "", "使现值等于价格的贴现率", "1 资产"),      # assets
    ("久期", "duration", r"$D$", r"$-\frac{1}{P}\frac{\partial P}{\partial y}$", "价格对利率的敏感度", "1 资产"),   # assets
    ("凸性", "convexity", "", "", "久期本身也随利率变化", "1 资产"),                                # assets
    ("收益率曲线", "yield curve", "", "", "不同期限的收益率", "1 资产"),                            # assets
    ("倒挂", "inverted yield curve", "", "", "短端高于长端，常被视为衰退信号", "1 资产"),            # assets
    ("总回报", "total return", "", "", "含分红与拆股的收益", "1 资产"),                              # assets
    ("对数收益率", "log return", r"$r_t=\ln(P_t/P_{t-1})$", "", "可加、近似百分比", "1 资产"),        # assets
    ("厚尾", "fat tails", "", "", "极端值比正态多", "1 资产"),                                       # assets
    ("峰度", "kurtosis", "", "", "尾部厚薄的度量", "1 资产"),                                        # assets
    ("远期", "forward", "", "", "场外约定未来交割", "1 资产"),                                       # assets
    ("期货", "futures", "", "", "交易所标准化合约，逐日盯市", "1 资产"),                              # assets
    ("期权", "option", "", "", "买方有权利无义务", "1 资产"),                                        # assets
    ("隐含波动率", "implied volatility", r"$\sigma_{IV}$", "", "由期权价格反解的波动率", "1 资产"),   # assets
    ("波动率微笑", "volatility smile", "", "", "IV 随行权价呈 U 形", "1 资产"),                      # assets
    ("另类投资", "alternative investment", "", "", "私募、房地产、大宗商品", "1 资产"),               # assets
    ("复权", "adjusted price", "", "", "剔除分红与拆股影响的连续价格", "2 数据"),                     # data
    ("时间戳对齐", "timestamp alignment", "", "", "跨市场对齐，避免未来信息", "2 数据"),               # data
    ("前视偏差", "look-ahead bias", "", "", "用了当时还拿不到的信息", "2 数据"),                       # data
    ("幸存者偏差", "survivorship bias", "", "", "只用活下来的样本", "2 数据"),                        # data
    ("另类数据", "alternative data", "", "", "卫星、信用卡、招聘等", "2 数据"),                       # data
    ("平稳性", "stationarity", "", "", "均值方差不随时间变", "3 时序"),                              # ts
    ("白噪声", "white noise", "", "", "无自相关的随机序列", "3 时序"),                                # ts
    ("自相关", "autocorrelation", r"$\rho_k$", "", "序列与自身滞后项的相关", "3 时序"),               # ts
    ("ACF", "autocorrelation function", "", "", "自相关随滞后的曲线", "3 时序"),                      # ts
    ("PACF", "partial autocorrelation function", "", "", "剔除中间滞后后的相关", "3 时序"),           # ts
    ("移动平均过程", "moving average process", "MA(q)", "", "当前值由过去的冲击决定", "3 时序"),      # ts
    ("自回归过程", "autoregressive process", "AR(p)", "", "当前值由过去的值决定", "3 时序"),          # ts
    ("ARIMA", "ARIMA(p,d,q)", "", "", "差分后的 ARMA 模型", "3 时序"),                                # ts
    ("AIC/BIC", "information criteria", "", "", "兼顾拟合与复杂度的模型选择", "3 时序"),              # ts
    ("趋势模型", "trend model", "", "", "确定性时间趋势", "3 时序"),                                   # ts
    ("随机游走", "random walk", "", "", "差分是白噪声，方差随时间线性增长", "3 时序"),                 # ts
    ("单位根", "unit root", "", "", "非平稳的一种，AR 系数为 1", "3 时序"),                            # ts
    ("ADF 检验", "augmented Dickey-Fuller test", "", "", "检验是否存在单位根", "3 时序"),              # ts
    ("差分", "differencing", "", "", "用变化量代替水平值", "3 时序"),                                  # ts
    ("季节性", "seasonality", "", "", "固定周期的波动", "3 时序"),                                     # ts
    ("STL 分解", "STL decomposition", "", "", "趋势-季节-残差分解", "3 时序"),                         # ts
    ("GARCH", "GARCH(1,1)", "", r"$\sigma_t^2=\omega+\alpha r_{t-1}^2+\beta\sigma_{t-1}^2$", "波动率聚类模型", "4 风险"),   # risk
    ("波动聚集", "volatility clustering", "", "", "大波动后跟大波动", "4 风险"),                       # risk
    ("持续性", "persistence", r"$\alpha+\beta$", "", "波动冲击的持久程度", "4 风险"),                  # risk
    ("半衰期", "half-life", "", r"$\ln 0.5/\ln(\alpha+\beta)$", "冲击衰减一半所需时间", "4 风险"),      # risk
    ("已实现波动率", "realised volatility", "", "", "用高频收益平方和估计", "4 风险"),                  # risk
    ("状态切换", "regime switching", "", "", "牛市/熊市两状态模型", "4 风险"),                         # risk
    ("VAR", "vector autoregression", "", "", "多变量自回归", "4 风险"),                                # risk
    ("协整", "cointegration", "", "", "非平稳序列的长期均衡关系", "4 风险"),                            # risk
    ("配对交易", "pairs trading", "", "", "用价差均值回复做多空", "4 风险"),                            # risk
    ("CAPM", "capital asset pricing model", "", "", "单因子资产定价", "4 风险"),                       # risk
    ("alpha", "alpha", r"$\alpha$", "", "超出风险补偿的收益", "4 风险"),                               # risk
    ("beta", "beta", r"$\beta$", "", "对市场因子的暴露", "4 风险"),                                   # risk
    ("因子模型", "factor model", "", "", "多因子解释收益", "4 风险"),                                  # risk
    ("有效前沿", "efficient frontier", "", "", "给定风险的最高收益组合", "5 组合"),                     # port
    ("均值—方差", "mean-variance", "", "", "Markowitz 的二次规划", "5 组合"),                          # port
    ("风险平价", "risk parity", "", "", "让每个资产的-risk贡献相等", "5 组合"),                          # port
    ("换手率", "turnover", "", "", "组合权重的日变化", "5 组合"),                                       # port
    ("滑点", "slippage", "", "", "成交价与预期价的差", "5 组合"),                                      # port
    ("回测", "backtest", "", "", "用历史数据模拟策略", "5 组合"),                                       # port
    ("Sharpe 比率", "Sharpe ratio", "", r"$\frac{\mu-r_f}{\sigma}$", "单位风险的超额收益", "5 组合"),    # port
    ("Sortino 比率", "Sortino ratio", "", "", "只用下行风险做分母", "5 组合"),                          # port
    ("最大回撤", "maximum drawdown", "", "", "从峰值到谷底的最大跌幅", "5 组合"),                       # port
    ("回测过拟合", "backtest overfitting", "", "", "试太多次后挑出的最优结果不可信", "5 组合"),                     # port
    ("Deflated Sharpe", "deflated Sharpe ratio", "", "", "对多重试验做修正的 Sharpe", "5 组合"),        # port
    ("情绪因子", "sentiment factor", "", "", "用文本情绪构造的信号", "6 风控"),                        # risk2
    ("信用风险", "credit risk", "", "", "对手方不履约的风险", "6 风控"),                               # risk2
    ("违约概率", "probability of default (PD)", "", "", "一年内违约的概率", "6 风控"),                 # risk2
    ("LGD", "loss given default", "", "", "违约后的损失比例", "6 风控"),                              # risk2
    ("评分卡", "scorecard", "", "", "把概率映射成分数", "6 风控"),                                     # risk2
    ("可解释性", "interpretability", "", "", "能说清为什么给这个分", "6 风控"),                        # risk2
    ("反欺诈", "fraud detection", "", "", "识别团伙与异常交易", "6 风控"),                             # risk2
    ("图神经网络", "graph neural network (GNN)", "", "", "在关系图上做消息传递", "6 风控"),            # risk2
    ("消息传递", "message passing", "", r"$h_i=\phi(Wx_i+\sum_j h_j)$", "聚合邻居信息", "6 风控"),      # risk2
    ("检索增强生成", "retrieval-augmented generation (RAG)", "", "", "先检索再生成，带引用", "6 风控"),   # risk2
    ("引用溯源", "citation tracing", "", "", "答案标注来源", "6 风控"),                                 # risk2
    ("模型治理", "model governance", "", "", "模型卡、监控、漂移与审计", "6 风控"),                     # risk2
    ("概念漂移", "concept drift", "", "", "数据分布随时间变化", "6 风控"),                              # risk2
    ("信噪比", "signal-to-noise ratio", "", "", "金融数据里通常极低", "6 风控"),                        # risk2
    ("交易成本", "transaction cost", "", "", "佣金、价差、冲击成本", "5 组合"),                         # port
    ("容量", "capacity", "", "", "策略能容纳的资金规模", "5 组合"),                                     # port
    ("风格漂移", "style drift", "", "", "组合的风险暴露悄悄改变", "5 组合"),                            # port
    ("CVaR", "conditional value at risk", "", "", "尾部平均损失", "4 风险"),                           # risk
    ("在险价值", "value at risk (VaR)", "", "", "给定置信度的最大损失", "4 风险"),                      # risk
    ("压力测试", "stress testing", "", "", "极端情景下的损失", "6 风控"),                              # risk2
]                                                                       # continuation of the call

# ---------------------------------------------------------------- persist the table
glossary = pd.DataFrame(TERMS, columns=["term_zh", "term_en", "symbol", "formula", "note", "part"])   # build the frame
dump(glossary, "l13_glossary.csv")                                 # machine-readable CSV
counts = glossary["part"].value_counts().sort_index()              # terms per part
print("glossary terms:", len(glossary))                            # report the total
print(counts.to_string())                                          # report the split
assert len(glossary) >= 60, "the glossary should hold at least sixty terms"   # acceptance check
md_lines = ["# Lecture 13 glossary", "",                            # markdown title
            "| 术语 (zh) | Term (en) | Symbol | Formula | Note | Part |",   # header
            "| --- | --- | --- | --- | --- | --- |"]                # separator
for _, r in glossary.iterrows():                                   # one row per term
    md_lines.append("| %s | %s | %s | %s | %s | %s |" % (r["term_zh"], r["term_en"],   # Chinese and English
                                                         r["symbol"], r["formula"], r["note"], r["part"]))   # the rest
(OUT / "l13_glossary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")   # markdown table
print("[artifact]", OUT / "l13_glossary.md")                        # log the path
print("done: glossary with %d terms" % len(glossary))              # closing log
