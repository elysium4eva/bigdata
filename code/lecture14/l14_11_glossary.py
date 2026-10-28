"""l14_11_glossary.py -- the bilingual glossary for Lecture 14.

Every line carries an English comment. The script writes the machine-readable term table   # continuation of the call
(Chinese, English, symbol, formula, note, part) that feeds the two glossary frames, the site   # continuation of the call
table and the printable Word/Excel handout.                             # continuation of the call
"""
import pandas as pd                                                # build and persist the table
from _common_l14 import dump, OUT                                  # shared helpers

# Columns: term_zh, term_en, symbol, formula, note, part.
TERMS = [                                                               # continuation of the call
    ("国别层数据", "country-level data", "", "", "国家-国家-年，引力回归的标配", "1 数据"),   # data
    ("CEPII", "CEPII", "", "", "距离、语言、殖民关系等引力变量", "1 数据"),                   # data
    ("UN Comtrade", "UN Comtrade", "", "", "最全的双边贸易数据库", "1 数据"),                 # data
    ("镜像数据", "mirror data", "", "", "出口国报告与进口国报告不一致", "1 数据"),             # data
    ("HS 编码", "Harmonized System code", "", "", "商品分类，版本会换", "1 数据"),             # data
    ("CIF/FOB", "CIF and FOB", "", "", "到岸价与离岸价的差额", "1 数据"),                     # data
    ("行业层数据", "industry-level data", "", "", "国家-行业-年", "1 数据"),                   # data
    ("BACI", "BACI", "", "", "调和后的双边贸易数据", "1 数据"),                               # data
    ("投入产出表", "input-output table", "", "", "行业间的中间投入关系", "1 数据"),             # data
    ("WIOD", "World Input-Output Database", "", "", "跨国投入产出数据库", "1 数据"),           # data
    ("企业层数据", "firm-level data", "", "", "企业-产品-国家-月", "1 数据"),                  # data
    ("工业企业数据库", "Chinese Industrial Enterprises Database", "", "", "规模以上工业企业", "1 数据"),   # data
    ("海关数据库", "customs transaction data", "", "", "逐笔报关记录", "1 数据"),               # data
    ("两库匹配", "firm-customs matching", "", "", "名称/电话/邮编连接两套数据", "1 数据"),       # data
    ("匹配率", "match rate", "", "", "可匹配记录占比", "1 数据"),                              # data
    ("幸存者偏差", "survivorship bias", "", "", "匹配上的样本系统性偏大", "1 数据"),            # data
    ("转口贸易", "entrepôt trade", "", "", "经第三地中转，扭曲双边统计", "1 数据"),             # data
    ("引力方程", "gravity equation", "", r"$X_{ij}=G\frac{Y_iY_j}{D_{ij}}$", "贸易与规模正比、与距离反比", "2 引力"),   # gravity
    ("贸易弹性", "trade elasticity", r"$\theta$", "", "贸易成本对贸易量的弹性", "2 引力"),      # gravity
    ("替代弹性", "substitution elasticity", r"$\sigma$", "", "消费者对品种的替代程度", "2 引力"),   # gravity
    ("多边阻力", "multilateral resistance", "", "", "对所有贸易伙伴的平均阻力", "2 引力"),       # gravity
    ("结构引力", "structural gravity", "", "", "与理论一致的引力设定", "2 引力"),               # gravity
    ("固定效应", "fixed effects", "", "", "用虚拟变量吸收国家层面因素", "2 引力"),              # gravity
    ("PPML", "Poisson pseudo-maximum likelihood", "", "", "能处理零贸易的估计方法", "2 引力"),   # gravity
    ("零贸易", "zero trade", "", "", "没记录到的双边贸易", "2 引力"),                          # gravity
    ("边界效应", "border effect", "", "", "同一国家内部贸易远高于跨境", "2 引力"),              # gravity
    ("元分析", "meta-analysis", "", "", "汇总大量估计的中位数", "2 引力"),                      # gravity
    ("贸易成本", "trade cost", "", "", "距离、关税、语言等综合成本", "2 引力"),                 # gravity
    ("关税等价", "tariff equivalent", "", r"$e^{\beta/\theta}-1$", "把系数换算成关税当量", "2 引力"),   # gravity
    ("RTA", "regional trade agreement", "", "", "区域贸易协定", "2 引力"),                    # gravity
    ("DID", "difference-in-differences", "", "", "处理组与对照组的双重差分", "3 识别"),         # id
    ("事件研究", "event study", "", "", "逐期估计处理效应", "3 识别"),                          # id
    ("平行趋势", "parallel trends", "", "", "处理前两组走势应相同", "3 识别"),                  # id
    ("合成控制", "synthetic control", "", "", "构造一个加权对照组", "3 识别"),                  # id
    ("PTI", "partial trade impact", "", "", "局部均衡的贸易效应", "3 识别"),                    # id
    ("MTI", "modelled trade impact", "", "", "含贸易条件的效应", "3 识别"),                     # id
    ("GETI", "general equilibrium trade impact", "", "", "全球一般均衡的效应", "3 识别"),        # id
    ("福利效应", "welfare effect", "", "", "消费者剩余与贸易条件之和", "3 识别"),                # id
    ("贸易政策不确定性", "trade policy uncertainty", "", "", "威胁加税也会影响贸易", "3 识别"),   # id
    ("关税传递", "tariff pass-through", "", "", "关税由进口商还是消费者承担", "3 识别"),         # id
    ("网络", "network", "", "", "节点与边构成的图", "4 网络"),                                  # network
    ("邻接矩阵", "adjacency matrix", r"$A$", "", "有边为 1 或权重，无边为 0", "4 网络"),         # network
    ("有向图", "directed graph", "", "", "边有方向（出口→进口）", "4 网络"),                     # network
    ("加权图", "weighted graph", "", "", "边带贸易额权重", "4 网络"),                            # network
    ("度", "degree", "", "", "与节点相连的边数", "4 网络"),                                     # network
    ("强度", "strength", "", "", "边的权重之和", "4 网络"),                                     # network
    ("中介中心性", "betweenness centrality", "", "", "有多少最短路经过该节点", "4 网络"),         # network
    ("社区发现", "community detection", "", "", "把网络分成相对紧密的块", "4 网络"),             # network
    ("模块度", "modularity", r"$Q$", "", "社区划分的好坏", "4 网络"),                            # network
    ("全球价值链", "global value chain (GVC)", "", "", "跨国的生产分工链条", "4 网络"),           # network
    ("前向参与度", "forward participation", "", "", "出口中被他国用作中间品的份额", "4 网络"),    # network
    ("后向参与度", "backward participation", "", "", "出口中包含的国外增加值份额", "4 网络"),     # network
    ("附加值", "value added", r"$VA$", "", "产出减去中间投入", "4 网络"),                       # network
    ("微笑曲线", "smile curve", "", "", "两端（研发/品牌）附加值高", "4 网络"),                  # network
    ("生产网络", "production network", "", "", "行业间的投入产出网络", "4 网络"),                # network
    ("冲击传导", "shock propagation", "", "", "一个环节断供会传多远", "4 网络"),                 # network
    ("图神经网络", "graph neural network (GNN)", "", "", "在图上做消息传递的模型", "4 网络"),     # network
    ("消息传递", "message passing", "", r"$h_i=\phi(Wx_i+\sum_j h_j)$", "聚合邻居信息", "4 网络"),   # network
    ("AIS", "automatic identification system", "", "", "船舶定位数据", "5 新数据"),             # new
    ("港口拥堵", "port congestion", "", "", "船等泊位的时间", "5 新数据"),                       # new
    ("卫星数据", "satellite data", "", "", "夜光、船舶探测等", "5 新数据"),                      # new
    ("供应链韧性", "supply-chain resilience", "", "", "受冲击后恢复的能力", "5 新数据"),         # new
    ("HHI", "Herfindahl-Hirschman index", "", r"$\sum s_i^2$", "集中度指标", "5 新数据"),       # new
    ("可替代性", "substitutability", "", r"$1-\mathrm{HHI}$", "供应商越分散越可替代", "5 新数据"),   # new
    ("跨境电商", "cross-border e-commerce", "", "", "B2C 包裹与平台数据", "5 新数据"),            # new
    ("数字贸易", "digital trade", "", "", "可数字交付的服务贸易", "5 新数据"),                   # new
    ("服务贸易模式", "modes of supply", "", "", "跨境交付/境外消费/商业存在/自然人移动", "5 新数据"),   # new
    ("报关单", "customs declaration", "", "", "逐笔申报文件", "5 新数据"),                       # new
    ("字段抽取", "field extraction", "", "", "从文本中抽结构化字段", "5 新数据"),                 # new
    ("地缘政治风险", "geopolitical risk", "", "", "冲突、制裁等风险", "5 新数据"),                # new
    ("GPR 指数", "geopolitical risk index", "", "", "基于新闻关键词的月度指数", "5 新数据"),      # new
    ("出口管制", "export controls", "", "", "对特定技术与产品的出口限制", "5 新数据"),            # new
    ("制裁", "sanctions", "", "", "对实体或国家的贸易限制", "5 新数据"),                         # new
    ("合规边界", "compliance boundary", "", "", "海关数据的保密与聚合要求", "5 新数据"),           # new
    ("引力残差", "gravity residual", "", "", "回归残差可当贸易成本代理", "2 引力"),               # gravity
    ("多边阻力项", "outward multilateral resistance", r"$\Pi_i$", "", "出口国的对外阻力", "2 引力"),   # gravity
    ("内向多边阻力", "inward multilateral resistance", r"$P_j$", "", "进口国的对外阻力", "2 引力"),   # gravity
    ("泊松回归", "Poisson regression", "", "", "计数/水平数据的回归", "2 引力"),                  # gravity
    ("聚类标准误", "clustered standard errors", "", "", "按国家对或国家聚类", "2 引力"),          # gravity
    ("弹性解释", "elasticity interpretation", "", "", "对数-对数模型的系数即弹性", "2 引力"),     # gravity
    ("网络密度", "network density", "", "", "实际边数/可能边数", "4 网络"),                      # network
    ("核心-边缘", "core-periphery", "", "", "少数大国连得多，多数小国连得少", "4 网络"),          # network
    ("面板数据", "panel data", "", "", "同一批个体多年观测", "1 数据"),                          # data
    ("口径变化", "definition change", "", "", "统计口径调整导致的断点", "1 数据"),                # data
]                                                                       # end of the term list

# ---------------------------------------------------------------- persist the table
glossary = pd.DataFrame(TERMS, columns=["term_zh", "term_en", "symbol", "formula", "note", "part"])   # build the frame
dump(glossary, "l14_glossary.csv")                                 # machine-readable CSV
counts = glossary["part"].value_counts().sort_index()              # terms per part
print("glossary terms:", len(glossary))                            # report the total
print(counts.to_string())                                          # report the split
assert len(glossary) >= 60, "the glossary should hold at least sixty terms"   # acceptance check
md_lines = ["# Lecture 14 glossary", "",                            # markdown title
            "| 术语 (zh) | Term (en) | Symbol | Formula | Note | Part |",   # header
            "| --- | --- | --- | --- | --- | --- |"]                # separator
for _, r in glossary.iterrows():                                   # one row per term
    md_lines.append("| %s | %s | %s | %s | %s | %s |" % (r["term_zh"], r["term_en"],   # Chinese and English
                                                         r["symbol"], r["formula"], r["note"], r["part"]))   # the rest
(OUT / "l14_glossary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")   # markdown table
print("[artifact]", OUT / "l14_glossary.md")                        # log the path
print("done: glossary with %d terms" % len(glossary))              # closing log
