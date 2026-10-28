# Lecture 14 glossary

| 术语 (zh) | Term (en) | Symbol | Formula | Note | Part |
| --- | --- | --- | --- | --- | --- |
| 国别层数据 | country-level data |  |  | 国家-国家-年，引力回归的标配 | 1 数据 |
| CEPII | CEPII |  |  | 距离、语言、殖民关系等引力变量 | 1 数据 |
| UN Comtrade | UN Comtrade |  |  | 最全的双边贸易数据库 | 1 数据 |
| 镜像数据 | mirror data |  |  | 出口国报告与进口国报告不一致 | 1 数据 |
| HS 编码 | Harmonized System code |  |  | 商品分类，版本会换 | 1 数据 |
| CIF/FOB | CIF and FOB |  |  | 到岸价与离岸价的差额 | 1 数据 |
| 行业层数据 | industry-level data |  |  | 国家-行业-年 | 1 数据 |
| BACI | BACI |  |  | 调和后的双边贸易数据 | 1 数据 |
| 投入产出表 | input-output table |  |  | 行业间的中间投入关系 | 1 数据 |
| WIOD | World Input-Output Database |  |  | 跨国投入产出数据库 | 1 数据 |
| 企业层数据 | firm-level data |  |  | 企业-产品-国家-月 | 1 数据 |
| 工业企业数据库 | Chinese Industrial Enterprises Database |  |  | 规模以上工业企业 | 1 数据 |
| 海关数据库 | customs transaction data |  |  | 逐笔报关记录 | 1 数据 |
| 两库匹配 | firm-customs matching |  |  | 名称/电话/邮编连接两套数据 | 1 数据 |
| 匹配率 | match rate |  |  | 可匹配记录占比 | 1 数据 |
| 幸存者偏差 | survivorship bias |  |  | 匹配上的样本系统性偏大 | 1 数据 |
| 转口贸易 | entrepôt trade |  |  | 经第三地中转，扭曲双边统计 | 1 数据 |
| 引力方程 | gravity equation |  | $X_{ij}=G\frac{Y_iY_j}{D_{ij}}$ | 贸易与规模正比、与距离反比 | 2 引力 |
| 贸易弹性 | trade elasticity | $\theta$ |  | 贸易成本对贸易量的弹性 | 2 引力 |
| 替代弹性 | substitution elasticity | $\sigma$ |  | 消费者对品种的替代程度 | 2 引力 |
| 多边阻力 | multilateral resistance |  |  | 对所有贸易伙伴的平均阻力 | 2 引力 |
| 结构引力 | structural gravity |  |  | 与理论一致的引力设定 | 2 引力 |
| 固定效应 | fixed effects |  |  | 用虚拟变量吸收国家层面因素 | 2 引力 |
| PPML | Poisson pseudo-maximum likelihood |  |  | 能处理零贸易的估计方法 | 2 引力 |
| 零贸易 | zero trade |  |  | 没记录到的双边贸易 | 2 引力 |
| 边界效应 | border effect |  |  | 同一国家内部贸易远高于跨境 | 2 引力 |
| 元分析 | meta-analysis |  |  | 汇总大量估计的中位数 | 2 引力 |
| 贸易成本 | trade cost |  |  | 距离、关税、语言等综合成本 | 2 引力 |
| 关税等价 | tariff equivalent |  | $e^{\beta/\theta}-1$ | 把系数换算成关税当量 | 2 引力 |
| RTA | regional trade agreement |  |  | 区域贸易协定 | 2 引力 |
| DID | difference-in-differences |  |  | 处理组与对照组的双重差分 | 3 识别 |
| 事件研究 | event study |  |  | 逐期估计处理效应 | 3 识别 |
| 平行趋势 | parallel trends |  |  | 处理前两组走势应相同 | 3 识别 |
| 合成控制 | synthetic control |  |  | 构造一个加权对照组 | 3 识别 |
| PTI | partial trade impact |  |  | 局部均衡的贸易效应 | 3 识别 |
| MTI | modelled trade impact |  |  | 含贸易条件的效应 | 3 识别 |
| GETI | general equilibrium trade impact |  |  | 全球一般均衡的效应 | 3 识别 |
| 福利效应 | welfare effect |  |  | 消费者剩余与贸易条件之和 | 3 识别 |
| 贸易政策不确定性 | trade policy uncertainty |  |  | 威胁加税也会影响贸易 | 3 识别 |
| 关税传递 | tariff pass-through |  |  | 关税由进口商还是消费者承担 | 3 识别 |
| 网络 | network |  |  | 节点与边构成的图 | 4 网络 |
| 邻接矩阵 | adjacency matrix | $A$ |  | 有边为 1 或权重，无边为 0 | 4 网络 |
| 有向图 | directed graph |  |  | 边有方向（出口→进口） | 4 网络 |
| 加权图 | weighted graph |  |  | 边带贸易额权重 | 4 网络 |
| 度 | degree |  |  | 与节点相连的边数 | 4 网络 |
| 强度 | strength |  |  | 边的权重之和 | 4 网络 |
| 中介中心性 | betweenness centrality |  |  | 有多少最短路经过该节点 | 4 网络 |
| 社区发现 | community detection |  |  | 把网络分成相对紧密的块 | 4 网络 |
| 模块度 | modularity | $Q$ |  | 社区划分的好坏 | 4 网络 |
| 全球价值链 | global value chain (GVC) |  |  | 跨国的生产分工链条 | 4 网络 |
| 前向参与度 | forward participation |  |  | 出口中被他国用作中间品的份额 | 4 网络 |
| 后向参与度 | backward participation |  |  | 出口中包含的国外增加值份额 | 4 网络 |
| 附加值 | value added | $VA$ |  | 产出减去中间投入 | 4 网络 |
| 微笑曲线 | smile curve |  |  | 两端（研发/品牌）附加值高 | 4 网络 |
| 生产网络 | production network |  |  | 行业间的投入产出网络 | 4 网络 |
| 冲击传导 | shock propagation |  |  | 一个环节断供会传多远 | 4 网络 |
| 图神经网络 | graph neural network (GNN) |  |  | 在图上做消息传递的模型 | 4 网络 |
| 消息传递 | message passing |  | $h_i=\phi(Wx_i+\sum_j h_j)$ | 聚合邻居信息 | 4 网络 |
| AIS | automatic identification system |  |  | 船舶定位数据 | 5 新数据 |
| 港口拥堵 | port congestion |  |  | 船等泊位的时间 | 5 新数据 |
| 卫星数据 | satellite data |  |  | 夜光、船舶探测等 | 5 新数据 |
| 供应链韧性 | supply-chain resilience |  |  | 受冲击后恢复的能力 | 5 新数据 |
| HHI | Herfindahl-Hirschman index |  | $\sum s_i^2$ | 集中度指标 | 5 新数据 |
| 可替代性 | substitutability |  | $1-\mathrm{HHI}$ | 供应商越分散越可替代 | 5 新数据 |
| 跨境电商 | cross-border e-commerce |  |  | B2C 包裹与平台数据 | 5 新数据 |
| 数字贸易 | digital trade |  |  | 可数字交付的服务贸易 | 5 新数据 |
| 服务贸易模式 | modes of supply |  |  | 跨境交付/境外消费/商业存在/自然人移动 | 5 新数据 |
| 报关单 | customs declaration |  |  | 逐笔申报文件 | 5 新数据 |
| 字段抽取 | field extraction |  |  | 从文本中抽结构化字段 | 5 新数据 |
| 地缘政治风险 | geopolitical risk |  |  | 冲突、制裁等风险 | 5 新数据 |
| GPR 指数 | geopolitical risk index |  |  | 基于新闻关键词的月度指数 | 5 新数据 |
| 出口管制 | export controls |  |  | 对特定技术与产品的出口限制 | 5 新数据 |
| 制裁 | sanctions |  |  | 对实体或国家的贸易限制 | 5 新数据 |
| 合规边界 | compliance boundary |  |  | 海关数据的保密与聚合要求 | 5 新数据 |
| 引力残差 | gravity residual |  |  | 回归残差可当贸易成本代理 | 2 引力 |
| 多边阻力项 | outward multilateral resistance | $\Pi_i$ |  | 出口国的对外阻力 | 2 引力 |
| 内向多边阻力 | inward multilateral resistance | $P_j$ |  | 进口国的对外阻力 | 2 引力 |
| 泊松回归 | Poisson regression |  |  | 计数/水平数据的回归 | 2 引力 |
| 聚类标准误 | clustered standard errors |  |  | 按国家对或国家聚类 | 2 引力 |
| 弹性解释 | elasticity interpretation |  |  | 对数-对数模型的系数即弹性 | 2 引力 |
| 网络密度 | network density |  |  | 实际边数/可能边数 | 4 网络 |
| 核心-边缘 | core-periphery |  |  | 少数大国连得多，多数小国连得少 | 4 网络 |
| 面板数据 | panel data |  |  | 同一批个体多年观测 | 1 数据 |
| 口径变化 | definition change |  |  | 统计口径调整导致的断点 | 1 数据 |
