"""03_datatype_text.py —— 文本型数据：样本语料 + 词频/长度可视化。"""  # 模块注释
import re                                                             # 正则表达式，用于切词
from collections import Counter                                       # 计数器，用于统计词频
import pandas as pd                                                   # 表格处理
import matplotlib.pyplot as plt                                       # 绘图
from _common import save, dump_csv                                    # 公共工具

# ---------------------------------------------------------------- 1) 样本语料（财经新闻标题风格）
docs = [                                                              # 定义一个 20 条短文本的样本语料
    "宁德时代 2025 年海外营收同比增长 32%",                             # 文本样本 1
    "贵州茅台 发布 2025 年 三季度 财报，毛利率 保持 稳定",                # 文本样本 2
    "央行 宣布 下调 存款准备金率 0.5 个 百分点",                          # 文本样本 3
    "比亚迪 新能源 汽车 出口 量 创 历史 新高",                            # 文本样本 4
    "沪深 300 指数 收涨 1.2%，成交额 突破 万亿",                          # 文本样本 5
    "跨境电商 出口 规模 连续 三年 两位数 增长",                           # 文本样本 6
    "半导体 设备 进口 额 环比 下降 8%",                                   # 文本样本 7
    "光伏 组件 价格 触底 反弹，行业 开工率 回升",                         # 文本样本 8
    "消费 信心 指数 连续 两个月 位于 荣枯线 上方",                        # 文本样本 9
    "钢材 期货 主力 合约 日内 振幅 扩大",                                 # 文本样本 10
    "港口 集装箱 吞吐量 同比 增长 6.4%",                                 # 文本样本 11
    "人工智能 算力 需求 带动 服务器 出货 增长",                           # 文本样本 12
    "人民币 汇率 双向 波动，跨境 结算 占比 提升",                         # 文本样本 13
    "房地产 销售 面积 降幅 收窄，政策 效应 显现",                         # 文本样本 14
    "航空 客运量 恢复 至 疫情 前 水平",                                   # 文本样本 15
    "储能 装机 规模 快速 扩张，锂价 震荡 运行",                           # 文本样本 16
    "生物医药 融资 回暖，创新药 出海 提速",                               # 文本样本 17
    "工业 机器人 产量 保持 两位数 增长",                                  # 文本样本 18
    "原油 价格 受 地缘 因素 影响 走强",                                   # 文本样本 19
    "数据 要素 市场 建设 加快，交易 规模 扩大",                           # 文本样本 20
]
df = pd.DataFrame({"doc_id": range(1, len(docs) + 1),                 # 文本编号列
                   "text": docs,                                       # 文本内容列
                   "length": [len(d) for d in docs]})                  # 字符长度列（文本的数值化特征）
dump_csv(df, "text_samples.csv")                                      # 落盘文本样本

# ---------------------------------------------------------------- 2) 特征化：切词 + 词频
def tokens(s):                                                        # 定义切词函数：中文按字、英文数字按词
    """把一句话切成 token 列表：连续英文/数字保留为整体，汉字逐字切分。"""  # 函数说明
    return re.findall(r"[A-Za-z0-9%\.]+|[\u4e00-\u9fff]", s)          # 正则：英文数字串或单个汉字
all_tokens = [t for d in docs for t in tokens(d)]                     # 把所有文档的 token 汇总成一个列表
freq = Counter(all_tokens).most_common(15)                            # 取出现频次最高的 15 个 token
words, counts = [w for w, _ in freq], [c for _, c in freq]            # 拆成词表与频数两个列表

# ---------------------------------------------------------------- 3) 可视化
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))                    # 一行三列：词频 / 文档长度 / 长度分布
axes[0].barh(range(len(words))[::-1], counts, color="#8A0C3C", alpha=0.85)  # 左图：Top15 频次横向条形图
axes[0].set_yticks(range(len(words))[::-1], words, fontsize=8)         # 设置 y 轴刻度为词表
axes[0].set_title("文本 → 词频（Top 15 token）")                        # 左图标题
axes[0].set_xlabel("出现次数")                                         # 左图横轴含义

axes[1].bar(df["doc_id"], df["length"], color="#66717D", alpha=0.9)    # 中图：每篇文本的字符数
axes[1].set_title("文本长度（数值化特征）")                              # 中图标题
axes[1].set_xlabel("文档编号")                                         # 中图横轴含义
axes[1].set_ylabel("字符数")                                           # 中图纵轴含义

axes[2].hist(df["length"], bins=8, color="#C9A227", alpha=0.9)         # 右图：文本长度分布直方图
axes[2].set_title("长度分布（用于截断/填充）")                          # 右图标题
axes[2].set_xlabel("字符数")                                           # 右图横轴含义
plt.tight_layout()                                                     # 调整布局
save(fig, "fig_text.png")                                              # 保存图片
