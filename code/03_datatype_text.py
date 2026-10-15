"""03_datatype_text.py -- text data: sample corpus plus word frequency and length plots."""  # module comment
import re                                                             # regular expressions for tokenization
from collections import Counter                                       # counter for word frequencies
import pandas as pd                                                   # table handling
import matplotlib.pyplot as plt                                       # plotting
from _common import save, dump_csv                                    # shared helpers

# ---------------------------------------------------------------- 1) sample corpus in financial headline style
docs = [                                                              # define a corpus of 20 short texts
    "宁德时代 2025 年海外营收同比增长 32%",                             # text sample 1
    "贵州茅台 发布 2025 年 三季度 财报，毛利率 保持 稳定",                # text sample 2
    "央行 宣布 下调 存款准备金率 0.5 个 百分点",                          # text sample 3
    "比亚迪 新能源 汽车 出口 量 创 历史 新高",                            # text sample 4
    "沪深 300 指数 收涨 1.2%，成交额 突破 万亿",                          # text sample 5
    "跨境电商 出口 规模 连续 三年 两位数 增长",                           # text sample 6
    "半导体 设备 进口 额 环比 下降 8%",                                   # text sample 7
    "光伏 组件 价格 触底 反弹，行业 开工率 回升",                         # text sample 8
    "消费 信心 指数 连续 两个月 位于 荣枯线 上方",                        # text sample 9
    "钢材 期货 主力 合约 日内 振幅 扩大",                                 # text sample 10
    "港口 集装箱 吞吐量 同比 增长 6.4%",                                 # text sample 11
    "人工智能 算力 需求 带动 服务器 出货 增长",                           # text sample 12
    "人民币 汇率 双向 波动，跨境 结算 占比 提升",                         # text sample 13
    "房地产 销售 面积 降幅 收窄，政策 效应 显现",                         # text sample 14
    "航空 客运量 恢复 至 疫情 前 水平",                                   # text sample 15
    "储能 装机 规模 快速 扩张，锂价 震荡 运行",                           # text sample 16
    "生物医药 融资 回暖，创新药 出海 提速",                               # text sample 17
    "工业 机器人 产量 保持 两位数 增长",                                  # text sample 18
    "原油 价格 受 地缘 因素 影响 走强",                                   # text sample 19
    "数据 要素 市场 建设 加快，交易 规模 扩大",                           # text sample 20
]                                                                     # end of the corpus list
df = pd.DataFrame({"doc_id": range(1, len(docs) + 1),                 # document id column
                   "text": docs,                                       # text content column
                   "length": [len(d) for d in docs]})                  # character length column, a numeric feature
dump_csv(df, "text_samples.csv")                                      # dump the text samples

# ---------------------------------------------------------------- 2) featurization: tokenization and word frequency
def tokens(s):                                                        # tokenizer: Chinese by character, alphanumerics by word
    """Split a sentence into tokens: runs of letters or digits stay whole, Chinese is split per character."""  # function docstring
    return re.findall(r"[A-Za-z0-9%\.]+|[\u4e00-\u9fff]", s)          # regex: alphanumeric run or a single Chinese character
all_tokens = [t for d in docs for t in tokens(d)]                     # collect all tokens of every document into one list
freq = Counter(all_tokens).most_common(15)                            # take the 15 most frequent tokens
words, counts = [w for w, _ in freq], [c for _, c in freq]            # split into a token list and a count list

# ---------------------------------------------------------------- 3) visualization
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))                    # one row, three columns: frequency / length / length distribution
axes[0].barh(range(len(words))[::-1], counts, color="#8A0C3C", alpha=0.85)  # left: horizontal bars of the top 15 tokens
axes[0].set_yticks(range(len(words))[::-1], words, fontsize=8)         # set y ticks to the token list
axes[0].set_title("文本 → 词频（Top 15 token）")                        # left panel title
axes[0].set_xlabel("出现次数")                                         # left panel x label

axes[1].bar(df["doc_id"], df["length"], color="#66717D", alpha=0.9)    # middle: character count of every document
axes[1].set_title("文本长度（数值化特征）")                              # middle panel title
axes[1].set_xlabel("文档编号")                                         # middle panel x label
axes[1].set_ylabel("字符数")                                           # middle panel y label

axes[2].hist(df["length"], bins=8, color="#C9A227", alpha=0.9)         # right: histogram of text lengths
axes[2].set_title("长度分布（用于截断/填充）")                          # right panel title
axes[2].set_xlabel("字符数")                                           # right panel x label
plt.tight_layout()                                                     # adjust the layout
save(fig, "fig_text.png")                                              # save the figure
