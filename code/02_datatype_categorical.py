"""02_datatype_categorical.py —— 类别型数据（名义 vs 有序）：样本生成 + 可视化。"""  # 模块注释
import numpy as np                                                    # 数值计算
import pandas as pd                                                   # 表格处理
import matplotlib.pyplot as plt                                       # 绘图
from _common import RNG, save, dump_csv                               # 公共工具

# ---------------------------------------------------------------- 1) 生成样本
industries = ["电子", "医药", "机械", "消费", "金融", "能源"]           # 名义型类别：行业（无大小顺序）
ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "C"]                   # 有序型类别：信用评级（有明确顺序）
n = 300                                                               # 样本量：300 家上市公司
ind = RNG.choice(industries, size=n, p=[0.28, 0.13, 0.18, 0.16, 0.15, 0.10])  # 按给定比例抽取行业（名义）
rate = RNG.choice(ratings, size=n, p=[0.03, 0.09, 0.18, 0.30, 0.22, 0.13, 0.05])  # 按给定比例抽取评级（有序）
year = RNG.choice([2023, 2024, 2025], size=n)                          # 记录年份，便于做交叉表
df = pd.DataFrame({"industry": ind, "rating": rate, "year": year})     # 组装样本表
dump_csv(df, "categorical_samples.csv")                                # 落盘为 CSV（仓库 data/ 目录）

# ---------------------------------------------------------------- 2) 可视化
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))                    # 一行三列：名义条形图 / 有序条形图 / 交叉热力图
counts_ind = df["industry"].value_counts().reindex(industries)         # 统计各行业数量（保持给定顺序）
axes[0].bar(industries, counts_ind.values, color="#8A0C3C", alpha=0.85)  # 左图：名义类别的频数条形图
axes[0].set_title("名义型：行业分布")                                    # 左图标题
axes[0].tick_params(axis="x", rotation=30)                             # 横轴标签旋转，避免重叠

counts_rate = df["rating"].value_counts().reindex(ratings)              # 统计各评级数量（按 AAA→C 的有序顺序）
axes[1].bar(ratings, counts_rate.values, color="#C9A227", alpha=0.9)    # 中图：有序类别的频数条形图
axes[1].set_title("有序型：信用评级（顺序有意义）")                        # 中图标题

cross = pd.crosstab(df["industry"], df["rating"]).reindex(index=industries, columns=ratings)  # 行业×评级交叉表
im = axes[2].imshow(cross.values, cmap="RdPu", aspect="auto")           # 右图：交叉表热力图
axes[2].set_xticks(range(len(ratings)), ratings, fontsize=8)            # 右图横轴：评级刻度
axes[2].set_yticks(range(len(industries)), industries, fontsize=8)      # 右图纵轴：行业刻度
axes[2].set_title("类别交叉：行业 × 评级")                               # 右图标题
axes[2].grid(False)                                                     # 热力图不需要网格线
plt.colorbar(im, ax=axes[2], fraction=0.046)                            # 添加颜色条表示频数
plt.tight_layout()                                                      # 调整布局
save(fig, "fig_categorical.png")                                        # 保存图片
