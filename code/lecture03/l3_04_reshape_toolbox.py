"""l3_04_reshape_toolbox.py —— 宽表与长表、透视与堆叠，以及各类编码工具。

同一份销售数据有三种形态：宽表（门店 × 月份）、长表（整洁数据）与透视表（聚合结果）。
脚本在三种形态之间来回转换，并把原课件"Reshaping and pivot tables"那一页列出的工具
（stack/unstack、explode、get_dummies/from_dummies、cut/qcut、crosstab、factorize）逐个演示。
"""
import numpy as np                                                 # 为销售样本造随机数
import pandas as pd                                                # 重塑与编码工具
import matplotlib.pyplot as plt                                    # 形态对照图
from _common_l3 import RNG, save, dump                             # 复用公共种子、出图与落盘函数

STORES = ["S01", "S02", "S03", "S04"]                              # 四家门店
MONTHS = [f"2026-{m:02d}" for m in range(1, 7)]                    # 六个月
wide = pd.DataFrame(                                                # 宽表：一行一家门店
    RNG.lognormal(mean=4.2, sigma=0.35, size=(len(STORES), len(MONTHS))).round(1),   # 各月销售额
    index=STORES, columns=MONTHS)                                  # 索引是门店，列是月份
dump(wide.reset_index(names="store_id"), "l3_sales_wide.csv")      # 落盘宽表样本

long = wide.reset_index(names="store_id").melt(                    # 长表：一行一个"门店-月"
    id_vars="store_id", var_name="month", value_name="sales")       # 指定保留列、列名与新值列名
dump(long, "l3_sales_long.csv")                                    # 落盘长表样本
back_to_wide = long.pivot(index="store_id", columns="month", values="sales")   # 从长表转回宽表
assert back_to_wide.shape == wide.shape, "pivot did not invert melt"   # 往返检查：形状一致
assert np.allclose(back_to_wide.values, wide.values), "values changed in the round trip"   # 数值一致

long["category"] = RNG.choice(["food", "drink", "home"], len(long))   # 加一列类别，用于聚合
pivot = long.pivot_table(index="store_id", columns="category",      # 按门店 × 类别做透视
                         values="sales", aggfunc="mean").round(1)   # 默认聚合是均值
pivot["total"] = pivot.sum(axis=1).round(1)                        # 追加每家门店的合计列
counts = pd.crosstab(long["store_id"], long["category"])           # 交叉表：统计的是频数
long["bucket"] = pd.cut(long["sales"], bins=[0, 40, 80, 140, 1000],   # 等宽分箱
                        labels=["<40", "40-80", "80-140", ">140"])     # 给分箱起可读的标签
long["quartile"] = pd.qcut(long["sales"], 4, labels=["Q1", "Q2", "Q3", "Q4"])   # 等频分箱
long["cat_code"] = pd.factorize(long["category"])[0]                # 类别转整数编码
long["cat_label"] = pd.factorize(long["category"])[1][long["cat_code"]]   # 反向解出类别名
dummies = pd.get_dummies(long["category"], prefix="cat")            # 独热编码
restored = pd.from_dummies(dummies).rename(columns=lambda c: "category")   # pandas ≥2.1 可逆变换
stacked = pivot.drop(columns="total").stack()                       # stack：把列层压到行层
unstacked = stacked.unstack()                                       # unstack：再恢复原状
assert unstacked.shape == pivot.drop(columns="total").shape, "stack/unstack round trip failed"   # 逆运算校验
items = pd.DataFrame({"order_id": ["O1", "O2", "O3"],               # explode 需要"列表型"单元格
                      "items": [["pen", "book"], ["mug"], ["pen", "mug", "bag"]]})   # 一单多品
exploded = items.explode("items")                                   # 一单拆成多行
print("wide -> long -> wide round trip OK:", back_to_wide.shape)    # 打印往返检查结果
print("explode: 3 orders ->", len(exploded), "item rows")           # 打印 explode 后的行数
print("buckets:\n", long["bucket"].value_counts().sort_index())     # 等宽分箱的分布
print("quartiles:", long["quartile"].value_counts().sort_index().to_dict())   # 等频分箱的分布
print("category codes:", dict(zip(long["cat_label"].head(3), long["cat_code"].head(3))))   # 编码对照
print("dummies restored == original:", bool((restored["category"].values   # 校验独热编码可逆
                                            == dummies.idxmax(axis=1).str.replace("cat_", "").values).all()))   # 逐行比较
print("pivot table:\n", pivot.to_string())                          # 打印透视表
dump(long, "l3_sales_long_enriched.csv")                            # 落盘带分箱的长表
dump(pivot.reset_index(), "l3_sales_pivot.csv")                     # 落盘透视表

# ---------------------------------------------------------------- 图 A：三种形态
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # 宽 / 长 / 透视 并排
im = axes[0].imshow(wide.values, cmap="RdPu", aspect="auto")        # 宽表用热力图表示
axes[0].set_xticks(range(len(MONTHS)), MONTHS, rotation=45, fontsize=8)   # 横轴：月份
axes[0].set_yticks(range(len(STORES)), STORES, fontsize=9)          # 纵轴：门店
axes[0].set_title("wide: stores x months")                          # 子图标题
axes[0].grid(False)                                                 # 热力图不需要网格
plt.colorbar(im, ax=axes[0], fraction=0.046)                        # 数值色带
axes[1].plot(range(len(long)), long["sales"], ".", ms=4, color="#8A0C3C", alpha=0.6)   # 长表：一行一个观测
axes[1].set_xlabel("row index (one row = store-month)")             # 说明"一行代表什么"
axes[1].set_title("long: one row per store-month")                  # 子图标题
im2 = axes[2].imshow(pivot.drop(columns="total").values, cmap="YlGnBu", aspect="auto")   # 透视结果
axes[2].set_xticks(range(pivot.shape[1] - 1), pivot.columns[:-1], fontsize=8)   # 横轴：类别
axes[2].set_yticks(range(len(STORES)), STORES, fontsize=9)          # 纵轴：门店
axes[2].set_title("pivot_table: store x category")                  # 子图标题
axes[2].grid(False)                                                 # 热力图不需要网格
plt.tight_layout()                                                  # 调整三个子图
save(fig, "fig_reshape_wide_long.png")                              # 保存：重塑一页用图

# ---------------------------------------------------------------- 图 B：分箱、交叉表与编码
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # 交叉表 / cut / qcut
im3 = axes[0].imshow(counts.values, cmap="OrRd", aspect="auto")     # 交叉表热力图
axes[0].set_xticks(range(counts.shape[1]), counts.columns, fontsize=8)   # 横轴：类别
axes[0].set_yticks(range(len(STORES)), STORES, fontsize=9)          # 纵轴：门店
axes[0].set_title("crosstab: store x category")                     # 子图标题
axes[0].grid(False)                                                 # 热力图不需要网格
axes[1].bar(long["bucket"].value_counts().sort_index().index.astype(str),   # 按分箱顺序排好标签
            long["bucket"].value_counts().sort_index().values, color="#8A0C3C", alpha=0.85)   # 等宽分箱频数
axes[1].set_title("cut(): equal-width bins")                        # 子图标题
axes[2].bar(long["quartile"].value_counts().sort_index().index.astype(str),   # 按分位顺序排好标签
            long["quartile"].value_counts().sort_index().values, color="#C9A227", alpha=0.9)   # 等频分箱频数
axes[2].set_title("qcut(): equal-count bins")                       # 子图标题
plt.tight_layout()                                                  # 调整三个子图
save(fig, "fig_reshape_bins.png")                                   # 保存：重塑工具箱一页用图
