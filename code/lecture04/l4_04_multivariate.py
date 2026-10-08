"""l4_04_multivariate.py —— 多变量：相关热力图、分组散点、小多图与条件分布。

一份企业面板（60 家 × 8 年）回答"多变量该看什么"：
先看两两相关性（热力图），再看关系是否被分组改变（分组散点与条件分布），
最后用"小多图"把趋势按行业摊开——同一张图里塞十种颜色是最常见的错误。
"""
import numpy as np                                                 # 数值计算
import pandas as pd                                                # 面板数据处理
import matplotlib.pyplot as plt                                    # 四联图
from _common_l4 import RNG, save, dump                             # 复用公共工具

N_FIRM, N_YEAR = 60, 8                                             # 60 家企业、8 年
INDUSTRIES = ["电子", "医药", "机械", "消费"]                       # 四个行业
firms = pd.DataFrame({                                             # 企业层面的静态特征
    "firm_id": [f"F{i:03d}" for i in range(1, N_FIRM + 1)],        # 企业编号
    "industry": RNG.choice(INDUSTRIES, N_FIRM),                    # 所属行业
    "size": RNG.choice(["小", "中", "大"], N_FIRM, p=[0.45, 0.35, 0.2]),   # 规模分档
    "base_rev": RNG.lognormal(mean=2.2, sigma=0.6, size=N_FIRM),   # 初始营收（亿元）
    "growth": RNG.normal(0.09, 0.05, N_FIRM),                      # 个体成长率
    "leverage": RNG.uniform(0.15, 0.75, N_FIRM).round(3),          # 资产负债率
})                                                                # 企业静态特征表定义结束
rows = []                                                          # 面板长表的行
for _, f in firms.iterrows():                                      # 逐家企业展开年份
    for year in range(2018, 2018 + N_YEAR):                        # 8 个年度
        rev = f["base_rev"] * (1 + f["growth"]) ** (year - 2018) * np.exp(RNG.normal(0, 0.06))   # 营收演化
        roa = (0.075 - 0.05 * (f["leverage"] - 0.45) + RNG.normal(0, 0.012))   # 资产收益率：杠杆高则偏低
        rows.append({"firm_id": f["firm_id"], "industry": f["industry"],   # 标识与行业
                     "size": f["size"], "year": year,              # 规模与年份
                     "revenue": round(float(rev), 3),              # 营收
                     "roa": round(float(roa), 4),                  # 资产收益率
                     "leverage": f["leverage"]})                   # 杠杆率（企业层面，年度不变）
panel = pd.DataFrame(rows)                                         # 汇总为面板
dump(panel, "l4_firm_panel.csv")                                   # 落盘样本

numeric_cols = ["revenue", "roa", "leverage"]                      # 参与相关分析的数值列
corr = panel[numeric_cols].corr().round(3)                         # 三变量相关矩阵
print("相关矩阵:\n", corr.to_string())                              # 终端打印
cond = (panel.groupby(["industry", "size"])["roa"].agg(["mean", "count"]).round(4).reset_index())   # 行业 × 规模的条件均值
dump(cond, "l4_conditional_roa.csv")                               # 落盘条件均值
print("条件均值（行业 × 规模，前 6 行）:\n", cond.head(6).to_string(index=False))   # 打印片段

# ---------------------------------------------------------------- 图：多变量四视图
fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.7))                # 热力图 / 分组散点 / 小多图 / 条件分布
im = axes[0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)   # 第一格：相关热力图（发散色板）
axes[0].set_xticks(range(len(numeric_cols)), ["营收", "ROA", "杠杆率"], fontsize=9)   # 横轴标签
axes[0].set_yticks(range(len(numeric_cols)), ["营收", "ROA", "杠杆率"], fontsize=9)   # 纵轴标签
axes[0].grid(False)                                                # 热力图不需要网格
for i in range(len(numeric_cols)):                                 # 在每个格子写数值
    for j in range(len(numeric_cols)):                             # 双层循环遍历矩阵
        axes[0].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8.6,   # 在热力图每个格子写相关系数
                     color="white" if abs(corr.values[i, j]) > 0.6 else "#1E2630")   # 深色底用白字
axes[0].set_title("相关矩阵（发散色板）")                            # 子图标题
plt.colorbar(im, ax=axes[0], fraction=0.046)                       # 颜色条
for ind, c in zip(INDUSTRIES, ["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"]):   # 按行业着色
    sub = panel[panel["industry"] == ind]                          # 取该行业子样本
    axes[1].scatter(sub["leverage"], sub["roa"], s=12, color=c, alpha=0.5, label=ind)   # 第二格：分组散点
axes[1].axhline(panel["roa"].mean(), color="#1E2630", lw=1.0, ls="--")   # 总体均值参考线
axes[1].set_title("关系是否被分组改变")                             # 子图标题
axes[1].set_xlabel("资产负债率"); axes[1].set_ylabel("ROA")         # 轴标签
axes[1].legend(fontsize=8.2, ncol=2)                               # 两列图例
for k, ind in enumerate(INDUSTRIES):                               # 小多图：每行业一格
    sub = panel[panel["industry"] == ind].groupby("year")["revenue"].mean()   # 行业年度均值
    axes[2].plot(sub.index, sub.values, "-o", ms=3, lw=1.5,        # 画行业趋势线
                 color=["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"][k], label=ind)   # 颜色与图例
axes[2].set_title("小多图：行业营收趋势")                            # 子图标题
axes[2].set_xlabel("年份"); axes[2].set_ylabel("平均营收（亿元）")    # 轴标签
axes[2].legend(fontsize=8.2)                                       # 图例
data = [panel.loc[panel["industry"] == ind, "roa"].values for ind in INDUSTRIES]   # 各行业 ROA
bp = axes[3].boxplot(data, patch_artist=True)                      # 第四格：条件分布（箱线）
axes[3].set_xticklabels(INDUSTRIES)                                # 分组标签（3.9 起用 set_xticklabels）
for patch, c in zip(bp["boxes"], ["#8A0C3C", "#C9A227", "#2E6F8E", "#66717D"]):   # 给箱体上色
    patch.set_facecolor(c); patch.set_alpha(0.55)                  # 设置填充色与透明度
axes[3].set_title("条件分布：各行业 ROA")                           # 子图标题
axes[3].set_ylabel("ROA")                                          # 轴标签
plt.tight_layout()                                                 # 调整四个子图
save(fig, "fig_multivariate.png")                                  # 保存：多变量一页用图
