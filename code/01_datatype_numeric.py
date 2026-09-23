"""01_datatype_numeric.py —— 数值型数据（连续 vs 离散）：样本生成 + 可视化。"""  # 模块注释
import numpy as np                                                    # 数值计算
import pandas as pd                                                   # 表格数据处理
import matplotlib.pyplot as plt                                       # 绘图
from _common import RNG, save, dump_csv                               # 复用公共工具（随机种子/保存/落盘）

# ---------------------------------------------------------------- 1) 生成样本
n = 800                                                               # 连续变量的样本量：800 个交易日
mu, sigma = 0.0004, 0.018                                             # 日收益率的均值与波动率（对数收益）
ret = RNG.normal(mu, sigma, n)                                        # 生成连续型样本：日对数收益率（近似正态）
volume = RNG.lognormal(mean=14.0, sigma=0.5, size=n)                  # 生成连续型样本：成交量（对数正态，右偏）
orders = RNG.poisson(lam=12, size=n)                                  # 生成离散型样本：每日订单数（泊松计数）

df = pd.DataFrame({"day": np.arange(1, n + 1),                       # 组装为一张表：第几交易日
                   "log_return": ret,                                 # 连续变量①：日对数收益率
                   "volume": volume,                                  # 连续变量②：成交量（右偏长尾）
                   "orders": orders})                                 # 离散变量：订单数（非负整数）
dump_csv(df, "numeric_samples.csv")                                   # 把样本写进仓库 data/ 目录

# ---------------------------------------------------------------- 2) 可视化
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))                   # 一行三列：直方图 / 箱线图 / 计数条形图
axes[0].hist(ret, bins=40, color="#8A0C3C", alpha=0.85)               # 左图：连续变量直方图（看分布形状）
axes[0].set_title("连续：日对数收益率分布")                            # 设置左图标题
axes[0].set_xlabel("收益率")                                          # 设置左图横轴含义
axes[0].set_ylabel("频数")                                            # 设置左图纵轴含义
axes[0].axvline(ret.mean(), color="#C9A227", lw=1.6,                 # 画一条均值参考线（金色）
                label="均值 %.4f" % ret.mean())                       # 图例中给出均值数值
axes[0].legend(fontsize=8)                                            # 显示图例

axes[1].boxplot([ret, np.log(volume)], showfliers=True,               # 中图：箱线图对比两个连续变量
                patch_artist=True,                                    # 允许自定义箱体填充色
                boxprops=dict(facecolor="#8A0C3C", alpha=0.35))       # 箱体颜色与透明度
axes[1].set_xticklabels(["收益率", "log 成交量"])                       # 给两组箱线图加上可读标签
axes[1].set_title("连续：箱线图（离群点）")                             # 设置中图标题

axes[2].bar(np.arange(1, 26), [np.sum(orders == k) for k in range(1, 26)],  # 右图：离散计数分布（1~25 单）
            color="#66717D", width=0.8)                               # 用灰色柱表示每个取值的频数
axes[2].set_title("离散：每日订单数（计数）")                           # 设置右图标题
axes[2].set_xlabel("订单数（整数取值）")                                # 设置右图横轴含义
plt.tight_layout()                                                    # 自动调整子图间距，避免重叠
save(fig, "fig_numeric.png")                                          # 保存图片到 website/figures/
