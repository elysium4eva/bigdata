"""l3_03_join_types_validate.py —— 内/左/右/全外连接，以及连接键的基数校验。

脚本复现课件里的两份教学数据（Dataset 1 / 2 / 3 / 4），逐个演示每种连接会保留或丢弃什么，
并展示 pandas 的 validate= 参数如何把"悄悄发生的行数爆炸"变成一声明确的报错。
"""
import numpy as np                                                 # 为样本表造随机数
import pandas as pd                                                # 连接与校验逻辑
import matplotlib.pyplot as plt                                    # 连接结果对照图
from _common_l3 import RNG, save, dump                             # 复用公共种子、出图与落盘函数

# ---------------------------------------------------------------- 1) 课件的 Dataset 1 / Dataset 2
dataset1 = pd.DataFrame({"person": ["A", "B", "C", "D"],           # Dataset 1：一人一行
                         "age": [21, 57, 35, 23]})                 # 年龄
dataset2 = pd.DataFrame({"person": ["A", "B", "C", "D"],           # Dataset 2：问卷第二部分
                         "sex": ["male", "female", "male", "female"]})   # 同一批人的性别
dataset3 = pd.DataFrame({"person": ["A", "A", "B", "C", "C", "C", "E"],   # Dataset 3：一人多行
                         "visit": [1, 2, 1, 1, 2, 3, 1],           # 第几次到访
                         "spend": [120.0, 80.5, 240.0, 60.0, 95.5, 45.0, 300.0]})   # 每次消费额
dataset4 = pd.DataFrame({"person": ["E", "F"],                     # Dataset 4：两个新的人
                         "age": [41, 29]})                         # 他们的年龄，用于 append
dump(dataset1, "l3_dataset1.csv")                                  # 落盘四份教学数据
dump(dataset2, "l3_dataset2.csv")                                  # 落盘 Dataset 2
dump(dataset3, "l3_dataset3.csv")                                  # 落盘 Dataset 3
dump(dataset4, "l3_dataset4.csv")                                  # 落盘 Dataset 4
print("merge 1:1 (Stata: merge 1:1 person using dataset2):")       # 对应原课件的 Stata 写法
print(dataset1.merge(dataset2, on="person", how="inner", validate="1:1").to_string(index=False))   # 1:1 合并
print("merge m:1 (Stata: merge m:1 person using dataset2):")       # 一人多次到访
print(dataset3.merge(dataset2, on="person", how="left", validate="m:1").head().to_string(index=False))   # m:1
print("append (Stata: append using dataset4):")                    # 纵向追加保留全部行
print(pd.concat([dataset1, dataset4], ignore_index=True).to_string(index=False))   # 等价于 append

# ---------------------------------------------------------------- 2) 业务键上的连接语义
customers = pd.DataFrame({                                         # 左表：一个客户一行
    "customer_id": ["C001", "C002", "C003", "C004"],               # 客户键
    "city": ["Shenzhen", "Guangzhou", "Beijing", "Chengdu"],       # 客户所在城市
    "segment": ["VIP", "Regular", "VIP", "Regular"],               # 客户分层
})                                                                 # 左表定义结束
orders = pd.DataFrame({                                            # 右表：一个客户多笔订单
    "order_id": [f"O{i:03d}" for i in range(1, 9)],                # 订单键
    "customer_id": ["C001", "C001", "C002", "C003", "C003", "C005", "C005", "C006"],   # 有匹配不上的
    "amount": np.round(RNG.lognormal(5.6, 0.7, 8), 2),             # 订单金额（元）
})                                                                 # 右表定义结束
dump(customers, "l3_customers.csv")                                # 落盘客户样本
dump(orders, "l3_orders.csv")                                      # 落盘订单样本
JOINS = ["inner", "left", "right", "outer"]                        # 课件里的四种连接
results = {}                                                       # 每种连接的结果存起来
for how in JOINS:                                                  # 用同一组键跑四种连接
    res = customers.merge(orders, on="customer_id", how=how,       # 按业务键合并
                          indicator=True)                          # indicator 记录每行的来源
    results[how] = res                                             # 保存结果供画图使用
    print(f"{how:6s} rows={len(res):2d}  "                        # 打印行数
          f"provenance={res['_merge'].value_counts().to_dict()}")  # 打印来源分布
union = pd.concat([orders, orders.iloc[:2]], ignore_index=True)    # union 是行追加，不是连接
print("union rows (orders + 2 duplicated orders):", len(union))    # union 全保留，重复也算

# ---------------------------------------------------------------- 3) validate= 让错误显式暴露
dup_keys = pd.concat([customers, customers.iloc[[0]]], ignore_index=True)   # 客户 C001 出现了两次
try:                                                               # 这里期望直接报错，而不是悄悄放大
    dup_keys.merge(orders, on="customer_id", how="left", validate="1:1")   # 声明是 1:1，实际不成立
    print("validate=1:1 did NOT complain -- this would be a silent row explosion")   # 正常情况走不到这里
except pd.errors.MergeError as err:                                # 违反声明时 pandas 抛 MergeError
    print("validate=1:1 correctly raised:", str(err).splitlines()[0])   # 打印报错的第一行
exploded = dup_keys.merge(orders, on="customer_id", how="left")    # 不加 validate：行数悄悄翻倍
print("rows without validate:", len(exploded), "vs customers:", len(dup_keys))   # 5 行变 8 行的证据
safe = orders.merge(customers, on="customer_id", how="left", validate="m:1")   # 多对一：订单多、客户唯一
print("rows with validate=m:1:", len(safe), "(one row per order, customer attributes attached)")   # 行数不变

# ---------------------------------------------------------------- 图：连接对照 + 行数
fig, axes = plt.subplots(2, 3, figsize=(13.6, 4.6))                # 五张结果表 + 一张行数柱状图
panels = JOINS + ["union"]                                         # 课件里展示的五种操作
for ax, name in zip(axes.ravel(), panels):                         # 每种操作各占一格
    ax.axis("off")                                                 # 表格不需要坐标轴
    if name == "union":                                            # union 是行追加，直接展示结果表
        data = union[["order_id", "customer_id"]].head(6)          # 取前六行
        title = "union: append rows (keeps duplicates)"            # 标题写明语义
    else:                                                          # 其余四种是连接
        res = results[name]                                        # 取出之前保存的结果
        data = res[["customer_id", "order_id", "amount"]].sort_values("customer_id").head(6)   # 前六行
        nulls = int(res["amount"].isna().sum())                    # 连接后产生多少空值
        title = f"{name} join: {len(res)} rows, {nulls} unmatched"  # 标题写出行数与空值数
    tbl = ax.table(cellText=data.fillna("NULL").astype(str).values,   # 渲染小结果表
                   colLabels=list(data.columns), loc="center", cellLoc="center")   # 居中显示
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.25)   # 字号紧凑但可读
    ax.set_title(title, fontsize=10.5)                             # 用标题说明该连接的语义
axes[1, 2].axis("on")                                              # 最后一格改回柱状图
counts = [len(results[j]) for j in JOINS] + [len(union)]          # 五种操作各自的结果行数
axes[1, 2].bar(JOINS + ["union"], counts, color="#8A0C3C", alpha=0.85)   # 一眼看出行数变化
axes[1, 2].set_title("rows returned per operation", fontsize=10.5)   # 子图标题
plt.tight_layout()                                                 # 调整六个子图
save(fig, "fig_join_types.png")                                    # 保存给课件使用
