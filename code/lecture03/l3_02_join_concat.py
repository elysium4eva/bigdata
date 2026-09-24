"""l3_02_join_concat.py —— 纵向合并两份抽取文件，并把它们的结构（schema）对齐。

纵向拼接看着简单，直到两份文件的列名、列序、类型或单位不一致。
脚本把两期订单抽取拼在一起，先展示"不处理会怎样"，再给出显式对齐后的干净结果，
并附带一列来源标记与两条断言。
"""
import numpy as np                                                 # 为两份抽取造随机数
import pandas as pd                                                # 拼接、类型处理与对账
from _common_l3 import RNG, dump                                   # 复用公共种子与落盘函数

N_Q1, N_Q2 = 240, 260                                              # 两份抽取各自的行数

q1 = pd.DataFrame({                                                # 抽取一：结构比较规范
    "order_id": [f"Q1-{1000 + i}" for i in range(N_Q1)],           # 订单编号
    "date": pd.to_datetime("2026-01-01") + pd.to_timedelta(RNG.integers(0, 90, N_Q1), "D"),   # 日期列
    "store_id": RNG.choice([f"S{i:02d}" for i in range(1, 9)], N_Q1),   # 门店代码
    "amount": np.round(RNG.lognormal(6.0, 0.6, N_Q1), 2),          # 金额（元）
    "qty": RNG.integers(1, 9, N_Q1),                               # 数量（整数）
})                                                                 # 抽取一定义结束
q2 = pd.DataFrame({                                                # 抽取二：结构略有差异
    "OrderID": [f"Q2-{2000 + i}" for i in range(N_Q2)],            # 主键的列名不同
    "date": (pd.to_datetime("2026-04-01") + pd.to_timedelta(RNG.integers(0, 91, N_Q2), "D")).strftime("%d/%m/%Y"),   # 日/月/年
    "store": RNG.choice([f"S{i:02d}" for i in range(1, 9)], N_Q2), # 门店列名也不同
    "amount": [f"¥{v:,.2f}" for v in RNG.lognormal(6.1, 0.6, N_Q2)],   # 金额是带符号的字符串
    "quantity": RNG.integers(1, 9, N_Q2),                         # 数量列名不同
    "channel": RNG.choice(["online", "store"], N_Q2),              # 只有这份抽取才有的列
})                                                                 # 抽取二定义结束
dump(q1, "l3_orders_2026Q1.csv")                                   # 落盘抽取一
dump(q2, "l3_orders_2026Q2.csv")                                   # 落盘抽取二

before = pd.DataFrame({"extract": ["q1", "q2", "q1+q2"],           # 清洗前的基础对账表
                       "rows": [len(q1), len(q2), len(q1) + len(q2)],   # 各自行数
                       "missing_cells": [int(q1.isna().sum().sum()),   # 抽取一的缺失单元格
                                         int(q2.isna().sum().sum()),   # 抽取二的缺失单元格
                                         0]})                      # 合计行不是真实抽取
print("columns q1:", list(q1.columns))                             # 打印抽取一的列
print("columns q2:", list(q2.columns))                             # 打印抽取二的列
print("columns only in q1:", sorted(set(q1.columns) - set(q2.columns)))   # 单向的结构差异
print("columns only in q2:", sorted(set(q2.columns) - set(q1.columns)))   # 反向的结构差异

# ---------------------------------------------------------------- 1) 朴素拼接：问题出在哪
naive = pd.concat([q1, q2], ignore_index=True)                     # 直接纵向堆叠
print("naive concat shape:", naive.shape)                          # 500 × 9：改名产生空列
print("naive missing per column:\n", naive.isna().sum())           # 每个被改名的列都成了一片空洞

# ---------------------------------------------------------------- 2) 显式对齐结构
RENAME = {"OrderID": "order_id", "store": "store_id", "quantity": "qty"}   # 旧列名 → 标准列名
def align(df, source):                                             # 把一份抽取规范到目标结构
    """改名、统一类型并打上来源标记，使两份文件拥有同一套结构。"""        # 函数约定
    out = df.rename(columns=RENAME).copy()                         # 先改名，再在副本上操作
    out["date"] = pd.to_datetime(out["date"], format="mixed", dayfirst=True, errors="coerce")   # 日期统一
    out["amount"] = pd.to_numeric(                                 # 金额里还带着货币符号
        out["amount"].astype(str).str.replace(r"[^\d.\-]", "", regex=True),   # 先剥掉符号与空格
        errors="coerce")                                           # 无法解析的变成 NaN
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce").astype("Int64")   # 可空整数类型
    out["order_id"] = out["order_id"].astype("string")             # 主键显式转为字符串类型
    out["store_id"] = out["store_id"].astype("string")             # 门店列同样显式转字符串
    out["channel"] = out.get("channel", pd.Series(pd.NA, index=out.index)).astype("string")   # 补齐缺失列
    out["source"] = source                                         # 来源列：这行来自哪份文件
    return out[["order_id", "date", "store_id", "amount", "qty", "channel", "source"]]   # 统一列序
combined = pd.concat([align(q1, "Q1"), align(q2, "Q2")], ignore_index=True)   # 500 × 7
print("combined shape:", combined.shape)                           # 行数保留、列数统一
print("rows per source:", combined["source"].value_counts().to_dict())   # 对账：两份各多少行
assert len(combined) == len(q1) + len(q2), "concat lost or invented rows"   # 行数守恒断言
assert combined["order_id"].is_unique, "duplicate order ids after concat"   # 主键唯一断言
dump(combined, "l3_orders_combined.csv")                           # 落盘对齐后的合并表
dump(before, "l3_concat_accounting.csv")                           # 落盘对账表
print(combined.head(3).to_string())                                # 终端打印前几行便于检查
