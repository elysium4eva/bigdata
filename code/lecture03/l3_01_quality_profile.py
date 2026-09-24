"""l3_01_quality_profile.py —— 造一份"脏数据"，并给它的五项质量特征打分。

脚本刻意造出订单表里最常见的缺陷（日期格式混用、主键重复、类别写法不一致、
金额里混入货币符号、取值缺失、不可能的量级），然后按课件里的五项质量特征
（有效性、准确性、完整性、一致性、单位一致）逐项量化。
"""
import numpy as np                                                 # 随机抽样与数值处理
import pandas as pd                                                # 构造表与做质量统计
import matplotlib.pyplot as plt                                    # 画四联质量仪表盘
from _common_l3 import RNG, save, dump                             # 复用公共种子、出图与落盘函数

N = 600                                                            # 合成的订单行数
CITIES = ["Shenzhen", "shenzhen", "SHENZHEN ", " Shenzhen",        # 同一个城市写成五种样子
          "Guangzhou", "Guangzhou City", "Guang Zhou"]             # 再补几种写法制造不一致
order_id = [f"SO{2026000 + i}" for i in range(1, N + 1)]           # 先造一列干净的订单号
for i in RNG.choice(range(1, N), size=18, replace=False):          # 随机挑 18 行准备复制主键
    order_id[i] = order_id[i - 1]                                  # 复制上一行订单号：制造重复主键
dates = []                                                         # 日期会用三种格式混着进来
for i in range(N):                                                 # 逐行生成原始日期字符串
    day = pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(RNG.integers(0, 120)))   # 随机取一天
    fmt = RNG.integers(0, 3)                                       # 随机选一种来源格式
    dates.append(day.strftime("%Y-%m-%d") if fmt == 0 else         # 格式一：2026-02-11
                 day.strftime("%d/%m/%Y") if fmt == 1 else         # 格式二：日/月/年
                 day.strftime("%Y%m%d"))                           # 格式三：紧凑写法
city = [str(RNG.choice(CITIES)) for _ in range(N)]                 # 城市列带上各种写法变体
amount_raw = []                                                    # 金额一律以原始字符串保存
for i in range(N):                                                 # 逐行生成原始金额字符串
    value = float(RNG.lognormal(mean=6.2, sigma=0.7))              # 对数正态的消费额：右偏
    style = RNG.integers(0, 4)                                     # 四种来源习惯
    if style == 0:                                                 # 习惯一：货币符号 + 千分位
        amount_raw.append(f"¥{value:,.2f}")                        # 例如 ¥1,234.56
    elif style == 1:                                               # 习惯二：货币代码后缀
        amount_raw.append(f"{value:.2f} RMB")                      # 例如 1234.56 RMB
    elif style == 2:                                               # 习惯三：千元为单位（单位不一致）
        amount_raw.append(f"{value / 1000:,.1f} thousand CNY")     # 例如 1.2 thousand CNY
    else:                                                          # 习惯四：纯数字还带空格
        amount_raw.append(f" {value:,.2f} ")                       # 首尾各留一个空格
qty = RNG.integers(1, 12, N).astype(float)                         # 数量列先做成浮点
qty[RNG.choice(N, size=70, replace=False)] = np.nan                # 随机让 70 行缺数量
amount_raw[3] = "-500.00"                                          # 埋一个不可能的负金额
amount_raw[7] = "0"                                                # 再埋一个不可能的零金额
payment = RNG.choice(["paid", "Paid", "PAID", "unpaid", "N/A",      # 支付状态有同义写法
                      "Not Applicable", "-"], size=N)              # "不适用"写成三种样子
raw = pd.DataFrame({"order_id": order_id,                          # 订单号（含重复主键）
                    "date_raw": dates,                             # 日期（格式混用）
                    "city_raw": city,                              # 城市（写法不一致）
                    "amount_raw": amount_raw,                      # 金额（符号与单位噪声）
                    "qty": qty,                                    # 数量（含缺失）
                    "payment_raw": payment})                       # 支付状态（同义词）
dump(raw, "l3_orders_dirty.csv")                                   # 落盘，供后续脚本读取

# ---------------------------------------------------------------- 质量画像：五项特征量化
parsed_date = pd.to_datetime(raw["date_raw"], format="mixed", dayfirst=True, errors="coerce")   # 尽力解析
amount = (raw["amount_raw"].str.replace(r"[^\d.\-]", "", regex=True)   # 先去掉符号做快速检查
          .pipe(pd.to_numeric, errors="coerce"))                       # 无法解析的变成 NaN
dup_rows = int(raw.duplicated().sum())                             # 完全重复的行数（本例为 0）
dup_keys = int(raw["order_id"].duplicated().sum())                 # 重复的业务主键（埋了 18 个）
missing_rate = raw.isna().mean().mul(100).round(2)                 # 完整性：各列缺失率
bad_amount = int(((amount <= 0) | (amount > 100000)).sum())        # 准确性：不可能的量级
city_raw_n = raw["city_raw"].nunique()                             # 一致性：原始列出现的变体数
city_clean_n = raw["city_raw"].str.strip().str.title().nunique()   # 一致性：归一化后的变体数
pay_raw_n = raw["payment_raw"].nunique()                           # 单位一致：同一状态有几种写法
pay_clean_n = (raw["payment_raw"].str.strip().str.lower()          # 先统一大小写与首尾空格
               .replace({"n/a": "not_applicable", "-": "not_applicable",   # 把"不适用"的同义写法
                         "not applicable": "not_applicable"}).nunique())   # 归并成同一取值
valid_date_rate = 100 * parsed_date.notna().mean()                 # 有效性：日期可解析比例
valid_id_rate = 100 * raw["order_id"].str.match(r"^SO\d{7}$").mean()   # 有效性：订单号是否符合格式
score = {                                                          # 质量评分卡：每项 0—100 分
    "Validity": round(0.5 * valid_date_rate + 0.5 * valid_id_rate, 1),   # 日期与订单号各占一半
    "Accuracy": round(100 - 100 * bad_amount / N, 1),              # 合理金额的占比
    "Completeness": round(100 - float(missing_rate.mean()), 1),    # 已填报单元格的占比
    "Consistency": round(100 * city_clean_n / max(city_raw_n, 1), 1),    # 归一后变体越少分越高
    "Uniqueness": round(100 * (N - dup_keys) / N, 1),              # 业务主键唯一的占比
}                                                                  # 评分字典到此结束
report = pd.DataFrame({"dimension": list(score), "score": list(score.values())})   # 整理成表
dump(report, "l3_quality_scorecard.csv")                           # 落盘评分卡
print(report.to_string(index=False))                               # 同时在终端打印

# ---------------------------------------------------------------- 图：质量仪表盘
fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.4))                 # 一行四格，适配 16:9 幻灯片
mr = missing_rate.sort_values(ascending=False)                     # 缺失最严重的列排前面
axes[0].bar(mr.index, mr.values, color="#8A0C3C", alpha=0.85)       # 第一格：完整性
axes[0].set_title("Completeness:\nmissing rate (%)", fontsize=11)   # 子图标题
axes[0].tick_params(axis="x", rotation=45, labelsize=8)             # 旋转横轴标签避免重叠
axes[1].bar(["full-row\nduplicates", "duplicate\nkeys"], [dup_rows, dup_keys],   # 两类重复计数
            color=["#66717D", "#8A0C3C"])                           # 第二格：唯一性
axes[1].set_title("Uniqueness:\nduplicated records", fontsize=11)   # 子图标题
axes[1].tick_params(axis="x", labelsize=8)                          # 缩小横轴标签字号
axes[2].bar(["raw", "normalised"], [city_raw_n, city_clean_n],      # 归一前后的变体数
            color=["#C9A227", "#711426"])                           # 第三格：一致性
axes[2].set_title("Consistency:\ncity spellings", fontsize=11)      # 子图标题
axes[2].tick_params(axis="x", labelsize=9)                          # 横轴标签字号
axes[3].barh(list(score)[::-1], list(score.values())[::-1], color="#8A0C3C", alpha=0.85)   # 第四格：评分卡
axes[3].set_xlim(0, 100)                                            # 分数按百分制
axes[3].tick_params(axis="y", labelsize=8.5)                        # 维度名称字号
axes[3].set_title("Quality scorecard\n(0-100)", fontsize=11)        # 子图标题
plt.tight_layout()                                                 # 自动调整四格间距
save(fig, "fig_quality_profile.png")                               # 保存给课件使用
print("payment spellings raw -> clean:", pay_raw_n, "->", pay_clean_n)   # 日志：单位一致的证据
print("valid dates %:", round(valid_date_rate, 1), "| implausible amounts:", bad_amount)   # 日志：有效性
