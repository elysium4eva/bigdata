"""l3_05_cleaning_pipeline.py —— 在一张脏订单表上走完清洗五步法。

第一步去掉重复与不相关观测，第二步修复结构性错误，第三步对异常值做分类（而不是一律删除），
第四步处理缺失值，第五步做验证与 QA。每一步都写入审计日志，让清洗过程可追溯、可复现——
这正是课件强调"必须先有模板"的原因。
"""
import numpy as np                                                 # 分位数等数值计算
import pandas as pd                                                # 清洗所需的表操作
import matplotlib.pyplot as plt                                    # 清洗前后对照图
from _common_l3 import DAT, save, dump                             # 路径常量与出图、落盘函数

raw = pd.read_csv(DAT / "l3_orders_dirty.csv", dtype=str)           # 读入 l3_01 生成的脏样本
raw["qty"] = pd.to_numeric(raw["qty"], errors="coerce")             # 数量列需要还原成数值
log = []                                                            # 每个清洗动作记一行日志
def record(step, before, after, note):                              # 把一步处理写进审计日志
    """向审计日志追加一条清洗记录。"""                                  # 函数约定
    log.append({"step": step, "rows_before": before, "rows_after": after,   # 处理前后的行数
                "rows_removed": before - after, "note": note})      # 移除了多少行、为什么
    print(f"{step:38s} {before:5d} -> {after:5d}   {note}")         # 同时在终端打印该步

# ---------------------------------------------------------------- 第一步：重复与不相关观测
n0 = len(raw)                                                       # 基线行数
df = raw.drop_duplicates().copy()                                   # 去掉完全重复的行
n1 = len(df)                                                        # 完全去重后的行数
record("Step 1a exact duplicate rows", n0, n1, "identical rows removed")   # 记录该步
df = df.drop_duplicates(subset="order_id", keep="first")            # 一个业务主键只留一行
n2 = len(df)                                                        # 主键去重后的行数
record("Step 1b duplicate business keys", n1, n2, "same order_id kept once")   # 记录该步
irrelevant = df["order_id"].isna()                                  # 不相关 = 连主键都没有
df = df.loc[~irrelevant].copy()                                     # 丢掉无法回联的行
n3 = len(df)                                                        # 相关性过滤后的行数
record("Step 1c irrelevant observations", n2, n3, "rows without a key removed")   # 记录该步

# ---------------------------------------------------------------- 第二步：结构性错误
city_map = {"shenzhen": "Shenzhen", "shēnzhèn": "Shenzhen",         # 各种拼法统一到一个标准写法
            "guangzhou city": "Guangzhou", "guang zhou": "Guangzhou"}   # 别名映射表
def normalise_city(value):                                          # 按忽略大小写的方式查别名表
    """去掉首尾空格、压缩空格，并把别名映射为标准城市名。"""              # 函数约定
    key = " ".join(str(value).strip().lower().split())              # 先做形式归一
    return city_map.get(key, key.title())                           # 命中映射就替换，否则首字母大写
df["city"] = df["city_raw"].map(normalise_city)                     # 逐行应用映射
pay_map = {"paid": "paid", "unpaid": "unpaid", "n/a": "not_applicable",   # 同一状态的不同写法
           "-": "not_applicable", "not applicable": "not_applicable"}   # 全部映射到同一取值
df["payment"] = (df["payment_raw"].str.strip().str.lower()          # 先去空格、统一小写
                 .map(pay_map).fillna("unknown"))                   # 再映射，未识别的显式记为 unknown
df["date"] = pd.to_datetime(df["date_raw"], format="mixed", dayfirst=True, errors="coerce")   # 第二步：解析日期
bad_dates = int(df["date"].isna().sum())                            # 统计无法解析的日期数
record("Step 2 structural errors", n3, len(df), f"cities/payments normalised, {bad_dates} bad dates")   # 记日志
print("city variants:", raw["city_raw"].nunique(), "->", df["city"].nunique())   # 一致性的证据
print("payment variants:", raw["payment_raw"].nunique(), "->", df["payment"].nunique())   # 单位一致的证据

# ---------------------------------------------------------------- 第三步：异常值（先分类，再处置）
amount = pd.to_numeric(df["amount_raw"].astype(str)                 # 金额字符串里还带符号
                       .str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")   # 剥符号后转数值
df["amount"] = amount                                               # 把解析结果挂回表上
impossible = (df["amount"] <= 0) | (df["amount"] > 100000)          # 规则判断：物理上不可能
q1, q3 = df["amount"].quantile(.25), df["amount"].quantile(.75)     # 计算四分位数
iqr = q3 - q1                                                       # 四分位距
statistical = (df["amount"] < q1 - 1.5 * iqr) | (df["amount"] > q3 + 1.5 * iqr)   # 统计意义上的离群
df["flag_impossible"] = impossible                                  # 只打标记，绝不悄悄删除
df["flag_outlier"] = statistical & ~impossible                      # 统计离群但业务上可能真实
record("Step 3 outlier classification", len(df), len(df),           # 这一步不减行
       f"{int(impossible.sum())} impossible, {int(statistical.sum())} statistical outliers")   # 记录数量

# ---------------------------------------------------------------- 第四步：处理缺失
miss_before = df[["qty", "amount", "date"]].isna().mean().mul(100).round(2)   # 处理前的缺失率
median_qty = df["qty"].median()                                     # 中位数对偏态与离群都稳健
df["qty_missing"] = df["qty"].isna().astype(int)                    # 缺失指示列：缺失本身就是信息
df["qty"] = df["qty"].fillna(median_qty)                            # 用中位数填充，并记录理由
miss_after = df[["qty", "amount", "date"]].isna().mean().mul(100).round(2)   # 处理后的缺失率
record("Step 4 missing handling", len(df), len(df),                 # 这一步也不减行
       f"qty filled with median {median_qty:.1f}, indicator column added")   # 记录处置方式
print("missing rate before:\n", miss_before.to_string())            # 打印处理前缺失率
print("missing rate after:\n", miss_after.to_string())              # 打印处理后缺失率

# ---------------------------------------------------------------- 第五步：验证与 QA
assert df["order_id"].is_unique, "order_id must be unique after cleaning"   # 主键唯一性断言
assert df["date"].notna().all(), "unparsable dates must be resolved"        # 日期有效性断言
assert df["amount"].notna().all(), "amount must be numeric after cleaning"  # 金额类型断言
assert set(df["payment"].unique()) <= {"paid", "unpaid", "not_applicable", "unknown"}, "unexpected payment"   # 取值域
assert df["city"].nunique() <= 5, "city normalisation failed"       # 归一化效果断言
clean = df[["order_id", "date", "city", "amount", "qty", "qty_missing", "payment",   # 统一最终列序
            "flag_impossible", "flag_outlier"]].copy()              # 分析层可用的列集合
dump(clean, "l3_orders_clean.csv")                                  # 落盘清洗后的表
dump(pd.DataFrame(log), "l3_cleaning_log.csv")                      # 落盘清洗审计日志

# ---------------------------------------------------------------- 结构性错误一页用的样本
city_variants = (raw["city_raw"].value_counts().rename_axis("city_raw")   # 统计每种原始写法
                 .reset_index(name="rows"))                         # 整理成两列
dump(city_variants, "l3_city_names_raw.csv")                        # 落盘原始变体
mapping = pd.DataFrame({"raw": list(city_map), "canonical": list(city_map.values())})   # 别名映射表
dump(mapping, "l3_city_names_map.csv")                              # 落盘映射表

# ---------------------------------------------------------------- 图 A：前后对照
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # 三格：删行 / 缺失 / 分布
steps = [r["step"].split()[1] for r in log[:3]] + ["Step 4"]        # 前三步的短标签（图注用）
axes[0].bar([r["step"].split()[1] for r in log[:3]],                # 前三个动作的名称
            [r["rows_removed"] for r in log[:3]], color="#8A0C3C", alpha=0.85)   # 各自动删了多少行
axes[0].set_title("Step 1: rows removed")                           # 子图标题
axes[0].tick_params(axis="x", rotation=20)                          # 标签旋转便于阅读
axes[0].set_ylabel("rows")                                          # 纵轴含义
x = np.arange(len(miss_before))                                     # 分组柱的横坐标
axes[1].bar(x - 0.2, miss_before.values, width=0.4, label="before", color="#C9A227")   # 处理前缺失率
axes[1].bar(x + 0.2, miss_after.values, width=0.4, label="after", color="#8A0C3C")     # 处理后缺失率
axes[1].set_xticks(x, miss_before.index)                            # 横轴：列名
axes[1].set_title("Step 4: missing rate (%)")                       # 子图标题
axes[1].legend()                                                    # 图例说明两种颜色
positive = df["amount"] > 0                                         # 取对数要求金额为正
vals = [np.log10(df.loc[df[k] & positive, "amount"]) for k in ["flag_outlier", "flag_impossible"]]   # 两类标记
axes[2].hist(np.log10(df.loc[positive, "amount"]), bins=30,         # 全部正金额的分布（对数尺度）
             color="#66717D", alpha=0.6, label="all rows")          # 灰色背景直方图
for v, lab, c in zip(vals, ["statistical", "impossible"], ["#C9A227", "#8A0C3C"]):   # 叠加两类标记
    axes[2].hist(v, bins=15, color=c, alpha=0.85, label=lab)        # 各自一张直方图
axes[2].set_title("Step 3: amount distribution (log10, positive only)")   # 子图标题
axes[2].legend()                                                    # 图例
plt.tight_layout()                                                  # 调整三个子图
save(fig, "fig_cleaning_before_after.png")                          # 保存：清洗一页用图

# ---------------------------------------------------------------- 图 B：结构性错误修复
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # 变体 / 映射 / 结果
v = city_variants.set_index("city_raw")["rows"].sort_values()       # 原始写法的出现次数
axes[0].barh(v.index, v.values, color="#8A0C3C", alpha=0.85)        # 原始列有多乱
axes[0].set_title(f"raw spellings ({len(v)} variants)")             # 子图标题
axes[0].tick_params(axis="y", labelsize=8)                          # 长标签缩小字号
axes[1].axis("off")                                                 # 映射用表格展示，不用坐标轴
tbl = axes[1].table(cellText=mapping.values, colLabels=["raw", "canonical"],   # 别名映射表
                    loc="center", cellLoc="center")                 # 居中显示
tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.3)   # 紧凑但可读
axes[1].set_title("alias mapping applied")                          # 子图标题
final_counts = df["city"].value_counts()                            # 清洗后的标准类别计数
axes[2].bar(final_counts.index, final_counts.values, color="#C9A227", alpha=0.9)   # 归一后的类别列
axes[2].set_title(f"after cleaning ({df['city'].nunique()} cities)")   # 子图标题
axes[2].tick_params(axis="x", rotation=20)                          # 标签旋转
plt.tight_layout()                                                  # 调整三个子图
save(fig, "fig_structural_errors.png")                              # 保存：结构错误一页用图
print("clean shape:", clean.shape, "| audit log rows:", len(log))   # 收尾日志
