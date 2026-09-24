"""l3_10_transform_assert.py —— 驯服右偏分布的几种变换，以及一份 schema 断言报告。

收入类变量通常右偏，少数极大值会主导模型与检验。脚本比较常见的几种变换，
量化它们对偏度与峰度的影响，最后用一份轻量的（pandera 风格的）schema 校验，
在发布数据前把关——这正是生产管道会做的事。
"""
import numpy as np                                                 # 对数与描述统计
import pandas as pd                                                # 表格与分位分箱
import matplotlib.pyplot as plt                                    # 分布对照图
from _common_l3 import RNG, save, dump                             # 复用公共种子、出图与落盘函数

# ---------------------------------------------------------------- 1) 一个右偏变量
N = 900                                                            # 企业数量
revenue = RNG.lognormal(mean=4.6, sigma=0.9, size=N)                # 强烈右偏的营收
region = RNG.choice(["East", "Central", "West"], N, p=[0.5, 0.3, 0.2])   # 一列类别变量
raw = pd.DataFrame({"firm_id": [f"F{i:04d}" for i in range(N)],     # 企业主键
                    "region": region,                              # 地区（类别）
                    "revenue": np.round(revenue, 2)})              # 营收（右偏数值）
dump(raw, "l3_income_skewed.csv")                                  # 落盘原始样本

# ---------------------------------------------------------------- 2) 五种变换
transformed = raw.copy()                                           # 在副本上做变换，保留原始表
transformed["log1p"] = np.log1p(transformed["revenue"])             # log(1+x)：处理零值并压缩右尾
transformed["sqrt"] = np.sqrt(transformed["revenue"])               # 平方根：比对数更温和
transformed["zscore"] = ((transformed["revenue"] - transformed["revenue"].mean())   # 先中心化
                         / transformed["revenue"].std(ddof=0))      # 再缩放：均值 0、标准差 1
transformed["minmax"] = ((transformed["revenue"] - transformed["revenue"].min())    # 平移到从 0 开始
                         / (transformed["revenue"].max() - transformed["revenue"].min()))   # 缩放到 [0,1]
transformed["quartile"] = pd.qcut(transformed["revenue"], 4, labels=["Q1", "Q2", "Q3", "Q4"])   # 等频分箱
transformed["decile"] = pd.qcut(transformed["revenue"], 10, labels=False)   # 十分位（整数编码）
summary = pd.DataFrame([                                          # 变换前后的形状对比
    {"variable": name, "mean": round(transformed[name].mean(), 3),   # 位置
     "sd": round(transformed[name].std(), 3),                      # 离散度
     "skew": round(transformed[name].skew(), 3),                   # 偏度：0 表示对称
     "kurtosis": round(transformed[name].kurt(), 3)}               # 超额峰度：0 接近正态
    for name in ["revenue", "log1p", "sqrt", "zscore", "minmax"]])  # 五个可比的列
print(summary.to_string(index=False))                               # 终端打印每种变换的效果
dump(summary, "l3_transform_summary.csv")                           # 落盘对比表
print("quartile counts:", transformed["quartile"].value_counts().sort_index().to_dict())   # 等频分箱计数
print("decile means (revenue):\n",                               # 十分位常用于业务分组汇报
      transformed.groupby("decile")["revenue"].mean().round(1).to_string())   # 各十分位的营收均值

# ---------------------------------------------------------------- 3) 发布前的 schema 断言
RULES = {                                                          # 一份最小化的 schema 契约
    "firm_id": {"dtype": "object", "unique": True, "regex": r"^F\d{4}$"},   # 主键：唯一且有格式
    "region": {"dtype": "object", "allowed": ["East", "Central", "West"]},   # 类别：取值域封闭
    "revenue": {"dtype": "float64", "min": 0.0, "max": 1_000_000.0, "nullable": False},   # 数值范围
    "log1p": {"dtype": "float64", "min": 0.0, "nullable": False},   # 派生列非负
}                                                                  # 契约定义到此结束
def check(table, rules):                                           # 逐条跑契约并收集违规项
    """返回一份违规明细，而不是遇到第一个问题就中断。"""                 # 函数约定
    report = []                                                    # 每条规则一行
    for column, spec in rules.items():                             # 遍历声明的列
        present = column in table.columns                          # 列不存在本身就是违规
        report.append({"column": column, "rule": "present",        # 记录存在性检查
                       "ok": present, "detail": "" if present else "missing column"})   # 失败时写明原因
        if not present:                                            # 列不存在就跳过其余规则
            continue                                               # 处理下一列
        series = table[column]                                     # 待检查的列
        if spec.get("unique"):                                     # 唯一性规则
            dup = int(series.duplicated().sum())                   # 重复值个数
            report.append({"column": column, "rule": "unique", "ok": dup == 0,   # 唯一性检查结果
                           "detail": f"{dup} duplicates"})         # 写明重复数量
        if spec.get("regex"):                                      # 标识符格式规则
            bad = int((~series.astype(str).str.match(spec["regex"])).sum())   # 不合格式的值
            report.append({"column": column, "rule": "regex", "ok": bad == 0,   # 格式检查结果
                           "detail": f"{bad} non-conforming"})     # 写明不合规数量
        if spec.get("allowed"):                                    # 取值域规则
            extra = sorted(set(series.dropna()) - set(spec["allowed"]))   # 出现过的意外类别
            report.append({"column": column, "rule": "allowed", "ok": not extra,   # 取值域检查结果
                           "detail": f"unexpected: {extra}"})      # 写明越界取值
        if spec.get("min") is not None:                            # 下界规则
            below = int((series < spec["min"]).sum())              # 小于下界的值
            report.append({"column": column, "rule": "min", "ok": below == 0,   # 下界检查结果
                           "detail": f"{below} below {spec['min']}"})   # 写明越界数量
        if spec.get("max") is not None:                            # 上界规则
            above = int((series > spec["max"]).sum())              # 大于上界的值
            report.append({"column": column, "rule": "max", "ok": above == 0,   # 上界检查结果
                           "detail": f"{above} above {spec['max']}"})   # 写明越界数量
        if spec.get("nullable") is False:                          # 非空规则
            miss = int(series.isna().sum())                        # 缺失值个数
            report.append({"column": column, "rule": "not-null", "ok": miss == 0,   # 非空检查结果
                           "detail": f"{miss} missing"})           # 写明缺失数量
    return pd.DataFrame(report)                                    # 返回明细表供页面展示
report = check(transformed, RULES)                                 # 对变换后的表执行校验
print("schema checks passed:", int(report["ok"].sum()), "of", len(report))   # 打印通过条数
print(report[~report["ok"]].to_string(index=False))                # 只打印失败的条目
dump(report, "l3_schema_report.csv")                               # 落盘校验报告
assert report["ok"].all(), "schema contract failed: fix the table before publishing it"   # 未通过就让管道停下

# ---------------------------------------------------------------- 图：变换前后
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))                 # 原始 / 变换后 / 形状指标
axes[0].hist(transformed["revenue"], bins=40, color="#8A0C3C", alpha=0.85)   # 原始分布：强烈右偏
axes[0].set_title(f"raw revenue (skew = {transformed['revenue'].skew():.2f})")   # 偏度写进标题
axes[0].set_xlabel("revenue")                                       # 横轴含义
axes[1].hist(transformed["log1p"], bins=40, color="#C9A227", alpha=0.9)   # log1p 之后
axes[1].set_title(f"log1p (skew = {transformed['log1p'].skew():.2f})")   # 偏度写进标题
axes[1].set_xlabel("log(1 + revenue)")                              # 横轴含义
metrics = summary.set_index("variable")[["skew", "kurtosis"]]       # 各变换的形状指标
x = np.arange(len(metrics))                                         # 柱状图横坐标
axes[2].bar(x - 0.2, metrics["skew"], width=0.4, label="skew", color="#8A0C3C")   # 偏度对比
axes[2].bar(x + 0.2, metrics["kurtosis"], width=0.4, label="kurtosis", color="#66717D")   # 峰度对比
axes[2].axhline(0, color="#1E2630", lw=1.0)                         # 零线：对称且接近正态
axes[2].set_xticks(x, metrics.index, rotation=20, fontsize=8.5)     # 变换名称
axes[2].set_title("shape metrics after transformation")             # 子图标题
axes[2].legend(fontsize=8.5)                                        # 图例
plt.tight_layout()                                                  # 调整三个子图
save(fig, "fig_transform_scale.png")                                # 保存：变换一页用图
