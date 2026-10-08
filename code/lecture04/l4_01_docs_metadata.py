"""l4_01_docs_metadata.py —— 造一份调研样本，并按原课件的 20 项清单体检文档完成度。

原课件的立场是：**在数据收集之前**就开始记录文档，至少和数据放在一起。
本脚本先把这份调研数据落盘，再自动生成"字段字典"（codebook），
最后按 20 项文档要素逐项打分，指出还缺什么。
"""
import numpy as np                                                 # 随机抽样
import pandas as pd                                                # 表格处理
import matplotlib.pyplot as plt                                    # 文档完成度图
from _common_l4 import RNG, DAT, save, dump, style_axes, annotated   # 复用公共工具与图注工具

N = 500                                                            # 调研样本量
CITIES = ["深圳", "上海", "北京", "广州", "成都", "杭州",           # 六个城市，权重之和为 1
          "武汉", "西安"]                                             # 六个城市名的第二行
CITY_P = [0.26, 0.2, 0.18, 0.12, 0.08, 0.08, 0.05, 0.03]           # 各城市抽样比例
CHANNELS = ["线上", "门店", "电话"]                                # 触达渠道
EDU = ["高中及以下", "本科", "硕士及以上"]                          # 学历分档

survey = pd.DataFrame({                                            # 组装调研样本
    "respondent_id": [f"R{i:04d}" for i in range(1, N + 1)],       # 受访者编号（主键）
    "city": RNG.choice(CITIES, N, p=CITY_P),                       # 城市（类别）
    "channel": RNG.choice(CHANNELS, N, p=[0.55, 0.3, 0.15]),       # 触达渠道（类别）
    "education": RNG.choice(EDU, N, p=[0.32, 0.55, 0.13]),         # 学历（有序类别）
    "age": RNG.integers(18, 66, N),                                # 年龄（数值）
    "income_k": np.round(RNG.lognormal(2.55, 0.5, N), 1),          # 月收入（千元，右偏）
    "satisfaction": RNG.choice([1, 2, 3, 4, 5], N, p=[0.05, 0.1, 0.25, 0.4, 0.2]),   # 满意度 1—5
    "spend_k": np.round(RNG.lognormal(1.6, 0.7, N), 2),            # 月消费（千元，右偏）
    "revisit": RNG.random(N) < 0.42,                               # 是否会复购（布尔）
})                                                                # 调研样本定义结束
survey.loc[RNG.choice(N, 12, replace=False), "income_k"] = np.nan   # 收入题敏感：人为造成 12 条缺失
survey.loc[RNG.choice(N, 5, replace=False), "satisfaction"] = np.nan   # 满意度也有少量缺失
dump(survey, "l4_survey_raw.csv")                                  # 落盘样本

# ---------------------------------------------------------------- 字段字典（codebook）自动生成
def field_dictionary(df, units):                                   # 为每一列生成字典条目
    """返回字段字典：列名、类型、非空率、唯一值数、取值范围/样例、单位。"""   # 函数约定
    rows = []                                                      # 每个字段一行
    for col in df.columns:                                         # 遍历所有列
        s = df[col]                                                # 取出该列
        if pd.api.types.is_numeric_dtype(s):                       # 数值型：给范围与分位数
            rng_txt = f"{s.min():.2f} ~ {s.max():.2f}"             # 最小到最大
        elif pd.api.types.is_bool_dtype(s):                        # 布尔型：给真值比例
            rng_txt = f"True 占比 {s.mean():.1%}"                   # 真值占比
        else:                                                      # 类别型：给取值个数与示例
            rng_txt = f"{s.nunique()} 类，如 {', '.join(map(str, s.dropna().unique()[:3]))}"   # 前三个取值
        rows.append({"字段": col, "类型": str(s.dtype),             # 列名与 dtype
                     "非空率": f"{s.notna().mean():.1%}",           # 非空率（完整性）
                     "唯一值": int(s.nunique()),                    # 唯一值个数
                     "取值范围/样例": rng_txt,                      # 取值域说明
                     "单位": units.get(col, "-")})                  # 单位（若已知）
    return pd.DataFrame(rows)                                      # 汇总成表
UNITS = {"income_k": "千元/月", "spend_k": "千元/月", "age": "岁",   # 关键字段的单位
         "satisfaction": "1—5 分", "revisit": "0/1"}                # 评分与布尔字段的说明
dictionary = field_dictionary(survey, UNITS)                       # 生成字段字典
dump(dictionary, "l4_data_dictionary.csv")                         # 落盘为 CSV
md = ["# 数据字典 Data Dictionary（l4_survey_raw.csv）", "",        # 同时写成 Markdown 便于放进报告
      "| 字段 | 类型 | 非空率 | 唯一值 | 取值范围/样例 | 单位 |",    # 表头
      "| --- | --- | --- | --- | --- | --- |"]                     # 分隔行
for _, r in dictionary.iterrows():                                 # 逐行拼表格
    md.append(f"| {r['字段']} | {r['类型']} | {r['非空率']} | {r['唯一值']} | {r['取值范围/样例']} | {r['单位']} |")   # 拼一行 Markdown 表格
(DAT / "l4_data_dictionary.md").write_text("\n".join(md), encoding="utf-8")   # 写入 Markdown
print("[data]", DAT / "l4_data_dictionary.md")                     # 打印产物路径

# ---------------------------------------------------------------- 按原课件 20 项清单体检文档
DOC_ITEMS = {                                                      # 20 项文档要素 → 本项目的完成情况
    "Title 标题": "l4_survey_raw.csv 消费者满意度和复购调研（2026 春）",   # 标题
    "Creator 创建者": "大数据分析课程组（教学用合成数据）",              # 创建者
    "Identifier 标识符": "SZU-BDA-2026-L4-SURVEY-01",              # 内部编号
    "Subject 主题词": "消费者满意度, 复购意愿, 渠道差异",                # 关键词
    "Funders 资助方": "无（课程自建）",                              # 资助信息
    "Rights 权利": "教学使用；不含真实个人信息",                       # 知识产权
    "Access 获取方式": "课程仓库 website/data/lecture04/",           # 获取路径
    "Language 语言": "中文（字段名）",                              # 内容语言
    "Dates 日期": "采集期 2026-03；发布于 2026-03-20",               # 生命周期日期
    "Location 空间范围": "中国 8 个城市（见 city 字段）",              # 空间覆盖
    "Methodology 方法": "配额抽样，线上/门店/电话三渠道，问卷 12 题",     # 生成方法
    "Data processing 处理": "缺失值标记；收入上限截尾 1%；未插补",       # 处理过程
    "Sources 来源": "课程合成样本（结构参考电商消费者调研）",            # 数据来源
    "File list 文件清单": "l4_survey_raw.csv, l4_data_dictionary.csv",   # 文件列表
    "File formats 格式": "CSV（UTF-8-SIG）",                        # 文件格式
    "File structure 结构": "一行 = 一位受访者（长表）",                 # 数据结构与粒度
    "Variable list 变量清单": "见 l4_data_dictionary.csv",           # 变量清单
    "Code lists 编码表": "satisfaction 1—5；channel={线上,门店,电话}",  # 编码说明
    "Versions 版本": "v1.0（2026-03-20）",                          # 版本与时间戳
    "Checksums 校验和": "SHA-256 见发布包 README",                   # 校验和
}                                                                 # 20 项文档要素字典结束
status = []                                                        # 逐项判定完成情况
for item, value in DOC_ITEMS.items():                              # 遍历 20 项
    filled = bool(str(value).strip()) and value != "-"             # 非空即视为"已填"
    weak = isinstance(value, str) and ("待补" in value or "无" == value.strip())   # 明确写"待补/无"视为薄弱
    status.append({"文档要素": item, "内容": value,                 # 记录要素与内容
                   "状态": "已填" if (filled and not weak) else ("薄弱" if filled else "缺失")})   # 三档状态
doc_status = pd.DataFrame(status)                                  # 汇总成表
dump(doc_status, "l4_docs_coverage.csv")                           # 落盘体检表
counts = doc_status["状态"].value_counts()                          # 统计三档数量
print("documentation coverage:", counts.to_dict())                 # 打印覆盖率概况

# ---------------------------------------------------------------- 图：文档完成度仪表盘
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), gridspec_kw={"width_ratios": [1, 1.5, 1.2]})   # 三格：状态计数 / 字段缺失率 / 字典规模
labels = ["已填", "薄弱", "缺失"]                                  # 三档状态的顺序
vals = [int(counts.get(k, 0)) for k in labels]                     # 各档数量
colors = ["#2E6F8E", "#C9A227", "#8A0C3C"]                         # 分别用蓝/金/红表示
axes[0].bar(labels, vals, color=colors, alpha=0.9)                 # 第一格：三档计数
style_axes(axes[0], "文档要素完成度", "状态", "要素个数")            # 统一风格
for i, v in enumerate(vals):                                       # 在柱顶标注数量
    axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=9.5)   # 数值标签
miss_rate = (survey.isna().mean() * 100).sort_values(ascending=False)   # 各列缺失率
mr = miss_rate[miss_rate > 0]                                      # 只画有缺失的列
axes[1].barh(mr.index[::-1], mr.values[::-1], color="#8A0C3C", alpha=0.85)   # 第二格：缺失率
style_axes(axes[1], "字段非空率（完整性）", "缺失率 %", "字段")        # 统一风格
annotated(axes[1], f"整体非空率 {survey.notna().mean().mean():.1%}", "lower right")   # 图内注释
axes[2].barh(["唯一值总数", "字段数"],                           # 第三格：字典规模
             [int(dictionary["唯一值"].sum()), len(dictionary)],   # 用条形展示量级
             color=["#C9A227", "#66717D"], alpha=0.9)              # 两种颜色
style_axes(axes[2], "数据字典规模", "数量", "")                     # 统一风格
plt.tight_layout()                                                 # 调整三个子图
save(fig, "fig_docs_coverage.png")                                 # 保存：文档与元数据一页用图
print("dictionary rows:", len(dictionary), "| survey shape:", survey.shape)   # 收尾日志
