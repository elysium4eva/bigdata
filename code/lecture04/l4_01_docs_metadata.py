"""l4_01_docs_metadata.py -- build a survey sample and audit documentation against the slide's 20-item list.

The slides argue that documentation starts **before** data collection, at least alongside the data.
The script writes the survey data first, then generates a field dictionary (codebook),
and finally scores all 20 documentation items and points out what is still missing.
"""
import numpy as np                                                 # random sampling
import pandas as pd                                                # table handling
import matplotlib.pyplot as plt                                    # documentation coverage figure
from _common_l4 import RNG, DAT, save, dump, style_axes, annotated   # reuse shared helpers and the annotation helper

N = 500                                                            # survey sample size
CITIES = ["深圳", "上海", "北京", "广州", "成都", "杭州",           # the cities; the weights sum to 1
          "武汉", "西安"]                                             # second line of the city name list
CITY_P = [0.26, 0.2, 0.18, 0.12, 0.08, 0.08, 0.05, 0.03]           # sampling proportion per city
CHANNELS = ["线上", "门店", "电话"]                                # outreach channels
EDU = ["高中及以下", "本科", "硕士及以上"]                          # education bands

survey = pd.DataFrame({                                            # assemble the survey sample
    "respondent_id": [f"R{i:04d}" for i in range(1, N + 1)],       # respondent id (primary key)
    "city": RNG.choice(CITIES, N, p=CITY_P),                       # city (categorical)
    "channel": RNG.choice(CHANNELS, N, p=[0.55, 0.3, 0.15]),       # outreach channel (categorical)
    "education": RNG.choice(EDU, N, p=[0.32, 0.55, 0.13]),         # education (ordinal)
    "age": RNG.integers(18, 66, N),                                # age (numeric)
    "income_k": np.round(RNG.lognormal(2.55, 0.5, N), 1),          # monthly income (thousand CNY, right-skewed)
    "satisfaction": RNG.choice([1, 2, 3, 4, 5], N, p=[0.05, 0.1, 0.25, 0.4, 0.2]),   # satisfaction 1-5
    "spend_k": np.round(RNG.lognormal(1.6, 0.7, N), 2),            # monthly spending (thousand CNY, right-skewed)
    "revisit": RNG.random(N) < 0.42,                               # will they buy again (boolean)
})                                                                # end of the survey sample definition
survey.loc[RNG.choice(N, 12, replace=False), "income_k"] = np.nan   # income is sensitive: create 12 missing values
survey.loc[RNG.choice(N, 5, replace=False), "satisfaction"] = np.nan   # satisfaction also has a few missing values
dump(survey, "l4_survey_raw.csv")                                  # write the sample

# ---------------------------------------------------------------- automatic field dictionary (codebook)
def field_dictionary(df, units):                                   # build one entry per column
    """Return the field dictionary: name, type, non-null rate, unique values, range/examples, unit."""   # function contract
    rows = []                                                      # one row per field
    for col in df.columns:                                         # loop over all columns
        s = df[col]                                                # take that column
        if pd.api.types.is_numeric_dtype(s):                       # numeric: give the range and quantiles
            rng_txt = f"{s.min():.2f} ~ {s.max():.2f}"             # minimum to maximum
        elif pd.api.types.is_bool_dtype(s):                        # boolean: give the true share
            rng_txt = f"True 占比 {s.mean():.1%}"                   # share of True
        else:                                                      # categorical: give the value count and examples
            rng_txt = f"{s.nunique()} 类，如 {', '.join(map(str, s.dropna().unique()[:3]))}"   # the first three values
        rows.append({"字段": col, "类型": str(s.dtype),             # column name and dtype
                     "非空率": f"{s.notna().mean():.1%}",           # non-null rate (completeness)
                     "唯一值": int(s.nunique()),                    # number of unique values
                     "取值范围/样例": rng_txt,                      # value domain description
                     "单位": units.get(col, "-")})                  # unit (when known)
    return pd.DataFrame(rows)                                      # collect into a table
UNITS = {"income_k": "千元/月", "spend_k": "千元/月", "age": "岁",   # units of the key fields
         "satisfaction": "1—5 分", "revisit": "0/1"}                # notes for the score and boolean fields
dictionary = field_dictionary(survey, UNITS)                       # generate the field dictionary
dump(dictionary, "l4_data_dictionary.csv")                         # write it as CSV
md = ["# 数据字典 Data Dictionary（l4_survey_raw.csv）", "",        # also write Markdown for the report
      "| 字段 | 类型 | 非空率 | 唯一值 | 取值范围/样例 | 单位 |",    # header row
      "| --- | --- | --- | --- | --- | --- |"]                     # separator row
for _, r in dictionary.iterrows():                                 # build the table row by row
    md.append(f"| {r['字段']} | {r['类型']} | {r['非空率']} | {r['唯一值']} | {r['取值范围/样例']} | {r['单位']} |")   # one Markdown row
(DAT / "l4_data_dictionary.md").write_text("\n".join(md), encoding="utf-8")   # write the Markdown file
print("[data]", DAT / "l4_data_dictionary.md")                     # print the output path

# ---------------------------------------------------------------- audit documentation against the slide's 20 items
DOC_ITEMS = {                                                      # 20 documentation items -> status in this project
    "Title 标题": "l4_survey_raw.csv 消费者满意度和复购调研（2026 春）",   # title
    "Creator 创建者": "大数据分析课程组（教学用合成数据）",              # creator
    "Identifier 标识符": "SZU-BDA-2026-L4-SURVEY-01",              # internal identifier
    "Subject 主题词": "消费者满意度, 复购意愿, 渠道差异",                # keywords
    "Funders 资助方": "无（课程自建）",                              # funding information
    "Rights 权利": "教学使用；不含真实个人信息",                       # intellectual property
    "Access 获取方式": "课程仓库 website/data/lecture04/",           # access path
    "Language 语言": "中文（字段名）",                              # content language
    "Dates 日期": "采集期 2026-03；发布于 2026-03-20",               # lifecycle dates
    "Location 空间范围": "中国 8 个城市（见 city 字段）",              # spatial coverage
    "Methodology 方法": "配额抽样，线上/门店/电话三渠道，问卷 12 题",     # generation method
    "Data processing 处理": "缺失值标记；收入上限截尾 1%；未插补",       # processing steps
    "Sources 来源": "课程合成样本（结构参考电商消费者调研）",            # data source
    "File list 文件清单": "l4_survey_raw.csv, l4_data_dictionary.csv",   # file list
    "File formats 格式": "CSV（UTF-8-SIG）",                        # file format
    "File structure 结构": "一行 = 一位受访者（长表）",                 # data structure and granularity
    "Variable list 变量清单": "见 l4_data_dictionary.csv",           # variable list
    "Code lists 编码表": "satisfaction 1—5；channel={线上,门店,电话}",  # code lists
    "Versions 版本": "v1.0（2026-03-20）",                          # version and timestamp
    "Checksums 校验和": "SHA-256 见发布包 README",                   # checksums
}                                                                 # end of the 20-item documentation dict
status = []                                                        # judge each item
for item, value in DOC_ITEMS.items():                              # loop over the 20 items
    filled = bool(str(value).strip()) and value != "-"             # non-empty counts as "present"
    weak = isinstance(value, str) and ("待补" in value or "无" == value.strip())   # explicit "to be added / none" counts as weak
    status.append({"文档要素": item, "内容": value,                 # record the item and its content
                   "状态": "已填" if (filled and not weak) else ("薄弱" if filled else "缺失")})   # three-level status
doc_status = pd.DataFrame(status)                                  # collect into a table
dump(doc_status, "l4_docs_coverage.csv")                           # write the coverage table
counts = doc_status["状态"].value_counts()                          # count the three levels
print("documentation coverage:", counts.to_dict())                 # print the coverage overview

# ---------------------------------------------------------------- figure: documentation coverage dashboard
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), gridspec_kw={"width_ratios": [1, 1.5, 1.2]})   # three panels: status counts / missing rate / dictionary size
labels = ["已填", "薄弱", "缺失"]                                  # order of the three statuses
vals = [int(counts.get(k, 0)) for k in labels]                     # count per status
colors = ["#2E6F8E", "#C9A227", "#8A0C3C"]                         # blue, gold and red respectively
axes[0].bar(labels, vals, color=colors, alpha=0.9)                 # first panel: counts of the three levels
style_axes(axes[0], "文档要素完成度", "状态", "要素个数")            # shared style
for i, v in enumerate(vals):                                       # annotate the count on each bar
    axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=9.5)   # value label
miss_rate = (survey.isna().mean() * 100).sort_values(ascending=False)   # missing rate per column
mr = miss_rate[miss_rate > 0]                                      # plot only columns with missing values
axes[1].barh(mr.index[::-1], mr.values[::-1], color="#8A0C3C", alpha=0.85)   # second panel: missing rate
style_axes(axes[1], "字段非空率（完整性）", "缺失率 %", "字段")        # shared style
annotated(axes[1], f"整体非空率 {survey.notna().mean().mean():.1%}", "lower right")   # in-figure note
axes[2].barh(["唯一值总数", "字段数"],                           # third panel: dictionary size
             [int(dictionary["唯一值"].sum()), len(dictionary)],   # show the magnitude with bars
             color=["#C9A227", "#66717D"], alpha=0.9)              # two colors
style_axes(axes[2], "数据字典规模", "数量", "")                     # shared style
plt.tight_layout()                                                 # adjust the three subplots
save(fig, "fig_docs_coverage.png")                                 # save: documentation and metadata one-pager
print("dictionary rows:", len(dictionary), "| survey shape:", survey.shape)   # closing log
